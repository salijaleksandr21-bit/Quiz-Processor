import pandas as pd
import json
import re

def extract_fields_from_text(raw: str):
    """
    Извлекает question, answer, justification, confidence из сырого текста,
    даже если JSON невалидный или обрезан.
    """
    if not isinstance(raw, str):
        return None
    
    # Удаляем HTML-теги и переносы строк
    clean = re.sub(r'<[^>]+>', ' ', raw)
    clean = clean.replace('\n', ' ').replace('\r', ' ')
    
    data = {}
    
    # Пытаемся извлечь значения через регулярки (ищем "ключ": "значение")
    patterns = {
        "question": r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "answer": r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "justification": r'"justification"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "confidence": r'"confidence"\s*:\s*([\d.]+)'
    }
    
    for key, pat in patterns.items():
        match = re.search(pat, clean)
        if match:
            value = match.group(1)
            if key == 'confidence':
                try:
                    data[key] = float(value)
                except:
                    data[key] = 0.0
            else:
                # Убираем экранированные кавычки
                value = value.replace('\\"', '"')
                data[key] = value
    
    if "question" in data and "answer" in data:
        return data
    return None

# Загружаем файл
df = pd.read_excel("quiz_results.xlsx")

fixed_count = 0
for i, row in df.iterrows():
    answer = str(row["answer"])
    if "Ошибка парсинга" in answer or "JSON не найден" in answer:
        justification = row["justification"]
        print(f"\n--- Строка {i+2} ---")
        print(f"raw justification: {repr(justification[:200])}")  # первые 200 символов
        
        # Метод 1: чиним JSON (удаляем <br>, пробуем parse)
        data = None
        try:
            clean_json = re.sub(r'<[^>]+>', ' ', justification)
            # Ищем JSON в фигурных скобках
            start = clean_json.find('{')
            end = clean_json.rfind('}')
            if start != -1 and end != -1:
                json_str = clean_json[start:end+1]
                # Попытка восстановить обрезанный JSON
                if not json_str.strip().endswith('}'):
                    json_str += '}'
                if json_str.count('"') % 2 != 0:
                    json_str += '"'
                data = json.loads(json_str)
        except:
            pass
        
        # Метод 2: прямое извлечение полей
        if not data:
            data = extract_fields_from_text(justification)
        
        if data:
            df.at[i, "question_text"] = data.get("question", row["question_text"])
            ans = data.get("answer", row["answer"])
            # Убираем возможные префиксы типа "А) "
            ans = re.sub(r'^[А-Я]\)\s*', '', ans).strip()
            df.at[i, "answer"] = ans
            df.at[i, "justification"] = data.get("justification", row["justification"])
            df.at[i, "confidence"] = data.get("confidence", row["confidence"])
            fixed_count += 1
            print(f"✅ Исправлена. answer = {ans}")
        else:
            print("❌ Не удалось извлечь данные")

df.to_excel("quiz_results_fixed.xlsx", index=False)
print(f"\n✅ Исправлено строк: {fixed_count}. Результат сохранён в quiz_results_fixed.xlsx")