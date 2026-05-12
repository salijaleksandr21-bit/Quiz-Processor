import os
import time
import json
import base64
import io
import requests
from PIL import Image
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ Не найден OPENROUTER_API_KEY в переменных окружения")

IMAGE_FOLDER = "путь_к_папке"
OUTPUT_FILE = "quiz_results"
SLEEP_BETWEEN_IMAGES = 3          # пауза после каждого запроса
MAX_RETRIES = 3

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

VISION_MODEL = "qwen/qwen3-vl-8b-instruct"          # чтение текста с фото
TEXT_MODEL   = "meta-llama/llama-3.1-8b-instruct"    # поиск ответа

def call_openrouter(messages, model, temperature=0.2):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1500      # увеличено, чтобы JSON не обрезался
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                if content and content.strip():
                    return content.strip()
                print("⚠️ Пустой ответ от модели")
            elif resp.status_code == 429:
                print("⚠️ Лимит запросов (429). Пауза 30 секунд...")
                time.sleep(30)
            else:
                print(f"⚠️ Ошибка {resp.status_code}: {resp.text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        except Exception as e:
            print(f"⚠️ Исключение: {e}. Повтор через {2**attempt} сек...")
            time.sleep(2 ** attempt)
    return None

def extract_full_text_vision(image_path):
    with Image.open(image_path) as img:
        img.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Извлеки **весь текст** с этого изображения викторины, "
                    "включая вопрос и все варианты ответов. "
                    "Верни ТОЛЬКО текст, сохраняя исходную нумерацию или буквенные метки вариантов. "
                    "Никаких приветствий, пояснений или перевода."
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            }
        ]
    }]
    return call_openrouter(messages, VISION_MODEL, temperature=0.1)

def parse_json_from_text(text):
    """Извлекает словарь из строки с JSON, учитывая обёртки и кавычки."""
    json_str = None

    # 1. Ищем внутри ```json ... ```
    if "```json" in text:
        parts = text.split("```json", 1)
        if len(parts) > 1:
            after = parts[1]
            if "```" in after:
                json_str = after.split("```", 1)[0].strip()

    # 2. Если нет, берём первую { и последнюю }
    if not json_str:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start:end + 1].strip()

    # 3. Если JSON обёрнут в кавычки (строка), раскрываем
    if json_str and json_str.startswith('"') and json_str.endswith('"'):
        try:
            decoded = json.loads(json_str)
            if isinstance(decoded, str):
                return parse_json_from_text(decoded)  # рекурсивно
            elif isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    return None

def get_answer_from_quiz_text(full_text: str):
    prompt = f"""
Ты — эксперт по викторинам. Дан текст, извлечённый с фотографии викторины.
Он содержит вопрос и варианты ответов (с буквенными или цифровыми метками, либо без).

Игнорируй строки-заголовки вроде «Вопрос: 1», «Задание 2».
Выбери **правильный ответ** ИЗ ПРЕДЛОЖЕННЫХ ВАРИАНТОВ, не придумывай свой.
Выдели точный текст вопроса.

Ответь ТОЛЬКО кодом JSON внутри маркера ```json ... ```, без пояснений до или после.
JSON строго такого вида:
```json
{{
    "question": "текст самого вопроса",
    "answer": "текст правильного варианта (как он записан в тексте, без добавления букв)",
    "justification": "краткое обоснование на русском",
    "confidence": 0.95
}}
Текст с изображения:
{full_text}
"""
    result = call_openrouter([{"role": "user", "content": prompt}], TEXT_MODEL, temperature=0.2)

    if not result:
        return (
            full_text.strip().split("\n")[0] if full_text else "Ошибка",
            "Не удалось получить ответ от API",
            "API вернул пустой ответ",
            0.0,
        )

    # Пытаемся извлечь данные
    data = parse_json_from_text(result)
    if data:
        return (
            data.get("question", full_text.strip().split("\n")[0]),
            data.get("answer", "Не найден"),
            data.get("justification", "Нет обоснования"),
            data.get("confidence", 0.0),
        )

    # Если не вышло, пробуем распарсить сырой ответ как JSON (fallback)
    fallback_data = parse_json_from_text(result)
    if fallback_data:
        return (
            fallback_data.get("question", full_text.strip().split("\n")[0]),
            fallback_data.get("answer", "Не найден"),
            fallback_data.get("justification", "Нет обоснования"),
            fallback_data.get("confidence", 0.0),
        )

    # Совсем крайний случай – отдаём сырой текст модели
    lines = [l.strip() for l in full_text.strip().split("\n") if l.strip()]
    question = lines[0] if lines else full_text
    return question, "Ошибка парсинга JSON", result[:500], 0.0

def process_all_images():
    image_files = sorted([
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not image_files:
        print(f"❌ Нет изображений в папке: {IMAGE_FOLDER}")
        return

    results = []
    for idx, img_file in enumerate(tqdm(image_files, desc="Обработка"), 1):
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        print(f"\n📸 [{idx}/{len(image_files)}] {img_file}")

        # Шаг 1: извлечение текста с фото
        full_text = extract_full_text_vision(img_path)
        time.sleep(SLEEP_BETWEEN_IMAGES)

        if not full_text or len(full_text) < 10:
            question = "ОШИБКА: не удалось извлечь текст"
            answer, justification, confidence = "Невозможно ответить", "Текст не распознан", 0.0
        else:
            print("   🤖 Ищем ответ...")
            question, answer, justification, confidence = get_answer_from_quiz_text(full_text)
            print(f"   🔍 Вопрос: {question[:80]}...")
            time.sleep(SLEEP_BETWEEN_IMAGES)

        results.append({
            "image_name": img_file,
            "question_text": question,
            "answer": answer,
            "justification": justification,
            "confidence": confidence
        })

        # Сохраняем прогресс после КАЖДОГО изображения
        try:
            pd.DataFrame(results).to_csv(
                f"{OUTPUT_FILE}_progress.csv", index=False, encoding='utf-8-sig'
            )
        except Exception as e:
            print(f"⚠️ Не удалось сохранить прогресс: {e}")

    # Финальный Excel
    df = pd.DataFrame(results)
    df.insert(0, 'question_number', range(1, len(df)+1))
    df[['question_number', 'question_text', 'answer', 'justification', 'confidence']].to_excel(
        f"{OUTPUT_FILE}.xlsx", index=False
    )
    print(f"\n✅ Готово! Результат сохранен в {OUTPUT_FILE}.xlsx")

if __name__ == "__main__":
    process_all_images()