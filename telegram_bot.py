import os
import requests
from dotenv import load_dotenv
from gradio_client import Client
from PIL import Image
import logging
from datetime import datetime

# === Настройки логирования ===
logging.basicConfig(
    filename='combined_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - IMAGE_PROMPT: %(message)s'
)

# === Загрузка окружения ===
load_dotenv()

class Config:
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')
    HF_TOKEN = os.getenv("HF_TOKEN")
    MODEL = os.getenv('MODEL', "qwen/qwen3-14b:free")
    IMAGE_MODEL = os.getenv('IMAGE_MODEL', "black-forest-labs/FLUX.1-schnell")

# === Настройки ===
INPUT_FILENAME = "input.txt"
OUTPUT_DIR = "images"
IMAGE_PROMPTS_FILE = "image_prompts.txt"
BACKUP_IMAGE_MODELS = [
    "black-forest-labs/FLUX.1-dev",
    "multimodalart/FLUX.1-merged",
    "runwayml/stable-diffusion-v1-5"
]

def setup():
    required_vars = ['OPENROUTER_API_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHANNEL_ID', 'HF_TOKEN']
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"❌ Не задана переменная {var} в .env файле")

def format_for_telegram(text):
    if not text:
        return ""
    text = text.replace("```markdown", "").replace("```", "")
    text = text.replace("*", "").replace("_", "").replace("`", "")
    return text.strip()

def read_queries(file_path=INPUT_FILENAME):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        logging.error(f"Ошибка чтения файла: {e}")
        return []

def remove_first_prompt():
    try:
        with open(INPUT_FILENAME, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if not lines:
            logging.info("Файл промптов пуст")
            return True
        
        with open(INPUT_FILENAME, "w", encoding="utf-8") as f:
            f.writelines(lines[1:])
        
        logging.info(f"Удален первый промпт. Осталось {len(lines)-1} строк.")
        return True
    
    except Exception as e:
        logging.error(f"Ошибка при обновлении файла: {e}")
        return False

def get_ai_response(query, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }

            data = {
                "model": Config.MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты - креативный контент-мейкер для Telegram канала. "
                            "Сгенерируй интересный, структурированный пост длиной 500–700 символов. "
                            "Добавь эмодзи, хэштеги и сделай его легко читаемым. "
                            "Не используй Markdown."
                        )
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.75
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content'].strip()
                return format_for_telegram(answer)

            else:
                logging.warning(f"Ошибка API при генерации поста (попытка {attempt}): {response.status_code}, {response.text}")

        except Exception as e:
            logging.warning(f"Ошибка запроса к ИИ при генерации поста (попытка {attempt}): {e}")

    logging.error("Не удалось сгенерировать пост за несколько попыток")
    return f"📌 {query}\n\nКонтент временно недоступен. Попробуйте позже."

def save_prompt_to_file(prompt):
    """Функция для гарантированного сохранения промта в файл"""
    try:
        os.makedirs(os.path.dirname(IMAGE_PROMPTS_FILE) or '.', exist_ok=True)
        with open(IMAGE_PROMPTS_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {prompt}\n\n")
        return True
    except Exception as e:
        logging.error(f"Критическая ошибка при сохранении промта: {e}")
        return False

def generate_image_prompt(post_text, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }

            user_content = (
                "Проанализируйте следующий текст из поста в Telegram и составьте подробный и ёмкий промт на английском языке "
                "для создания реалистичного изображения, которое идеально передаст его смысл и эмоциональную атмосферу. "
                "Учитывайте стиль подачи информации, ключевые темы, настроение и возможную целевую аудиторию.\n\n"
                f"{post_text}\n\n"
                "На изображении не должно быть, текста, надписей, букв. Изображение должно быть реалисчичным и современным, привлекать внимание и подходить для использования в социальных сетях."
            )

            data = {
                "model": Config.MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert prompt engineer for AI image generation. "
                            "Create detailed, vivid prompts in English that perfectly capture "
                            "the essence of the provided text. Focus on visual storytelling."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.5
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    prompt = result['choices'][0]['message']['content'].strip()
                    if prompt:
                        prompt = prompt.replace('"', '').replace("'", "")
                        logging.info(f"Generated image prompt: {prompt}")
                        save_prompt_to_file(prompt)
                        return prompt

            logging.warning(f"Ошибка API при генерации промта (попытка {attempt}): {response.status_code}, {response.text}")

        except Exception as e:
            logging.warning(f"Ошибка при генерации промта (попытка {attempt}): {e}")

    default_prompt = (
        "modern digital art, trending on artstation, ultra-detailed, "
        "vibrant colors, professional composition, social media style, "
        "eye-catching visual, 8k resolution"
    )
    logging.info(f"Using default prompt: {default_prompt}")
    save_prompt_to_file(default_prompt)
    return default_prompt

def get_working_image_model():
    model_list = [Config.IMAGE_MODEL] if Config.IMAGE_MODEL else []
    model_list.extend(BACKUP_IMAGE_MODELS)
    
    for model in model_list:
        try:
            client = Client(model, hf_token=Config.HF_TOKEN)
            return client, model
        except Exception as e:
            continue
    
    raise ValueError("Ни одна из моделей изображений не доступна!")

def generate_image(client, prompt, model_name):
    try:
        if "FLUX.1-schnell" in model_name:
            result = client.predict(
                prompt=prompt,
                api_name="/infer"
            )
        else:
            result = client.predict(
                prompt=prompt,
                seed=0,
                width=1024,
                height=1024,
                guidance_scale=3.5,
                num_inference_steps=28,
                api_name="/infer"
            )
        
        return result[0]
    except Exception as e:
        logging.error(f"Ошибка генерации изображения: {e}")
        return None

def save_image(temp_path, prompt):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    safe_prompt = "".join(
        c for c in prompt[:30] 
        if c.isalnum() or c in " _-"
    ).rstrip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{safe_prompt}.png")
    
    try:
        with Image.open(temp_path) as img:
            img.save(output_path, "PNG", quality=95)
        return output_path
    except Exception as e:
        logging.error(f"Ошибка сохранения изображения: {e}")
        return None

def send_post_with_image_to_telegram(text, image_path):
    try:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(image_path, "rb") as photo:
            files = {"photo": photo}
            data = {"chat_id": Config.TELEGRAM_CHANNEL_ID, "caption": text[:1024]}
            response = requests.post(url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                return True
            else:
                logging.error(f"Ошибка Telegram API: {response.status_code}, {response.text}")
                return False
    except Exception as e:
        logging.error(f"Ошибка отправки в Telegram: {e}")
        return False

def main():
    try:
        setup()
    except ValueError as e:
        logging.error(f"Ошибка конфигурации: {e}")
        return

    queries = read_queries()
    if not queries:
        logging.error("В файле input.txt нет вопросов")
        return

    query = queries[0]

    # 1. Генерация поста
    post_text = get_ai_response(query)
    if not post_text:
        return

    # 2. Генерация промта для изображения
    image_prompt = generate_image_prompt(post_text)

    # 3. Генерация изображения
    try:
        client, model_name = get_working_image_model()
        temp_image_path = generate_image(client, image_prompt, model_name)
        if not temp_image_path:
            return

        final_image_path = save_image(temp_image_path, image_prompt)
        if not final_image_path:
            return

        # 4. Отправка в Telegram
        if send_post_with_image_to_telegram(post_text, final_image_path):
            remove_first_prompt()

    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    main()