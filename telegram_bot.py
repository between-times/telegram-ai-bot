import os
import requests
from dotenv import load_dotenv
from gradio_client import Client
from PIL import Image
import logging
from datetime import datetime
import sys

# === Настройки логирования ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('combined_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
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

def setup_environment():
    """Проверка и создание необходимых файлов и директорий"""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if not os.path.exists(INPUT_FILENAME):
            with open(INPUT_FILENAME, 'w', encoding='utf-8') as f:
                f.write("Пример промпта для теста\n")
                logging.info(f"Создан новый файл {INPUT_FILENAME}")
                
        return True
    except Exception as e:
        logging.error(f"Ошибка при настройке окружения: {e}")
        return False

def setup():
    """Проверка обязательных переменных окружения"""
    required_vars = ['OPENROUTER_API_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHANNEL_ID', 'HF_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    logging.info("Все необходимые переменные окружения установлены")

def format_for_telegram(text):
    """Форматирование текста для Telegram"""
    if not text:
        return ""
    
    replacements = {
        "```markdown": "",
        "```": "",
        "*": "",
        "_": "",
        "`": "",
        "\\[\\!\\[\\]": ""  # Удаление markdown изображений
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip()

def read_queries(file_path=INPUT_FILENAME):
    """Чтение промптов из файла"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            queries = [line.strip() for line in file if line.strip()]
            
        if not queries:
            logging.warning(f"Файл {file_path} пуст")
            return []
            
        logging.info(f"Прочитано {len(queries)} промптов из файла")
        return queries
        
    except Exception as e:
        logging.error(f"Ошибка чтения файла {file_path}: {e}")
        return []

def remove_first_prompt():
    """Удаление обработанного промпта из файла"""
    try:
        with open(INPUT_FILENAME, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if not lines:
            logging.info("Файл промптов пуст - нечего удалять")
            return False
        
        with open(INPUT_FILENAME, "w", encoding="utf-8") as f:
            f.writelines(lines[1:])
        
        remaining = len(lines) - 1
        logging.info(f"Удален первый промпт. Осталось: {remaining}")
        return True
    
    except Exception as e:
        logging.error(f"Ошибка при обновлении файла промптов: {e}")
        return False

def get_ai_response(query, max_retries=3):
    """Получение ответа от ИИ через OpenRouter"""
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
                logging.info(f"Успешно получен ответ от ИИ (попытка {attempt})")
                return format_for_telegram(answer)

            logging.warning(f"Ошибка API (попытка {attempt}): {response.status_code} - {response.text}")
            
        except Exception as e:
            logging.warning(f"Ошибка запроса к ИИ (попытка {attempt}): {e}")

    logging.error("Не удалось сгенерировать пост после нескольких попыток")
    return f"📌 {query}\n\nКонтент временно недоступен. Попробуйте позже."

def save_prompt_to_file(prompt):
    """Сохранение промта для изображения в файл"""
    try:
        os.makedirs(os.path.dirname(IMAGE_PROMPTS_FILE) or '.', exist_ok=True)
        with open(IMAGE_PROMPTS_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {prompt}\n\n")
        logging.info("Промт для изображения сохранен в файл")
        return True
    except Exception as e:
        logging.error(f"Ошибка сохранения промта: {e}")
        return False

def generate_image_prompt(post_text, max_retries=3):
    """Генерация промта для изображения"""
    for attempt in range(1, max_retries + 1):
        try:
            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }

            user_content = (
                "Проанализируйте текст и составьте подробный промт на английском языке "
                "для создания реалистичного изображения, которое передаст его смысл и атмосферу.\n\n"
                f"{post_text}\n\n"
                "Изображение должно быть реалистичным, без текста и надписей, привлекать внимание."
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
                        logging.info(f"Сгенерирован промт для изображения: {prompt[:100]}...")
                        save_prompt_to_file(prompt)
                        return prompt

            logging.warning(f"Ошибка API при генерации промта (попытка {attempt}): {response.status_code}")

        except Exception as e:
            logging.warning(f"Ошибка при генерации промта (попытка {attempt}): {e}")

    default_prompt = (
        "modern digital art, trending on artstation, ultra-detailed, "
        "vibrant colors, professional composition, social media style, "
        "eye-catching visual, 8k resolution"
    )
    logging.info(f"Используется стандартный промт: {default_prompt}")
    save_prompt_to_file(default_prompt)
    return default_prompt

def get_working_image_model():
    """Поиск рабочей модели для генерации изображений"""
    model_list = [Config.IMAGE_MODEL] if Config.IMAGE_MODEL else []
    model_list.extend(BACKUP_IMAGE_MODELS)
    
    for model in model_list:
        try:
            logging.info(f"Попытка подключения к модели: {model}")
            client = Client(model, hf_token=Config.HF_TOKEN)
            logging.info(f"Успешное подключение к модели: {model}")
            return client, model
        except Exception as e:
            logging.warning(f"Ошибка подключения к модели {model}: {e}")
            continue
    
    raise ValueError("Ни одна из моделей изображений не доступна!")

def generate_image(client, prompt, model_name):
    """Генерация изображения по промту"""
    try:
        logging.info(f"Генерация изображения с моделью: {model_name}")
        
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
        
        if result and len(result) > 0:
            logging.info("Изображение успешно сгенерировано")
            return result[0]
            
        logging.error("Пустой результат генерации изображения")
        return None
        
    except Exception as e:
        logging.error(f"Ошибка генерации изображения: {e}")
        return None

def save_image(temp_path, prompt):
    """Сохранение изображения на диск"""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        safe_prompt = "".join(
            c for c in prompt[:30] 
            if c.isalnum() or c in " _-"
        ).rstrip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{safe_prompt}.png")
        
        with Image.open(temp_path) as img:
            img.save(output_path, "PNG", quality=95)
        
        logging.info(f"Изображение сохранено: {output_path}")
        return output_path
        
    except Exception as e:
        logging.error(f"Ошибка сохранения изображения: {e}")
        return None

def send_post_with_image_to_telegram(text, image_path):
    """Отправка поста с изображением в Telegram"""
    try:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendPhoto"
        
        with open(image_path, "rb") as photo:
            files = {"photo": photo}
            data = {"chat_id": Config.TELEGRAM_CHANNEL_ID, "caption": text[:1024]}
            response = requests.post(url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                logging.info("Пост успешно отправлен в Telegram")
                return True
                
            logging.error(f"Ошибка Telegram API: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Ошибка отправки в Telegram: {e}")
        return False

def main():
    """Основная функция выполнения"""
    try:
        logging.info("=== Запуск бота ===")
        setup()
        setup_environment()
        
        queries = read_queries()
        if not queries:
            logging.error("Нет промптов для обработки")
            return
            
        query = queries[0]
        logging.info(f"Обработка промпта: {query[:50]}...")
        
        # 1. Генерация поста
        post_text = get_ai_response(query)
        if not post_text:
            logging.error("Не удалось сгенерировать текст поста")
            return
            
        logging.info(f"Сгенерирован текст поста ({len(post_text)} символов)")
        
        # 2. Генерация промта для изображения
        image_prompt = generate_image_prompt(post_text)
        if not image_prompt:
            logging.error("Не удалось сгенерировать промт для изображения")
            return
            
        # 3. Генерация изображения
        try:
            client, model_name = get_working_image_model()
            temp_image_path = generate_image(client, image_prompt, model_name)
            
            if not temp_image_path or not os.path.exists(temp_image_path):
                logging.error("Не удалось сгенерировать изображение")
                return
                
            final_image_path = save_image(temp_image_path, image_prompt)
            
            if not final_image_path:
                return
                
            # 4. Отправка в Telegram
            if send_post_with_image_to_telegram(post_text, final_image_path):
                remove_first_prompt()
                
        except Exception as e:
            logging.error(f"Ошибка в процессе генерации изображения: {e}", exc_info=True)
            
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}", exc_info=True)
    finally:
        logging.info("=== Завершение работы бота ===")

if __name__ == "__main__":
    main()