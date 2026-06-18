import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "sqlite:///literadb.sqlite",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv("FLASK_SECRET", "secret")
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    OCR_PROVIDER = os.getenv("OCR_PROVIDER", "yandex")
    VISION_API_KEY = os.getenv("VISION_API_KEY")
    VISION_API_URL = os.getenv("VISION_API_URL")
    LLM_MODEL = os.getenv("LLM_MODEL")
    SA_KEY_PATH = os.getenv("SA_KEY_PATH", "sa-key.json")

    TESSERACT_CMD = os.getenv("TESSERACT_CMD")
    TESSERACT_LANG = os.getenv("TESSERACT_LANG", "rus+eng")
    TESSERACT_PSM = os.getenv("TESSERACT_PSM", "6")

    EASYOCR_LANGS = os.getenv("EASYOCR_LANGS", "ru,en")
    EASYOCR_GPU = os.getenv("EASYOCR_GPU", "false")

    TROCR_MODEL = os.getenv("TROCR_MODEL", "microsoft/trocr-base-handwritten")

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    VLM_OCR_PROMPT = os.getenv(
        "VLM_OCR_PROMPT",
        "Transcribe this Russian handwritten text exactly. Preserve line breaks. Return only the transcribed text.",
    )
    VLM_OCR_MAX_TOKENS = os.getenv("VLM_OCR_MAX_TOKENS", "2048")

    QWEN25_VL_7B_MODEL = os.getenv("QWEN25_VL_7B_MODEL", "qwen/qwen2.5-vl-7b-instruct")
    QWEN25_VL_32B_MODEL = os.getenv("QWEN25_VL_32B_MODEL", "qwen/qwen2.5-vl-32b-instruct")
    QWEN25_VL_72B_MODEL = os.getenv("QWEN25_VL_72B_MODEL", "qwen/qwen2.5-vl-72b-instruct")
    GEMMA3_VISION_MODEL = os.getenv("GEMMA3_VISION_MODEL", "google/gemma-3-27b-it")
    INTERNVL_MODEL = os.getenv("INTERNVL_MODEL", "opengvlab/internvl3-14b")
    MINICPM_V_MODEL = os.getenv("MINICPM_V_MODEL", "openbmb/minicpm-v-2.6")

    FLORENCE2_MODEL = os.getenv("FLORENCE2_MODEL", "microsoft/Florence-2-base")
    FLORENCE2_TASK_PROMPT = os.getenv("FLORENCE2_TASK_PROMPT", "<OCR>")
    FLORENCE2_MAX_TOKENS = os.getenv("FLORENCE2_MAX_TOKENS", "1024")
