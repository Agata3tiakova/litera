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

    PADDLEOCR_LANG = os.getenv("PADDLEOCR_LANG", "cyrillic")
    PADDLEOCR_USE_ANGLE_CLS = os.getenv("PADDLEOCR_USE_ANGLE_CLS", "true")

    GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")

    AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
    AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
