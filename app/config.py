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
    VISION_API_KEY = os.getenv("VISION_API_KEY")
    VISION_API_URL = os.getenv("VISION_API_URL")
    LLM_MODEL = os.getenv("LLM_MODEL")
    SA_KEY_PATH = os.getenv("SA_KEY_PATH", "sa-key.json")
