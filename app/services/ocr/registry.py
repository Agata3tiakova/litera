from app.services.ocr.base import OCRProvider, OCRResult
from app.services.ocr.azure_vision_provider import AzureVisionProvider
from app.services.ocr.easyocr_provider import EasyOCRProvider
from app.services.ocr.florence2_provider import Florence2Provider
from app.services.ocr.google_vision_provider import GoogleVisionProvider
from app.services.ocr.openrouter_vlm_provider import (
    Gemma3VisionProvider,
    InternVLProvider,
    MiniCPMVProvider,
    Qwen25VL7BProvider,
    Qwen25VL32BProvider,
    Qwen25VL72BProvider,
)
from app.services.ocr.paddleocr_provider import PaddleOCRProvider
from app.services.ocr.tesseract_provider import TesseractProvider
from app.services.ocr.trocr_provider import TrOCRProvider
from app.services.ocr.yandex_provider import YandexVisionProvider

PROVIDERS = {
    "azure_vision": AzureVisionProvider,
    "google_vision": GoogleVisionProvider,
    "yandex": YandexVisionProvider,
    "tesseract": TesseractProvider,
    "easyocr": EasyOCRProvider,
    "trocr": TrOCRProvider,
    "paddleocr": PaddleOCRProvider,
    "qwen25_vl_7b": Qwen25VL7BProvider,
    "qwen25_vl_32b": Qwen25VL32BProvider,
    "qwen25_vl_72b": Qwen25VL72BProvider,
    "gemma3_vision": Gemma3VisionProvider,
    "internvl": InternVLProvider,
    "minicpm_v": MiniCPMVProvider,
    "florence2": Florence2Provider,
}


def list_ocr_providers() -> list[str]:
    return sorted(PROVIDERS)


def get_ocr_provider(name: str) -> OCRProvider:
    provider_name = (name or "yandex").lower()
    provider_class = PROVIDERS.get(provider_name)
    if provider_class is None:
        available = ", ".join(list_ocr_providers())
        raise ValueError(f"Unknown OCR provider '{name}'. Available: {available}")
    return provider_class()


def run_ocr(image_path: str, config: dict, provider_name: str | None = None) -> OCRResult:
    selected_provider = provider_name or config.get("OCR_PROVIDER") or "yandex"
    provider = get_ocr_provider(selected_provider)
    return provider.recognize(image_path, config)
