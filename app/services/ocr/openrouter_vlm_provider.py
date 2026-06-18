import base64
import mimetypes
import os
import time

import requests

from app.services.ocr.base import OCRResult

DEFAULT_PROMPT = (
    "Transcribe this Russian handwritten text exactly. "
    "Preserve line breaks. Return only the transcribed text."
)


class OpenRouterVLMProvider:
    name = "openrouter_vlm"
    model_config_key = "OPENROUTER_VLM_MODEL"
    default_model = "qwen/qwen2.5-vl-7b-instruct"

    def recognize(self, image_path: str, config: dict) -> OCRResult:
        started_at = time.perf_counter()
        api_key = config.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        base_url = (
            config.get("OPENROUTER_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        model = config.get(self.model_config_key) or os.getenv(self.model_config_key) or self.default_model
        prompt = config.get("VLM_OCR_PROMPT") or os.getenv("VLM_OCR_PROMPT") or DEFAULT_PROMPT

        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image_path)},
                    },
                ],
            }],
            "temperature": 0,
            "max_tokens": int(config.get("VLM_OCR_MAX_TOKENS") or os.getenv("VLM_OCR_MAX_TOKENS", "2048")),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]

        return OCRResult(
            provider=self.name,
            text=_clean_vlm_text(text),
            elapsed_seconds=time.perf_counter() - started_at,
            metadata={
                "model": model,
                "base_url": base_url,
                "usage": data.get("usage", {}),
            },
        )


class Qwen25VL7BProvider(OpenRouterVLMProvider):
    name = "qwen25_vl_7b"
    model_config_key = "QWEN25_VL_7B_MODEL"
    default_model = "qwen/qwen2.5-vl-7b-instruct"


class Qwen25VL32BProvider(OpenRouterVLMProvider):
    name = "qwen25_vl_32b"
    model_config_key = "QWEN25_VL_32B_MODEL"
    default_model = "qwen/qwen2.5-vl-32b-instruct"


class Qwen25VL72BProvider(OpenRouterVLMProvider):
    name = "qwen25_vl_72b"
    model_config_key = "QWEN25_VL_72B_MODEL"
    default_model = "qwen/qwen2.5-vl-72b-instruct"


class Gemma3VisionProvider(OpenRouterVLMProvider):
    name = "gemma3_vision"
    model_config_key = "GEMMA3_VISION_MODEL"
    default_model = "google/gemma-3-27b-it"


class InternVLProvider(OpenRouterVLMProvider):
    name = "internvl"
    model_config_key = "INTERNVL_MODEL"
    default_model = "opengvlab/internvl3-14b"


class MiniCPMVProvider(OpenRouterVLMProvider):
    name = "minicpm_v"
    model_config_key = "MINICPM_V_MODEL"
    default_model = "openbmb/minicpm-v-2.6"


def _image_data_url(image_path: str) -> str:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _clean_vlm_text(text: str) -> str:
    return text.strip().strip("`").strip()
