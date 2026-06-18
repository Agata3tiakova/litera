import os
import time

from PIL import Image

from app.services.ocr.base import OCRResult

_models = {}


class Florence2Provider:
    name = "florence2"

    def recognize(self, image_path: str, config: dict) -> OCRResult:
        started_at = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Florence-2 dependencies are not installed. Install optional OCR dependencies "
                "from requirements-ocr.txt."
            ) from exc

        model_name = config.get("FLORENCE2_MODEL") or os.getenv(
            "FLORENCE2_MODEL",
            "microsoft/Florence-2-base",
        )
        task_prompt = config.get("FLORENCE2_TASK_PROMPT") or os.getenv("FLORENCE2_TASK_PROMPT", "<OCR>")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor, model = _get_model(model_name, device)

        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=task_prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=int(config.get("FLORENCE2_MAX_TOKENS") or os.getenv("FLORENCE2_MAX_TOKENS", "1024")),
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        text = parsed.get(task_prompt, generated_text)

        return OCRResult(
            provider=self.name,
            text=str(text).strip(),
            elapsed_seconds=time.perf_counter() - started_at,
            metadata={"model": model_name, "task_prompt": task_prompt, "device": device},
        )


def _get_model(model_name: str, device: str):
    key = (model_name, device)
    if key not in _models:
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        _models[key] = (processor, model)
    return _models[key]
