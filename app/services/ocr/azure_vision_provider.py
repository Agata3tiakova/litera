import os
import time

import requests

from app.services.ocr.base import OCRResult


class AzureVisionProvider:
    name = "azure_vision"

    def recognize(self, image_path: str, config: dict) -> OCRResult:
        started_at = time.perf_counter()
        endpoint = _clean_endpoint(
            config.get("AZURE_VISION_ENDPOINT") or os.getenv("AZURE_VISION_ENDPOINT")
        )
        key = config.get("AZURE_VISION_KEY") or os.getenv("AZURE_VISION_KEY")
        if not endpoint or not key:
            raise RuntimeError("AZURE_VISION_ENDPOINT or AZURE_VISION_KEY is not set")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        analyze_response = requests.post(
            f"{endpoint}/vision/v3.2/read/analyze",
            params={"language": "ru"},
            headers={
                "Ocp-Apim-Subscription-Key": key,
                "Content-Type": "application/octet-stream",
            },
            data=image_bytes,
            timeout=60,
        )
        analyze_response.raise_for_status()
        operation_url = analyze_response.headers.get("Operation-Location")
        if not operation_url:
            raise RuntimeError("Azure Vision response did not include Operation-Location")

        data = _poll_result(operation_url, key)
        lines = []
        for page in data.get("analyzeResult", {}).get("readResults", []):
            for line in page.get("lines", []):
                lines.append(line.get("text", ""))

        return OCRResult(
            provider=self.name,
            text="\n".join(lines).strip(),
            elapsed_seconds=time.perf_counter() - started_at,
            metadata={"status": data.get("status"), "pages": len(data.get("analyzeResult", {}).get("readResults", []))},
        )


def _poll_result(operation_url: str, key: str) -> dict:
    for _ in range(30):
        response = requests.get(
            operation_url,
            headers={"Ocp-Apim-Subscription-Key": key},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        if status in {"succeeded", "failed"}:
            if status == "failed":
                raise RuntimeError(f"Azure Vision OCR failed: {data}")
            return data
        time.sleep(1)
    raise TimeoutError("Azure Vision OCR timed out")


def _clean_endpoint(value: str | None) -> str | None:
    if not value:
        return None
    return value.rstrip("/")
