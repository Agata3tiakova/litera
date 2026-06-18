import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config
from app.services.image_processor import preprocess_image
from app.services.ocr import run_ocr

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR providers on labeled images.")
    parser.add_argument("--images", required=True, help="Directory with images.")
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Directory with .txt files matching image stems.",
    )
    parser.add_argument(
        "--providers",
        default=(
            "yandex,tesseract,easyocr,trocr,"
            "qwen25_vl_7b,qwen25_vl_32b,qwen25_vl_72b,gemma3_vision,"
            "internvl,minicpm_v,florence2"
        ),
        help="Comma-separated OCR providers.",
    )
    parser.add_argument("--output", default="ocr_benchmark_results.csv")
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Do not preprocess a temporary image copy before OCR.",
    )
    args = parser.parse_args()

    load_dotenv()
    config = {
        key: getattr(Config, key)
        for key in dir(Config)
        if key.isupper()
    }
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    rows = []
    for image_path in iter_images(Path(args.images)):
        gt_path = Path(args.ground_truth) / f"{image_path.stem}.txt"
        if not gt_path.exists():
            print(f"skip {image_path.name}: missing {gt_path.name}")
            continue

        expected = gt_path.read_text(encoding="utf-8").strip()
        candidate_path = image_path

        if not args.keep_original:
            candidate_path = make_preprocessed_copy(image_path)

        for provider in providers:
            started_at = time.perf_counter()
            error = ""
            recognized = ""
            metadata = {}

            try:
                result = run_ocr(str(candidate_path), config, provider)
                recognized = result.text
                metadata = result.metadata
                elapsed = result.elapsed_seconds
            except Exception as exc:
                elapsed = time.perf_counter() - started_at
                error = str(exc)

            rows.append({
                "image": image_path.name,
                "provider": provider,
                "cer": cer(expected, recognized) if not error else "",
                "wer": wer(expected, recognized) if not error else "",
                "elapsed_seconds": round(elapsed, 3),
                "expected_chars": len(expected),
                "recognized_chars": len(recognized),
                "error": error,
                "recognized_text": recognized,
                "metadata": json.dumps(metadata, ensure_ascii=False),
            })

    write_csv(Path(args.output), rows)
    print(f"wrote {args.output}")


def iter_images(directory: Path):
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def make_preprocessed_copy(image_path: Path) -> Path:
    output_dir = Path("tmp") / "ocr_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / image_path.name
    output_path.write_bytes(image_path.read_bytes())
    preprocess_image(str(output_path))
    return output_path


def cer(expected: str, actual: str) -> float:
    if not expected:
        return 0.0 if not actual else 1.0
    return levenshtein(expected, actual) / len(expected)


def wer(expected: str, actual: str) -> float:
    expected_words = expected.split()
    actual_words = actual.split()
    if not expected_words:
        return 0.0 if not actual_words else 1.0
    return levenshtein(expected_words, actual_words) / len(expected_words)


def levenshtein(left, right) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_item != right_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def write_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "image",
        "provider",
        "cer",
        "wer",
        "elapsed_seconds",
        "expected_chars",
        "recognized_chars",
        "error",
        "recognized_text",
        "metadata",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
