import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
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
from app.services.image_variants import DEFAULT_VARIANTS, create_image_variants
from app.services.ocr_error_analysis import analyze_ocr_errors, normalize_for_text_comparison
from app.services.ocr import run_ocr

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROMPT_VARIANTS = {
    "default": (
        "Transcribe this Russian handwritten text exactly. "
        "Preserve line breaks. Return only the transcribed text."
    ),
    "exact_no_correction": (
        "Transcribe the Russian handwritten text exactly as written. "
        "Do not correct spelling, grammar, punctuation, or word choice. "
        "Preserve line breaks and hyphenation. Return only the transcribed text."
    ),
    "school_notebook": (
        "This is a Russian school notebook page with handwritten text. "
        "Transcribe it exactly. Preserve student mistakes, line breaks, punctuation, "
        "and hyphenated word breaks. Return only the text."
    ),
    "literal_uncertain": (
        "Read the Russian handwriting literally. Do not infer missing words from context. "
        "If a word is unclear, output your best literal reading instead of correcting it. "
        "Preserve line breaks. Return only the transcription."
    ),
}
VLM_PROVIDERS = {
    "qwen25_vl_7b",
    "qwen25_vl_32b",
    "qwen25_vl_72b",
    "gemma3_vision",
    "internvl",
    "minicpm_v",
}


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
    parser.add_argument(
        "--variants",
        default="",
        help=(
            "Comma-separated image variants to benchmark. "
            f"Available defaults: {','.join(DEFAULT_VARIANTS)}. "
            "Also available: otsu_binary."
        ),
    )
    parser.add_argument(
        "--variant-output-dir",
        default="tmp/ocr_benchmark_variants",
        help="Directory for generated image variants.",
    )
    parser.add_argument(
        "--analysis-output",
        default="",
        help="Optional Markdown report with OCR error analysis.",
    )
    parser.add_argument(
        "--prompt-variants",
        default="",
        help=(
            "Comma-separated VLM prompt variants to benchmark. "
            f"Available: {','.join(PROMPT_VARIANTS)}."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs for each image/provider/variant/prompt combination.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be 1 or greater")

    load_dotenv()
    config = {
        key: getattr(Config, key)
        for key in dir(Config)
        if key.isupper()
    }
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    prompt_variant_names = [p.strip() for p in args.prompt_variants.split(",") if p.strip()]

    rows = []
    for image_path in iter_images(Path(args.images)):
        gt_path = Path(args.ground_truth) / f"{image_path.stem}.txt"
        if not gt_path.exists():
            print(f"skip {image_path.name}: missing {gt_path.name}")
            continue

        expected = gt_path.read_text(encoding="utf-8").strip()
        candidates = get_candidate_images(image_path, args, variant_names)

        for variant_name, candidate_path in candidates:
            for provider in providers:
                for prompt_name, prompt_text in get_provider_prompts(provider, prompt_variant_names):
                    for repeat_index in range(1, args.repeats + 1):
                        run_config = dict(config)
                        if prompt_text:
                            run_config["VLM_OCR_PROMPT"] = prompt_text

                        started_at = time.perf_counter()
                        error = ""
                        recognized = ""
                        metadata = {}
                        error_analysis = {}

                        try:
                            result = run_ocr(str(candidate_path), run_config, provider)
                            recognized = result.text
                            metadata = result.metadata
                            elapsed = result.elapsed_seconds
                            error_analysis = analyze_ocr_errors(expected, recognized)
                        except Exception as exc:
                            elapsed = time.perf_counter() - started_at
                            error = str(exc)

                        rows.append({
                            "image": image_path.name,
                            "source_image": str(candidate_path),
                            "variant": variant_name,
                            "prompt_variant": prompt_name,
                            "repeat_index": repeat_index,
                            "provider": provider,
                            "cer": cer(expected, recognized) if not error else "",
                            "wer": wer(expected, recognized) if not error else "",
                            "normalized_cer": (
                                cer(normalize_for_text_comparison(expected), normalize_for_text_comparison(recognized))
                                if not error else ""
                            ),
                            "normalized_wer": (
                                wer(normalize_for_text_comparison(expected), normalize_for_text_comparison(recognized))
                                if not error else ""
                            ),
                            "elapsed_seconds": round(elapsed, 3),
                            "expected_chars": len(expected),
                            "recognized_chars": len(recognized),
                            "word_substitutions_count": error_analysis.get("word_substitutions_count", ""),
                            "missing_words_count": error_analysis.get("missing_words_count", ""),
                            "extra_words_count": error_analysis.get("extra_words_count", ""),
                            "punctuation_delta_count": error_analysis.get("punctuation_delta_count", ""),
                            "line_break_delta": error_analysis.get("line_break_delta", ""),
                            "hyphenated_line_breaks": error_analysis.get("hyphenated_line_breaks", ""),
                            "error": error,
                            "recognized_text": recognized,
                            "error_analysis": json.dumps(error_analysis, ensure_ascii=False),
                            "metadata": json.dumps(metadata, ensure_ascii=False),
                        })

    write_csv(Path(args.output), rows)
    if args.analysis_output:
        write_markdown_report(Path(args.analysis_output), rows)
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


def get_candidate_images(
    image_path: Path,
    args: argparse.Namespace,
    variant_names: list[str],
) -> list[tuple[str, Path]]:
    if variant_names:
        variant_dir = Path(args.variant_output_dir) / image_path.stem
        return [
            (variant.name, variant.path)
            for variant in create_image_variants(image_path, variant_dir, variant_names)
        ]

    if args.keep_original:
        return [("original", image_path)]

    return [("legacy_preprocess", make_preprocessed_copy(image_path))]


def get_provider_prompts(
    provider: str,
    prompt_variant_names: list[str],
) -> list[tuple[str, str | None]]:
    if not prompt_variant_names or provider not in VLM_PROVIDERS:
        return [("default", None)]

    prompts = []
    for prompt_name in prompt_variant_names:
        prompt = PROMPT_VARIANTS.get(prompt_name)
        if prompt is None:
            available = ", ".join(PROMPT_VARIANTS)
            raise ValueError(f"Unknown prompt variant '{prompt_name}'. Available: {available}")
        prompts.append((prompt_name, prompt))
    return prompts


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
        "source_image",
        "variant",
        "prompt_variant",
        "repeat_index",
        "provider",
        "cer",
        "wer",
        "normalized_cer",
        "normalized_wer",
        "elapsed_seconds",
        "expected_chars",
        "recognized_chars",
        "word_substitutions_count",
        "missing_words_count",
        "extra_words_count",
        "punctuation_delta_count",
        "line_break_delta",
        "hyphenated_line_breaks",
        "error",
        "recognized_text",
        "error_analysis",
        "metadata",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    successful_rows = [row for row in rows if not row.get("error")]
    sorted_rows = sorted(
        successful_rows,
        key=lambda row: (
            float(row["cer"]) if row["cer"] != "" else 999,
            float(row["wer"]) if row["wer"] != "" else 999,
        ),
    )

    lines = [
        "# OCR Error Analysis",
        "",
    ]
    lines.extend(build_aggregate_report(sorted_rows))
    lines.extend([
        "",
        "# Runs",
        "",
        "| Image | Variant | Prompt | Repeat | Provider | CER | WER | Norm. CER | Norm. WER | Subst. | Missing | Extra | Punct. delta | Hyphen breaks |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in sorted_rows:
        lines.append(
            "| {image} | {variant} | {prompt} | {repeat} | {provider} | {cer:.3f} | {wer:.3f} | {norm_cer:.3f} | {norm_wer:.3f} | {subs} | {missing} | {extra} | {punct} | {hyphen} |".format(
                image=row["image"],
                variant=row["variant"],
                prompt=row["prompt_variant"],
                repeat=row["repeat_index"],
                provider=row["provider"],
                cer=float(row["cer"]),
                wer=float(row["wer"]),
                norm_cer=float(row["normalized_cer"]),
                norm_wer=float(row["normalized_wer"]),
                subs=row["word_substitutions_count"],
                missing=row["missing_words_count"],
                extra=row["extra_words_count"],
                punct=row["punctuation_delta_count"],
                hyphen=row["hyphenated_line_breaks"],
            )
        )

    for row in sorted_rows:
        analysis = json.loads(row["error_analysis"] or "{}")
        lines.extend([
            "",
            f"## {row['image']} / {row['variant']} / {row['prompt_variant']} / repeat {row['repeat_index']} / {row['provider']}",
            "",
            f"- CER: {float(row['cer']):.3f}",
            f"- WER: {float(row['wer']):.3f}",
            f"- Normalized CER: {float(row['normalized_cer']):.3f}",
            f"- Normalized WER: {float(row['normalized_wer']):.3f}",
            f"- Word substitutions: {row['word_substitutions_count']}",
            f"- Missing words: {row['missing_words_count']}",
            f"- Extra words: {row['extra_words_count']}",
            f"- Punctuation delta: {row['punctuation_delta_count']}",
            f"- Hyphenated line breaks in OCR: {row['hyphenated_line_breaks']}",
            "",
            "Top substitutions:",
        ])
        substitutions = analysis.get("top_substitutions", [])
        if substitutions:
            lines.extend(
                f"- `{item['expected']}` -> `{item['actual']}`"
                for item in substitutions
            )
        else:
            lines.append("- none")

        lines.extend(["", "Missing words:"])
        missing_words = analysis.get("missing_words", [])
        lines.append("- " + ", ".join(f"`{word}`" for word in missing_words) if missing_words else "- none")

        lines.extend(["", "Extra words:"])
        extra_words = analysis.get("extra_words", [])
        lines.append("- " + ", ".join(f"`{word}`" for word in extra_words) if extra_words else "- none")

    failed_rows = [row for row in rows if row.get("error")]
    if failed_rows:
        lines.extend(["", "# Failed Runs", ""])
        for row in failed_rows:
            lines.append(
                f"- {row['image']} / {row['variant']} / {row['prompt_variant']} / repeat {row['repeat_index']} / {row['provider']}: {row['error']}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_aggregate_report(rows: list[dict]) -> list[str]:
    if not rows:
        return ["No successful runs."]

    groups = defaultdict(list)
    for row in rows:
        key = (row["provider"], row["variant"], row["prompt_variant"])
        groups[key].append(row)

    aggregate_rows = []
    for (provider, variant, prompt), group_rows in groups.items():
        aggregate_rows.append({
            "provider": provider,
            "variant": variant,
            "prompt": prompt,
            "count": len(group_rows),
            "cer": average_float(group_rows, "cer"),
            "wer": average_float(group_rows, "wer"),
            "normalized_cer": average_float(group_rows, "normalized_cer"),
            "normalized_wer": average_float(group_rows, "normalized_wer"),
            "normalized_cer_min": min_float(group_rows, "normalized_cer"),
            "normalized_cer_max": max_float(group_rows, "normalized_cer"),
            "normalized_cer_std": std_float(group_rows, "normalized_cer"),
            "word_substitutions_count": average_float(group_rows, "word_substitutions_count"),
            "missing_words_count": average_float(group_rows, "missing_words_count"),
            "extra_words_count": average_float(group_rows, "extra_words_count"),
            "elapsed_seconds": average_float(group_rows, "elapsed_seconds"),
        })

    aggregate_rows.sort(key=lambda row: (row["normalized_cer"], row["normalized_wer"], row["cer"]))
    lines = [
        "## Aggregate Results",
        "",
        "| Provider | Variant | Prompt | Runs | Avg CER | Avg WER | Avg Norm. CER | Norm. CER min | Norm. CER max | Norm. CER std | Avg Norm. WER | Avg Subst. | Avg Missing | Avg Extra | Avg Time |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {provider} | {variant} | {prompt} | {count} | {cer:.3f} | {wer:.3f} | {norm_cer:.3f} | {norm_cer_min:.3f} | {norm_cer_max:.3f} | {norm_cer_std:.3f} | {norm_wer:.3f} | {subs:.2f} | {missing:.2f} | {extra:.2f} | {time:.3f}s |".format(
                provider=row["provider"],
                variant=row["variant"],
                prompt=row["prompt"],
                count=row["count"],
                cer=row["cer"],
                wer=row["wer"],
                norm_cer=row["normalized_cer"],
                norm_cer_min=row["normalized_cer_min"],
                norm_cer_max=row["normalized_cer_max"],
                norm_cer_std=row["normalized_cer_std"],
                norm_wer=row["normalized_wer"],
                subs=row["word_substitutions_count"],
                missing=row["missing_words_count"],
                extra=row["extra_words_count"],
                time=row["elapsed_seconds"],
            )
        )
    return lines


def average_float(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return sum(values) / len(values) if values else 0.0


def min_float(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return min(values) if values else 0.0


def max_float(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return max(values) if values else 0.0


def std_float(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    if len(values) < 2:
        return 0.0

    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return variance ** 0.5


if __name__ == "__main__":
    main()
