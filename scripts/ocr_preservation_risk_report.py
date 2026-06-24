import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Find places where a high-CER OCR winner may have corrected student mistakes."
    )
    parser.add_argument(
        "--preservation",
        required=True,
        help="CSV produced by scripts/ocr_student_mistake_preservation.py.",
    )
    parser.add_argument(
        "--base-provider",
        default="qwen25_vl_72b",
        help="Provider used as the main transcription candidate.",
    )
    parser.add_argument(
        "--support-providers",
        default="yandex,cyrillic_trocr,gemma3_vision",
        help="Comma-separated providers used as preservation signals.",
    )
    parser.add_argument("--output", required=True, help="Detailed CSV output path.")
    parser.add_argument("--report", required=True, help="Markdown report output path.")
    args = parser.parse_args()

    rows = read_rows(Path(args.preservation))
    support_providers = {item.strip() for item in args.support_providers.split(",") if item.strip()}
    detail_rows = build_risk_rows(rows, args.base_provider, support_providers)
    write_csv(Path(args.output), detail_rows)
    write_report(Path(args.report), detail_rows, args.base_provider, support_providers)
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_risk_rows(
    rows: list[dict],
    base_provider: str,
    support_providers: set[str],
) -> list[dict]:
    by_mistake = defaultdict(dict)
    for row in rows:
        key = (row["image"], row["mistake_index"])
        by_mistake[key][row["provider"]] = row

    detail_rows = []
    for (image, mistake_index), provider_rows in sorted(by_mistake.items()):
        base = provider_rows.get(base_provider)
        if not base:
            continue

        preserving = sorted(
            provider
            for provider in support_providers
            if provider_rows.get(provider, {}).get("classification") == "preserved"
        )
        correcting = sorted(
            provider
            for provider in support_providers
            if provider_rows.get(provider, {}).get("classification") == "corrected"
        )
        lost = sorted(
            provider
            for provider in support_providers
            if provider_rows.get(provider, {}).get("classification") == "lost"
        )

        status = classify_risk(base["classification"], preserving, correcting)
        detail_rows.append({
            "image": image,
            "mistake_index": mistake_index,
            "mistake_type": base["mistake_type"],
            "clean_text": base["clean_text"],
            "student_text": base["student_text"],
            "base_provider": base_provider,
            "base_classification": base["classification"],
            "risk_status": status,
            "preserving_support": ",".join(preserving),
            "correcting_support": ",".join(correcting),
            "lost_support": ",".join(lost),
            "support_preserved_count": len(preserving),
            "support_corrected_count": len(correcting),
            "support_lost_count": len(lost),
        })
    return detail_rows


def classify_risk(
    base_classification: str,
    preserving: list[str],
    correcting: list[str],
) -> str:
    if base_classification == "preserved":
        return "base_preserved_student_mistake"
    if base_classification == "corrected" and preserving:
        return "probable_base_correction_with_preservation_support"
    if base_classification == "corrected":
        return "probable_base_correction_no_support"
    if base_classification == "lost" and preserving:
        return "base_lost_but_support_preserved"
    if base_classification == "lost" and correcting:
        return "base_lost_support_corrected"
    return "base_lost_no_signal"


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "mistake_index",
        "mistake_type",
        "clean_text",
        "student_text",
        "base_provider",
        "base_classification",
        "risk_status",
        "preserving_support",
        "correcting_support",
        "lost_support",
        "support_preserved_count",
        "support_corrected_count",
        "support_lost_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    rows: list[dict],
    base_provider: str,
    support_providers: set[str],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(row["risk_status"] for row in rows)
    support_counts = Counter()
    for row in rows:
        for provider in row["preserving_support"].split(","):
            if provider:
                support_counts[provider] += 1

    lines = [
        "# OCR Preservation Risk Report",
        "",
        f"Base provider: `{base_provider}`",
        f"Support providers: {', '.join(f'`{provider}`' for provider in sorted(support_providers))}",
        f"Checked student mistakes: {len(rows)}",
        "",
        "## Risk Summary",
        "",
        "| Risk status | Count | Meaning |",
        "| --- | ---: | --- |",
    ]
    meanings = {
        "base_preserved_student_mistake": "Base OCR already kept the student variant.",
        "probable_base_correction_with_preservation_support": "Base OCR used the clean variant while another provider kept the student variant.",
        "probable_base_correction_no_support": "Base OCR used the clean variant and no support provider kept the student variant.",
        "base_lost_but_support_preserved": "Base OCR lost the place, but another provider kept the student variant.",
        "base_lost_support_corrected": "Base OCR lost the place and a support provider used the clean variant.",
        "base_lost_no_signal": "No provider gave a useful preservation/correction signal.",
    }
    for status, count in status_counts.most_common():
        lines.append(f"| `{status}` | {count} | {meanings.get(status, '')} |")

    lines.extend([
        "",
        "## Preservation Support Providers",
        "",
        "| Provider | Preserved student variant count |",
        "| --- | ---: |",
    ])
    for provider, count in support_counts.most_common():
        lines.append(f"| `{provider}` | {count} |")

    high_risk = [
        row for row in rows
        if row["risk_status"] == "probable_base_correction_with_preservation_support"
    ]
    missed = [
        row for row in rows
        if row["risk_status"] == "base_lost_but_support_preserved"
    ]

    lines.extend([
        "",
        "## High-Risk Base Corrections",
        "",
        "| Image | Mistake | Clean text | Student text | Preserving support |",
        "| --- | ---: | --- | --- | --- |",
    ])
    if high_risk:
        for row in high_risk:
            lines.append(
                "| {image} | {mistake} | `{clean}` | `{student}` | {support} |".format(
                    image=row["image"],
                    mistake=row["mistake_index"],
                    clean=row["clean_text"],
                    student=row["student_text"],
                    support=", ".join(f"`{provider}`" for provider in row["preserving_support"].split(",") if provider),
                )
            )
    else:
        lines.append("| - | - | - | - | - |")

    lines.extend([
        "",
        "## Base Lost, Support Preserved",
        "",
        "| Image | Mistake | Clean text | Student text | Preserving support |",
        "| --- | ---: | --- | --- | --- |",
    ])
    if missed:
        for row in missed:
            lines.append(
                "| {image} | {mistake} | `{clean}` | `{student}` | {support} |".format(
                    image=row["image"],
                    mistake=row["mistake_index"],
                    clean=row["clean_text"],
                    student=row["student_text"],
                    support=", ".join(f"`{provider}`" for provider in row["preserving_support"].split(",") if provider),
                )
            )
    else:
        lines.append("| - | - | - | - | - |")

    lines.extend([
        "",
        "## Recommended Use",
        "",
        "Do not automatically replace the base OCR text. Pass high-risk rows to the next LLM analysis step as suspicious places where OCR may have corrected or lost a real student mistake.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
