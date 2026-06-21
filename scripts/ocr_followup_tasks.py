import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create follow-up OCR review tasks from ensemble confidence.")
    parser.add_argument("--confidence", required=True, help="word_confidence.csv from scripts/ocr_ensemble.py.")
    parser.add_argument("--ensemble-report", default="", help="Optional ensemble_report.md for context.")
    parser.add_argument("--output", required=True, help="Markdown file with follow-up tasks.")
    parser.add_argument(
        "--rerun-list",
        default="",
        help="Optional text file with image names recommended for second-pass OCR.",
    )
    parser.add_argument(
        "--statuses",
        default="base_only,uncertain,alternative_supported",
        help="Comma-separated confidence statuses that should become follow-up tasks.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=20,
        help="Maximum suspicious words to list per image.",
    )
    args = parser.parse_args()

    statuses = {item.strip() for item in args.statuses.split(",") if item.strip()}
    rows = read_confidence_rows(Path(args.confidence))
    grouped = group_suspicious_rows(rows, statuses)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_markdown(grouped, rows, args.max_words),
        encoding="utf-8",
    )
    if args.rerun_list:
        write_rerun_list(Path(args.rerun_list), grouped)
    print(f"wrote {output_path}")


def read_confidence_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_suspicious_rows(rows: list[dict], statuses: set[str]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] in statuses:
            grouped[row["image"]].append(row)
    return grouped


def build_markdown(grouped: dict[str, list[dict]], all_rows: list[dict], max_words: int) -> str:
    totals = count_statuses(all_rows)
    lines = [
        "# OCR Follow-up Tasks",
        "",
        "These tasks are based on ensemble word confidence. They are not student grading tasks.",
        "",
        "## Status Summary",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(totals.items()):
        lines.append(f"| {status} | {count} |")

    lines.extend([
        "",
        "## Image Tasks",
        "",
    ])

    for image, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        status_counts = count_statuses(rows)
        action = recommended_action(rows)
        suspicious_words = rows[:max_words]
        lines.extend([
            f"### {image}",
            "",
            f"- Recommended action: {action}",
            f"- Suspicious words: {len(rows)}",
            f"- Status counts: {format_status_counts(status_counts)}",
            "",
            "| Position | Word | Status | Support | Candidate count | Top share |",
            "| ---: | --- | --- | --- | ---: | ---: |",
        ])
        for row in suspicious_words:
            lines.append(
                "| {position} | `{word}` | {status} | {support} | {candidate_count} | {top_share} |".format(
                    position=row["position"],
                    word=row["base_word"],
                    status=row["status"],
                    support=row["base_support"] or row["top_support"],
                    candidate_count=row["candidate_count"],
                    top_share=row["top_share"],
                )
            )
        if len(rows) > max_words:
            lines.append(f"| ... | ... | {len(rows) - max_words} more | ... | ... | ... |")
        lines.append("")

    return "\n".join(lines) + "\n"


def count_statuses(rows: list[dict]) -> dict[str, int]:
    counts = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    return dict(counts)


def format_status_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))


def recommended_action(rows: list[dict]) -> str:
    total = len(rows)
    if total >= 30:
        return "rerun full-page VLM with an alternative prompt and manually review result"
    if total >= 10:
        return "rerun VLM for this page and compare with current base text"
    return "review listed words only"


def write_rerun_list(path: Path, grouped: dict[str, list[dict]]):
    images = [
        image
        for image, rows in sorted(grouped.items())
        if len(rows) >= 10
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(images) + ("\n" if images else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
