import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.ocr_error_analysis import extract_words, normalize_for_text_comparison, normalize_word


@dataclass(frozen=True)
class StudentMistake:
    image_stem: str
    index: int
    mistake_type: str
    clean_words: tuple[str, ...]
    student_words: tuple[str, ...]


def main():
    parser = argparse.ArgumentParser(
        description="Measure how well OCR outputs preserve real student mistakes."
    )
    parser.add_argument(
        "--benchmarks",
        required=True,
        help="Comma-separated benchmark CSV files with recognized_text columns.",
    )
    parser.add_argument(
        "--student-text",
        required=True,
        help="Directory with manually checked student transcriptions.",
    )
    parser.add_argument(
        "--correct-text",
        required=True,
        help="Directory with corrected reference texts.",
    )
    parser.add_argument("--output", required=True, help="Detailed CSV output path.")
    parser.add_argument("--report", required=True, help="Markdown report output path.")
    args = parser.parse_args()

    benchmark_paths = [Path(item.strip()) for item in args.benchmarks.split(",") if item.strip()]
    student_dir = Path(args.student_text)
    correct_dir = Path(args.correct_text)

    mistakes_by_stem = load_mistakes(student_dir, correct_dir)
    detail_rows = []
    for row in load_benchmark_rows(benchmark_paths):
        image_stem = Path(row["image"]).stem
        mistakes = mistakes_by_stem.get(image_stem, [])
        if not mistakes or row.get("error"):
            continue

        ocr_words = normalized_words(row.get("recognized_text", ""))
        for mistake in mistakes:
            classification = classify_mistake(mistake, ocr_words)
            detail_rows.append({
                "image": row["image"],
                "provider": row["provider"],
                "variant": row.get("variant", ""),
                "prompt_variant": row.get("prompt_variant", ""),
                "repeat_index": row.get("repeat_index", ""),
                "mistake_index": mistake.index,
                "mistake_type": mistake.mistake_type,
                "clean_text": " ".join(mistake.clean_words),
                "student_text": " ".join(mistake.student_words),
                "classification": classification,
            })

    write_csv(Path(args.output), detail_rows)
    write_report(Path(args.report), detail_rows, mistakes_by_stem)
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


def load_mistakes(student_dir: Path, correct_dir: Path) -> dict[str, list[StudentMistake]]:
    mistakes_by_stem = {}
    for student_path in sorted(student_dir.glob("*.txt")):
        if student_path.name.upper().startswith("REVIEW_"):
            continue

        correct_path = correct_dir / student_path.name
        if not correct_path.exists():
            continue

        stem = student_path.stem
        student_words = extract_words(normalize_for_text_comparison(student_path.read_text(encoding="utf-8")))
        clean_words = extract_words(normalize_for_text_comparison(correct_path.read_text(encoding="utf-8")))
        mistakes_by_stem[stem] = find_student_mistakes(stem, clean_words, student_words)
    return mistakes_by_stem


def find_student_mistakes(
    image_stem: str,
    clean_words: list[str],
    student_words: list[str],
) -> list[StudentMistake]:
    clean_normalized = [normalize_word(word) for word in clean_words]
    student_normalized = [normalize_word(word) for word in student_words]
    matcher = SequenceMatcher(None, clean_normalized, student_normalized, autojunk=False)

    mistakes = []
    index = 1
    for tag, clean_start, clean_end, student_start, student_end in matcher.get_opcodes():
        if tag == "equal":
            continue

        clean_chunk = tuple(clean_words[clean_start:clean_end])
        student_chunk = tuple(student_words[student_start:student_end])
        if tag == "replace":
            mistake_type = "substitution"
        elif tag == "delete":
            mistake_type = "omission"
        elif tag == "insert":
            mistake_type = "insertion"
        else:
            mistake_type = tag

        mistakes.append(StudentMistake(
            image_stem=image_stem,
            index=index,
            mistake_type=mistake_type,
            clean_words=clean_chunk,
            student_words=student_chunk,
        ))
        index += 1
    return mistakes


def load_benchmark_rows(paths: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row.get("image"),
                    row.get("provider"),
                    row.get("variant"),
                    row.get("prompt_variant"),
                    row.get("repeat_index"),
                    row.get("recognized_text"),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def normalized_words(text: str) -> list[str]:
    return [
        normalize_word(word)
        for word in extract_words(normalize_for_text_comparison(text))
    ]


def classify_mistake(mistake: StudentMistake, ocr_words: list[str]) -> str:
    clean_words = tuple(normalize_word(word) for word in mistake.clean_words)
    student_words = tuple(normalize_word(word) for word in mistake.student_words)

    clean_present = phrase_present(ocr_words, clean_words) if clean_words else False
    student_present = phrase_present(ocr_words, student_words) if student_words else False

    if mistake.mistake_type == "omission":
        return "corrected" if clean_present else "preserved"

    if mistake.mistake_type == "insertion":
        return "preserved" if student_present else "lost"

    if student_present and not clean_present:
        return "preserved"
    if clean_present and not student_present:
        return "corrected"
    if student_present and clean_present:
        return "ambiguous"
    return "lost"


def phrase_present(words: list[str], phrase: tuple[str, ...]) -> bool:
    if not phrase:
        return False
    if len(phrase) > len(words):
        return False

    for index in range(0, len(words) - len(phrase) + 1):
        if tuple(words[index:index + len(phrase)]) == phrase:
            return True
    return False


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "provider",
        "variant",
        "prompt_variant",
        "repeat_index",
        "mistake_index",
        "mistake_type",
        "clean_text",
        "student_text",
        "classification",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], mistakes_by_stem: dict[str, list[StudentMistake]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    total_mistakes = sum(len(mistakes) for mistakes in mistakes_by_stem.values())
    by_provider = aggregate_by_provider(rows)
    by_type = aggregate_by_provider_and_type(rows)

    lines = [
        "# Student Mistake Preservation Report",
        "",
        f"Images with checked student/correct references: {len(mistakes_by_stem)}",
        f"Detected student-vs-correct differences: {total_mistakes}",
        "",
        "## Provider Summary",
        "",
        "| Provider | Checked mistakes | Preserved | Corrected | Lost | Ambiguous | Preservation rate | Correction rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for provider, counts in sorted(
        by_provider.items(),
        key=lambda item: (-rate(item[1], "preserved"), item[0]),
    ):
        total = sum(counts.values())
        lines.append(
            "| {provider} | {total} | {preserved} | {corrected} | {lost} | {ambiguous} | {preservation:.3f} | {correction:.3f} |".format(
                provider=provider,
                total=total,
                preserved=counts["preserved"],
                corrected=counts["corrected"],
                lost=counts["lost"],
                ambiguous=counts["ambiguous"],
                preservation=rate(counts, "preserved"),
                correction=rate(counts, "corrected"),
            )
        )

    lines.extend([
        "",
        "## Provider And Mistake Type",
        "",
        "| Provider | Mistake type | Checked | Preserved | Corrected | Lost | Ambiguous | Preservation rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for (provider, mistake_type), counts in sorted(by_type.items()):
        total = sum(counts.values())
        lines.append(
            "| {provider} | {mistake_type} | {total} | {preserved} | {corrected} | {lost} | {ambiguous} | {preservation:.3f} |".format(
                provider=provider,
                mistake_type=mistake_type,
                total=total,
                preserved=counts["preserved"],
                corrected=counts["corrected"],
                lost=counts["lost"],
                ambiguous=counts["ambiguous"],
                preservation=rate(counts, "preserved"),
            )
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- `preserved` means OCR kept the student's variant rather than the clean reference variant.",
        "- `corrected` means OCR output contains the clean reference variant instead of the student's variant.",
        "- `lost` means neither variant was found in OCR output.",
        "- `ambiguous` means both variants were found, usually because OCR duplicated or hallucinated nearby text.",
        "- For student omissions, `preserved` means the clean omitted phrase was not found in OCR output.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_by_provider(rows: list[dict]) -> dict[str, defaultdict[str, int]]:
    counts_by_provider = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts_by_provider[row["provider"]][row["classification"]] += 1
    return counts_by_provider


def aggregate_by_provider_and_type(rows: list[dict]) -> dict[tuple[str, str], defaultdict[str, int]]:
    counts_by_key = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts_by_key[(row["provider"], row["mistake_type"])][row["classification"]] += 1
    return counts_by_key


def rate(counts: dict[str, int], key: str) -> float:
    total = sum(counts.values())
    return counts[key] / total if total else 0.0


if __name__ == "__main__":
    main()
