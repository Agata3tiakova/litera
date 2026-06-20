import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.ocr_error_analysis import normalize_for_text_comparison

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)?")


@dataclass
class Token:
    text: str
    is_word: bool
    word_index: int | None = None


def main():
    parser = argparse.ArgumentParser(description="Build weighted OCR ensemble text from benchmark CSV.")
    parser.add_argument("--input", required=True, help="Benchmark CSV produced by scripts/ocr_benchmark.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for consensus texts and reports.")
    parser.add_argument(
        "--base-provider",
        default="",
        help="Provider used as the structural base. Defaults to best average normalized CER in the CSV.",
    )
    parser.add_argument(
        "--trusted-providers",
        default="qwen25_vl_72b,gemma3_vision",
        help="Comma-separated providers allowed to vote on replacements. Empty value uses all providers.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=2,
        help="Minimum number of providers supporting a replacement candidate.",
    )
    parser.add_argument(
        "--replace-margin",
        type=float,
        default=1.25,
        help="Alternative candidate must exceed base vote weight by this multiplier.",
    )
    parser.add_argument(
        "--ground-truth",
        default="",
        help="Optional student-text reference directory for evaluating consensus output.",
    )
    parser.add_argument(
        "--clean-ground-truth",
        default="",
        help="Optional clean reference directory for secondary metrics.",
    )
    args = parser.parse_args()

    rows = read_successful_rows(Path(args.input))
    if not rows:
        raise SystemExit("No successful OCR rows found.")

    output_dir = Path(args.output_dir)
    text_dir = output_dir / "consensus_text"
    text_dir.mkdir(parents=True, exist_ok=True)

    provider_weights = calculate_provider_weights(rows)
    base_provider = args.base_provider or min(
        provider_weights,
        key=lambda provider: provider_weights[provider]["avg_norm_cer"],
    )
    trusted_providers = parse_provider_list(args.trusted_providers)

    grouped_rows = group_rows(rows)
    report_rows = []
    decision_rows = []
    confidence_rows = []

    for image, image_rows in sorted(grouped_rows.items()):
        result = build_consensus(
            image_rows=image_rows,
            provider_weights=provider_weights,
            base_provider=base_provider,
            trusted_providers=trusted_providers,
            min_support=args.min_support,
            replace_margin=args.replace_margin,
        )

        consensus_path = text_dir / f"{Path(image).stem}.txt"
        consensus_path.write_text(result["text"], encoding="utf-8")

        student_reference = read_reference(args.ground_truth, image)
        clean_reference = read_reference(args.clean_ground_truth, image)
        metrics = evaluate_text(result["text"], student_reference, clean_reference)

        report_rows.append({
            "image": image,
            "base_provider": result["base_provider"],
            "providers": ", ".join(sorted(row["provider"] for row in image_rows)),
            "replacements": result["replacement_count"],
            "supported_words": count_status(result["confidence"], "supported"),
            "base_dominant_words": count_status(result["confidence"], "base_dominant"),
            "base_only_words": count_status(result["confidence"], "base_only"),
            "alternative_supported_words": count_status(result["confidence"], "alternative_supported"),
            "uncertain_words": count_status(result["confidence"], "uncertain"),
            "supported_ratio": supported_ratio(result["confidence"]),
            "review_flag": review_flag(result["confidence"]),
            **metrics,
            "consensus_path": str(consensus_path),
        })
        decision_rows.extend(result["decisions"])
        confidence_rows.extend(result["confidence"])

    write_report(output_dir / "ensemble_report.md", report_rows, provider_weights, base_provider)
    write_decisions(output_dir / "ensemble_decisions.csv", decision_rows)
    write_confidence(output_dir / "word_confidence.csv", confidence_rows)
    print(f"wrote {output_dir}")


def read_successful_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            row
            for row in csv.DictReader(f)
            if not row.get("error") and row.get("recognized_text")
        ]


def calculate_provider_weights(rows: list[dict]) -> dict[str, dict]:
    by_provider = defaultdict(list)
    for row in rows:
        if row.get("normalized_cer") not in {"", None}:
            by_provider[row["provider"]].append(float(row["normalized_cer"]))

    weights = {}
    for provider, values in by_provider.items():
        avg = sum(values) / len(values)
        weights[provider] = {
            "avg_norm_cer": avg,
            "weight": 1.0 / max(avg, 0.02),
            "runs": len(values),
        }
    return weights


def parse_provider_list(value: str) -> set[str] | None:
    providers = {item.strip() for item in value.split(",") if item.strip()}
    return providers or None


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(row)
    return grouped


def build_consensus(
    image_rows: list[dict],
    provider_weights: dict[str, dict],
    base_provider: str,
    trusted_providers: set[str] | None,
    min_support: int,
    replace_margin: float,
) -> dict:
    base_row = find_base_row(image_rows, provider_weights, base_provider)
    base_tokens = tokenize_with_separators(base_row["recognized_text"])
    base_words = [token.text for token in base_tokens if token.is_word]

    votes = [
        {
            normalize_word(word): {
                "weight": provider_weights[base_row["provider"]]["weight"],
                "support": {base_row["provider"]},
                "forms": Counter({word: 1}),
            }
        }
        for word in base_words
    ]

    for row in image_rows:
        provider = row["provider"]
        if provider == base_row["provider"]:
            continue
        if trusted_providers is not None and provider not in trusted_providers:
            continue

        provider_weight = provider_weights.get(provider, {}).get("weight", 0.0)
        if provider_weight <= 0:
            continue

        add_aligned_votes(
            base_words=base_words,
            candidate_words=extract_words(row["recognized_text"]),
            votes=votes,
            provider=provider,
            provider_weight=provider_weight,
        )

    decisions = []
    confidence_rows = []
    replacement_count = 0
    for token in base_tokens:
        if not token.is_word or token.word_index is None:
            continue

        base_word = token.text
        base_norm = normalize_word(base_word)
        word_votes = votes[token.word_index]
        ranked = sorted(
            word_votes.items(),
            key=lambda item: (item[1]["weight"], len(item[1]["support"])),
            reverse=True,
        )
        top_norm, top_vote = ranked[0]
        base_vote = word_votes.get(base_norm, {"weight": 0.0, "support": set(), "forms": Counter()})
        total_weight = sum(vote["weight"] for vote in word_votes.values())
        top_share = top_vote["weight"] / total_weight if total_weight else 0.0
        confidence_rows.append({
            "image": base_row["image"],
            "position": token.word_index,
            "base_word": base_word,
            "top_word": top_vote["forms"].most_common(1)[0][0],
            "base_support": ",".join(sorted(base_vote["support"])),
            "top_support": ",".join(sorted(top_vote["support"])),
            "candidate_count": len(word_votes),
            "top_weight": round(top_vote["weight"], 4),
            "base_weight": round(base_vote["weight"], 4),
            "top_share": round(top_share, 4),
            "status": classify_word_status(base_norm, top_norm, base_vote, top_vote, top_share),
        })

        if (
            top_norm != base_norm
            and len(top_vote["support"]) >= min_support
            and top_vote["weight"] >= base_vote["weight"] * replace_margin
        ):
            replacement = top_vote["forms"].most_common(1)[0][0]
            decisions.append({
                "image": base_row["image"],
                "position": token.word_index,
                "base": base_word,
                "replacement": replacement,
                "support": ",".join(sorted(top_vote["support"])),
                "base_support": ",".join(sorted(base_vote["support"])),
                "top_weight": round(top_vote["weight"], 4),
                "base_weight": round(base_vote["weight"], 4),
            })
            token.text = replacement
            replacement_count += 1

    return {
        "base_provider": base_row["provider"],
        "text": "".join(token.text for token in base_tokens),
        "replacement_count": replacement_count,
        "decisions": decisions,
        "confidence": confidence_rows,
    }


def classify_word_status(
    base_norm: str,
    top_norm: str,
    base_vote: dict,
    top_vote: dict,
    top_share: float,
) -> str:
    if top_norm == base_norm and len(base_vote["support"]) >= 2:
        return "supported"
    if top_norm == base_norm and len(base_vote["support"]) == 1:
        return "base_only"
    if top_norm == base_norm and top_share >= 0.75:
        return "base_dominant"
    if top_norm != base_norm and len(top_vote["support"]) >= 2:
        return "alternative_supported"
    return "uncertain"


def find_base_row(rows: list[dict], provider_weights: dict[str, dict], preferred_provider: str) -> dict:
    for row in rows:
        if row["provider"] == preferred_provider:
            return row
    return min(rows, key=lambda row: provider_weights.get(row["provider"], {}).get("avg_norm_cer", 999.0))


def tokenize_with_separators(text: str) -> list[Token]:
    tokens = []
    word_index = 0
    position = 0
    for match in WORD_RE.finditer(text):
        if match.start() > position:
            tokens.append(Token(text=text[position:match.start()], is_word=False))
        tokens.append(Token(text=match.group(0), is_word=True, word_index=word_index))
        word_index += 1
        position = match.end()
    if position < len(text):
        tokens.append(Token(text=text[position:], is_word=False))
    return tokens


def extract_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def add_aligned_votes(
    base_words: list[str],
    candidate_words: list[str],
    votes: list[dict],
    provider: str,
    provider_weight: float,
):
    base_norm = [normalize_word(word) for word in base_words]
    candidate_norm = [normalize_word(word) for word in candidate_words]
    matcher = SequenceMatcher(None, base_norm, candidate_norm, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                add_vote(votes[i1 + offset], candidate_words[j1 + offset], provider, provider_weight)
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            for offset in range(i2 - i1):
                add_vote(votes[i1 + offset], candidate_words[j1 + offset], provider, provider_weight)


def add_vote(word_votes: dict, word: str, provider: str, provider_weight: float):
    normalized = normalize_word(word)
    current = word_votes.setdefault(
        normalized,
        {"weight": 0.0, "support": set(), "forms": Counter()},
    )
    current["weight"] += provider_weight
    current["support"].add(provider)
    current["forms"][word] += 1


def normalize_word(word: str) -> str:
    return word.lower().replace("ё", "е")


def read_reference(directory: str, image_name: str) -> str:
    if not directory:
        return ""
    path = Path(directory) / f"{Path(image_name).stem}.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def evaluate_text(text: str, student_reference: str, clean_reference: str) -> dict:
    metrics = {}
    if student_reference:
        metrics["student_normalized_cer"] = normalized_cer(student_reference, text)
        metrics["student_normalized_wer"] = normalized_wer(student_reference, text)
    else:
        metrics["student_normalized_cer"] = ""
        metrics["student_normalized_wer"] = ""

    if clean_reference:
        metrics["clean_normalized_cer"] = normalized_cer(clean_reference, text)
        metrics["clean_normalized_wer"] = normalized_wer(clean_reference, text)
    else:
        metrics["clean_normalized_cer"] = ""
        metrics["clean_normalized_wer"] = ""
    return metrics


def normalized_cer(expected: str, actual: str) -> float:
    expected = normalize_for_text_comparison(expected)
    actual = normalize_for_text_comparison(actual)
    return levenshtein(expected, actual) / len(expected) if expected else float(bool(actual))


def normalized_wer(expected: str, actual: str) -> float:
    expected_words = normalize_for_text_comparison(expected).split()
    actual_words = normalize_for_text_comparison(actual).split()
    return levenshtein(expected_words, actual_words) / len(expected_words) if expected_words else float(bool(actual_words))


def levenshtein(left, right) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def write_report(path: Path, rows: list[dict], provider_weights: dict[str, dict], base_provider: str):
    lines = [
        "# OCR Ensemble Report",
        "",
        f"Base provider: `{base_provider}`",
        "",
        "## Provider Weights",
        "",
        "| Provider | Runs | Avg Norm. CER | Weight |",
        "| --- | ---: | ---: | ---: |",
    ]
    for provider, stats in sorted(provider_weights.items(), key=lambda item: item[1]["avg_norm_cer"]):
        lines.append(
            f"| {provider} | {stats['runs']} | {stats['avg_norm_cer']:.3f} | {stats['weight']:.3f} |"
        )

    lines.extend([
        "",
        "## Consensus Outputs",
        "",
        "| Image | Review | Base | Replacements | Supported Ratio | Supported | Base Only | Alternative | Uncertain | Student Norm. CER | Student Norm. WER | Clean Norm. CER | Clean Norm. WER | Output |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in rows:
        lines.append(
            "| {image} | {review} | {base_provider} | {replacements} | {supported_ratio:.2f} | {supported} | {base_only} | {alternative} | {uncertain} | {student_cer} | {student_wer} | {clean_cer} | {clean_wer} | {path} |".format(
                image=row["image"],
                review=row["review_flag"],
                base_provider=row["base_provider"],
                replacements=row["replacements"],
                supported_ratio=row["supported_ratio"],
                supported=row["supported_words"],
                base_only=row["base_only_words"],
                alternative=row["alternative_supported_words"],
                uncertain=row["uncertain_words"],
                student_cer=format_metric(row["student_normalized_cer"]),
                student_wer=format_metric(row["student_normalized_wer"]),
                clean_cer=format_metric(row["clean_normalized_cer"]),
                clean_wer=format_metric(row["clean_normalized_wer"]),
                path=row["consensus_path"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_status(rows: list[dict], status: str) -> int:
    return sum(1 for row in rows if row["status"] == status)


def supported_ratio(rows: list[dict]) -> float:
    return count_status(rows, "supported") / len(rows) if rows else 0.0


def review_flag(rows: list[dict]) -> str:
    ratio = supported_ratio(rows)
    if ratio < 0.35:
        return "needs_review"
    if ratio < 0.60:
        return "partial_review"
    return "ok"


def write_decisions(path: Path, rows: list[dict]):
    fieldnames = ["image", "position", "base", "replacement", "support", "base_support", "top_weight", "base_weight"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confidence(path: Path, rows: list[dict]):
    fieldnames = [
        "image",
        "position",
        "base_word",
        "top_word",
        "status",
        "candidate_count",
        "base_support",
        "top_support",
        "top_weight",
        "base_weight",
        "top_share",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value) -> str:
    return "" if value == "" else f"{float(value):.3f}"


if __name__ == "__main__":
    main()
