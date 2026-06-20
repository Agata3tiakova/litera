from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)?")
PUNCT_RE = re.compile(r"[.,!?;:()\"«»]")


@dataclass(frozen=True)
class WordDiff:
    expected: str
    actual: str


def analyze_ocr_errors(expected: str, actual: str) -> dict[str, Any]:
    expected_normalized_text = normalize_for_text_comparison(expected)
    actual_normalized_text = normalize_for_text_comparison(actual)
    expected_words = extract_words(expected_normalized_text)
    actual_words = extract_words(actual_normalized_text)
    operations = align_words(expected_words, actual_words)

    substitutions = []
    missing_words = []
    extra_words = []

    for tag, left_items, right_items in operations:
        if tag == "replace":
            substitutions.extend(
                WordDiff(expected=left, actual=right)
                for left, right in pair_replacements(left_items, right_items)
            )
            pair_count = min(len(left_items), len(right_items))
            missing_words.extend(left_items[pair_count:])
            extra_words.extend(right_items[pair_count:])
        elif tag == "delete":
            missing_words.extend(left_items)
        elif tag == "insert":
            extra_words.extend(right_items)

    expected_punctuation = Counter(PUNCT_RE.findall(expected))
    actual_punctuation = Counter(PUNCT_RE.findall(actual))

    return {
        "word_substitutions_count": len(substitutions),
        "missing_words_count": len(missing_words),
        "extra_words_count": len(extra_words),
        "punctuation_delta_count": punctuation_delta_count(expected_punctuation, actual_punctuation),
        "line_break_delta": actual.count("\n") - expected.count("\n"),
        "hyphenated_line_breaks": actual.count("-\n"),
        "top_substitutions": [
            {"expected": diff.expected, "actual": diff.actual}
            for diff in substitutions[:12]
        ],
        "missing_words": missing_words[:12],
        "extra_words": extra_words[:12],
        "expected_punctuation": dict(expected_punctuation),
        "actual_punctuation": dict(actual_punctuation),
    }


def extract_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def normalize_for_text_comparison(text: str) -> str:
    text = re.sub(
        r"([A-Za-zА-Яа-яЁё]+)-\s*\n\s*([A-Za-zА-Яа-яЁё]+)",
        r"\1\2",
        text,
    )
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_word(word: str) -> str:
    return word.replace("ё", "е").replace("Ё", "Е").lower()


def align_words(
    expected_words: list[str],
    actual_words: list[str],
) -> list[tuple[str, list[str], list[str]]]:
    expected_normalized = [normalize_word(word) for word in expected_words]
    actual_normalized = [normalize_word(word) for word in actual_words]
    matcher = SequenceMatcher(None, expected_normalized, actual_normalized, autojunk=False)

    operations = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        operations.append((tag, expected_words[i1:i2], actual_words[j1:j2]))
    return operations


def pair_replacements(left_items: list[str], right_items: list[str]) -> list[tuple[str, str]]:
    pair_count = min(len(left_items), len(right_items))
    return [
        (left_items[index], right_items[index])
        for index in range(pair_count)
    ]


def punctuation_delta_count(expected: Counter[str], actual: Counter[str]) -> int:
    chars = set(expected) | set(actual)
    return sum(abs(expected.get(char, 0) - actual.get(char, 0)) for char in chars)
