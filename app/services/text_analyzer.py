import re
import json
import logging
import requests
from typing import Dict, List
from .yandex_iam import YandexIAMTokenProvider

logger = logging.getLogger(__name__)

def clean_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    return re.split(r"[.!?]+", text)

def build_prompt(text: str) -> str:
    return f"""
Ты — профессиональный лингвист и корректор русского языка.

Проанализируй текст и найди ВСЕ ошибки.

Для каждой ошибки укажи:
- fragment — слово или фрагмент с ошибкой
- type — один из: Орфография, Пунктуация, Грамматика, Лексика, Стилистика
- explanation — краткое объяснение
- correction — исправленный вариант (если применимо)
- rule — правило русского языка (если применимо)

Верни строго валидный JSON с ключом "errors".

Текст:
{text}
"""

_iam_providers = {}


def get_iam_provider(sa_key_path: str) -> YandexIAMTokenProvider:
    if sa_key_path not in _iam_providers:
        _iam_providers[sa_key_path] = YandexIAMTokenProvider(sa_key_path)
    return _iam_providers[sa_key_path]


def call_llm(text: str, model_uri: str, sa_key_path: str) -> dict:
    if not model_uri:
        return {"errors": []}

    iam_provider = get_iam_provider(sa_key_path)
    iam_token = iam_provider.get_token()
    payload = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 2000
        },
        "messages": [
            {"role": "system", "text": "Ты профессиональный корректор русского языка."},
            {"role": "user", "text": build_prompt(text)}
        ]
    }

    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json"
    }
    r = None
    text_result = ""
    try:
        r = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers,
            json=payload,
            timeout=30
        )
        r.raise_for_status()
        response_json = r.json()

        text_result = response_json["result"]["alternatives"][0]["message"]["text"]

        match = re.search(r"{[\s\S]*}", text_result)
        if match:
            return json.loads(match.group(0))
        else:
            return {"errors": []}

    except requests.HTTPError as e:
        logger.exception("YandexGPT HTTP error: %s", r.text if r else "")
        return {"errors": []}
    except json.JSONDecodeError as e:
        logger.exception("JSON decode error. Raw response: %s", text_result)
        return {"errors": []}
    except Exception:
        logger.exception("Unexpected YandexGPT error")
        return {"errors": []}

def map_severity(error_type: str) -> str:
    return {
        "Орфография": "высокая",
        "Грамматика": "высокая",
        "Пунктуация": "средняя",
        "Лексика": "средняя",
        "Стилистика": "низкая"
    }.get(error_type, "средняя")


def build_error_ui(err: Dict, text: str) -> Dict:
    fragment = err.get("fragment", "")
    pos = text.find(fragment)

    before = text[max(0, pos - 30):pos] if pos != -1 else ""
    after = text[pos + len(fragment):pos + len(fragment) + 30] if pos != -1 else ""

    return {
        "type": f"{err.get('type', '')} ошибка",
        "severity": map_severity(err.get("type")),
        "description": err.get("explanation", ""),
        "word": fragment,
        "context_before": before,
        "context_after": after,
        "correction": err.get("correction"),
        "rule": err.get("rule")
    }


def group_errors(errors: List[Dict], text: str) -> Dict:
    grouped = {
        "spelling": [],
        "punctuation": [],
        "style": []
    }

    for err in errors:
        ui_error = build_error_ui(err, text)
        etype = err.get("type")

        if etype in {"Орфография", "Грамматика"}:
            grouped["spelling"].append(ui_error)
        elif etype == "Пунктуация":
            grouped["punctuation"].append(ui_error)
        elif etype in {"Стилистика", "Лексика"}:
            grouped["style"].append(ui_error)

    return grouped


def score_from_count(count: int) -> int:
    return max(0, 10 - count)


def lexical_richness(text: str) -> str:
    words = re.findall(r"\w+", text.lower())
    if not words:
        return "низкое"

    ratio = len(set(words)) / len(words)

    if ratio > 0.6:
        return "высокое"
    if ratio > 0.4:
        return "среднее"
    return "низкое"

def build_recommendations(stats: Dict) -> List[Dict]:
    recs = []

    if stats["spelling_errors"] > 3:
        recs.append({
            "priority": "high",
            "title": "Улучшите орфографию",
            "description": "В тексте много орфографических ошибок.",
            "example": "Используйте автоматическую проверку орфографии."
        })

    if stats["style_errors"] > 2:
        recs.append({
            "priority": "medium",
            "title": "Поработайте над стилем",
            "description": "Некоторые формулировки можно сделать более выразительными."
        })

    return recs

def analyze_text(text: str, config) -> Dict:
    text = clean_text(text)
    llm_result = call_llm(text, config.get("LLM_MODEL"), config.get("SA_KEY_PATH"))
    raw_errors = llm_result.get("errors", [])

    errors_grouped = group_errors(raw_errors, text)

    metrics = {
        "spelling": {
            "score": score_from_count(len(errors_grouped["spelling"])),
            "error_count": len(errors_grouped["spelling"])
        },
        "punctuation": {
            "score": score_from_count(len(errors_grouped["punctuation"])),
            "error_count": len(errors_grouped["punctuation"])
        },
        "style": {
            "score": score_from_count(len(errors_grouped["style"])),
            "error_count": len(errors_grouped["style"])
        },
        "lexical": {
            "score": 8,
            "richness": lexical_richness(text)
        }
    }

    overall_score = round(
        (metrics["spelling"]["score"]
         + metrics["punctuation"]["score"]
         + metrics["style"]["score"]
         + metrics["lexical"]["score"]) / 4
    )

    words = re.findall(r"\w+", text.lower())
    sentences = split_sentences(text)

    statistics = {
        "total_errors": sum(len(v) for v in errors_grouped.values()),
        "spelling_errors": len(errors_grouped["spelling"]),
        "punctuation_errors": len(errors_grouped["punctuation"]),
        "style_errors": len(errors_grouped["style"]),
        "unique_words": len(set(words)),
        "avg_sentence_length": round(len(words) / max(1, len(sentences)))
    }

    recommendations = build_recommendations(statistics)

    return {
        "overall_score": overall_score,
        "metrics": metrics,
        "errors": errors_grouped,
        "recommendations": recommendations,
        "statistics": statistics
    }
