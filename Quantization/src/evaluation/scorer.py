from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Try to parse the first JSON object in the text."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def score_output(prompt_id: str, text: str) -> dict[str, Any]:
    """Run lightweight task-compliance checks."""
    lower = text.lower()
    checks: list[tuple[str, bool]] = []

    if prompt_id == "math_reasoning":
        checks = [
            (
                "Contains the required FINAL label",
                bool(re.search(r"FINAL\s*:\s*174\b", text, re.I)),
            ),
            (
                "Shows subtraction/shipping calculation",
                "84" in text or "156" in text,
            ),
            ("Final answer is correct", "174" in text),
            (
                "Avoids a conflicting final number",
                not bool(
                    re.search(
                        r"FINAL\s*:\s*(?!174\b)\d+",
                        text,
                        re.I,
                    )
                ),
            ),
        ]

    elif prompt_id == "json_extraction":
        obj = extract_json_object(text)
        expected_keys = {
            "name",
            "city",
            "product",
            "price_usd",
            "purchase_date",
        }

        checks = [
            ("Output contains parseable JSON", obj is not None),
            (
                "Uses exactly the requested keys",
                obj is not None and set(obj.keys()) == expected_keys,
            ),
            (
                "Extracts Nora, Cairo, and laptop",
                obj is not None
                and str(obj.get("name", "")).lower() == "nora"
                and str(obj.get("city", "")).lower() == "cairo"
                and str(obj.get("product", "")).lower() == "laptop",
            ),
            (
                "Extracts price and date",
                obj is not None
                and str(obj.get("price_usd", "")) in {"1250", "1250.0"}
                and "2025" in str(obj.get("purchase_date", "")),
            ),
        ]

    elif prompt_id == "coding":
        checks = [
            (
                "Defines chunk_list",
                bool(re.search(r"def\s+chunk_list\s*\(", text)),
            ),
            ("Raises ValueError", "raise ValueError" in text),
            (
                "Checks size <= 0",
                bool(re.search(r"size\s*<=\s*0", text)),
            ),
            (
                "Builds consecutive slices",
                "range(" in text and "items[" in text and ":" in text,
            ),
        ]

    elif prompt_id == "rag_explanation":
        words = re.findall(
            r"\b[\w'-]+\b",
            text,
            flags=re.UNICODE,
        )

        analogy_markers = [
            "like",
            "imagine",
            "as if",
            "مثل",
            "تخيل",
            "كأن",
        ]
        retrieval_markers = [
            "retriev",
            "search",
            "look up",
            "find",
            "يبحث",
            "استرجاع",
        ]
        generation_markers = [
            "answer",
            "generate",
            "write",
            "respond",
            "إجابة",
            "يجيب",
            "يولّد",
        ]

        checks = [
            ("At most 80 words", len(words) <= 80),
            (
                "Includes an analogy",
                any(marker in lower for marker in analogy_markers),
            ),
            (
                "Explains retrieval",
                any(marker in lower for marker in retrieval_markers),
            ),
            (
                "Explains answer generation",
                any(marker in lower for marker in generation_markers),
            ),
        ]

    elif prompt_id == "quantization_summary":
        bullet_lines = [
            line.strip()
            for line in text.splitlines()
            if re.match(r"^\s*[-*•]\s+", line)
        ]

        bullet_word_counts = [
            len(
                re.findall(
                    r"\b[\w'-]+\b",
                    line,
                    flags=re.UNICODE,
                )
            )
            for line in bullet_lines
        ]

        checks = [
            ("Has exactly 3 bullet points", len(bullet_lines) == 3),
            (
                "Every bullet is under 18 words",
                len(bullet_lines) == 3
                and all(count < 18 for count in bullet_word_counts),
            ),
            ("Mentions memory", "memory" in lower),
            (
                "Preserves speed and quality ideas",
                "speed" in lower and "quality" in lower,
            ),
        ]

    passed = sum(int(value) for _, value in checks)
    score = round(100 * passed / len(checks), 1) if checks else 0.0

    return {
        "automatic_compliance_score": score,
        "checks": [
            {"criterion": name, "passed": value}
            for name, value in checks
        ],
    }
