"""Optional LLM judge: is every sentence of an answer supported by the tool outputs? (off by default)"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from gta.llm import LLM

CALIBRATION_FILE = Path(__file__).resolve().parent / "judge_calibration.yaml"
TRUST_THRESHOLD = 0.8
MAX_EVIDENCE_CHARS = 6000

JUDGE_PROMPT = (
    "You are a strict fact checker. Given TOOL OUTPUTS and one SENTENCE from an assistant's answer, decide "
    "whether the sentence is fully supported by the tool outputs (numbers, ids and gene or pathway names must "
    "match). Sentences that only report an error, say something was not found, or refuse are supported if the "
    'tool outputs agree. Reply with JSON only: {"supported": true} or {"supported": false}.'
)
_VERDICT = re.compile(r'"supported"\s*:\s*(true|false)', re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(])|\n+")


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences (and list lines)."""
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def judge_sentence(llm: LLM, sentence: str, evidence: str) -> bool | None:
    """One verdict from the judge, or None if its reply cannot be parsed."""
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": f"TOOL OUTPUTS:\n{evidence[:MAX_EVIDENCE_CHARS]}\n\nSENTENCE:\n{sentence}",
        },
    ]
    match = _VERDICT.search(llm.chat(messages, []).content or "")
    return None if match is None else match.group(1).lower() == "true"


def judge_answer(llm: LLM, answer: str, evidence: str) -> float | None:
    """Fraction of answer sentences the judge deems supported (unparseable verdicts count as unsupported)."""
    sentences = split_sentences(answer)
    if not sentences:
        return None
    return sum(judge_sentence(llm, s, evidence) is True for s in sentences) / len(sentences)


def calibrate(llm: LLM, items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Agreement between the judge and hand labels (an answer is 'supported' iff all its sentences are)."""
    items = items if items is not None else yaml.safe_load(CALIBRATION_FILE.read_text())["items"]
    verdicts = [judge_answer(llm, it["answer"], it["tool_outputs"]) == 1.0 for it in items]
    wrong = [it["id"] for it, v in zip(items, verdicts, strict=True) if v != it["supported"]]
    agreement = 1 - len(wrong) / len(items)
    return {
        "agreement": agreement,
        "n": len(items),
        "disagreements": wrong,
        "trusted": agreement >= TRUST_THRESHOLD,
    }
