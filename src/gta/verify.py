"""Deterministic, rule-based answer verifier.

The verifier sees ONLY: the user prompt, the tool calls (name, args, ok, error, result_text /
result_summary) and the final answer. It never reads expected values, task files or graders.

Checks
------
unsupported_number    a number in the answer that is in neither the prompt nor any tool output
unsupported_citation  a GO:/R-HSA- id in the answer that is in neither the prompt nor any tool output
input_substitution    a user-specified input failed (not found / unknown), a later call of the same tool
                      succeeded with a different value, and the answer never mentions the original value
unreported_error      the run ended on a not-found style error and the answer neither names the missing
                      input nor says anything is missing
unresolved_error      the run ended on a failed call and the answer is empty, or states results without
                      acknowledging the failure
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

_NUM = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+")
_ID = re.compile(r"\b(?:GO:\d{7}|R-HSA-\d+)\b")
_IDLIKE = re.compile(r"\b(?:GO:\d+|R-HSA-\d+)\b")
_TOKEN_WITH_DIGITS = re.compile(r"\b[A-Za-z][A-Za-z0-9]*[_-]\d+\b")  # GENE_0076, patient_001, SET_02

# error text that means "the thing you asked for is not there"
NOT_FOUND = re.compile(
    r"not found|does not exist|doesn't exist|no such|missing|unknown|not registered|not in the|"
    r"not allowed|unavailable|cannot find|could not be found|not present|no column",
    re.IGNORECASE,
)
# answer text that acknowledges something is missing
ACKNOWLEDGES = re.compile(
    r"not found|does not exist|doesn't exist|no such|missing|unknown|not registered|not in the|"
    r"not allowed|unavailable|cannot find|can't find|could not be found|could not find|couldn't find|"
    r"not present|not available|no column|failed|error",
    re.IGNORECASE,
)
_PATHLIKE = re.compile(r"[\w./-]+\.[A-Za-z]{2,5}(?:\.gz)?")


@dataclass
class Issue:
    type: str
    detail: str
    evidence: str = ""


@dataclass
class VerifyReport:
    issues: list[Issue] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.issues)

    @property
    def types(self) -> list[str]:
        return [i.type for i in self.issues]


# ---------------------------------------------------------------- number handling


def _clean_for_numbers(text: str) -> str:
    text = _IDLIKE.sub(" ", text)
    text = _TOKEN_WITH_DIGITS.sub(" ", text)
    return _LIST_MARKER.sub("", text)


def extract_numbers(text: str) -> list[tuple[str, float]]:
    """Numbers in free text, ignoring ids, gene/sample tokens, list markers and 4-digit years."""
    out: list[tuple[str, float]] = []
    for m in _NUM.finditer(_clean_for_numbers(text or "")):
        s = m.group(0)
        try:
            v = float(s.replace(",", ""))
        except ValueError:
            continue
        if re.fullmatch(r"(19|20)\d\d", s):
            continue
        out.append((s, v))
    return out


def _decimals(s: str) -> int:
    mant = re.split(r"[eE]", s)[0]
    return len(mant.split(".")[1]) if "." in mant else 0


def _matches(s: str, x: float, y: float) -> bool:
    if x == y:
        return True
    if "e" in s.lower() or (y != 0 and abs(y) < 1e-3):
        return abs(x - y) <= 0.01 * max(abs(y), 1e-300)
    return abs(x - y) <= 0.5 * 10 ** (-_decimals(s)) + 1e-12


def _supported(s: str, x: float, pool: list[float]) -> bool:
    return any(_matches(s, x, y) for y in pool)


# ---------------------------------------------------------------- helpers


def _output_text(call: dict[str, Any]) -> str:
    text = call.get("result_text")
    if text:
        return str(text)
    summary = call.get("result_summary")
    return json.dumps(summary) if summary else ""


def _tool_text(calls: list[dict[str, Any]]) -> str:
    parts = []
    for c in calls:
        parts.append(_output_text(c))
        if c.get("error"):
            parts.append(str(c["error"]))
    return "\n".join(parts)


def _mentions(answer: str, value: str) -> bool:
    a = (answer or "").lower()
    v = value.lower()
    return v in a or os.path.basename(v) in a


# ---------------------------------------------------------------- checks


def _check_numbers_and_citations(prompt: str, calls: list[dict[str, Any]], answer: str, rep: VerifyReport) -> None:
    ok_outputs = [c for c in calls if c.get("ok") and _output_text(c)]
    ans_nums = extract_numbers(answer)
    ans_ids = set(_ID.findall(answer or ""))
    prompt_pool = [v for _, v in extract_numbers(prompt)]
    if not ok_outputs:
        new_ids = [i for i in ans_ids if i not in (prompt or "")]
        if [1 for s, x in ans_nums if not _supported(s, x, prompt_pool)] or new_ids:
            rep.skipped.append("number/citation checks skipped: no tool output stored for this run")
        return
    pool = prompt_pool + [v for _, v in extract_numbers(_tool_text(calls))]
    for s, x in ans_nums:
        if not _supported(s, x, pool):
            rep.issues.append(Issue("unsupported_number", f"{s} is in neither the prompt nor any tool output", s))
    haystack = (prompt or "") + "\n" + _tool_text(calls)
    for i in sorted(ans_ids):
        if i not in haystack:
            rep.issues.append(Issue("unsupported_citation", f"{i} was never returned by a tool", i))


def _check_substitution(prompt: str, calls: list[dict[str, Any]], answer: str, rep: VerifyReport) -> None:
    seen: set[tuple[str, str, str]] = set()
    for i, call in enumerate(calls):
        if call.get("ok") or not NOT_FOUND.search(str(call.get("error") or "")):
            continue
        for key, val in (call.get("args") or {}).items():
            if not isinstance(val, str) or len(val) < 3 or val.lower() in {"null", "none"}:
                continue
            if val.lower() not in (prompt or "").lower():
                continue  # the model chose this value itself, the user did not specify it
            for later in calls[i + 1 :]:
                new = (later.get("args") or {}).get(key)
                if later.get("name") == call.get("name") and later.get("ok") and isinstance(new, str) and new != val:
                    tag = (call["name"], key, val)
                    if tag not in seen and not _mentions(answer, val):
                        seen.add(tag)
                        rep.issues.append(
                            Issue(
                                "input_substitution",
                                f"{call['name']}.{key}: '{val}' failed, then '{new}' was used; the answer never mentions '{val}'",
                                str(call.get("error")),
                            )
                        )
                    break


def _check_final_failure(prompt: str, calls: list[dict[str, Any]], answer: str, rep: VerifyReport) -> None:
    if not calls or calls[-1].get("ok"):
        return
    last = calls[-1]
    err = str(last.get("error") or "")
    if not (answer or "").strip():
        rep.issues.append(Issue("unresolved_error", "run ended on a failed call and the answer is empty", err))
        return
    acknowledges = bool(ACKNOWLEDGES.search(answer))
    prompt_pool = [v for _, v in extract_numbers(prompt)]
    states_results = any(not _supported(s, x, prompt_pool) for s, x in extract_numbers(answer))
    if states_results and not acknowledges:
        rep.issues.append(Issue("unresolved_error", "answer states numbers but the last call failed", err))
        return
    if NOT_FOUND.search(err):
        missing = [t for t in _PATHLIKE.findall(err)]
        missing += [v for v in (last.get("args") or {}).values() if isinstance(v, str) and len(v) >= 3 and v in err]
        names_it = any(_mentions(answer, m) for m in missing)
        if not names_it and not acknowledges:
            rep.issues.append(
                Issue("unreported_error", "run ended on a not-found error; the answer does not say what is missing", err)
            )


def verify(prompt: str, calls: list[dict[str, Any]], answer: str) -> VerifyReport:
    """Run all checks. `calls` items: name, args, ok, error, and result_text and/or result_summary."""
    rep = VerifyReport()
    calls = calls or []
    _check_numbers_and_citations(prompt or "", calls, answer or "", rep)
    _check_substitution(prompt or "", calls, answer or "", rep)
    _check_final_failure(prompt or "", calls, answer or "", rep)
    return rep


def format_issues(rep: VerifyReport) -> str:
    """Plain-text list for a revision message."""
    return "\n".join(f"- {i.type}: {i.detail}" for i in rep.issues)
