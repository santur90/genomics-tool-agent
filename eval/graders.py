"""Deterministic graders for agent runs."""
from __future__ import annotations

import math
import re
from typing import Any

from gta.agent import AgentResult

_NUM = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+")


def extract_numbers(text: str) -> list[str]:
    """Numeric literals in free text, ignoring list markers and identifiers like GENE_0001."""
    return [m.replace(",", "") for m in _NUM.findall(_LIST_MARKER.sub("", text))]


def _walk_numbers(node: Any, out: set[float]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.add(float(node))
    elif isinstance(node, str):
        out.update(float(n) for n in extract_numbers(node))
    elif isinstance(node, dict):
        for v in node.values():
            _walk_numbers(v, out)
    elif isinstance(node, list):
        out.add(float(len(node)))
        for v in node:
            _walk_numbers(v, out)


def _matches(literal: str, known: set[float]) -> bool:
    value = float(literal)
    scientific = "e" in literal.lower()
    decimals = len(literal.split(".")[1]) if "." in literal and not scientific else 0
    for k in known:
        for cand in (k, k * 100):
            if scientific:
                if math.isclose(value, cand, rel_tol=0.05):
                    return True
            elif abs(value - cand) <= 0.5 * 10 ** (-decimals) + 1e-12:
                return True
    return False


def hallucinated_number_rate(answer: str, prompt: str, tool_results: list[dict[str, Any]]) -> float:
    """Fraction of distinct numbers in the answer found neither in the prompt nor in any tool output."""
    known: set[float] = set()
    for r in tool_results:
        _walk_numbers(r, known)
    known.update(float(n) for n in extract_numbers(prompt))
    numbers = set(extract_numbers(answer))
    if not numbers:
        return 0.0
    return sum(not _matches(n, known) for n in numbers) / len(numbers)


def tool_selection(expected: list[str], called: list[str], must_refuse: bool) -> float:
    """1.0 if expected tools appear in order (or no tools were called for refusals)."""
    if must_refuse:
        return float(not called)
    it = iter(called)
    return float(all(name in it for name in expected))


def _arg_ok(check: dict[str, Any], calls: list[Any]) -> bool:
    for c in calls:
        if c.name != check["tool"] or check["arg"] not in c.args:
            continue
        v = c.args[check["arg"]]
        if "equals" in check:
            exp = check["equals"]
            if isinstance(exp, (int, float)) and isinstance(v, (int, float)):
                if math.isclose(float(v), float(exp), abs_tol=1e-9):
                    return True
            elif v == exp:
                return True
        if "contains" in check and check["contains"] in str(v):
            return True
        if "regex" in check and re.search(check["regex"], str(v)):
            return True
    return False


def arg_accuracy(arg_checks: list[dict[str, Any]], calls: list[Any]) -> float:
    """Fraction of argument checks satisfied by at least one call."""
    if not arg_checks:
        return 1.0
    return sum(_arg_ok(c, calls) for c in arg_checks) / len(arg_checks)


def _lookup_truth(ref: str, truth: dict[str, Any]) -> float:
    node: Any = truth
    for part in ref.split("."):
        node = node[part]
    return float(node)


def check_answer(check: dict[str, Any], answer: str, truth: dict[str, Any]) -> bool:
    """Evaluate one answer check (number with tolerance, or regex)."""
    if check["type"] == "regex":
        return re.search(check["pattern"], answer, re.IGNORECASE) is not None
    expected = _lookup_truth(check["truth"], truth) if "truth" in check else float(check["value"])
    tol = float(check.get("tol", 0))
    return any(abs(float(n) - expected) <= tol for n in extract_numbers(answer))


def grade_task(task: dict[str, Any], result: AgentResult, truth: dict[str, Any]) -> dict[str, float]:
    """Compute all per-task metrics."""
    called = [c.name for c in result.calls]
    must_refuse = bool(task.get("must_refuse"))
    sel = tool_selection(task.get("expected_tools", []), called, must_refuse)
    args = arg_accuracy(task.get("arg_checks", []), result.calls)
    answers_ok = all(check_answer(c, result.answer, truth) for c in task.get("answer_checks", []))
    invalid = sum(c.error_kind in ("unknown_tool", "invalid_arguments") for c in result.calls)
    return {
        "task_success": float(sel == 1.0 and args == 1.0 and answers_ok and bool(result.answer)),
        "tool_selection_accuracy": sel,
        "arg_accuracy": args,
        "invalid_call_rate": invalid / len(result.calls) if result.calls else 0.0,
        "hallucinated_number_rate": hallucinated_number_rate(result.answer, task["prompt"], [c.result for c in result.calls]),
        "n_steps": float(result.n_steps),
        "latency_s": result.latency_s,
        "tokens": float(result.prompt_tokens + result.completion_tokens),
    }
