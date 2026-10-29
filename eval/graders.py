"""Deterministic graders for agent runs."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from gta.agent import AgentResult
from gta.tools import registry

_NUM = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+")
_CITE = re.compile(r"\b(?:GO:\d{7}(?!\d)|R-HSA-\d+)")
KNOWLEDGE_TOOLS = {"search_knowledge", "get_term", "annotate_genes"}


def extract_numbers(text: str) -> list[str]:
    """Numeric literals in free text, ignoring list markers and identifiers like GENE_0001."""
    cleaned = _CITE.sub(" ", _LIST_MARKER.sub("", text))
    return [m.replace(",", "") for m in _NUM.findall(cleaned)]


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


def _known_numbers(prompt: str, tool_results: list[dict[str, Any]]) -> set[float]:
    known: set[float] = set()
    for r in tool_results:
        _walk_numbers(r, known)
    known.update(float(n) for n in extract_numbers(prompt))
    return known


def _strictly_matches(literal: str, known: set[float], percent: bool) -> bool:
    value = float(literal)
    scientific = "e" in literal.lower()
    decimals = len(literal.split(".")[1]) if "." in literal and not scientific else 0
    for k in known:
        for cand in (k, k * 100) if percent else (k,):
            if scientific:
                if math.isclose(value, cand, rel_tol=0.05):
                    return True
            elif abs(value - cand) <= 0.5 * 10 ** (-decimals) + 1e-12:
                return True
    return False


def numeric_grounding(answer: str, prompt: str, tool_results: list[dict[str, Any]]) -> float | None:
    """Fraction of answer numbers that match a tool-output (or prompt) value within rounding tolerance.

    Stricter than `1 - hallucinated_number_rate`: the x100 percentage allowance applies only to numbers
    written with a trailing '%'. None if the answer contains no numbers.
    """
    known = _known_numbers(prompt, tool_results)
    cleaned = _CITE.sub(" ", _LIST_MARKER.sub("", answer))
    found = [
        (m.group().replace(",", ""), cleaned[m.end() : m.end() + 1] == "%") for m in _NUM.finditer(cleaned)
    ]
    if not found:
        return None
    return sum(_strictly_matches(n, known, pct) for n, pct in found) / len(found)


def cited_ids(text: str) -> list[str]:
    """GO / Reactome ids in `text`, de-duplicated in order of appearance."""
    return list(dict.fromkeys(_CITE.findall(text)))


def citation_validity(answer: str, prompt: str, tool_results: list[dict[str, Any]]) -> float | None:
    """Fraction of cited ids that some tool returned. Ids quoted from the prompt are not citations. None if none cited."""
    cited = [i for i in cited_ids(answer) if i not in set(cited_ids(prompt))]
    if not cited:
        return None
    retrieved = set(_CITE.findall(json.dumps(tool_results)))
    return sum(i in retrieved for i in cited) / len(cited)


def citation_correctness(
    answer: str, prompt: str, expected: list[str] | None, allowed: list[str] | None
) -> tuple[float | None, float | None]:
    """(precision, recall) of cited ids; None when the task defines no expected/allowed ids.

    Recall: expected ids that are cited. Precision: cited ids (excluding ones given in the prompt) that are in
    the allowed set (expected plus `allowed_ids`); 1.0 when nothing is cited.
    """
    if expected is None and allowed is None:
        return None, None
    expected_set = set(expected or [])
    allowed_set = expected_set | set(allowed or [])
    cited = set(cited_ids(answer))
    own = cited - set(cited_ids(prompt))
    precision = len(own & allowed_set) / len(own) if own else 1.0
    recall = len(expected_set & cited) / len(expected_set) if expected_set else None
    return precision, recall


def tool_selection(expected: list[str], called: list[str], must_refuse: bool) -> float:
    """1.0 if expected tools appear in order (or no tools were called for refusals)."""
    if must_refuse:
        return float(not called)
    it = iter(called)
    return float(all(name in it for name in expected))


_MISSING = object()


def _as_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def values_equal(actual: Any, expected: Any) -> bool:
    """Numbers compare numerically (30 == 30.0 == "30"); strings case-insensitively after strip."""
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected if isinstance(actual, bool) and isinstance(expected, bool) else False
    a, e = _as_number(actual), _as_number(expected)
    if a is not None and e is not None:
        return math.isclose(a, e, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().casefold() == expected.strip().casefold()
    return actual == expected


def _resolve_arg(call: Any, arg: str) -> Any:
    """Argument value of a call; omitted optional args resolve to the tool's Pydantic model default."""
    args = call.args
    if arg in args:
        return args[arg]
    spec = registry.get_spec(call.name)
    field = spec.model.model_fields.get(arg) if spec else None
    if field is None or field.is_required():
        return _MISSING
    return field.get_default(call_default_factory=True)


def _arg_ok(check: dict[str, Any], calls: list[Any]) -> bool:
    """Evaluate a check against the last successful call of the named tool."""
    matching = [c for c in calls if c.name == check["tool"] and not c.is_error]
    if not matching:
        return False
    v = _resolve_arg(matching[-1], check["arg"])
    if v is _MISSING:
        return False
    if "equals" in check and values_equal(v, check["equals"]):
        return True
    if "contains" in check and check["contains"] in str(v):
        return True
    return "regex" in check and re.search(check["regex"], str(v)) is not None


def arg_accuracy(arg_checks: list[dict[str, Any]], calls: list[Any]) -> float:
    """Fraction of argument checks satisfied by the last successful call of the named tool."""
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
    if check["type"] == "not_regex":
        return re.search(check["pattern"], answer, re.IGNORECASE) is None
    expected = _lookup_truth(check["truth"], truth) if "truth" in check else float(check["value"])
    tol = float(check.get("tol", 0))
    return any(abs(float(n) - expected) <= tol for n in extract_numbers(answer))


def grade_task(task: dict[str, Any], result: AgentResult, truth: dict[str, Any]) -> dict[str, float | None]:
    """Compute all per-task metrics."""
    called = [c.name for c in result.calls]
    must_refuse = bool(task.get("must_refuse"))
    sel = tool_selection(task.get("expected_tools", []), called, must_refuse)
    args = arg_accuracy(task.get("arg_checks", []), result.calls)
    answers_ok = all(check_answer(c, result.answer, truth) for c in task.get("answer_checks", []))
    invalid = sum(c.error_kind in ("unknown_tool", "invalid_arguments") for c in result.calls)
    results = [c.result for c in result.calls]
    validity = citation_validity(result.answer, task["prompt"], results)
    precision, recall = citation_correctness(
        result.answer, task["prompt"], task.get("expected_ids"), task.get("allowed_ids")
    )
    retrieval = (
        float(any(c.name in KNOWLEDGE_TOOLS for c in result.calls)) if task.get("must_retrieve") else None
    )
    grounded = all(x in (None, 1.0) for x in (validity, precision, recall, retrieval))
    return {
        "task_success": float(sel == 1.0 and args == 1.0 and answers_ok and grounded and bool(result.answer)),
        "citation_validity": validity,
        "citation_precision": precision,
        "citation_recall": recall,
        "retrieval_used": retrieval,
        "numeric_grounding": numeric_grounding(result.answer, task["prompt"], results),
        "tool_selection_accuracy": sel,
        "arg_accuracy": args,
        "invalid_call_rate": invalid / len(result.calls) if result.calls else 0.0,
        "hallucinated_number_rate": hallucinated_number_rate(result.answer, task["prompt"], results),
        "n_steps": float(result.n_steps),
        "n_duplicate_calls": float(result.n_duplicate_calls),
        "latency_s": result.latency_s,
        "tokens": float(result.prompt_tokens + result.completion_tokens),
    }
