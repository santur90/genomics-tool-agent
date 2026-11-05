"""Agent loop with tool calling and JSONL tracing."""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

from gta import tools as _tools  # noqa: F401  (registers tools)
from gta.llm import LLM, LLMError, QuotaExhausted
from gta.sandbox import RUNS_DIR
from gta.tools import registry

MAX_TOOL_CHARS = 4000
MAX_REPEATED_VALIDATION_ERRORS = 3

SYSTEM_PROMPT = (
    (resources.files("gta") / "resources" / "system_prompt.md").read_text(encoding="utf-8").strip()
)


@dataclass
class ToolCallRecord:
    """One executed tool call."""

    name: str
    arguments: str
    result: dict[str, Any]
    latency_s: float
    duplicate: bool = False
    validated_args: dict[str, Any] | None = None

    @property
    def is_error(self) -> bool:
        return "error" in self.result

    @property
    def error_kind(self) -> str | None:
        return self.result.get("kind") if self.is_error else None

    @property
    def args(self) -> dict[str, Any]:
        if self.validated_args is not None:
            return self.validated_args
        try:
            parsed = json.loads(self.arguments)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}

    def to_log(self) -> dict[str, Any]:
        """Audit record: only tool name, arguments, outcome and the tool's own summary (no prompts/secrets)."""
        summary = self.result.get("summary")
        error = self.result.get("error")
        return {
            "name": self.name,
            "args": self.args,
            "ok": not self.is_error,
            "error": None if error is None else str(error),
            "result_summary": summary if isinstance(summary, dict) else None,
            "result_text": _truncate(redact_secrets(json.dumps(self.result, default=str))),
        }


@dataclass
class AgentResult:
    """Outcome of one agent run."""

    task: str
    answer: str
    calls: list[ToolCallRecord] = field(default_factory=list)
    n_steps: int = 0
    n_duplicate_calls: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stopped: str = "final_answer"
    termination_reason: str = "final_answer"
    trace_path: Path | None = None
    llm_error: LLMError | None = None
    messages: list[dict[str, Any]] = field(default_factory=list, repr=False)
    verifier: str = "off"
    verifier_issues: list[str] = field(default_factory=list)
    verifier_issues_after: list[str] = field(default_factory=list)
    answer_before_revision: str | None = None
    n_revisions: int = 0
    extra_llm_calls: int = 0


def _call_key(name: str, args: str) -> str:
    try:
        return name + json.dumps(json.loads(args), sort_keys=True)
    except ValueError:
        return name + args


def _truncate(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)
_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.DOTALL),
    re.compile(
        r"""(?i)\b([\w-]*(?:api[_-]?key|token|secret|password)[\w-]*["']?\s*[:=]\s*["']?)[^\s"',}]+"""
    ),
]


def redact_secrets(text: str) -> str:
    """Mask environment secret values and common credential patterns."""
    for name, value in os.environ.items():
        if _SECRET_ENV.search(name) and len(value) >= 6:
            text = text.replace(value, "[REDACTED]")
    text = _SECRET_PATTERNS[0].sub(r"\1 [REDACTED]", text)
    text = _SECRET_PATTERNS[1].sub("[REDACTED]", text)
    text = _SECRET_PATTERNS[2].sub("[REDACTED]", text)
    return _SECRET_PATTERNS[3].sub(r"\1[REDACTED]", text)


class _Trace:
    def __init__(self, trace_dir: Path | None) -> None:
        if trace_dir is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            trace_dir = RUNS_DIR / f"{stamp}-{secrets.token_hex(2)}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = trace_dir / "trace.jsonl"

    def log(self, **event: Any) -> None:
        with open(self.path, "a") as fh:
            fh.write(
                json.dumps({"ts": datetime.now().isoformat(timespec="milliseconds"), **event}, default=str)
                + "\n"
            )


def _force_final_answer(
    llm: LLM,
    messages: list[dict[str, Any]],
    result: AgentResult,
    trace: _Trace,
    repeated: tuple[str, str],
    step: int,
) -> None:
    """Stop tool use after repeated identical validation errors and obtain a final report."""
    name, error = repeated
    result.stopped = "repeated_validation_error"
    trace.log(type="repeated_validation_error", step=step, name=name, error=error)
    messages.append(
        {
            "role": "user",
            "content": (
                f"Tool calls are disabled: {name} failed with the same validation error "
                f"{MAX_REPEATED_VALIDATION_ERRORS} times. Do not call any more tools; "
                f"report this validation error to the user: {error}"
            ),
        }
    )
    fallback = f"I could not complete the request: {name} repeatedly failed validation. {error}"
    try:
        resp = llm.chat(messages, [])
        result.prompt_tokens += resp.usage.get("prompt_tokens", 0)
        result.completion_tokens += resp.usage.get("completion_tokens", 0)
        answer = resp.content if resp.content and not resp.tool_calls else fallback
    except QuotaExhausted:
        raise
    except Exception as exc:  # the forced-final call must never crash a run
        trace.log(type="forced_final_error", step=step, error=f"{type(exc).__name__}: {exc}")
        answer = fallback
    result.answer = answer


def _run_loop(
    task: str,
    llm: LLM,
    max_steps: int = 8,
    trace_dir: Path | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    _messages: list[dict[str, Any]] | None = None,
) -> AgentResult:
    """Run the tool-calling loop until a final answer or max_steps."""
    trace = _Trace(trace_dir)
    emit = on_event or (lambda e: None)
    specs = registry.openai_tool_specs()
    messages: list[dict[str, Any]] = (
        list(_messages)
        if _messages is not None
        else [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
    )
    trace.log(type="start", model=getattr(llm, "model", ""), messages=messages)
    result = AgentResult(task=task, answer="", trace_path=trace.path)
    cache: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    last_invalid: tuple[str, str] | None = None
    invalid_streak = 0
    repeated: tuple[str, str] | None = None
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        t = time.perf_counter()
        try:
            resp = llm.chat(messages, specs)
        except LLMError as exc:
            trace.log(
                type="llm_error",
                step=step,
                error=str(exc),
                retries=exc.retries,
                quota=isinstance(exc, QuotaExhausted),
            )
            if isinstance(exc, QuotaExhausted):
                raise
            result.n_steps = step
            result.stopped = "llm_error"
            result.llm_error = exc
            break
        result.n_steps = step
        result.prompt_tokens += resp.usage.get("prompt_tokens", 0)
        result.completion_tokens += resp.usage.get("completion_tokens", 0)
        trace.log(
            type="llm",
            step=step,
            content=resp.content,
            tool_calls=[c.name for c in resp.tool_calls],
            latency_s=time.perf_counter() - t,
            usage=resp.usage,
            retries=resp.retries,
        )
        if not resp.tool_calls:
            result.answer = resp.content or ""
            emit({"type": "answer", "content": result.answer})
            break
        calls = [
            (
                c.id or f"call_{step}_{i}",
                c.name,
                c.arguments if isinstance(c.arguments, str) else json.dumps(c.arguments or {}),
            )
            for i, c in enumerate(resp.tool_calls)
        ]
        messages.append(
            {
                "role": "assistant",
                "content": resp.content or "",
                "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
                    for cid, name, args in calls
                ],
            }
        )
        for cid, name, args in calls:
            if repeated is not None:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": json.dumps({"error": "Tool execution stopped.", "kind": "skipped"}),
                    }
                )
                continue
            ts = time.perf_counter()
            key = _call_key(name, args)
            duplicate = key in cache
            if duplicate:
                cached, validated = cache[key]
                out = {**cached, "note": "duplicate call, cached result"}
                result.n_duplicate_calls += 1
            else:
                out, validated = registry.call_validated(name, args)
                if "error" not in out:
                    cache[key] = (out, validated)
            rec = ToolCallRecord(name, args, out, time.perf_counter() - ts, duplicate, validated)
            result.calls.append(rec)
            trace.log(
                type="tool",
                step=step,
                name=name,
                arguments=args,
                latency_s=rec.latency_s,
                error=out.get("error"),
                duplicate=duplicate,
                result=out,
            )
            emit({"type": "tool", "step": step, "name": name, "arguments": args, "result": out})
            messages.append(
                {"role": "tool", "tool_call_id": cid, "content": _truncate(json.dumps(out, default=str))}
            )
            if rec.error_kind == "invalid_arguments":
                ident = (name, str(out.get("error")))
                invalid_streak = invalid_streak + 1 if ident == last_invalid else 1
                last_invalid = ident
                if invalid_streak >= MAX_REPEATED_VALIDATION_ERRORS:
                    repeated = ident
            else:
                last_invalid, invalid_streak = None, 0
        if repeated is not None:
            _force_final_answer(llm, messages, result, trace, repeated, step)
            break
    else:
        result.stopped = "max_steps"
    result.termination_reason = result.stopped
    result.messages = messages
    result.latency_s = time.perf_counter() - t0
    trace.log(
        type="end",
        stopped=result.stopped,
        n_steps=result.n_steps,
        n_duplicate_calls=result.n_duplicate_calls,
        answer=result.answer,
    )
    return result


# ---------------------------------------------------------------- optional verifier stage

REVISION_PROMPT = (
    "Your answer has these problems:\n{issues}\n"
    "Correct the answer using only the tool results above. If an input the user specified was "
    "unavailable, say so explicitly and name it; do not silently use a different input and do not "
    "invent results. If you cannot complete the task, explain why."
)
MAX_REVISION_STEPS = 3


def _record_to_call(rec: Any) -> dict[str, Any]:
    """ToolCallRecord -> the plain dict the verifier reads (name, args, ok, error, result_text)."""
    import dataclasses

    if dataclasses.is_dataclass(rec):
        vals = [getattr(rec, f.name) for f in dataclasses.fields(rec)]
        name, raw_args, out, validated = vals[0], vals[1], vals[2], vals[5]  # constructor order
    else:  # pragma: no cover - fallback if the record is not a dataclass
        name, raw_args, out, validated = rec.name, rec.args, rec.out, rec.validated
    if isinstance(validated, dict):
        args = validated
    else:
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
        except ValueError:
            args = {}
    ok = "error" not in out
    return {
        "name": name,
        "args": args,
        "ok": ok,
        "error": None if ok else str(out.get("error")),
        "result_text": json.dumps(out, default=str)[:20000] if ok else None,
    }


def _verify_and_revise(
    task: str,
    llm: LLM,
    result: AgentResult,
    max_steps: int,
    trace_dir: Path | None,
    on_event: Callable[[dict[str, Any]], None] | None,
) -> AgentResult:
    from gta.verify import format_issues, verify

    if result.stopped == "llm_error":
        return result
    report = verify(task, [_record_to_call(r) for r in result.calls], result.answer)
    result.verifier_issues = report.types
    result.verifier_issues_after = report.types
    if not report.flagged:
        return result

    messages = list(result.messages)
    if not messages or messages[-1].get("role") != "assistant":
        messages.append({"role": "assistant", "content": result.answer or "(no final answer was produced)"})
    messages.append({"role": "user", "content": REVISION_PROMPT.format(issues=format_issues(report))})
    try:
        rev = _run_loop(task, llm, MAX_REVISION_STEPS, trace_dir, on_event, _messages=messages)
    except QuotaExhausted:
        raise
    except LLMError:  # keep the first answer if the revision call fails
        return result
    if rev.stopped == "llm_error":
        return result

    result.answer_before_revision = result.answer
    result.n_revisions = 1
    result.extra_llm_calls = rev.n_steps
    result.calls.extend(rev.calls)
    result.n_steps += rev.n_steps
    result.n_duplicate_calls += rev.n_duplicate_calls
    result.prompt_tokens += rev.prompt_tokens
    result.completion_tokens += rev.completion_tokens
    result.latency_s += rev.latency_s
    result.answer = rev.answer
    result.stopped = rev.stopped
    result.termination_reason = rev.termination_reason
    result.messages = rev.messages
    after = verify(task, [_record_to_call(r) for r in result.calls], result.answer)
    result.verifier_issues_after = after.types
    return result


def run_agent(
    task: str,
    llm: LLM,
    max_steps: int = 8,
    trace_dir: Path | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    verifier: str | None = None,
) -> AgentResult:
    """Run the tool-calling loop; optionally verify the answer and allow ONE revision round.

    verifier: "off" (default) or "rules". When None, the GTA_VERIFIER environment variable is used.
    """
    import os

    mode = (verifier or os.environ.get("GTA_VERIFIER") or "off").strip().lower()
    if mode not in {"off", "rules"}:
        raise ValueError(f"unknown verifier mode: {mode!r} (use 'off' or 'rules')")
    result = _run_loop(task, llm, max_steps, trace_dir, on_event)
    result.verifier = mode
    if mode == "off":
        return result
    return _verify_and_revise(task, llm, result, max_steps, trace_dir, on_event)
