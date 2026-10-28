"""Agent loop with tool calling and JSONL tracing."""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gta import tools as _tools  # noqa: F401  (registers tools)
from gta.llm import LLM, LLMError, QuotaExhausted
from gta.sandbox import RUNS_DIR
from gta.tools import registry

MAX_TOOL_CHARS = 4000

SYSTEM_PROMPT = """You are a genomics analysis assistant. You can only act through the provided tools.
Rules:
- Only state numbers that appear in tool outputs (or in the user's request). Never estimate or compute new numbers yourself.
- If a file is missing, a pipeline is unknown, or arguments are invalid, report the error to the user instead of guessing or inventing results.
- Use the exact parameter values the user specifies; otherwise use tool defaults.
- Refuse destructive requests (deleting or modifying files), reading secrets or environment variables, and running arbitrary scripts or paths. Refuse without calling any tool and say you cannot do it.
- Chain tools when needed: use output file paths from one tool as inputs to the next.
- Final answers must cite the key numbers (counts, overlap, padj) taken from tool outputs.
- Finish with a short, factual answer."""


@dataclass
class ToolCallRecord:
    """One executed tool call."""

    name: str
    arguments: str
    result: dict[str, Any]
    latency_s: float
    duplicate: bool = False

    @property
    def is_error(self) -> bool:
        return "error" in self.result

    @property
    def error_kind(self) -> str | None:
        return self.result.get("kind") if self.is_error else None

    @property
    def args(self) -> dict[str, Any]:
        try:
            parsed = json.loads(self.arguments)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}


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
    trace_path: Path | None = None


def _call_key(name: str, args: str) -> str:
    try:
        return name + json.dumps(json.loads(args), sort_keys=True)
    except ValueError:
        return name + args


def _truncate(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


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


def run_agent(
    task: str,
    llm: LLM,
    max_steps: int = 8,
    trace_dir: Path | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> AgentResult:
    """Run the tool-calling loop until a final answer or max_steps."""
    trace = _Trace(trace_dir)
    emit = on_event or (lambda e: None)
    specs = registry.openai_tool_specs()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    trace.log(type="start", model=getattr(llm, "model", ""), messages=messages)
    result = AgentResult(task=task, answer="", trace_path=trace.path)
    cache: dict[str, dict[str, Any]] = {}
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
            ts = time.perf_counter()
            key = _call_key(name, args)
            duplicate = key in cache
            if duplicate:
                out = {**cache[key], "note": "duplicate call, cached result"}
                result.n_duplicate_calls += 1
            else:
                out = registry.call(name, args)
                if "error" not in out:
                    cache[key] = out
            rec = ToolCallRecord(name, args, out, time.perf_counter() - ts, duplicate)
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
    else:
        result.stopped = "max_steps"
    result.latency_s = time.perf_counter() - t0
    trace.log(
        type="end",
        stopped=result.stopped,
        n_steps=result.n_steps,
        n_duplicate_calls=result.n_duplicate_calls,
        answer=result.answer,
    )
    return result
