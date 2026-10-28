"""LLM backends: OpenAI-compatible client and a scripted mock for CI."""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from gta.sandbox import PROJECT_ROOT


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: str  # raw JSON string


@dataclass
class LLMResponse:
    """Normalised chat completion."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    retries: list[dict[str, Any]] = field(default_factory=list)


class LLMError(RuntimeError):
    """A transient LLM failure that persisted after all retries."""

    def __init__(self, message: str, retries: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.retries = retries or []


class QuotaExhausted(LLMError):
    """The provider's daily allocation is used up; retrying today is pointless."""


MAX_RETRIES = 5
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0
_QUOTA_CODE = re.compile(r"code\W{1,4}4006\b")


def is_quota_error(exc: Exception) -> bool:
    """True for a 429 saying the daily free allocation is exhausted (e.g. Cloudflare code 4006)."""
    if getattr(exc, "status_code", None) != 429:
        return False
    text = f"{exc} {getattr(exc, 'body', '')}"
    return "daily free allocation" in text.lower() or bool(_QUOTA_CODE.search(text))


def is_retryable(exc: Exception) -> bool:
    """429, 5xx, and connection/timeout errors are retried."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    try:
        import openai

        return isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError))
    except ImportError:
        return False


def backoff_delay(attempt: int, exc: Exception | None = None) -> float:
    """Exponential backoff with jitter; honours a Retry-After header when present."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        retry_after = float(headers.get("retry-after", 0))
    except (TypeError, ValueError):
        retry_after = 0.0
    return min(BACKOFF_CAP_S, max(retry_after, BACKOFF_BASE_S * 2**attempt) + random.uniform(0, 1))


class LLM(Protocol):
    """Anything that can answer a chat request."""

    model: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines (optionally prefixed with `export`) from .env without overriding existing vars."""
    env_file = path or PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class OpenAICompatLLM:
    """Chat client for any OpenAI-compatible endpoint, configured only via env vars."""

    def __init__(
        self, base_url: str | None = None, api_key: str | None = None, model: str | None = None
    ) -> None:
        from openai import OpenAI

        self.base_url = base_url or os.environ.get("LLM_BASE_URL")
        self.model = model or os.environ.get("LLM_MODEL", "")
        key = api_key or os.environ.get("LLM_API_KEY")
        missing = [
            n
            for n, v in [("LLM_BASE_URL", self.base_url), ("LLM_API_KEY", key), ("LLM_MODEL", self.model)]
            if not v
        ]
        if missing:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
        self.client = OpenAI(base_url=self.base_url, api_key=key, max_retries=0)
        self._sleep = time.sleep

    def _create(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Call the API, retrying 429/5xx with backoff; returns (response, retry events)."""
        retries: list[dict[str, Any]] = []
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools, tool_choice="auto", temperature=0
                )
                return resp, retries
            except Exception as exc:
                if is_quota_error(exc):
                    raise QuotaExhausted(str(exc), retries) from exc
                if not is_retryable(exc):
                    raise
                if attempt == MAX_RETRIES:
                    raise LLMError(f"giving up after {MAX_RETRIES} retries: {exc}", retries) from exc
                wait = backoff_delay(attempt, exc)
                retries.append(
                    {
                        "attempt": attempt + 1,
                        "status": getattr(exc, "status_code", None),
                        "wait_s": round(wait, 2),
                        "error": str(exc)[:200],
                    }
                )
                self._sleep(wait)
        raise AssertionError("unreachable")

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        """Send one chat completion request."""
        resp, retries = self._create(messages, tools)
        msg = resp.choices[0].message
        calls = []
        for i, tc in enumerate(msg.tool_calls or []):
            args = tc.function.arguments
            calls.append(
                ToolCall(
                    tc.id or f"call_{i}",
                    tc.function.name,
                    args if isinstance(args, str) else json.dumps(args or {}),
                )
            )
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
        }
        return LLMResponse(msg.content or "", calls, usage, retries)


_REF = re.compile(r"\{r(\d+)((?:\.\w+|\[\d+\])*)\}")
_STEP = re.compile(r"\.(\w+)|\[(\d+)\]")


def _lookup(results: list[dict[str, Any]], idx: int, path: str) -> str:
    try:
        node: Any = results[idx]
        for key, pos in _STEP.findall(path):
            node = node[int(pos)] if pos else node[key]
        return str(node)
    except (IndexError, KeyError, TypeError):
        return ""


def render(value: Any, results: list[dict[str, Any]]) -> Any:
    """Fill `{r0.summary.n_up}`-style placeholders using earlier tool results."""
    if isinstance(value, str):
        return _REF.sub(lambda m: _lookup(results, int(m.group(1)), m.group(2)), value)
    if isinstance(value, list):
        return [render(v, results) for v in value]
    if isinstance(value, dict):
        return {k: render(v, results) for k, v in value.items()}
    return value


def _tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        if m.get("role") == "tool":
            try:
                out.append(json.loads(m["content"]))
            except (ValueError, TypeError):
                out.append({})
    return out


class MockLLM:
    """Replays a script: steps of `{tool_calls: [{name, arguments}]}` or `{content: text}`."""

    model = "mock"

    def __init__(self, steps: list[dict[str, Any]]) -> None:
        self.steps = steps
        self.i = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> MockLLM:
        """Load a script from a YAML file with a top-level `steps` list."""
        return cls(yaml.safe_load(Path(path).read_text())["steps"])

    @classmethod
    def from_calls(cls, calls: list[dict[str, Any]], answer: str) -> MockLLM:
        """Build a script that makes the given calls one per step, then answers."""
        return cls([{"tool_calls": [c]} for c in calls] + [{"content": answer}])

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        """Return the next scripted step."""
        results = _tool_results(messages)
        if self.i >= len(self.steps):
            return LLMResponse(content="(mock script exhausted)")
        step = self.steps[self.i]
        self.i += 1
        if "tool_calls" in step:
            calls = [
                ToolCall(f"mock_{self.i}_{j}", c["name"], json.dumps(render(c.get("arguments", {}), results)))
                for j, c in enumerate(step["tool_calls"])
            ]
            return LLMResponse(None, calls)
        return LLMResponse(render(step["content"], results))
