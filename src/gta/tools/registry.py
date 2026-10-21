"""Tool registry: decorator, OpenAI tool specs, and validated dispatch."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from gta.sandbox import ToolError


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool."""

    name: str
    description: str
    model: type[BaseModel]
    func: Callable[..., dict[str, Any]]


_REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, description: str, model: type[BaseModel]) -> Callable:
    """Register a function as an agent tool."""

    def decorator(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        _REGISTRY[name] = ToolSpec(name, description, model, func)
        return func

    return decorator


def tool_names() -> list[str]:
    """Names of all registered tools."""
    return sorted(_REGISTRY)


def openai_tool_specs() -> list[dict[str, Any]]:
    """Tool list for the `tools=` parameter of a chat completion."""
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.model.model_json_schema()}}
        for t in _REGISTRY.values()
    ]


def _error(kind: str, message: str, hint: str) -> dict[str, Any]:
    return {"error": message, "hint": hint, "kind": kind}


def call(name: str, args_json: str | dict[str, Any] | None) -> dict[str, Any]:
    """Validate arguments and run a tool; return its dict or a structured error."""
    spec = _REGISTRY.get(name)
    if spec is None:
        return _error("unknown_tool", f"Unknown tool: {name}", f"Available tools: {tool_names()}")
    try:
        raw = json.loads(args_json) if isinstance(args_json, str) and args_json.strip() else (args_json or {})
        if not isinstance(raw, dict):
            raise ValueError("arguments must be a JSON object")
        args = spec.model.model_validate(raw)
    except (ValueError, ValidationError) as exc:
        detail = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()] if isinstance(exc, ValidationError) else str(exc)
        return _error("invalid_arguments", f"Invalid arguments for {name}: {detail}", "Fix the arguments to match the tool's JSON schema.")
    try:
        result = spec.func(**args.model_dump())
        return json.loads(json.dumps(result, default=str))
    except ToolError as exc:
        return _error("tool_error", exc.message, exc.hint)
    except Exception as exc:  # unexpected failures must not crash the agent loop
        return _error("tool_error", f"{type(exc).__name__}: {exc}", "Report this failure to the user instead of guessing results.")
