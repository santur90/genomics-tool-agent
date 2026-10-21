"""Command line interface: `gta run` and `gta eval`."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from gta.agent import run_agent
from gta.llm import OpenAICompatLLM, load_env
from gta.sandbox import PROJECT_ROOT

app = typer.Typer(add_completion=False, help="Genomics tool agent")
console = Console()


def _print_event(event: dict[str, Any]) -> None:
    if event["type"] == "tool":
        failed = "error" in event["result"]
        style = "red" if failed else "green"
        body = event["result"].get("error", json.dumps(event["result"].get("summary", {}), indent=2)[:800])
        console.print(Panel(body, title=f"step {event['step']}: {event['name']}({event['arguments']})", border_style=style))
    else:
        console.print(Panel(event["content"], title="answer", border_style="cyan"))


@app.command()
def run(
    task: str = typer.Argument(..., help="Natural-language task"),
    max_steps: int = typer.Option(8, help="Maximum agent steps"),
    model: Optional[str] = typer.Option(None, help="Override LLM_MODEL"),
) -> None:
    """Run the agent on a task."""
    load_env()
    if model:
        os.environ["LLM_MODEL"] = model
    llm = OpenAICompatLLM()
    result = run_agent(task, llm, max_steps=max_steps, on_event=_print_event)
    console.print(f"[dim]{result.n_steps} steps, {result.latency_s:.1f}s, trace: {result.trace_path}[/dim]")


@app.command("eval")
def eval_cmd(
    mock: bool = typer.Option(False, "--mock", help="Use scripted MockLLM answers (CI)"),
    model: Optional[str] = typer.Option(None, help="Override LLM_MODEL"),
    repeats: int = typer.Option(1, help="Repeats per task, reports mean±sd"),
    category: Optional[str] = typer.Option(None, help="Only run one category"),
) -> None:
    """Run the evaluation benchmark."""
    load_env()
    if model:
        os.environ["LLM_MODEL"] = model
    sys.path.insert(0, str(PROJECT_ROOT))
    from eval.run_eval import run_eval

    report = run_eval(mock=mock, repeats=repeats, category=category)
    console.print(f"model={report['model']}  success={report['aggregate']['overall']['task_success']['mean']:.2f}")
    console.print("Results written to eval/results/")
