"""Command line interface: `gta run` and `gta eval`."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

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
        console.print(
            Panel(
                body, title=f"step {event['step']}: {event['name']}({event['arguments']})", border_style=style
            )
        )
    else:
        console.print(Panel(event["content"], title="answer", border_style="cyan"))


@app.command()
def run(
    task: str = typer.Argument(..., help="Natural-language task"),
    max_steps: int = typer.Option(8, help="Maximum agent steps"),
    model: str | None = typer.Option(None, help="Override LLM_MODEL"),
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
    mock: bool = typer.Option(False, "--mock", help="Use scripted MockLLM answers (no network; for CI)"),
    model: str | None = typer.Option(None, "--model", help="Override LLM_MODEL"),
    repeats: int = typer.Option(1, "--repeats", min=1, help="Repeats per task; reports mean±sd"),
    sleep: float = typer.Option(0.0, "--sleep", min=0.0, help="Seconds to wait between tasks (rate limits)"),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Continue the latest incomplete (.partial.jsonl) run for this model, skipping finished task/repeat pairs",
    ),
    tasks: str | None = typer.Option(
        None, "--tasks", help="Comma-separated task ids and/or categories to run"
    ),
    category: str | None = typer.Option(None, "--category", help="Only run tasks of this category"),
) -> None:
    """Run the evaluation benchmark."""
    load_env()
    if model:
        os.environ["LLM_MODEL"] = model
    sys.path.insert(0, str(PROJECT_ROOT))
    from eval.run_eval import QUOTA_MESSAGE, run_eval

    report = run_eval(mock=mock, repeats=repeats, tasks=tasks, sleep=sleep, resume=resume, category=category)
    if report["aggregate"]:
        console.print(
            f"model={report['model']}  rows={report['n_rows']}  success={report['aggregate']['overall']['task_success']['mean']:.2f}"
        )
    if report["failed"]:
        console.print(
            f"[yellow]LLM errors (not recorded, rerun with --resume): {', '.join(report['failed'])}[/yellow]"
        )
    console.print(f"Progress file: {report['partial_file']}")
    if report["quota_exhausted"]:
        console.print(f"[red]{QUOTA_MESSAGE}[/red]")
        raise typer.Exit(code=2)
    console.print("Results written to eval/results/")
