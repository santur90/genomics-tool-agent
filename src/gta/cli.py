"""Command line interface: `gta run`, `gta eval` and `gta doctor`."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gta import __version__, doctor
from gta.agent import run_agent
from gta.llm import LLMError, OpenAICompatLLM, QuotaExhausted, load_env
from gta.sandbox import PROJECT_ROOT

app = typer.Typer(add_completion=False, help="Genomics tool agent")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"genomics-tool-agent {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit"
    ),
) -> None:
    """Genomics tool agent."""


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
    debug: bool = typer.Option(False, "--debug", help="Show full tracebacks for LLM errors"),
) -> None:
    """Run the agent on a task."""
    load_env()
    if model:
        os.environ["LLM_MODEL"] = model
    llm = OpenAICompatLLM()
    try:
        result = run_agent(task, llm, max_steps=max_steps, on_event=_print_event)
        if result.llm_error is not None:
            raise result.llm_error
    except LLMError as exc:
        if debug:
            raise
        if isinstance(exc, QuotaExhausted):
            console.print(
                "[red]LLM daily quota exhausted (provider 429). Try again after the quota resets "
                "or set LLM_MODEL/LLM_BASE_URL to another provider.[/red]"
            )
            raise typer.Exit(2) from None
        console.print(f"[red]LLM request failed: {exc}[/red]")
        raise typer.Exit(1) from None
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
    judge: bool = typer.Option(
        False,
        "--judge",
        help="Also have an LLM judge check each answer sentence against tool outputs "
        "(calibrated on eval/judge_calibration.yaml first; judge model: LLM_JUDGE_MODEL or LLM_MODEL)",
    ),
    rescore: Path | None = typer.Option(  # noqa: B008
        None,
        "--rescore",
        help="Recompute metrics offline from a saved .done.jsonl with the current graders (no LLM calls); "
        "writes <name>.rescored_v1.2.jsonl next to it and prints a per-task diff",
    ),
) -> None:
    """Run the evaluation benchmark."""
    load_env()
    if model:
        os.environ["LLM_MODEL"] = model
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from eval.run_eval import QUOTA_MESSAGE, run_eval
        from eval.run_eval import rescore as rescore_rows
    except ModuleNotFoundError as exc:
        if exc.name not in ("eval", "eval.run_eval"):
            raise
        console.print(
            f"[red]The benchmark (eval/) is not part of the installed package; it was not found in {PROJECT_ROOT}.[/red]\n"
            "Run from a repository checkout or set GTA_ROOT to it."
        )
        raise typer.Exit(code=1) from exc

    if rescore is not None:
        if not rescore.is_file():
            console.print(f"[red]No such file: {rescore}[/red]")
            raise typer.Exit(code=1)
        out, diffs = rescore_rows(rescore)
        console.print("task id | repeat | old success | new success | changed metrics", markup=False)
        for d in diffs:
            console.print(
                f"{d['id']} | {d['repeat']} | {d['old']} | {d['new']} | {', '.join(d['changed']) or '-'}",
                markup=False,
                highlight=False,
            )
        n_legacy = sum(not d.get("has_result_text", True) for d in diffs)
        if n_legacy:
            console.print(
                f"WARNING: {n_legacy} of {len(diffs)} rows have no saved tool output (calls[].result_text); "
                "citation validity is unreliable for them (computed from logged summaries only).",
                markup=False,
                style="yellow",
            )
        n = sum(d["old"] != d["new"] for d in diffs)
        console.print(f"{n} of {len(diffs)} rows changed success. Rescored rows: {out}", markup=False)
        return

    report = run_eval(
        mock=mock, repeats=repeats, tasks=tasks, sleep=sleep, resume=resume, category=category, judge=judge
    )
    if report.get("judge"):
        j = report["judge"]
        verdict = "trusted" if j["trusted"] else "[yellow]NOT trusted (below 80%)[/yellow]"
        console.print(
            f"Judge agreement with hand labels: {j['agreement']:.0%} on {j['n']} answers, {verdict}"
        )
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


@app.command("doctor")
def doctor_cmd() -> None:
    """Check Python packages, external binaries and LLM environment variables."""
    load_env()
    checks = doctor.run_checks()
    table = Table(title=f"gta doctor (genomics-tool-agent {__version__})")
    for col in ("Kind", "Name", "Status", "Version / value", "Install hint"):
        table.add_column(col, overflow="fold")
    for c in checks:
        status = (
            "[green]OK[/green]"
            if c.ok
            else ("[red]MISSING[/red]" if c.required else "[yellow]MISSING[/yellow]")
        )
        table.add_row(c.kind, c.name, status, c.detail if c.ok else "", c.hint)
    console.print(table)
    code = doctor.exit_code(checks)
    if code:
        console.print("[red]Required components are missing; see the install hints above.[/red]")
    raise typer.Exit(code=code)


@app.command("fetch-knowledge")
def fetch_knowledge_cmd(
    dest: str | None = typer.Option(
        None, "--dest", help="Target dir (default: GTA_KNOWLEDGE_DIR or ~/.cache/gta/knowledge)"
    ),
    force: bool = typer.Option(False, "--force", help="Re-download existing files"),
) -> None:
    """Download GO + Reactome and build the knowledge store used by the RAG tools."""
    from gta.rag.fetch import main as fetch_main

    argv = (["--dest", dest] if dest else []) + (["--force"] if force else [])
    raise typer.Exit(code=fetch_main(argv))
