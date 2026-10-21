"""Evaluation harness: run tasks, grade, aggregate, write JSON + markdown."""
from __future__ import annotations

import json
import os
import re
import statistics
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from eval.graders import grade_task
from gta.agent import run_agent
from gta.llm import LLM, MockLLM, OpenAICompatLLM

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
RESULTS_DIR = EVAL_DIR / "results"
METRICS = ["task_success", "tool_selection_accuracy", "arg_accuracy", "invalid_call_rate",
           "hallucinated_number_rate", "n_steps", "latency_s", "tokens"]


def load_truth() -> dict[str, Any]:
    """Load ground-truth JSON files produced by `make data`."""
    try:
        return {
            "rnaseq": json.loads((ROOT / "data/rnaseq/ground_truth.json").read_text()),
            "vcf": json.loads((ROOT / "data/variants/vcf_truth.json").read_text()),
        }
    except FileNotFoundError as exc:
        raise SystemExit("Ground-truth files missing; run `make data` first.") from exc


def load_tasks(category: str | None = None, path: Path = EVAL_DIR / "tasks.yaml") -> list[dict[str, Any]]:
    """Load (and optionally filter) tasks."""
    tasks = yaml.safe_load(path.read_text())["tasks"]
    return [t for t in tasks if category is None or t["category"] == category]


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def aggregate(rows: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    """Mean±sd across repeats, overall and per category."""
    def over_repeats(subset: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        out = {}
        for m in METRICS:
            per_rep = [statistics.fmean(r["metrics"][m] for r in subset if r["repeat"] == k) for k in range(repeats)]
            out[m] = _mean_sd(per_rep)
        return out

    cats = sorted({r["category"] for r in rows})
    return {"overall": over_repeats(rows), "by_category": {c: over_repeats([r for r in rows if r["category"] == c]) for c in cats}}


def _make_llm(task: dict[str, Any], mock: bool) -> LLM:
    if mock:
        spec = task["mock"]
        return MockLLM.from_calls(spec.get("calls", []), spec["answer"])
    return OpenAICompatLLM()


def _fmt(m: dict[str, float], pct: bool = False) -> str:
    return f"{m['mean'] * 100:.0f}±{m['sd'] * 100:.0f}%" if pct else f"{m['mean']:.2f}±{m['sd']:.2f}"


def write_summary(results_dir: Path = RESULTS_DIR) -> None:
    """Render summary.md from every results JSON (latest file per model)."""
    latest: dict[str, dict[str, Any]] = {}
    for f in sorted(results_dir.glob("*.json")):
        rep = json.loads(f.read_text())
        latest[rep["model"]] = rep
    lines = ["| model | repeats | success | tool sel. | arg acc. | invalid calls | halluc. numbers | steps | latency (s) | tokens |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for model, rep in latest.items():
        o = rep["aggregate"]["overall"]
        lines.append(f"| {model} | {rep['repeats']} | {_fmt(o['task_success'], True)} | {_fmt(o['tool_selection_accuracy'], True)} | "
                     f"{_fmt(o['arg_accuracy'], True)} | {_fmt(o['invalid_call_rate'], True)} | {_fmt(o['hallucinated_number_rate'], True)} | "
                     f"{_fmt(o['n_steps'])} | {_fmt(o['latency_s'])} | {o['tokens']['mean']:.0f} |")
    for model, rep in latest.items():
        lines += ["", f"### {model}: task success by category", "", "| category | success | tool sel. | arg acc. | halluc. numbers |", "|---|---|---|---|---|"]
        for cat, m in rep["aggregate"]["by_category"].items():
            lines.append(f"| {cat} | {_fmt(m['task_success'], True)} | {_fmt(m['tool_selection_accuracy'], True)} | "
                         f"{_fmt(m['arg_accuracy'], True)} | {_fmt(m['hallucinated_number_rate'], True)} |")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")


def run_eval(mock: bool = False, repeats: int = 1, category: str | None = None, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Run all tasks `repeats` times and write results."""
    truth, tasks = load_truth(), load_tasks(category)
    rows: list[dict[str, Any]] = []
    model = "mock"
    for k in range(repeats):
        for task in tasks:
            llm = _make_llm(task, mock)
            model = getattr(llm, "model", model)
            result = run_agent(task["prompt"], llm)
            rows.append({"id": task["id"], "category": task["category"], "repeat": k, "answer": result.answer,
                         "tools_called": [c.name for c in result.calls], "metrics": grade_task(task, result, truth)})
    report = {"model": model, "date": date.today().isoformat(), "repeats": repeats, "n_tasks": len(tasks),
              "aggregate": aggregate(rows, repeats), "tasks": rows}
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^\w.-]+", "_", model)
    (results_dir / f"{safe_model}_{report['date']}.json").write_text(json.dumps(report, indent=2))
    write_summary(results_dir)
    return report


def main() -> None:
    """Entry point: python -m eval.run_eval [--mock] [--model M] [--repeats N]."""
    import argparse

    from gta.llm import load_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--category")
    a = ap.parse_args()
    load_env()
    if a.model:
        os.environ["LLM_MODEL"] = a.model
    run_eval(mock=a.mock, repeats=a.repeats, category=a.category)


if __name__ == "__main__":
    main()
