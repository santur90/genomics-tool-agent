"""Evaluation harness: run tasks, grade, aggregate, write JSON + markdown."""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from eval.graders import grade_task
from gta.agent import run_agent
from gta.llm import LLM, MockLLM, OpenAICompatLLM, QuotaExhausted

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
RESULTS_DIR = EVAL_DIR / "results"
METRICS = [
    "task_success",
    "tool_selection_accuracy",
    "arg_accuracy",
    "invalid_call_rate",
    "hallucinated_number_rate",
    "n_steps",
    "n_duplicate_calls",
    "latency_s",
    "tokens",
]


def load_truth() -> dict[str, Any]:
    """Load ground-truth JSON files produced by `make data`."""
    try:
        return {
            "rnaseq": json.loads((ROOT / "data/rnaseq/ground_truth.json").read_text()),
            "vcf": json.loads((ROOT / "data/variants/vcf_truth.json").read_text()),
        }
    except FileNotFoundError as exc:
        raise SystemExit("Ground-truth files missing; run `make data` first.") from exc


QUOTA_MESSAGE = "daily quota exhausted, rerun with --resume tomorrow"
PCT_METRICS = {
    "task_success",
    "tool_selection_accuracy",
    "arg_accuracy",
    "invalid_call_rate",
    "hallucinated_number_rate",
}
HEADERS = {
    "task_success": "success",
    "tool_selection_accuracy": "tool sel.",
    "arg_accuracy": "arg acc.",
    "invalid_call_rate": "invalid calls",
    "hallucinated_number_rate": "halluc. numbers",
    "n_steps": "steps",
    "n_duplicate_calls": "dup. calls",
    "latency_s": "latency (s)",
    "tokens": "tokens",
}


def load_tasks(
    selectors: str | list[str] | None = None, path: Path = EVAL_DIR / "tasks.yaml"
) -> list[dict[str, Any]]:
    """Load tasks, optionally filtered by comma-separated task ids and/or category names."""
    tasks = yaml.safe_load(path.read_text())["tasks"]
    if isinstance(selectors, str):
        selectors = [x.strip() for x in selectors.split(",") if x.strip()]
    if not selectors:
        return tasks
    known = {t["id"] for t in tasks} | {t["category"] for t in tasks}
    unknown = [x for x in selectors if x not in known]
    if unknown:
        raise SystemExit(f"Unknown task id or category: {unknown}")
    return [t for t in tasks if t["id"] in selectors or t["category"] in selectors]


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def aggregate(rows: list[dict[str, Any]], repeats: int | None = None) -> dict[str, Any]:
    """Mean±sd across repeats (per-repeat means), overall and per category."""

    def over_repeats(subset: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        reps = sorted({r["repeat"] for r in subset})
        return {
            m: _mean_sd(
                [statistics.fmean(r["metrics"].get(m, 0.0) for r in subset if r["repeat"] == k) for k in reps]
            )
            for m in METRICS
        }

    cats = sorted({r["category"] for r in rows})
    return {
        "overall": over_repeats(rows),
        "by_category": {c: over_repeats([r for r in rows if r["category"] == c]) for c in cats},
    }


def _make_llm(task: dict[str, Any], mock: bool) -> LLM:
    if mock:
        spec = task["mock"]
        return MockLLM.from_calls(spec.get("calls", []), spec["answer"])
    return OpenAICompatLLM()


def _fmt(metric: str, m: dict[str, float] | None) -> str:
    m = m or {"mean": 0.0, "sd": 0.0}
    if metric in PCT_METRICS:
        return f"{m['mean'] * 100:.0f}±{m['sd'] * 100:.0f}%"
    if metric == "tokens":
        return f"{m['mean']:.0f}±{m['sd']:.0f}"
    return f"{m['mean']:.2f}±{m['sd']:.2f}"


def _table(first_col: str, rows: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    lines = [
        f"| {first_col} | " + " | ".join(HEADERS[m] for m in METRICS) + " |",
        "|" + "---|" * (len(METRICS) + 1),
    ]
    for name, agg in rows.items():
        lines.append(f"| {name} | " + " | ".join(_fmt(m, agg.get(m)) for m in METRICS) + " |")
    return lines


def write_summary(results_dir: Path = RESULTS_DIR) -> None:
    """Render summary.md from the latest result file of every real (non-mock) model."""
    latest: dict[str, dict[str, Any]] = {}
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith("mock_"):
            continue
        rep = json.loads(f.read_text())
        if rep.get("model") != "mock":
            latest[rep["model"]] = rep
    if not latest:
        (results_dir / "summary.md").write_text("_No real-model results yet. Run `gta eval --repeats 3`._\n")
        return
    lines = ["Mean±sd across repeats (per-repeat means).", ""]
    lines += _table(
        "model", {f"{m} (n={r['repeats']})": r["aggregate"]["overall"] for m, r in latest.items()}
    )
    for model, rep in latest.items():
        lines += ["", f"### {model}: by category", ""] + _table("category", rep["aggregate"]["by_category"])
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _read_partial(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _find_partial(results_dir: Path, slug: str) -> Path | None:
    candidates = sorted(results_dir.glob(f"{slug}_*.partial.jsonl"))
    return candidates[-1] if candidates else None


def _finalize(rows: list[dict[str, Any]], model: str, partial: Path, n_tasks: int) -> dict[str, Any]:
    repeats = max((r["repeat"] for r in rows), default=-1) + 1
    report = {
        "model": model,
        "date": datetime.now().isoformat(timespec="minutes"),
        "repeats": repeats,
        "n_tasks": n_tasks,
        "n_rows": len(rows),
        "aggregate": aggregate(rows) if rows else {},
        "tasks": rows,
    }
    if rows:
        final = partial.with_name(partial.name.removesuffix(".partial.jsonl") + ".json")
        final.write_text(json.dumps(report, indent=2))
        write_summary(partial.parent)
    return report


def run_eval(
    mock: bool = False,
    repeats: int = 1,
    tasks: str | list[str] | None = None,
    sleep: float = 0.0,
    resume: bool = False,
    results_dir: Path = RESULTS_DIR,
    category: str | None = None,
) -> dict[str, Any]:
    """Run tasks `repeats` times, appending each finished row to a .partial.jsonl; write final JSON + summary at the end.

    A fully completed run renames its partial file to .done.jsonl, so only incomplete partials are resumed.
    """
    truth, selected = load_truth(), load_tasks(tasks)
    if category:
        selected = [t for t in selected if t["category"] == category]
        if not selected:
            raise SystemExit(f"No tasks in category '{category}'.")
    model = "mock" if mock else os.environ.get("LLM_MODEL", "")
    if not model:
        raise SystemExit("LLM_MODEL is not set (use --model or .env).")
    slug = re.sub(r"[^\w.-]+", "_", model)
    results_dir.mkdir(parents=True, exist_ok=True)
    partial = (
        _find_partial(results_dir, slug) if resume else None
    ) or results_dir / f"{slug}_{datetime.now():%Y%m%d-%H%M}.partial.jsonl"
    rows = _read_partial(partial)
    done = {(r["id"], r["repeat"]) for r in rows}
    quota_hit, failed, ran = False, [], 0
    try:
        for k in range(repeats):
            for task in selected:
                if (task["id"], k) in done:
                    continue
                if ran and sleep:
                    time.sleep(sleep)
                ran += 1
                result = run_agent(task["prompt"], _make_llm(task, mock))
                if result.stopped == "llm_error":
                    failed.append(f"{task['id']}#{k}")  # not recorded, so --resume retries it
                    continue
                row = {
                    "id": task["id"],
                    "category": task["category"],
                    "repeat": k,
                    "model": model,
                    "answer": result.answer,
                    "tools_called": [c.name for c in result.calls],
                    "metrics": grade_task(task, result, truth),
                }
                with open(partial, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                rows.append(row)
    except QuotaExhausted:
        quota_hit = True
    report = _finalize(rows, model, partial, len(selected))
    done = {(r["id"], r["repeat"]) for r in rows}
    complete = (
        not quota_hit and not failed and all((t["id"], k) in done for t in selected for k in range(repeats))
    )
    if complete and partial.exists():
        finished = partial.with_name(partial.name.removesuffix(".partial.jsonl") + ".done.jsonl")
        partial.rename(finished)
        partial = finished
    report.update(quota_exhausted=quota_hit, failed=failed, partial_file=str(partial), complete=complete)
    return report


def main() -> None:
    """Entry point: python -m eval.run_eval (same options as `gta eval`)."""
    import argparse

    from gta.llm import load_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tasks")
    ap.add_argument("--category")
    a = ap.parse_args()
    load_env()
    if a.model:
        os.environ["LLM_MODEL"] = a.model
    report = run_eval(
        mock=a.mock, repeats=a.repeats, tasks=a.tasks, sleep=a.sleep, resume=a.resume, category=a.category
    )
    if report["quota_exhausted"]:
        print(QUOTA_MESSAGE)


if __name__ == "__main__":
    main()
