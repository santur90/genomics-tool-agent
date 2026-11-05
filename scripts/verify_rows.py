#!/usr/bin/env python
"""Offline check: run the rule-based verifier over saved eval rows. No LLM calls, no quota.

Usage:
    python scripts/verify_rows.py eval/results/<file>.done.jsonl [--tasks eval/tasks.yaml]

The task file is used ONLY to look up each task's prompt text by id. Expected values are never read.
task_success is read from the saved metrics ONLY to build the confusion table afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from gta.verify import verify


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=Path)
    ap.add_argument("--tasks", type=Path, default=Path("eval/tasks.yaml"))
    args = ap.parse_args()

    prompts = {t["id"]: t["prompt"] for t in yaml.safe_load(args.tasks.read_text())["tasks"]}
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]

    table = Counter()
    false_pos, missed, caught = [], [], []
    skipped_rows = 0
    no_calls = 0
    print(f"{'task':32s} {'repeat':>6s} {'success':>7s}  flagged issues")
    for r in rows:
        if "calls" not in r:
            no_calls += 1
            continue
        rep = verify(prompts.get(r["id"], ""), r["calls"], r.get("answer") or "")
        ok = bool(r["metrics"].get("task_success"))
        if rep.skipped:
            skipped_rows += 1
        flagged = rep.flagged
        table[(flagged, ok)] += 1
        label = f"{r['id']}#{r.get('repeat', 0)}"
        if flagged and ok:
            false_pos.append(label)
        elif flagged and not ok:
            caught.append(label)
        elif not flagged and not ok:
            missed.append(label)
        if flagged:
            print(f"{r['id']:32s} {r.get('repeat', 0):>6} {int(ok):>7}  {', '.join(rep.types)}")
            for i in rep.issues:
                print(f"{'':48s}{i.detail[:150]}")

    print("\nConfusion (rows):")
    print(f"                  failed   passed")
    print(f"  flagged        {table[(True, False)]:>6d}   {table[(True, True)]:>6d}")
    print(f"  not flagged    {table[(False, False)]:>6d}   {table[(False, True)]:>6d}")
    n_pass = table[(True, True)] + table[(False, True)]
    n_fail = table[(True, False)] + table[(False, False)]
    if n_pass:
        print(f"\nFalse-flag rate on passed rows: {table[(True, True)]}/{n_pass}")
    if n_fail:
        print(f"Detection rate on failed rows:  {table[(True, False)]}/{n_fail}")
    print("\nFlagged and failed (caught):", caught or "none")
    print("Flagged but PASSED (false flags):", false_pos or "none")
    print("Failed but NOT flagged (verifier blind spots):", missed or "none")
    if skipped_rows:
        print(f"\nNote: number/citation checks were skipped for {skipped_rows} rows (no stored tool output).")
    if no_calls:
        print(f"Note: {no_calls} rows have no 'calls' (pre-v1.1 results) and were skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
