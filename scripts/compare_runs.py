#!/usr/bin/env python
"""Paired comparison of a baseline run and a verifier run (same model, same tasks_version).

Usage: python scripts/compare_runs.py BASELINE.done.jsonl VERIFIER.done.jsonl [--category error_handling]
Rows are paired on (task id, repeat). No LLM calls. Never merges the two conditions.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict


def load(p, cat):
    rows = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            if cat and r.get("category") != cat:
                continue
            rows[(r["id"], r.get("repeat", 0))] = r
    return rows


ap = argparse.ArgumentParser()
ap.add_argument("baseline")
ap.add_argument("verifier")
ap.add_argument("--category")
a = ap.parse_args()
base, ver = load(a.baseline, a.category), load(a.verifier, a.category)
keys = sorted(set(base) & set(ver))
if not keys:
    sys.exit("no (task, repeat) pairs in common")
only = (len(base) - len(keys), len(ver) - len(keys))


def ok(r):
    return bool(r["metrics"].get("task_success"))


c, by_cat, fixed, broken = Counter(), defaultdict(Counter), [], []
for k in keys:
    b, v = ok(base[k]), ok(ver[k])
    tag = "fixed" if (not b and v) else "broken" if (b and not v) else ("both_pass" if b else "both_fail")
    c[tag] += 1
    by_cat[base[k]["category"]][tag] += 1
    (fixed if tag == "fixed" else broken if tag == "broken" else []).append(f"{k[0]}#{k[1]}")
n = len(keys)
print(f"paired rows: {n}   (unpaired: baseline {only[0]}, verifier {only[1]})")
print(
    f"baseline success: {sum(ok(base[k]) for k in keys)}/{n}   verifier success: {sum(ok(ver[k]) for k in keys)}/{n}"
)
print(
    f"fixed: {c['fixed']}   broken: {c['broken']}   both pass: {c['both_pass']}   both fail: {c['both_fail']}"
)
bp = c["both_pass"] + c["broken"]
print(f"broken among baseline passes: {c['broken']}/{bp}")
flagged_pass = [k for k in keys if ok(base[k]) and ver[k].get("verifier_issues")]
if any("verifier_issues" in ver[k] for k in keys):
    print(f"verifier flagged {len(flagged_pass)}/{bp} baseline-passing rows (false-flag rate)")
else:
    print("note: verifier rows carry no verifier_issues field (run_eval not wired yet)")
print("\nby category (fixed / broken / both pass / both fail):")
for cat, cc in sorted(by_cat.items()):
    print(f"  {cat:20s} {cc['fixed']} / {cc['broken']} / {cc['both_pass']} / {cc['both_fail']}")
print("\nfixed :", fixed or "none")
print("broken:", broken or "none")
extra = [ver[k].get("extra_llm_calls", 0) for k in keys]
print(f"\nextra LLM calls in verifier run: {sum(extra)} over {n} rows")
