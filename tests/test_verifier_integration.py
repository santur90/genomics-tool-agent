"""Agent + verifier integration, using a scripted LLM and a fake tool layer (no data, no binaries)."""

import json

import pytest

from gta import agent
from gta.llm import LLMResponse, ToolCall


class Scripted:
    model = "scripted"

    def __init__(self, steps):
        self.steps = list(steps)
        self.n_calls = 0
        self.seen = []

    def chat(self, messages, tools):
        self.n_calls += 1
        self.seen.append([dict(m) for m in messages])
        kind, payload = self.steps.pop(0)
        if kind == "call":
            name, args = payload
            return LLMResponse(None, [ToolCall(f"c{self.n_calls}", name, json.dumps(args))], {})
        return LLMResponse(payload, [], {})


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    def call_validated(name, args):
        a = json.loads(args)
        path = a.get("path", "")
        if path.endswith("counts_v2.tsv"):
            return {"error": "File not found: " + path, "hint": "x", "kind": "tool_error"}, None
        return {"summary": {"n_rows": 12}}, {"path": path}

    monkeypatch.setattr(agent.registry, "call_validated", call_validated)


PROMPT = "Inspect data/rnaseq/counts_v2.tsv and tell me what it contains."
SUBSTITUTE = [
    ("call", ("inspect_file", {"path": "data/rnaseq/counts_v2.tsv"})),
    ("call", ("inspect_file", {"path": "data/rnaseq/counts.tsv"})),
]


def test_off_by_default_never_revises(monkeypatch):
    monkeypatch.delenv("GTA_VERIFIER", raising=False)
    llm = Scripted(SUBSTITUTE + [("answer", "The file has 12 rows.")])
    res = agent.run_agent(PROMPT, llm)
    assert res.verifier == "off" and res.n_revisions == 0 and llm.n_calls == 3
    assert res.answer == "The file has 12 rows."


def test_clean_run_has_no_flag_and_no_extra_calls():
    llm = Scripted(
        [("call", ("inspect_file", {"path": "data/rnaseq/counts.tsv"})), ("answer", "It has 12 rows.")]
    )
    res = agent.run_agent("Inspect data/rnaseq/counts.tsv", llm, verifier="rules")
    assert (
        res.verifier_issues == [] and res.n_revisions == 0 and res.extra_llm_calls == 0 and llm.n_calls == 2
    )


def test_flagged_run_is_revised_into_a_disclosed_answer():
    llm = Scripted(
        SUBSTITUTE
        + [
            ("answer", "The file has 12 rows."),
            ("answer", "counts_v2.tsv was not found, so I inspected counts.tsv: 12 rows."),
        ]
    )
    res = agent.run_agent(PROMPT, llm, verifier="rules")
    assert res.verifier_issues == ["input_substitution"]
    assert res.verifier_issues_after == []
    assert res.n_revisions == 1 and res.extra_llm_calls == 1
    assert res.answer_before_revision == "The file has 12 rows."
    assert "counts_v2.tsv" in res.answer
    # the revision message was sent, and it lists the issue; tool outputs were not altered
    last_user = llm.seen[-1][-1]
    assert last_user["role"] == "user" and "input_substitution" in last_user["content"]
    assert [m["role"] for m in llm.seen[-1]].count("tool") == 2


def test_revision_that_still_fails_is_kept_and_reported():
    llm = Scripted(SUBSTITUTE + [("answer", "12 rows."), ("answer", "Still 12 rows.")])
    res = agent.run_agent(PROMPT, llm, verifier="rules")
    assert res.n_revisions == 1 and res.verifier_issues_after == ["input_substitution"]


def test_only_one_revision_round():
    llm = Scripted(
        SUBSTITUTE + [("answer", "12 rows."), ("answer", "12 rows again."), ("answer", "never used")]
    )
    agent.run_agent(PROMPT, llm, verifier="rules")
    assert llm.n_calls == 4 and len(llm.steps) == 1


def test_env_switch_and_bad_value(monkeypatch):
    monkeypatch.setenv("GTA_VERIFIER", "rules")
    llm = Scripted(
        SUBSTITUTE + [("answer", "12 rows."), ("answer", "counts_v2.tsv was not found; used counts.tsv.")]
    )
    assert agent.run_agent(PROMPT, llm).verifier == "rules"
    with pytest.raises(ValueError):
        agent.run_agent(PROMPT, Scripted([]), verifier="llm")


def test_verifier_never_sees_expected_values_by_construction():
    import inspect

    src = inspect.getsource(agent._verify_and_revise)
    assert "yaml" not in src and "eval" not in src.replace("evaluat", "")
