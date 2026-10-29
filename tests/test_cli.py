import os

import pytest
from typer.testing import CliRunner

import eval.run_eval as run_eval_mod
from gta import cli
from gta.llm import LLMError, MockLLM, QuotaExhausted

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    monkeypatch.setattr(cli, "load_env", lambda: None)  # never read the developer's real .env


def test_run_prints_tool_panels_and_answer(monkeypatch):
    seen = {}

    def factory():
        seen["model"] = os.environ.get("LLM_MODEL")
        calls = [{"name": "inspect_file", "arguments": {"path": "data/definitely_missing.tsv"}}]
        return MockLLM.from_calls(calls, "Could not read it: {r0.error}")

    monkeypatch.setattr(cli, "OpenAICompatLLM", factory)
    res = runner.invoke(cli.app, ["run", "inspect it", "--model", "my-model", "--max-steps", "3"])
    assert res.exit_code == 0, res.output
    assert seen["model"] == "my-model"
    assert "inspect_file" in res.output and "File not found" in res.output
    assert "2 steps" in res.output


def test_run_shows_summary_for_successful_tool(monkeypatch):
    monkeypatch.setattr(
        cli,
        "OpenAICompatLLM",
        lambda: MockLLM.from_calls(
            [{"name": "summarize_vcf", "arguments": {"vcf": "data/variants/toy.vcf.gz"}}], "done"
        ),
    )
    res = runner.invoke(cli.app, ["run", "summarize"])
    assert res.exit_code == 0
    # either a summary (data present) or a not-found error panel; both are rendered, never a traceback
    assert "summarize_vcf" in res.output and "Traceback" not in res.output


def _fake_report(**over):
    base = {
        "model": "m", "n_rows": 5, "aggregate": {"overall": {"task_success": {"mean": 0.8}}},
        "failed": [], "partial_file": "/tmp/x.partial.jsonl", "quota_exhausted": False,
    }  # fmt: skip
    return {**base, **over}


def _patch_eval(monkeypatch, report, captured=None):
    def fake(**kw):
        if captured is not None:
            captured.update(kw)
        return report

    monkeypatch.setattr(run_eval_mod, "run_eval", fake)


def test_eval_forwards_options_and_reports(monkeypatch):
    got = {}
    _patch_eval(monkeypatch, _fake_report(), got)
    res = runner.invoke(
        cli.app,
        [
            "eval",
            "--model",
            "abc",
            "--repeats",
            "3",
            "--sleep",
            "1.5",
            "--resume",
            "--tasks",
            "t1,t2",
            "--category",
            "safety",
        ],
    )
    assert res.exit_code == 0, res.output
    assert got == {
        "mock": False,
        "repeats": 3,
        "tasks": "t1,t2",
        "sleep": 1.5,
        "resume": True,
        "category": "safety",
        "judge": False,
    }
    assert os.environ["LLM_MODEL"] == "abc"
    assert "success=0.80" in res.output and "Results written" in res.output


def test_eval_warns_about_failed_tasks(monkeypatch):
    _patch_eval(monkeypatch, _fake_report(failed=["t1#0", "t2#1"]))
    res = runner.invoke(cli.app, ["eval"])
    assert res.exit_code == 0 and "t1#0" in res.output and "--resume" in res.output


def test_eval_quota_exhausted_exits_2_with_message(monkeypatch):
    _patch_eval(monkeypatch, _fake_report(quota_exhausted=True))
    res = runner.invoke(cli.app, ["eval"])
    assert res.exit_code == 2
    assert "daily quota exhausted, rerun with --resume tomorrow" in res.output
    assert "Results written" not in res.output


def test_eval_with_no_rows_skips_success_line(monkeypatch):
    _patch_eval(monkeypatch, _fake_report(aggregate={}, n_rows=0))
    res = runner.invoke(cli.app, ["eval"])
    assert "success=" not in res.output


def test_print_event_styles_errors_and_answers(capsys):
    cli._print_event({"type": "tool", "step": 1, "name": "x", "arguments": "{}", "result": {"error": "bad"}})
    cli._print_event({"type": "answer", "content": "final words"})
    out = capsys.readouterr().out
    assert "bad" in out and "final words" in out


class _FailingLLM:
    model = "fake"

    def __init__(self, exc):
        self.exc = exc

    def chat(self, messages, tools):
        raise self.exc


def _fail_with(monkeypatch, exc):
    monkeypatch.setattr(cli, "OpenAICompatLLM", lambda: _FailingLLM(exc))


def test_run_quota_exhausted_exits_2_with_one_line_message(monkeypatch):
    _fail_with(monkeypatch, QuotaExhausted("4006 daily free allocation"))
    res = runner.invoke(cli.app, ["run", "task"])
    assert res.exit_code == 2
    assert "LLM daily quota exhausted (provider 429)" in res.output
    assert "Traceback" not in res.output


def test_run_other_llm_error_exits_1_without_traceback(monkeypatch):
    _fail_with(monkeypatch, LLMError("HTTP 503 after retries"))
    res = runner.invoke(cli.app, ["run", "task"])
    assert res.exit_code == 1
    assert "HTTP 503 after retries" in res.output and "Traceback" not in res.output


@pytest.mark.parametrize("exc_type", [QuotaExhausted, LLMError])
def test_run_debug_reraises(monkeypatch, exc_type):
    _fail_with(monkeypatch, exc_type("boom"))
    res = runner.invoke(cli.app, ["run", "task", "--debug"])
    assert isinstance(res.exception, exc_type)


def test_eval_judge_flag_reports_calibration_agreement(monkeypatch):
    got = {}
    judge = {"agreement": 0.5, "n": 10, "disagreements": ["c02"], "trusted": False}
    _patch_eval(monkeypatch, _fake_report(judge=judge), got)
    res = runner.invoke(cli.app, ["eval", "--judge"])
    assert res.exit_code == 0, res.output
    assert got["judge"] is True
    assert "Judge agreement with hand labels: 50% on 10 answers" in res.output and "NOT trusted" in res.output
