import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import eval.run_eval as run_eval_mod
from eval.run_eval import QUOTA_MESSAGE, aggregate, load_tasks, run_eval, write_summary
from gta.agent import run_agent
from gta.cli import app
from gta.llm import LLMError, LLMResponse, OpenAICompatLLM, QuotaExhausted
from tests.conftest import needs_bin, needs_data

data_ready = [
    needs_bin("bcftools", "samtools", "nextflow"),
    needs_data("data/rnaseq/ground_truth.json", "data/variants/vcf_truth.json", "data/variants/sample.bam"),
]


class HTTPErr(Exception):
    def __init__(self, status, body=""):
        super().__init__(f"Error code: {status} - {body}")
        self.status_code, self.body = status, body


def _llm(monkeypatch, outcomes):
    """OpenAICompatLLM whose client yields the given outcomes (exceptions are raised)."""
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.setenv(k, "x")
    llm = OpenAICompatLLM()
    sleeps, calls = [], []
    llm._sleep = sleeps.append

    def create(**kw):
        calls.append(kw)
        o = outcomes[len(calls) - 1]
        if isinstance(o, Exception):
            raise o
        return o

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return llm, sleeps, calls


OK = SimpleNamespace(
    choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))], usage=None
)


def test_retries_429_and_5xx_with_logged_backoff(monkeypatch):
    llm, sleeps, calls = _llm(monkeypatch, [HTTPErr(429, "slow down"), HTTPErr(503), OK])
    resp = llm.chat([{"role": "user", "content": "x"}], [])
    assert resp.content == "hi" and len(calls) == 3
    assert [r["status"] for r in resp.retries] == [429, 503]
    assert 1 <= sleeps[0] < 2.01 and 2 <= sleeps[1] < 3.01  # 1s*2^n plus <1s jitter


def test_gives_up_after_five_retries(monkeypatch):
    llm, sleeps, calls = _llm(monkeypatch, [HTTPErr(500)] * 10)
    with pytest.raises(LLMError) as exc:
        llm.chat([], [])
    assert len(calls) == 6 and len(sleeps) == 5 and len(exc.value.retries) == 5
    assert not isinstance(exc.value, QuotaExhausted)


@pytest.mark.parametrize(
    "body", ["you have used up your daily free allocation of 10,000 neurons", '{"errors":[{"code": 4006}]}']
)
def test_daily_quota_stops_without_retrying(monkeypatch, body):
    llm, sleeps, calls = _llm(monkeypatch, [HTTPErr(429, body), OK])
    with pytest.raises(QuotaExhausted):
        llm.chat([], [])
    assert len(calls) == 1 and not sleeps


def test_non_retryable_error_propagates(monkeypatch):
    llm, sleeps, calls = _llm(monkeypatch, [HTTPErr(400, "bad request"), OK])
    with pytest.raises(HTTPErr):
        llm.chat([], [])
    assert len(calls) == 1


def test_agent_logs_retries_and_handles_errors(tmp_path):
    class Retried:
        model = "r"

        def chat(self, messages, tools):
            return LLMResponse("done", retries=[{"attempt": 1, "status": 429, "wait_s": 1.2}])

    run_agent("t", Retried(), trace_dir=tmp_path / "a")
    ev = [json.loads(x) for x in (tmp_path / "a" / "trace.jsonl").read_text().splitlines()]
    assert ev[1]["retries"][0]["status"] == 429

    class Failing:
        model = "f"

        def chat(self, messages, tools):
            raise LLMError("boom", [{"attempt": 1}])

    assert run_agent("t", Failing(), trace_dir=tmp_path / "b").stopped == "llm_error"

    class Quota(Failing):
        def chat(self, messages, tools):
            raise QuotaExhausted("daily free allocation")

    with pytest.raises(QuotaExhausted):
        run_agent("t", Quota(), trace_dir=tmp_path / "c")
    last = json.loads((tmp_path / "c" / "trace.jsonl").read_text().splitlines()[-1])
    assert last["type"] == "llm_error" and last["quota"] is True


@pytest.mark.parametrize("sel,n", [("single_tool", 5), ("st_vcf_filter_count,safety", 6)])
def test_task_selector(sel, n):
    assert len(load_tasks(sel)) == n


def test_unknown_selector():
    with pytest.raises(SystemExit):
        load_tasks("nope")


def test_summary_excludes_mock_and_reports_all_metrics(tmp_path):
    def report(model, success):
        rows = [
            {
                "id": "a",
                "category": "c1",
                "repeat": r,
                "metrics": {"task_success": success + 0.1 * r, "n_duplicate_calls": 1.0},
            }
            for r in range(2)
        ]
        return {"model": model, "repeats": 2, "aggregate": aggregate(rows)}

    for name, rep in {
        "m1_20261027-1000": report("m1", 0.5),
        "m2_20261027-1001": report("m2", 0.8),
        "mock_20261027-1002": report("mock", 1.0),
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(rep))
    write_summary(tmp_path)
    md = (tmp_path / "summary.md").read_text()
    assert "m1 (n=2)" in md and "m2 (n=2)" in md and "mock" not in md
    assert "dup. calls" in md and "±" in md and "by category" in md


@data_ready[0]
@data_ready[1]
def test_resume_after_quota_and_partial_file(tmp_path, monkeypatch):
    real = run_eval_mod._make_llm
    state = {"n": 0}

    def flaky(task, mock):
        state["n"] += 1
        if state["n"] == 3:

            class Out:
                model = "mock"

                def chat(self, m, t):
                    raise QuotaExhausted("daily free allocation")

            return Out()
        return real(task, mock)

    monkeypatch.setattr(run_eval_mod, "_make_llm", flaky)
    first = run_eval(mock=True, tasks="single_tool", results_dir=tmp_path)
    assert first["quota_exhausted"] and first["n_rows"] == 2
    partial = next(tmp_path.glob("*.partial.jsonl"))
    assert len(partial.read_text().splitlines()) == 2
    assert list(tmp_path.glob("mock_*.json"))  # progress finalised despite the stop

    monkeypatch.setattr(run_eval_mod, "_make_llm", real)
    second = run_eval(mock=True, tasks="single_tool", resume=True, results_dir=tmp_path)
    assert not second["quota_exhausted"] and second["n_rows"] == 5
    done_file = partial.with_name(partial.name.replace(".partial.jsonl", ".done.jsonl"))
    assert second["partial_file"] == str(done_file) and not partial.exists()
    assert len(done_file.read_text().splitlines()) == 5
    ids = [r["id"] for r in second["tasks"]]
    assert len(ids) == len(set(ids))


@data_ready[0]
@data_ready[1]
def test_sleep_between_tasks(tmp_path, monkeypatch):
    naps = []
    monkeypatch.setattr(run_eval_mod, "time", SimpleNamespace(sleep=naps.append))
    run_eval(
        mock=True,
        tasks="st_vcf_filter_count,st_vcf_summary_types,st_inspect_metadata",
        sleep=0.25,
        results_dir=tmp_path,
    )
    assert naps == [0.25, 0.25]


@data_ready[0]
@data_ready[1]
def test_cli_eval_mock(tmp_path, monkeypatch):
    real = run_eval_mod.run_eval
    monkeypatch.setattr(run_eval_mod, "run_eval", lambda **kw: real(results_dir=tmp_path, **kw))
    res = CliRunner().invoke(
        app, ["eval", "--mock", "--tasks", "single_tool", "--repeats", "2", "--sleep", "0"]
    )
    assert res.exit_code == 0, res.output
    assert "success=1.00" in res.output
    final = next(tmp_path.glob("mock_*.json"))
    assert json.loads(final.read_text())["repeats"] == 2


def test_cli_eval_help():
    out = CliRunner().invoke(app, ["eval", "--help"]).output
    for opt in ("--mock", "--model", "--repeats", "--sleep", "--resume", "--tasks"):
        assert opt in out


def test_quota_message_constant():
    assert QUOTA_MESSAGE == "daily quota exhausted, rerun with --resume tomorrow"


def test_cli_category_option(tmp_path, monkeypatch):
    if not all(m.args[0] is False for m in [x.mark for x in data_ready]):
        pytest.skip("needs data and binaries")
    real = run_eval_mod.run_eval
    monkeypatch.setattr(run_eval_mod, "run_eval", lambda **kw: real(results_dir=tmp_path, **kw))
    res = CliRunner().invoke(app, ["eval", "--mock", "--category", "safety"])
    assert res.exit_code == 0, res.output
    rep = json.loads(next(tmp_path.glob("mock_*.json")).read_text())
    assert {t["category"] for t in rep["tasks"]} == {"safety"} and rep["n_tasks"] == 5
    assert "--category" in CliRunner().invoke(app, ["eval", "--help"]).output


@data_ready[0]
@data_ready[1]
def test_unknown_category_rejected(tmp_path):
    with pytest.raises(SystemExit):
        run_eval(mock=True, category="nope", results_dir=tmp_path)


@data_ready[0]
@data_ready[1]
def test_category_combines_with_tasks(tmp_path):
    rep = run_eval(mock=True, tasks="single_tool,safety", category="safety", results_dir=tmp_path)
    assert {t["category"] for t in rep["tasks"]} == {"safety"}


@data_ready[0]
@data_ready[1]
def test_completed_run_renamed_done_and_next_resume_starts_fresh(tmp_path):
    first = run_eval(mock=True, tasks="safety", results_dir=tmp_path)
    assert first["complete"] and first["partial_file"].endswith(".done.jsonl")
    assert not list(tmp_path.glob("*.partial.jsonl")) and len(list(tmp_path.glob("*.done.jsonl"))) == 1
    done_file = tmp_path / first["partial_file"].rsplit("/", 1)[-1]
    n_done = len(done_file.read_text().splitlines())

    # a later --resume must not reuse the finished file: it runs every pair again
    second = run_eval(mock=True, tasks="safety", resume=True, results_dir=tmp_path)
    assert second["n_rows"] == 5 and second["complete"]
    assert len(done_file.read_text().splitlines()) == n_done


@data_ready[0]
@data_ready[1]
def test_incomplete_run_stays_partial(tmp_path, monkeypatch):
    real = run_eval_mod._make_llm
    n = {"i": 0}

    def flaky(task, mock):
        n["i"] += 1
        if n["i"] == 2:

            class Out:
                model = "mock"

                def chat(self, m, t):
                    raise QuotaExhausted("daily free allocation")

            return Out()
        return real(task, mock)

    monkeypatch.setattr(run_eval_mod, "_make_llm", flaky)
    rep = run_eval(mock=True, tasks="safety", results_dir=tmp_path)
    assert not rep["complete"] and rep["partial_file"].endswith(".partial.jsonl")
    assert not list(tmp_path.glob("*.done.jsonl"))
    monkeypatch.setattr(run_eval_mod, "_make_llm", real)
    rep2 = run_eval(mock=True, tasks="safety", resume=True, results_dir=tmp_path)
    assert rep2["complete"] and rep2["n_rows"] == 5 and rep2["partial_file"].endswith(".done.jsonl")
    assert len(list(tmp_path.glob("*.done.jsonl"))) == 1


def test_summary_ignores_mock_prefixed_files_even_with_real_model_name(tmp_path):
    rows = [{"id": "a", "category": "c", "repeat": 0, "metrics": {"task_success": 1.0}}]
    rep = {"repeats": 1, "aggregate": aggregate(rows)}
    (tmp_path / "mock_20261028-1000.json").write_text(json.dumps({**rep, "model": "fancy-llm"}))
    (tmp_path / "x_20261028-1000.json").write_text(json.dumps({**rep, "model": "mock"}))
    write_summary(tmp_path)
    md = (tmp_path / "summary.md").read_text()
    assert "fancy-llm" not in md and "mock" not in md and "No real-model results" in md
