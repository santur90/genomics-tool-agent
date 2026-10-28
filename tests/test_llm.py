import openai
import pytest

from gta import llm as llm_mod
from gta.llm import MockLLM, OpenAICompatLLM, backoff_delay, is_quota_error, is_retryable, load_env, render


def test_load_env_parses_export_quotes_comments_and_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nexport LLM_BASE_URL='http://x/v1'\nLLM_API_KEY=\"k\"\nLLM_MODEL=file-model\nnot a pair\n\n"
    )
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "already-set")
    load_env(env)
    import os

    assert os.environ["LLM_BASE_URL"] == "http://x/v1" and os.environ["LLM_API_KEY"] == "k"
    assert os.environ["LLM_MODEL"] == "already-set"
    load_env(tmp_path / "missing.env")  # no file: silently ignored


def test_client_reports_all_missing_env_vars(monkeypatch):
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError) as exc:
        OpenAICompatLLM()
    assert all(k in str(exc.value) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))


class _Err(Exception):
    def __init__(self, status, body="", headers=None):
        super().__init__(f"{status} {body}")
        self.status_code, self.body = status, body
        self.response = type("R", (), {"headers": headers or {}})()


def test_quota_detection_requires_429():
    assert is_quota_error(_Err(429, "daily free allocation used"))
    assert is_quota_error(_Err(429, {"errors": [{"code": 4006}]}))
    assert not is_quota_error(_Err(500, "daily free allocation"))
    assert not is_quota_error(_Err(429, "slow down, rate limited"))
    assert not is_quota_error(ValueError("no status"))


def test_retryable_classification_includes_connection_errors():
    assert is_retryable(_Err(429)) and is_retryable(_Err(502))
    assert not is_retryable(_Err(400)) and not is_retryable(_Err(401))
    req = object()  # the exception only stores the request
    assert is_retryable(openai.APIConnectionError(request=req))
    assert is_retryable(openai.APITimeoutError(request=req))
    assert not is_retryable(ValueError("x"))


def test_backoff_grows_honours_retry_after_and_is_capped():
    d0, d3 = backoff_delay(0), backoff_delay(3)
    assert 1 <= d0 <= 2 and 8 <= d3 <= 9
    assert backoff_delay(0, _Err(429, headers={"retry-after": "30"})) >= 30
    assert backoff_delay(0, _Err(429, headers={"retry-after": "soon"})) <= 2  # unparsable header ignored
    assert backoff_delay(10) <= llm_mod.BACKOFF_CAP_S


def test_render_placeholders():
    results = [{"summary": {"n": 7, "top": [{"set": "A"}, {"set": "B"}]}, "outputs": {"f": "x.txt"}}]
    assert render("n={r0.summary.n} {r0.summary.top[1].set}", results) == "n=7 B"
    assert render({"a": ["{r0.outputs.f}", 3]}, results) == {"a": ["x.txt", 3]}
    assert render("{r5.summary.n} {r0.nope}", results) == " "  # unresolved references become empty


def test_mock_llm_script_and_exhaustion(tmp_path):
    script = tmp_path / "s.yaml"
    script.write_text(
        "steps:\n  - tool_calls:\n      - {name: inspect_file, arguments: {path: a.tsv}}\n  - content: 'got {r0.error}'\n"
    )
    mock = MockLLM.from_yaml(script)
    first = mock.chat([{"role": "user", "content": "q"}], [])
    assert first.tool_calls[0].name == "inspect_file" and first.content is None
    msgs = [{"role": "tool", "content": '{"error": "boom"}'}, {"role": "tool", "content": "not json"}]
    assert mock.chat(msgs, []).content == "got boom"
    assert mock.chat(msgs, []).content == "(mock script exhausted)"
