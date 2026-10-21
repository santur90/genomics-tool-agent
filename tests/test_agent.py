import json

from gta.agent import run_agent
from gta.llm import MockLLM


def test_agent_runs_tools_and_logs_trace(tmp_path):
    llm = MockLLM.from_calls(
        [{"name": "inspect_file", "arguments": {"path": "data/nope.tsv"}}],
        "Error: {r0.error}",
    )
    res = run_agent("inspect data/nope.tsv", llm, trace_dir=tmp_path)
    assert res.calls[0].is_error and "not found" in res.answer.lower()
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events] == ["start", "llm", "tool", "llm", "end"]
    assert events[2]["name"] == "inspect_file" and events[2]["error"]


def test_agent_stops_at_max_steps(tmp_path):
    llm = MockLLM([{"tool_calls": [{"name": "inspect_file", "arguments": {"path": "x.tsv"}}]}] * 10)
    res = run_agent("loop", llm, max_steps=3, trace_dir=tmp_path)
    assert res.stopped == "max_steps" and res.n_steps == 3


def test_tool_output_truncated(tmp_path):
    captured = {}

    class Spy(MockLLM):
        def chat(self, messages, tools):
            captured["msgs"] = messages
            return super().chat(messages, tools)

    import gta.agent as agent

    old = agent.registry.call
    agent.registry.call = lambda name, args: {"summary": {"blob": "x" * 10000}, "outputs": {}}
    try:
        run_agent("t", Spy.from_calls([{"name": "inspect_file", "arguments": {}}], "done"), trace_dir=tmp_path)
    finally:
        agent.registry.call = old
    tool_msg = [m for m in captured["msgs"] if m["role"] == "tool"][0]
    assert len(tool_msg["content"]) < 4100


def test_mock_from_yaml(tmp_path):
    f = tmp_path / "script.yaml"
    f.write_text("steps:\n  - content: hello\n")
    assert run_agent("hi", MockLLM.from_yaml(f), trace_dir=tmp_path / "t").answer == "hello"
