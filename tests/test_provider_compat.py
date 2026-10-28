import json
from types import SimpleNamespace

import pytest

from gta.agent import run_agent
from gta.llm import LLMResponse, OpenAICompatLLM, ToolCall
from gta.sandbox import ToolError
from gta.tools.enrichment import ORAInput, run_ora
from tests.conftest import needs_data


class QuirkyLLM:
    """Returns a tool call with no id, dict arguments and content=None, then a final answer."""

    model = "quirky"

    def __init__(self):
        self.seen: list[list[dict]] = []
        self.n = 0

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        self.n += 1
        if self.n == 1:
            return LLMResponse(None, [ToolCall("", "inspect_file", {"path": "data/nope.tsv"})])
        return LLMResponse(None)


def test_outgoing_messages_have_string_content(tmp_path):
    llm = QuirkyLLM()
    run_agent("inspect it", llm, trace_dir=tmp_path)
    last = llm.seen[-1]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "tool"]
    for m in last:
        assert isinstance(m.get("content"), str), m
    assistant, tool = last[2], last[3]
    call = assistant["tool_calls"][0]
    assert call["id"] == "call_1_0" and tool["tool_call_id"] == call["id"]
    assert json.loads(call["function"]["arguments"]) == {"path": "data/nope.tsv"}
    json.loads(tool["content"])


def test_openai_client_normalises_missing_id_dict_args_and_null_content(monkeypatch):
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.setenv(k, "x")
    llm = OpenAICompatLLM()
    tc = SimpleNamespace(id=None, function=SimpleNamespace(name="run_ora", arguments={"gmt": "g.gmt"}))
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    fake = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: fake)))
    resp = llm.chat([{"role": "user", "content": "hi"}], [])
    assert resp.content == "" and resp.tool_calls[0].id == "call_0"
    assert json.loads(resp.tool_calls[0].arguments) == {"gmt": "g.gmt"}


@pytest.mark.parametrize(
    "raw",
    [
        ["GENE_0001", "GENE_0002"],
        '["GENE_0001", "GENE_0002"]',
        "GENE_0001,GENE_0002",
        "GENE_0001\nGENE_0002",
        "GENE_0001 GENE_0002",
        "GENE_0001, GENE_0002\n",
    ],
)
def test_genes_validator_accepts_formats(raw):
    assert ORAInput(gmt="x.gmt", genes=raw).genes == ["GENE_0001", "GENE_0002"]


def test_genes_validator_empty_is_none_and_bad_type_rejected():
    assert ORAInput(gmt="x.gmt", genes="  ").genes is None
    with pytest.raises(ValueError):
        ORAInput(gmt="x.gmt", genes=5)


def test_ora_requires_genes_or_file():
    with pytest.raises(ToolError) as exc:
        run_ora("data/genesets/toy.gmt")
    assert "run_deseq2" in exc.value.hint


@needs_data("data/genesets/toy.gmt", "data/rnaseq/planted_up.txt")
def test_ora_with_string_genes_and_file_alternative():
    up = open("data/rnaseq/planted_up.txt").read().split()
    from gta.tools import registry

    out = registry.call("run_ora", {"gmt": "data/genesets/toy.gmt", "genes": ",".join(up)})
    assert out["summary"]["top"][0]["set"] == "PATHWAY_UP"
    out = registry.call(
        "run_ora", {"gmt": "data/genesets/toy.gmt", "gene_list_file": "data/rnaseq/planted_up.txt"}
    )
    assert out["summary"]["top"][0]["set"] == "PATHWAY_UP"
