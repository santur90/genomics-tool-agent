# Project rules for genomics-tool-agent

- Project: genomics-tool-agent, a Python 3.11 LLM agent that calls genomics analysis tools via OpenAI-style function calling, plus an evaluation benchmark.
- Package layout: `src/gta/` (`llm.py`, `agent.py`, `sandbox.py`, `cli.py`, `tools/`), `eval/`, `data/`, `tests/`.
- LLM access ONLY through an OpenAI-compatible client configured by env vars `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. Never hardcode model names or keys.
- Every tool: a pydantic input model, auto-generated JSON schema, a pure Python function returning a JSON-serializable dict with `"summary"` and `"outputs"` (file paths). Tools never print; they return.
- All subprocess calls go through `sandbox.run()`: command allowlist (bcftools, samtools, bwa, wgsim, nextflow), timeout, working directory restricted to a run folder, no `shell=True`.
- Parse VCFs with `bcftools query`, not cyvcf2/pysam.
- Use type hints, docstrings, pytest tests for every module. Keep functions small.
