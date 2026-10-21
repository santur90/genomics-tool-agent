"""Importing this package registers all tools."""
from gta.tools import enrichment, files, nextflow, rnaseq, variants  # noqa: F401
from gta.tools.registry import call, openai_tool_specs, tool_names  # noqa: F401
