import re
import shutil
import sys
from importlib import resources
from pathlib import Path

import pytest
from typer.testing import CliRunner

import gta
from gta import cli, doctor, sandbox
from gta.agent import SYSTEM_PROMPT
from gta.sandbox import SandboxError
from gta.tools import nextflow

runner = CliRunner()


def test_version_is_single_sourced_and_shown():
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", gta.__version__)
    res = runner.invoke(cli.app, ["--version"])
    assert res.exit_code == 0 and gta.__version__ in res.output


def test_packaged_resources_exist_and_are_used():
    root = resources.files("gta") / "resources"
    assert (root / "pipelines.yaml").is_file() and (root / "pipelines" / "rnaseq_mini" / "main.nf").is_file()
    assert (root / "pipelines" / "rnaseq_mini" / "example" / "counts.tsv").is_file()
    assert "Only state numbers that appear in tool outputs" in SYSTEM_PROMPT
    assert "rnaseq_mini" in nextflow.load_pipelines()


def test_project_config_extends_and_overrides_packaged_pipelines(tmp_path, monkeypatch):
    cfg = tmp_path / "pipelines.yaml"
    cfg.write_text("pipelines:\n  mine:\n    path: flows/main.nf\n")
    monkeypatch.setattr(nextflow, "PROJECT_CONFIG", cfg)
    loaded = nextflow.load_pipelines()
    assert loaded["mine"]["_packaged"] is False and loaded["rnaseq_mini"]["_packaged"] is True


def test_root_is_gta_root_then_checkout_then_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("GTA_ROOT", str(tmp_path))
    assert sandbox._default_root() == tmp_path
    monkeypatch.delenv("GTA_ROOT")
    assert (sandbox._default_root() / "pyproject.toml").is_file()  # running from the checkout
    fake_site_packages = tmp_path / "lib" / "python3.11" / "site-packages" / "gta"
    fake_site_packages.mkdir(parents=True)
    monkeypatch.setattr(sandbox, "__file__", str(fake_site_packages / "sandbox.py"))
    monkeypatch.chdir(tmp_path)
    assert sandbox._default_root() == Path.cwd()  # installed wheel: work in the current directory


def test_missing_binary_error_points_to_doctor(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    with pytest.raises(SandboxError) as exc:
        sandbox.run(["bcftools", "--version"])
    assert "gta doctor" in exc.value.hint and "bioconda" in exc.value.hint


def test_nextflow_without_java_points_to_doctor(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None if name == "java" else "/bin/" + name)
    with pytest.raises(SandboxError) as exc:
        sandbox.run(["nextflow", "-version"])
    assert "Java" in exc.value.message and "gta doctor" in exc.value.hint


def test_version_parsing_handles_stderr_banners():
    assert doctor._first_version("bcftools 1.19\nUsing htslib 1.19") == "1.19"
    assert doctor._first_version('openjdk version "17.0.9" 2023-10-17') == "17.0.9"
    assert doctor._first_version("no digits") == "unknown"


def test_doctor_table_and_exit_codes(monkeypatch):
    monkeypatch.setattr(cli, "load_env", lambda: None)
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None if name == "bwa" else "/bin/" + name)
    monkeypatch.setattr(doctor, "binary_version", lambda cmd: "9.9")
    res = runner.invoke(cli.app, ["doctor"], env={"COLUMNS": "200"})
    assert res.exit_code == 1  # bwa missing is a required component
    out = res.output
    assert "MISSING" in out and "conda install -c conda-forge -c bioconda bwa" in out and "9.9" in out
    assert "LLM_API_KEY" in out

    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    res = runner.invoke(cli.app, ["doctor"], env={"COLUMNS": "200"})
    assert res.exit_code == 0  # missing LLM vars alone only warn
    assert "sk-secret" not in res.output


def test_doctor_flags_missing_python_package(monkeypatch):
    def fake_version(name):
        raise doctor.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(doctor.metadata, "version", fake_version)
    rows = doctor.check_python_deps()
    assert all(not r.ok and r.hint.startswith("pip install") for r in rows)
    assert doctor.exit_code(rows) == 1


def test_eval_without_checkout_gives_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    for mod in [m for m in sys.modules if m == "eval" or m.startswith("eval.")]:
        monkeypatch.delitem(sys.modules, mod)
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p or ".").resolve() != repo])
    res = runner.invoke(cli.app, ["eval", "--mock"])
    assert res.exit_code == 1 and "not part of the installed package" in res.output


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow not installed")
def test_packaged_pipeline_runs_in_stub_mode_with_defaults():
    from gta.tools import registry

    out = registry.call("run_nextflow", {"pipeline": "rnaseq_mini", "stub": True})
    assert "error" not in out, out
    assert out["summary"]["n_output_files"] >= 1
