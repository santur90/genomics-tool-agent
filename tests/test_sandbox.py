import pytest

from gta import sandbox
from gta.sandbox import SandboxError, ToolError
from tests.conftest import needs_bin


def test_disallowed_command():
    with pytest.raises(SandboxError):
        sandbox.run(["rm", "-rf", "/"])


def test_no_path_in_command_name():
    with pytest.raises(SandboxError):
        sandbox.run(["/bin/echo", "hi"])


def test_cwd_must_be_run_folder(tmp_path):
    with pytest.raises(SandboxError):
        sandbox.run(["samtools", "--version"], cwd=tmp_path)


def test_safe_path_blocks_outside_and_secrets():
    with pytest.raises(SandboxError):
        sandbox.safe_path("/etc/passwd")
    with pytest.raises(SandboxError):
        sandbox.safe_path("~/.ssh/id_rsa")
    with pytest.raises(SandboxError):
        sandbox.safe_path(".env", must_exist=False)


def test_safe_path_missing_file_is_tool_error():
    with pytest.raises(ToolError) as exc:
        sandbox.safe_path("data/nope.tsv")
    assert "not found" in exc.value.message.lower()


def test_empty_or_non_string_command_rejected():
    for bad in ([], ["samtools", 1]):
        with pytest.raises(SandboxError, match="non-empty list of strings"):
            sandbox.run(bad)


def test_missing_executable_reported(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    with pytest.raises(SandboxError) as exc:
        sandbox.run(["bcftools", "--version"])
    assert "not installed" in exc.value.message and "conda" in exc.value.hint


def test_timeout_becomes_sandbox_error(monkeypatch):
    def slow(*args, **kwargs):
        raise sandbox.subprocess.TimeoutExpired(cmd="samtools", timeout=1)

    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(sandbox.subprocess, "run", slow)
    with pytest.raises(SandboxError, match="timed out after 1s"):
        sandbox.run(["samtools", "view", "x.bam"], timeout=1)


@needs_bin("samtools")
def test_run_executes_without_shell_in_run_folder():
    folder = sandbox.new_run_dir("unit")
    assert folder.is_dir() and folder.is_relative_to(sandbox.RUNS_DIR)
    ok = sandbox.run(["samtools", "--version"], cwd=folder)
    assert ok.returncode == 0 and ok.stdout.startswith("samtools")
    # shell metacharacters are plain arguments, never interpreted
    bad = sandbox.run(["samtools", "view", "a.bam; touch pwned"], cwd=folder, check=False)
    assert bad.returncode != 0 and not (folder / "pwned").exists()


@needs_bin("samtools")
def test_nonzero_exit_raises_tool_error_with_stderr():
    with pytest.raises(ToolError) as exc:
        sandbox.run(["samtools", "view", "does_not_exist.bam"], cwd=sandbox.new_run_dir("fail"))
    assert "samtools failed" in exc.value.message and "exit" in exc.value.message
