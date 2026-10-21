import pytest

from gta import sandbox
from gta.sandbox import SandboxError, ToolError


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
