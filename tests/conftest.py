"""Shared fixtures: skip data/tool-dependent tests when prerequisites are missing."""

import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def needs_data(*paths: str):
    """Skip unless all given data files exist (run `make data`)."""
    missing = [p for p in paths if not (ROOT / p).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"missing data: {missing}; run `make data`")


def needs_bin(*names: str):
    """Skip unless all binaries are on PATH."""
    missing = [n for n in names if shutil.which(n) is None]
    return pytest.mark.skipif(bool(missing), reason=f"missing binaries: {missing}")


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


def pytest_addoption(parser):
    parser.addoption(
        "--strict-skips",
        action="store_true",
        default=False,
        help="fail (instead of skip) tests skipped because data or binaries are missing",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not item.config.getoption("--strict-skips") or not report.skipped:
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    if "missing data" in reason or "missing binaries" in reason:
        report.outcome = "failed"
        report.longrepr = f"--strict-skips: {reason}"
