from pathlib import Path

CONFTEST = Path(__file__).with_name("conftest.py")
pytest_plugins = ["pytester"]

TEST = """
import pytest
from tests.conftest import needs_data

@needs_data("data/nope.tsv")
def test_needs_missing_data():
    pass

@pytest.mark.skip(reason="unrelated")
def test_other_skip():
    pass
"""


def _run(pytester, *args):
    pytester.makeconftest(CONFTEST.read_text())
    pytester.makepyfile(test_x=TEST.replace("tests.conftest", "conftest"))
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", *args)


def test_missing_data_skip_is_failure_with_flag(pytester):
    res = _run(pytester, "--strict-skips")
    res.assert_outcomes(errors=1, skipped=1)
    assert "--strict-skips: " in res.stdout.str()


def test_default_mode_still_skips(pytester):
    _run(pytester).assert_outcomes(skipped=2)
