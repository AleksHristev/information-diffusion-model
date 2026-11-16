"""Runs every script in examples/ and checks that it exits cleanly and prints a result."""

import pathlib
import subprocess
import sys

import pytest

EXAMPLES = sorted((pathlib.Path(__file__).resolve().parents[1] / "examples").glob("*.py"))


def test_there_are_examples():
    assert len(EXAMPLES) >= 3


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_example_runs(script):
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300, cwd=script.parent)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
