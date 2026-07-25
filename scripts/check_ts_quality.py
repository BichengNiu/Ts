"""Run the reproducible quality gate for TsSims and TsTests."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("TsSims", "TsTests")
TESTS = ("TsSims/tests", "TsTests/tests")


def _run(label: str, args: list[str]) -> None:
    print(f"\n== {label} ==", flush=True)
    completed = subprocess.run(args, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    python = sys.executable
    _run(
        "Ruff format",
        [python, "-m", "ruff", "format", "--check", *PACKAGES],
    )
    _run(
        "Ruff lint",
        [python, "-m", "ruff", "check", *PACKAGES],
    )
    _run(
        "Tests and branch coverage",
        [
            python,
            "-m",
            "pytest",
            *TESTS,
            "--cov=TsSims",
            "--cov=TsTests",
            "--cov-branch",
            "--cov-report=term-missing",
            "-q",
        ],
    )


if __name__ == "__main__":
    main()
