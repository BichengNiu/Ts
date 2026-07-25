"""Run the reproducible quality gate for the complete Ts package."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = ROOT / "pyproject.toml"


def _discover_packages() -> tuple[str, ...]:
    """Return every top-level Ts subpackage in deterministic order."""
    packages = tuple(
        sorted(
            path.name
            for path in ROOT.iterdir()
            if path.is_dir()
            and path.name.startswith("Ts")
            and (path / "__init__.py").is_file()
        )
    )
    if not packages:
        raise RuntimeError("no Ts subpackages were found")
    return packages


def _coverage_sources() -> tuple[str, ...]:
    """Read coverage sources from the canonical project configuration."""
    with PROJECT_CONFIG.open("rb") as file:
        config = tomllib.load(file)
    sources = config["tool"]["coverage"]["run"]["source"]
    if not isinstance(sources, list) or not all(
        isinstance(source, str) and source for source in sources
    ):
        raise TypeError("coverage source must be a list of non-empty strings")
    return tuple(sources)


def _run(label: str, args: list[str]) -> None:
    print(f"\n== {label} ==", flush=True)
    completed = subprocess.run(args, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    python = sys.executable
    root_modules = sorted(path.name for path in ROOT.glob("*.py"))
    lint_targets = [*_discover_packages(), *root_modules, "scripts"]
    coverage_args = [
        argument for source in _coverage_sources() for argument in ("--cov", source)
    ]
    _run(
        "Ruff format",
        [python, "-m", "ruff", "format", "--check", *lint_targets],
    )
    _run(
        "Ruff lint",
        [python, "-m", "ruff", "check", *lint_targets],
    )
    _run(
        "Tests and branch coverage",
        [
            python,
            "-m",
            "pytest",
            *coverage_args,
            "--cov-branch",
            "--cov-report=term-missing",
            "-q",
        ],
    )


if __name__ == "__main__":
    main()
