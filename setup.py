"""Build configuration for the Ts time-series toolkit."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
SUBPACKAGES = [
    package
    for package in find_packages(where=str(ROOT))
    if ".tests" not in package
]
REQUIREMENTS = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]


setup(
    name="Ts",
    version="0.1.0",
    description="Time series econometrics toolkit",
    packages=["Ts", *(f"Ts.{package}" for package in SUBPACKAGES)],
    package_dir={"Ts": "."},
    python_requires=">=3.13",
    install_requires=REQUIREMENTS,
)
