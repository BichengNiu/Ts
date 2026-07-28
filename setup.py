"""Build configuration for the Ts time-series toolkit."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
SUBPACKAGES = [
    package
    for package in find_packages(where=str(ROOT))
    if ".tests" not in package and not package.endswith(".tests")
]


setup(
    name="Ts",
    version="0.1.0",
    description="Time series econometrics toolkit",
    packages=["Ts", *(f"Ts.{package}" for package in SUBPACKAGES)],
    package_dir={"Ts": "."},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.0,<3",
        "pandas>=2.2,<3",
        "scipy>=1.13,<2",
        "matplotlib>=3.8,<4",
        "statsmodels>=0.14.5,<0.15",
        "arch>=7,<9",
    ],
)
