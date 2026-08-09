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
    python_requires=">=3.13",
    install_requires=[
        "numpy>=2.5.1,<3",
        "pandas>=3.0.5,<4",
        "scipy>=1.18.0,<2",
        "matplotlib>=3.11.1,<4",
        "statsmodels>=0.14.6,<0.15",
        "arch>=8.0.0,<9",
    ],
)
