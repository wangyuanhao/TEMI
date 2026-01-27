from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).parent

def read_requirements(filename="requirements.txt"):
    """Read requirements from requirements.txt if it exists."""
    path = ROOT / filename
    if not path.exists():
        return []
    reqs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Avoid editable installs / local paths in install_requires
        if line.startswith(("-e ", "--editable")):
            continue
        reqs.append(line)
    return reqs

setup(
    name="temi",
    version="0.1.0",
    description="",
    python_requires=">=3.7.0",


    packages=find_packages(exclude=("tests", "docs", "examples")),


    install_requires=read_requirements(),


    entry_points={
        "console_scripts": [
            "temi-apr=temi.run_TEMI_APR:cli_main",
            "temi-aod=temi.run_TEMI_AOD:cli_main",
            "temi-aodpr=temi.run_TEMI_AODPR:cli_main",
        ]
    },
)
