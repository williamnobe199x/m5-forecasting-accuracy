"""Run the reproducible M5 optimization and reporting pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    run(["scripts/optimize_forecast_candidates.py"])
    run(["scripts/analyze_project_contribution.py"])
    run(["scripts/build_visual_assets.py"])
    run(["scripts/build_project_showcase.py"])


if __name__ == "__main__":
    main()
