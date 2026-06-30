"""Run the reproducible M5 optimization and reporting pipeline."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str], python_executable: Path | str | None = None) -> None:
    print("+", " ".join(args), flush=True)
    executable = str(python_executable or sys.executable)
    subprocess.run([executable, *args], cwd=ROOT, check=True)


def docx_python() -> Path | None:
    env_path = os.environ.get("DOCX_PYTHON")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    bundled = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "python"
        / "python.exe"
    )
    return bundled if bundled.exists() else None


def main() -> None:
    run(["scripts/optimize_forecast_candidates.py"])
    run(["scripts/train_direct_segment_lightgbm.py"])
    run(
        [
            "scripts/score_m5_wrmsse.py",
            "--predictions",
            "output/direct_segment_lightgbm_predictions.csv",
            "--details-out",
            "output/direct_segment_lightgbm_wrmsse_by_level.csv",
        ]
    )
    run(["scripts/analyze_project_contribution.py"])
    run(["scripts/build_visual_assets.py"])
    run(["scripts/build_job_alignment_assets.py"])
    run(["scripts/build_project_showcase.py"])
    run(["scripts/build_report_presentation.py"])
    docx_runner = docx_python()
    if docx_runner is None:
        print("+ skipping scripts/build_jd_gap_docx.py; set DOCX_PYTHON to a Python with python-docx", flush=True)
    else:
        run(["scripts/build_jd_gap_docx.py"], python_executable=docx_runner)
    run(["scripts/sanitize_artifacts.py"])


if __name__ == "__main__":
    main()
