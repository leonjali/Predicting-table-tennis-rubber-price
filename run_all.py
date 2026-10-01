"""Run every experiment and regenerate all paper figures."""

from pathlib import Path
import os
import subprocess
import sys


BASE = Path(__file__).resolve().parent
CACHE = BASE / ".cache" / "matplotlib"
CACHE.mkdir(parents=True, exist_ok=True)
ENV = os.environ.copy()
ENV.setdefault("MPLCONFIGDIR", str(CACHE))
ENV.setdefault("XDG_CACHE_HOME", str(BASE / ".cache"))
ENV.setdefault("MPLBACKEND", "Agg")
SCRIPTS = [
    "verify_data.py",
    "run_experiment.py",
    "run_camera_ready_experiments.py",
    "generate_figures.py",
    "generate_camera_ready_figure.py",
]


for script in SCRIPTS:
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run(
        [sys.executable, str(BASE / script)], cwd=BASE, env=ENV, check=True
    )

print("\nAll experiments and figures completed successfully.")
