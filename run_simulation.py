"""Run seeded OFF with local probes and an optional turbine action hook.

Run: python run_simulation.py --seed 42 --time-end 8
Use --point X Y Z repeatedly to set local observation coordinates.
See README_SIMULATION_HOOKS.md for action-hook instructions and output files.
"""

import os
from pathlib import Path
import runpy
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "off_mplconfig"))
import matplotlib


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent / "OFF"
    sys.path.insert(0, str(repo / "03_Code"))
    sys.path.insert(0, str(repo.parent))
    os.chdir(repo)
    os.environ["OFF_PATH"] = str(repo)
    matplotlib.use("Agg")
    runpy.run_path(str(repo / "03_Code/main_001.py"), run_name="__main__")
