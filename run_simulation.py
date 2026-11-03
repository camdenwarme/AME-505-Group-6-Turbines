"""Run OFF's original main_001.py with portable paths and headless plotting.

Install: python -m pip install -r requirements.txt
Run:     python run_simulation.py
Configure the case selected in OFF/03_Code/main_001.py.
"""

import os
from pathlib import Path
import runpy
import sys

import matplotlib


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent / "OFF"
    sys.path.insert(0, str(repo / "03_Code"))
    os.chdir(repo)
    os.environ["OFF_PATH"] = str(repo)
    matplotlib.use("Agg") 
    runpy.run_path(str(repo / "03_Code/main_001.py"), run_name="__main__")
