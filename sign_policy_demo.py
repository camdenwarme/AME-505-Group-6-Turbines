"""Test independent sign policies, optionally with a CPU process pool.

python sign_policy_demo.py --duration 120
python sign_policy_demo.py --duration 120 --workers 2 --rollouts 4
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import os
from pathlib import Path
import uuid

import numpy as np
import yaml

from fdynsim import FDynSim, DEFAULT_CASE, ROOT


@dataclass
class SignPolicy:
    angular_speed: float = 0.3  # degrees per second

    def __call__(self, wind_speed, wind_direction, heading):
        # Changed to a smooth sigmoid policy.
        return self.angular_speed * float(1/(1+np.exp(heading - wind_direction))-0.5)


def run_one(task, case=DEFAULT_CASE):
    episode, W, W_theta, dt, rate, lambda_, experiment_dir, keep_results = task
    settings = yaml.safe_load(Path(case).read_text())
    count = len(settings["wind_farm"]["farm"]["layout_x"])
    policies = [SignPolicy(rate) for _ in range(count)]  # independent callables
    result = FDynSim(policies, W, W_theta, dt=dt, lambda_=lambda_,
                     work_dir=experiment_dir, worker_id=f"worker_{os.getpid()}", keep_results=keep_results, case=case)
    destination = Path(experiment_dir) / f"trajectory_{episode:05d}.npz"
    np.savez_compressed(destination, times=result["times"], headings=result["headings"],
        actions=result["actions"], farm_power=result["farm_power"], R=result["R"], energy=result["energy"],
        states=np.stack([x["states"] for x in result["trajectories"]]),
        next_states=np.stack([x["next_states"] for x in result["trajectories"]]),
        rewards=np.stack([x["rewards"] for x in result["trajectories"]]),
        terminated=result["trajectories"][0]["terminated"])
    return f"Rollout {episode}: energy={result['energy']/3.6e9:.6f} MWh; trajectory={destination}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE, help="Base YAML for wind, layout and solver settings")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--angular-speed", type=float, default=0.3)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--keep-results", action="store_true", help="Also retain native OFF files/CSVs")
    args = parser.parse_args()
    if args.duration <= 0 or args.dt <= 0 or not np.isclose(args.duration/args.dt, round(args.duration/args.dt)):
        parser.error("Use a positive duration that is an integer multiple of a positive dt.")
    if args.workers < 1 or args.rollouts < 1 or args.angular_speed < 0:
        parser.error("workers/rollouts must be positive; angular speed must be nonnegative.")
    args.case = args.case.expanduser().resolve()
    settings = yaml.safe_load(args.case.read_text())
    ambient = settings["ambient"]["flow_field"]
    start = float(settings["sim"]["sim"]["time start"])
    times = start + np.arange(round(args.duration/args.dt)+1) * args.dt
    W = np.interp(times, ambient.get("wind_speeds_t", [start]), ambient["wind_speeds"])
    W_theta = np.interp(times, ambient.get("wind_directions_t", [start]), ambient["wind_directions"])
    experiment_dir = ROOT / "rl_runs" / f"sign_test_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    experiment_dir.mkdir(parents=True)
    tasks = [(i, W, W_theta, args.dt, args.angular_speed, args.lambda_, str(experiment_dir), args.keep_results)
             for i in range(args.rollouts)]
    if args.workers == 1:
        for task in tasks:
            print(run_one(task, case=args.case), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for message in pool.map(partial(run_one, case=args.case), tasks):
                print(message, flush=True)
    print("Results:", experiment_dir)


if __name__ == "__main__":
    main()
