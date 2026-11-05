# Copyright (C) <2024>, M Becker (TUDelft), M Lejeune (UCLouvain)

# List of the contributors to the development of OFF: see LICENSE file.
# Description and complete License: see LICENSE file.

# This program (OFF) is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.

"""Entry point used by the repository's run_simulation.py wrapper.

This file wires OFF to wind generation, local observations, and the optional
per-tick turbine action callback. See README_SIMULATION_HOOKS.md at the project
root for the callback contract and examples of manual changes.
"""

import argparse
from collections.abc import Mapping
import csv
from contextlib import ExitStack
import json
from pathlib import Path
import secrets
import sys
import tempfile

import numpy as np
import yaml

from off.off_interface import OFFInterface

# Also support running this file directly from the OFF checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL_agents import DEFAULT_COORDINATES, gaussian_wind, make_wind_profile, sample_local_wind


DEFAULT_CASE = PROJECT_ROOT / "OFF/02_Examples_and_Cases/02_Example_Cases/001_two_turbines_yaw_step_floris.yaml"
SAMPLE_FIELDS = ("time_s", "point_index", "x_m", "y_m", "z_m",
                 "effective_wind_speed_m_s", "ambient_wind_speed_m_s",
                 "ambient_direction_from_deg", "nearest_op_chain_turbine",
                 "distance_to_op_chain_m")


def choose_wind_function(args):
    """Change this function to provide a custom seeded wind perturbation.

    The returned function receives (time, base_speed, base_direction, rng) and
    returns (speed, direction). The RNG is seeded once for each run.
    """
    return gaussian_wind(args.speed_std, args.direction_std)


def choose_actions(time_s, off_sim, observations):
    """Return optional turbine commands based on this tick's local observations.

    This is the manual integration point for a future agent. ``observations``
    is a tuple of one dictionary per --point, in point_index order. Return
    ``{turbine_index: {"yaw_deg": value, "axial_induction": value}}``. The
    default empty result leaves the case YAML controller in charge. Manual
    --yaw-action / --axial-action CLI values are merged after this function
    and take priority for any field they specify.

    Example manual action: return {0: {"yaw_deg": 10.0}}
    """
    return {}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE,
                        help="OFF case YAML (default: bundled FLORIS example)")
    parser.add_argument("--seed", type=int, help="Random seed; omitted generates and logs one")
    parser.add_argument("--point", nargs=3, type=float, action="append", metavar=("X", "Y", "Z"),
                        help="World coordinate in metres; repeat for multiple points")
    parser.add_argument("--speed-std", type=float, default=0.5,
                        help="Gaussian wind speed standard deviation in m/s")
    parser.add_argument("--direction-std", type=float, default=3.0,
                        help="Gaussian wind direction standard deviation in degrees")
    parser.add_argument("--time-end", type=float, help="Optional simulation end time in seconds")
    parser.add_argument("--yaw-action", nargs=2, action="append", metavar=("TURBINE", "DEG"),
                        help="Apply constant yaw misalignment to a zero-based turbine index")
    parser.add_argument("--axial-action", nargs=2, action="append", metavar=("TURBINE", "VALUE"),
                        help="Apply constant axial induction to a zero-based turbine index")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed = secrets.randbits(64) if args.seed is None else args.seed
    coordinates = tuple(tuple(point) for point in (args.point or DEFAULT_COORDINATES))
    if any(np.asarray(point).shape != (3,) or not np.all(np.isfinite(point)) for point in coordinates):
        raise ValueError("Every --point must contain three finite coordinates")

    manual_actions = {}
    for entries, field in ((args.yaw_action or [], "yaw_deg"),
                           (args.axial_action or [], "axial_induction")):
        for index_text, value_text in entries:
            manual_actions.setdefault(int(index_text), {})[field] = float(value_text)

    wind_function = choose_wind_function(args)
    case_path = args.case.resolve()
    with case_path.open() as stream:
        base_case = yaml.safe_load(stream)
    case, profile = make_wind_profile(base_case, seed, wind_function, args.time_end)
    function_name = getattr(wind_function, "__name__", type(wind_function).__name__)
    case["randomization"] = {
        "seed": seed,
        "function": function_name,
        "speed_std_m_s": args.speed_std if function_name == "gaussian_wind" else None,
        "direction_std_deg": args.direction_std if function_name == "gaussian_wind" else None,
        "numpy_bit_generator": "PCG64",
        "source_case": str(case_path),
        "observation_coordinates_m": coordinates,
        "note": "Ambient feed times are shifted back one tick to match OFF's correction order.",
    }

    with tempfile.TemporaryDirectory(prefix="off_case_") as temp_dir:
        generated_case = Path(temp_dir) / "seeded_case.yaml"
        with generated_case.open("w") as stream:
            yaml.safe_dump(case, stream, sort_keys=False)

        interface = OFFInterface()
        interface.init_simulation_by_path(generated_case)
        run_dir = Path(interface.off_sim.sim_dir)
        interface.store_run_file()
        with (run_dir / "randomization.json").open("w") as stream:
            json.dump(case["randomization"], stream, indent=2)
            stream.write("\n")
        with (run_dir / "wind_profile.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time_s", "ambient_wind_speed_m_s", "ambient_direction_from_deg"))
            writer.writerows(profile)

        latest_observations = []
        with ExitStack() as stack:
            all_stream = stack.enter_context((run_dir / "local_wind_samples.csv").open("w", newline=""))
            all_writer = csv.DictWriter(all_stream, fieldnames=SAMPLE_FIELDS)
            all_writer.writeheader()
            point_writers = []
            for index in range(len(coordinates)):
                stream = stack.enter_context((run_dir / f"local_wind_point_{index}.csv").open("w", newline=""))
                writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDS)
                writer.writeheader()
                point_writers.append(writer)

            def observe_tick(time_s, off_sim):
                latest_observations.clear()
                for index, point in enumerate(coordinates):
                    row = {"time_s": time_s, "point_index": index,
                           **sample_local_wind(off_sim, point)}
                    latest_observations.append(row)
                    all_writer.writerow(row)
                    point_writers[index].writerow(row)

            def action_hook(time_s, off_sim):
                actions = choose_actions(time_s, off_sim, tuple(latest_observations))
                if not manual_actions:
                    return actions
                if actions is not None and not isinstance(actions, Mapping):
                    raise TypeError("choose_actions must return a mapping or None")
                merged = {index: dict(action) for index, action in (actions or {}).items()}
                for index, action in manual_actions.items():
                    merged.setdefault(index, {}).update(action)
                return merged

            interface.run_sim(on_tick=observe_tick, on_action=action_hook)

        interface.store_measurements()
        interface.store_applied_control()

    print(f"Run saved to {run_dir}")
    print(f"Seed: {seed}")
    return run_dir


if __name__ == "__main__":
    main()
