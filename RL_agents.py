"""Wind forcing and local observation helpers used by run_simulation.py.

This module contains no agent or learning policy. Extend the observation
function here if a future controller needs more local state variables.
"""

from copy import deepcopy

import numpy as np


# Default probes along the first row of the bundled six-turbine example.
# All coordinates are world [x, y, z] positions in metres.
DEFAULT_COORDINATES = ((250.0, -495.0, 119.0),
                       (750.0, -495.0, 119.0),
                       (1250.0, -495.0, 119.0))


def gaussian_wind(speed_std=0.5, direction_std=3.0):
    """Return a perturbation function using the supplied random generator."""
    if speed_std < 0 or direction_std < 0:
        raise ValueError("Standard deviations must be nonnegative")

    def perturb(time, base_speed, base_direction, rng):
        speed = max(0.0, base_speed + rng.normal(0.0, speed_std))
        direction = (base_direction + rng.normal(0.0, direction_std)) % 360.0
        return speed, direction

    perturb.__name__ = "gaussian_wind"
    return perturb


def make_wind_profile(case, seed, wind_function, time_end=None):
    """Build a reproducible forcing series at every OFF simulation time."""
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    case = deepcopy(case)
    sim = case["sim"]["sim"]
    if time_end is not None:
        sim["time end"] = float(time_end)
    dt = float(sim["time step"])
    start, end = float(sim["time start"]), float(sim["time end"])
    if dt <= 0 or end < start:
        raise ValueError("Simulation needs a positive time step and end >= start")
    # Match OFF.run_sim's time grid exactly.
    ticks = np.arange(start, end + dt, dt)
    ambient = case["ambient"]["flow_field"]
    speeds = np.asarray(ambient["wind_speeds"], dtype=float)
    directions = np.asarray(ambient["wind_directions"], dtype=float)
    speed_times = np.asarray(ambient.get("wind_speeds_t", [start]), dtype=float)
    direction_times = np.asarray(ambient.get("wind_directions_t", [start]), dtype=float)
    if len(speeds) != len(speed_times) or len(directions) != len(direction_times):
        raise ValueError("Ambient values and time arrays must have matching lengths")

    rng = np.random.default_rng(int(seed))
    profile = []
    for time in ticks:
        base_speed = float(np.interp(time, speed_times, speeds))
        base_direction = float(np.interp(time, direction_times, directions))
        speed, direction = map(float, wind_function(float(time), base_speed, base_direction, rng))
        if not np.isfinite(speed) or speed < 0 or not np.isfinite(direction):
            raise ValueError(f"Invalid wind at t={time}: {(speed, direction)}")
        profile.append((float(time), speed, direction % 360.0))

    # OFF updates ambient states after measuring each tick. Shifting the feed
    # times aligns the requested wind value with the measurement time.
    feed_times = [time - dt for time, _, _ in profile]
    ambient["wind_speeds_t"] = feed_times
    ambient["wind_directions_t"] = feed_times
    ambient["wind_speeds"] = [speed for _, speed, _ in profile]
    ambient["wind_directions"] = [direction for _, _, direction in profile]
    ambient["corr_overwrite_direction"] = False  # retain moving OP history
    return case, profile


def sample_local_wind(off_sim, coordinate):
    """Sample local wind variables at a world [x, y, z] coordinate.

    Effective speed comes from the wake solver. Ambient speed and direction
    interpolate the nearest observation-point chain. OFF does not expose a
    wake-deflected direction at arbitrary locations.
    """
    point = np.asarray(coordinate, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("coordinate must be finite [x, y, z] metres")
    best = None
    for turbine_index, turbine in enumerate(off_sim.wind_farm.turbines):
        positions = np.asarray(turbine.observation_points.get_world_coord(), dtype=float)
        if len(positions) < 2:
            continue
        starts, segments = positions[:-1], np.diff(positions, axis=0)
        lengths_sq = np.einsum("ij,ij->i", segments, segments)
        fractions = np.clip(np.divide(
            np.einsum("ij,ij->i", point - starts, segments), lengths_sq,
            out=np.zeros_like(lengths_sq), where=lengths_sq > 0), 0.0, 1.0)
        nearest = starts + fractions[:, None] * segments
        distances_sq = np.einsum("ij,ij->i", nearest - point, nearest - point)
        segment_index = int(np.argmin(distances_sq))
        candidate = (float(distances_sq[segment_index]), turbine_index,
                     segment_index, float(fractions[segment_index]))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("No observation-point chain is available")

    distance_sq, turbine_index, segment_index, fraction = best
    states = off_sim.wind_farm.turbines[turbine_index].ambient_states.get_all_states()
    speeds = states[segment_index:segment_index + 2, 0]
    directions = states[segment_index:segment_index + 2, 1]
    phi = np.deg2rad(270.0 - directions)
    u = float(np.interp(fraction, [0, 1], speeds * np.cos(phi)))
    v = float(np.interp(fraction, [0, 1], speeds * np.sin(phi)))
    ambient_speed = float(np.hypot(u, v))
    direction = float((270.0 - np.degrees(np.arctan2(v, u))) % 360.0) if ambient_speed > 0 else float("nan")
    effective_speed = off_sim.wake_solver.sample_effective_speed_at(point, off_sim.wind_farm)
    return {
        "x_m": float(point[0]), "y_m": float(point[1]), "z_m": float(point[2]),
        "effective_wind_speed_m_s": effective_speed,
        "ambient_wind_speed_m_s": ambient_speed,
        "ambient_direction_from_deg": direction,
        "nearest_op_chain_turbine": turbine_index,
        "distance_to_op_chain_m": float(np.sqrt(distance_sq)),
    }
