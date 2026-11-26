"""Seeded, replayable wind functions for training and evaluation episodes.

Random values are sampled at time knots. Speed is interpolated linearly and
direction is held until the next knot. Profiles use explicit episode identifiers,
not the process-global NumPy RNG; evaluation cannot consume training wind draws.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


MODES = ("off", "speed", "direction", "both")
_LINEAR_DIRECTION = "linear_unwrapped_shortest_path"
_STEPPED_DIRECTION = "step_hold"
_LINEAR_GENERATOR = "piecewise_linear_wind_v1"
_STEPPED_GENERATOR = "linear_speed_stepped_direction_v1"


def _hold_direction(time, knots, directions):
    indices = np.searchsorted(knots, time, side="right") - 1
    return directions[np.maximum(indices, 0)]


def _identifier(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


@dataclass(frozen=True)
class WindProfileConfig:
    """Settings for wind with linearly changing speed and stepped direction.

    Speed and direction knots are sampled within their configured bounds.
    A direction half-width of 180 degrees covers the full circle. Speed changes
    are at most (speed_max-speed_min)/knot_interval_s per second, including in
    the last partial interval. Direction switches at each knot.
    """

    mode: str = "off"
    seed: int = 1
    speed_min: float = 6.0
    speed_max: float = 10.0
    direction_half_width: float = 15.0
    knot_interval_s: float = 120.0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Wind mode must be one of {MODES}.")
        object.__setattr__(self, "seed", _identifier(self.seed, "wind seed"))
        for name in ("speed_min", "speed_max", "direction_half_width", "knot_interval_s"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite.") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if not 0 < self.speed_min <= self.speed_max:
            raise ValueError("Wind speed knot bounds must satisfy 0 < minimum <= maximum.")
        if not 0 <= self.direction_half_width <= 180:
            raise ValueError("Direction half-width must be between 0 and 180 degrees.")
        if self.knot_interval_s <= 0:
            raise ValueError("Wind knot interval must be positive.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True, eq=False)
class WindScenario:
    """An explicit wind realization that can be replayed without RNG calls.

    ``directions_deg`` contains unwrapped angles at the recorded knots.
    ``sample(t)`` returns the meteorological wind-from angle wrapped to [0,360).
    Values outside the recorded time interval use the nearest endpoint. Arrays
    are copied and marked read-only to prevent a replay from mutating its source.
    """

    times_s: Any
    speeds_m_s: Any
    directions_deg: Any
    mode: str = "off"
    master_seed: int = 1
    episode_seed: int = 1
    episode_id: int = 0
    stream_id: int = 0
    environment_id: int = 0
    scenario_id: str = ""
    direction_interpolation: str = _LINEAR_DIRECTION

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Wind mode must be one of {MODES}.")
        if self.direction_interpolation not in (_LINEAR_DIRECTION, _STEPPED_DIRECTION):
            raise ValueError("Unsupported wind direction interpolation.")
        for name in ("master_seed", "episode_seed", "episode_id", "stream_id", "environment_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        arrays = []
        for name in ("times_s", "speeds_m_s", "directions_deg"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
                raise ValueError(f"{name} must be a finite one-dimensional array with at least two samples.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
            arrays.append(values)
        if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
            raise ValueError("Wind scenario arrays must have matching shapes.")
        if self.times_s[0] != 0 or not np.all(np.diff(self.times_s) > 0):
            raise ValueError("Wind times must start at zero and increase strictly.")
        if not np.all(self.speeds_m_s > 0):
            raise ValueError("Wind speeds must be positive.")
        if not isinstance(self.scenario_id, str):
            raise ValueError("scenario_id must be a string.")
        if not self.scenario_id:
            identity = self._payload()
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                                 allow_nan=False).encode()).hexdigest()[:16]
            object.__setattr__(self, "scenario_id", f"wind-e{self.episode_id:06d}-{digest}")

    @property
    def duration_s(self):
        return float(self.times_s[-1])

    def _payload(self):
        generator = (_STEPPED_GENERATOR if self.direction_interpolation == _STEPPED_DIRECTION
                     else _LINEAR_GENERATOR)
        return {"format_version": 1, "generator": generator,
                "direction_interpolation": self.direction_interpolation,
                "mode": self.mode, "master_seed": self.master_seed,
                "episode_seed": self.episode_seed, "episode_id": self.episode_id,
                "stream_id": self.stream_id, "environment_id": self.environment_id,
                "times_s": self.times_s.tolist(), "speeds_m_s": self.speeds_m_s.tolist(),
                "directions_deg": self.directions_deg.tolist()}

    def to_dict(self):
        return {**self._payload(), "scenario_id": self.scenario_id}

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data.pop("format_version", 1) != 1:
            raise ValueError("Unsupported wind scenario format.")
        if data.pop("generator", _LINEAR_GENERATOR) not in (_LINEAR_GENERATOR, _STEPPED_GENERATOR):
            raise ValueError("Unsupported wind scenario generator.")
        return cls(**data)

    def direction_unwrapped(self, time_s):
        time = np.asarray(time_s, dtype=float)
        if not np.isfinite(time).all():
            raise ValueError("Wind sample time must be finite.")
        if self.direction_interpolation == _STEPPED_DIRECTION:
            direction = _hold_direction(time, self.times_s, self.directions_deg)
        else:
            direction = np.interp(time, self.times_s, self.directions_deg)
        return float(direction) if time.ndim == 0 else direction

    def sample(self, time_s):
        time = np.asarray(time_s, dtype=float)
        direction = self.direction_unwrapped(time) % 360.0
        speed = np.interp(time, self.times_s, self.speeds_m_s)
        if time.ndim == 0:
            return float(speed), float(direction)
        return speed, direction


def make_wind_scenario(config, duration_s, base_speed, base_direction, episode_id,
                       stream_id=0, environment_id=0):
    """Generate one independent seeded episode profile or an explicit fixed one.

    Use a stable stream ID for training and another for held-out evaluation.
    Reusing the same identifiers regenerates the same profile even if other
    episodes or evaluations have been generated in between.
    """
    if config is None:
        config = WindProfileConfig()
    if not isinstance(config, WindProfileConfig):
        config = WindProfileConfig.from_dict(config)
    duration_s, base_speed, base_direction = map(float, (duration_s, base_speed, base_direction))
    if not np.isfinite([duration_s, base_speed, base_direction]).all() or duration_s <= 0 or base_speed <= 0:
        raise ValueError("Wind duration and base speed must be positive, and base direction finite.")
    episode_id = _identifier(episode_id, "episode_id")
    stream_id = _identifier(stream_id, "stream_id")
    environment_id = _identifier(environment_id, "environment_id")
    sequence = np.random.SeedSequence([config.seed, stream_id, environment_id, episode_id])
    episode_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
    # Separate dimension streams keep direction-only and both-mode realizations
    # matched when switching the speed randomizer on or off, and vice versa.
    speed_sequence, direction_sequence = np.random.SeedSequence(episode_seed).spawn(2)
    speed_rng = np.random.default_rng(speed_sequence)
    direction_rng = np.random.default_rng(direction_sequence)
    if config.mode == "off":
        full_times = np.array([0.0, duration_s])
    else:
        full_times = np.arange(int(np.ceil(duration_s / config.knot_interval_s)) + 1,
                               dtype=float) * config.knot_interval_s
    speeds = (speed_rng.uniform(config.speed_min, config.speed_max, size=len(full_times))
              if config.mode in ("speed", "both") else np.full(len(full_times), base_speed))
    varying_direction = config.mode in ("direction", "both")
    if varying_direction:
        knots = (base_direction + direction_rng.uniform(-config.direction_half_width,
                                                        config.direction_half_width,
                                                        size=len(full_times))) % 360
        directions = np.rad2deg(np.unwrap(np.deg2rad(knots)))
    else:
        directions = np.full(len(full_times), base_direction % 360)
    # The episode endpoint samples the ongoing speed ramp and holds the last
    # direction until the next actual knot, even in a partial interval.
    times = np.append(full_times[full_times < duration_s], duration_s)
    speeds = np.interp(times, full_times, speeds)
    directions = _hold_direction(times, full_times, directions)
    return WindScenario(times, speeds, directions, mode=config.mode,
                        master_seed=config.seed, episode_seed=episode_seed,
                        episode_id=episode_id, stream_id=stream_id, environment_id=environment_id,
                        direction_interpolation=_STEPPED_DIRECTION if varying_direction else _LINEAR_DIRECTION)
