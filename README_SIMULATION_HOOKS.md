# Running OFF with local observations and turbine commands

Run the simulation from the project root through `run_simulation.py`:

```sh
python3 run_simulation.py --seed 42 --time-end 8
```

The wrapper sets OFF's paths and headless plotting, then executes
`OFF/03_Code/main_001.py`. That file builds a seeded wind profile, records
local observations, and connects optional turbine actions to OFF.

## Choose observation points

The default locations are defined as `DEFAULT_COORDINATES` in
`RL_agents.py`. Each is a world `(x, y, z)` coordinate in metres.
Override the defaults for one run by repeating `--point`:

```sh
python3 run_simulation.py --seed 42 --time-end 8 \
  --point 250 -495 119 \
  --point 750 -495 119 \
  --point 1250 -495 119
```

Point indexes start at 0 in the same order as the command line. The outputs
are `OFF/runs/off_run_*/local_wind_samples.csv` (all points) and
`local_wind_point_0.csv`, `local_wind_point_1.csv`, etc. (one profile per
point). Each row includes the tick time, world coordinate, wake-reduced
effective wind speed, ambient speed and direction, and distance to the
nearest observation-point chain.

`sample_local_wind(off_sim, coordinate)` in `RL_agents.py` defines the
sampled variables. Edit it to add more variables. Also add each new CSV
field name to `SAMPLE_FIELDS` in `OFF/03_Code/main_001.py`. The effective
speed is a scalar from OFF's wake solver. The direction is interpolated from
the nearest ambient observation-point chain; it is not a wake-deflected
direction. The distance field helps identify points far from any chain.

The random wind forcing is farm-wide. The local profiles differ because OFF
evaluates the wake and ambient state at each coordinate; they do not have
independent random wind generators.

## Use the action hook

The simplest manual check is a constant action from the command line:

```sh
python3 run_simulation.py --seed 42 --time-end 8 \
  --yaw-action 0 10 \
  --axial-action 0 0.30
```

This commands turbine index 0 to a 10-degree yaw misalignment and an axial
induction factor of 0.30 at every tick. Repeat either option for other
turbines. These are manual commands, not an RL agent. They override the
corresponding fields returned by `choose_actions` in `main_001.py`.

For a changing policy, edit **`choose_actions(time_s, off_sim, observations)`**
in `OFF/03_Code/main_001.py`. It receives the current tick's observations
as a tuple in point-index order. For example:

```python
def choose_actions(time_s, off_sim, observations):
    speed = observations[0]["effective_wind_speed_m_s"]
    if speed > 7.5:
        return {0: {"yaw_deg": 10.0}}
    return {0: {"yaw_deg": 0.0}}
```

The return value is a mapping from **zero-based turbine index** to a
nonempty action mapping:

```python
{
    0: {"yaw_deg": 10.0, "axial_induction": 0.30},
    2: {"yaw_deg": -5.0},
}
```

- `yaw_deg` is yaw **misalignment relative to that turbine's corrected
  local wind direction**, in degrees, from -180 to 180. It is not an absolute
  compass orientation.
- `axial_induction` is a value from 0 to 0.5.
- Each field is optional, but an action mapping must contain at least one.
  A listed turbine uses the action hook in place of the YAML controller for
  that tick. Its omitted field retains its previous value.
- Turbines absent from the mapping use the controller specified by the case
  YAML. Return `{}` or `None` to use that controller for every turbine.
- OFF checks indexes, field names, ranges, and finite numeric values before
  applying any action. An invalid command raises an error.

To connect a future agent in another Python file, import its callable in
`main_001.py` and call it from `choose_actions`. Keep the function's return
format above. For example:

```python
from my_controller import decide

def choose_actions(time_s, off_sim, observations):
    return decide(time_s, observations)
```

No `reset()` or `step()` agent implementation is included here.

### Where OFF receives and applies actions

`OFF/03_Code/off/off.py` is the central simulation loop:

1. OFF calculates rotor measurements and power for time `t`.
2. `on_tick(t, off_sim)` samples every configured local point.
3. `on_action(t, off_sim)` chooses commands from the same tick's state.
4. OFF corrects ambient conditions and propagates observation points.
5. In the `CONTROL` block, OFF applies each supplied action; otherwise it
   calls the case YAML controller for that turbine.
6. Those settings affect the **next measurement tick** (`t + time step`).

`OFFInterface.run_sim(on_tick=..., on_action=...)` forwards both callbacks
to this loop. `main_001.py` supplies `observe_tick` and `action_hook`,
with `action_hook` calling your `choose_actions` function. If you need
to change the action API itself, edit the `run_sim` method and
`_validate_actions` in `OFF/03_Code/off/off.py`; keep the forwarding
signature in `OFF/03_Code/off/off_interface.py` in sync.

## Seed and change the wind function

```sh
python3 run_simulation.py --seed 42 --speed-std 0.5 --direction-std 3
```

With no `--seed`, a seed is generated and printed. The seed, function name,
parameters, input case, and observation coordinates are saved in
`randomization.json`. The generated farm-wide forcing at each tick is
saved in `wind_profile.csv`; the generated case is saved as
`seeded_case.yaml`. Reusing the same case, seed, function, and options
reproduces the wind forcing.

To replace Gaussian perturbations, edit `choose_wind_function(args)` in
`main_001.py`. Return a callable with this signature:

```python
def my_wind(time_s, base_speed_m_s, base_direction_deg, rng):
    speed = max(0.0, base_speed_m_s + rng.normal(0.0, 0.2))
    direction = (base_direction_deg + rng.normal(0.0, 1.0)) % 360
    return speed, direction

def choose_wind_function(args):
    return my_wind
```

`rng` is a NumPy generator seeded once per run. The saved
`wind_profile.csv` records the actual generated values even for a custom
function.

## Output files

Each run creates `OFF/runs/off_run_*/` with:

| File | Contents |
| --- | --- |
| `randomization.json` | Seed, function, parameters, coordinates, source case |
| `wind_profile.csv` | Generated ambient speed and direction at every tick |
| `local_wind_samples.csv` | All local observations with point indexes |
| `local_wind_point_N.csv` | One local wind profile for point N |
| `measurements.csv` | OFF rotor and power measurements |
| `applied_control.csv` | Measured turbine settings, axial induction, and `control_source` |
| `seeded_case.yaml` | Exact generated OFF input case |

`applied_control.csv` records the settings used for each measurement tick.
`control_source` is `initial` at the first tick, then `action_hook` or
`case_controller` according to the command chosen on the preceding tick.
