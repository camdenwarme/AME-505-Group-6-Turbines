# Four turbines and seeded dynamic wind

Implementation record: **2026-11-14**. This document describes the implemented
square-farm and dynamic-wind changes, their interfaces, validation, and limits.
The adjacent `light_mappo` checkout and the bundled OFF source are unchanged by
this batch. New runs use checkpoint format 6 and should start from fresh models.

## 1. Resulting farm and learning architecture

The default layout uses the same IEA 10 MW turbine model and 198 m diameter:

| Agent / CSV index | Display label | Position in diameters | Position in meters |
|---|---|---|---|
| 0 | Turbine 1 | (0, 0, 0) | (0, 0, 0) |
| 1 | Turbine 2 | (0, 5, 0) | (0, 990, 0) |
| 2 | Turbine 3 | (5, 0, 0) | (990, 0, 0) |
| 3 | Turbine 4 | (5, 5, 0) | (990, 990, 0) |

`WindFarmEnv.layout_d` is the authoritative layout. The agent count, controller
arrays, spaces, measurements, energy arrays, infos, and termination flags derive
from its length. `num_agents` and the compatibility alias `num_agent` agree.
Native case bounds expand around the layout. A custom layout can use any positive
number of distinct finite `[x,y]` or `[x,y,z]` rows, in rotor diameters.

There are four independent actors, four individual value models, and one shared
farm critic. Actors retain three local inputs: speed in m/s, meteorological
wind-from direction in degrees, and local wind-relative yaw in [-180,180).
Each actor retains the corrected tanh-squashed Gaussian yaw-rate head, bounded
to [-6,6] deg/s. The defaults use Adam learning rates 0.00035 and
1 PPO epoch, using each episode's data for one training pass. These are starting
settings, not a demonstrated optimum for the expanded farm or randomized wind.

### Critic state and targets

For N turbines the centralized state has `9*N + 1` entries:

1. `3*N` local observation values, scaled by [10,360,180].
2. Remaining episode fraction at index `3*N`.
3. N instantaneous turbine powers divided by the 10 MW rating.
4. Five yaw-history samples per turbine at lags 5,10,20,30,40 ticks.

For the default square this is **37 inputs**, with time remaining at **index 12**.
The runner obtains the index from the environment and passes it explicitly to
`HorizonScaledCritic`; it no longer assumes index 6.

Farm power is the exact sum of the N measured turbine powers. Physical interval
energy uses the trapezoid rule. Individual reward is turbine interval energy
divided by `10e6 * dt`; farm reward is total interval energy divided by
`total_rated_power_w * dt`. The default farm rating is **40 MW**. With identical
ratings, farm reward equals the mean individual reward. The farm critic learns
expected remaining normalized farm return under the current joint policy; it
does not solve for an oracle-optimal energy trajectory.

Individual GAE still uses each turbine's aligned power-curve reference plus a
learned per-step residual multiplied by the discounted remaining horizon. The
farm critic also retains horizon scaling and per-horizon target regression.
ValueNorm/PopArt remain disabled for this integration. Each actor mixes separately
standardized individual and farm GAE with `--team-reward-weight`. The shared critic
uses one farm trajectory per optimizer epoch/minibatch, rather than N copies.

The analytic individual reference evaluates the turbine curve at current local
speed. It creates no additional simulator rollout. With changing winds it is an
approximation that assumes current local inflow persists; learned residuals must
account for departures from that approximation.

## 2. Wind randomizer controls

| Option | Default | Meaning |
|---|---|---|
| `--wind-mode` | `off` | `off`, `speed`, `direction`, or `both` |
| `--wind-speed` | 8.0 | Fixed speed when speed randomization is off |
| `--wind-direction` | 270.0 | Fixed direction or center of randomized direction knots |
| `--wind-seed` | model `--seed` | Seed for training wind profiles |
| `--eval-wind-seed` | model `--seed` | Seed for a distinct evaluation wind stream |
| `--wind-speed-min` / `--wind-speed-max` | 6.0 / 10.0 | Uniform speed-knot bounds in m/s |
| `--wind-direction-half-width` | 15.0 | Uniform direction-knot half-width in degrees |
| `--wind-knot-interval` | 120.0 | Nominal time in seconds between random knots |
| `--layout-file` | none | JSON layout array; omission selects square4 |
| `--numeric-eval-interval` | 10 | Completed training episodes between evaluations |
| `--wind-aligned-episode-zero` | true | Run an additional ideal local-wind benchmark at episode 0. Disable with `--no-wind-aligned-episode-zero`. |
| `--no-video` | false | Keep numeric evaluation and trajectory logs without MP4 rendering |

`off` produces a constant explicit scenario. `speed` varies speed over time while
holding direction fixed; `direction` varies direction while holding speed fixed;
`both` varies both. In a randomized dimension the first value is sampled too.
For example, `--wind-speed` is not the initial random speed in speed/both modes;
the speed bounds determine its sampled values.

### How the seeded functions are constructed

`wind_profiles.py` defines `WindProfileConfig`, `WindScenario`, and
`make_wind_scenario`. Episode seeds derive from a NumPy SeedSequence using:

```text
[master wind seed, stream ID, environment ID, zero-based wind episode ID]
```

Training uses stream 0 and a shared wind-generation environment ID of 0. The
wind episode ID is the logged training episode number minus one, independent
of the worker assigned to it. Evaluation uses stream 1. Each episode has separate speed
and direction substreams, so toggling one dimension preserves the realization of
the other when all other inputs match. Generation does not consume global NumPy,
Python, or PyTorch random numbers. Model initialization/exploration continue to
use the model seed separately.

Uniform random values are drawn at knots. Speed interpolates linearly between
knots. Direction switches to the next sampled value at each knot and holds until
the following knot. Values displayed/logged as ambient direction are wrapped to
[0,360). The final partial interval follows the ongoing speed ramp and keeps the
last sampled direction. A direction half-width of 180 degrees covers the entire circle.

All realized arrays and seed identifiers are serialized. Replaying those arrays
avoids dependence on future changes to random-generation code. `reset(scenario=...)`
accepts either a WindScenario or its dictionary. Explicit replay does not consume
the environment's training-wind episode counter. `make_scenario(episode_id=...)`
constructs a scenario without resetting or advancing the simulator.

## 3. Dynamic wind, actions, and simulation ticks

At measurement tick k the environment exposes ambient wind from `scenario.sample(k*dt)`.
Actors see the local wake-model wind at that tick, which can differ from ambient.
The selected yaw rate changes absolute rotor heading over the next interval.

OFF resumes after its measurement yield, applies the current ambient head,
propagates existing histories, and invokes the controller. The controller's update
then installs wind at `(k+1)*dt` into the new ambient-history head, before the next
measurement. Its native ambient feed is initialized as a constant buffer; runtime
buffer updates preserve the measured current wind during the following correction
and propagation. `corr_overwrite_direction=False` prevents replacement of the full
historical chain at each wind change.

The controller now tracks continuous absolute headings independently of the wind.
It calculates OFF yaw from unwrapped wind direction minus unwrapped heading.
Thus a zero rate holds absolute heading while changing wind changes relative yaw.
No automatic rotor jump occurs when direction crosses 0/360 degrees. Old wind and
yaw history continue to influence wakes. This is a global ambient boundary profile
with OFF history propagation, not a spatially resolved turbulence field.

## 4. Reproducible evaluation and video correspondence

Every checkpoint evaluates the same held-out explicit wind profile: evaluation
stream 1, wind episode ID 0. It uses the configured wind mode/ranges but is
independent of the training profile stream. This makes checkpoint comparisons
meaningful at one fixed dynamic test condition; it is not a multi-scenario
generalization test.

Episode 0 uses the untrained greedy actors. The additional `wind_aligned` rollout
sets each turbine's heading to the current local wind-from direction without a
yaw-rate limit. It saves `wind_aligned_steps.csv` and supplies the dashed benchmark
line on both farm-energy plots. Later evaluations use the trained greedy actors.
Numeric evaluation and MP4 replay the same greedy policy and wind scenario at each
checkpoint. Under fixed wind the optional reference retains
the historical initially aligned fixed-heading baseline. With dynamic wind it is
called `yaw_tracking_baseline` and tracks measured local wind with the same
yaw-rate limits; it does not anticipate future wind. Enable the extra reference
rollout with `--evaluate-aligned-baseline`.

`evaluate_episode(..., trajectory_path=...)` and `record_policy_video(..., scenario=...)`
write a common per-measurement schema. Numeric evaluations produce
`evaluation_episodeN_steps.csv`; videos produce `wind_farm_episodeN_steps.csv`.
Each contains the initial tick and every interval endpoint, with scenario IDs,
ambient wind, physical turbine positions, headings, local wind/yaw, rates, power,
and accumulated energy. A normal T-step video has T+1 frames corresponding to
those T+1 rows. The numeric and rendered trajectories can be compared directly.

## 5. Preserved information and artifacts

| File | Purpose |
|---|---|
| `config.json` | Parsed options, realized layout/count/rating/time index, training/evaluation wind settings |
| `runtime_versions.json` | Python and key package versions |
| `wind_scenarios.jsonl` | Training reset/completed events with full realized knots, seeds and IDs |
| `evaluation_wind_scenario.json` | Exact held-out profile reused at checkpoints and by video |
| `training_metrics.csv` | Per-episode energy, returns, actor/critic errors/clipping, scenario/seed and ending ambient conditions |
| `rollout_steps.csv` | Every training interval, all agents, rewards/values/advantages plus scenario/seed and interval-end ambient conditions |
| `evaluation_metrics.csv` | Policy, farm and individual energies/returns, scenario ID, seed, turbine count, and rating |
| `*_steps.csv` evaluation files | Physical per-tick states matching numeric evaluation or MP4 frames |
| `models/checkpoint_metadata.json` | Format 6, N, layout, rating, time index, critic input shape and value conventions |
| `models/actor_agentI.pt` | One actor per turbine |
| `models/individual_critic_agentI.pt` | One individual residual value model per turbine |
| `models/critic.pt` | The single shared farm critic |
| PNG/HTML diagnostic reports | Training/greedy energy, losses, clipping, yaw and critic error for every agent |

DummyVecEnv resets a completed environment before returning. Terminal infos carry
the completed scenario's IDs and full function so the runner does not accidentally
attribute its energy to the newly reset wind. The JSONL registry can contain a
final `reset` event for an unused next episode; a `completed` event and CSV rows
identify the episodes actually trained. Training episode numbers are one-based;
wind episode IDs and ticks are zero-based. Tick observations/actions are at the
interval start, while physical power and ambient values are at its endpoint.

Checkpoints reject incompatible counts/layouts/ratings/critic schemas, including
older two-turbine formats. Checkpoints still restore model weights rather than
optimizer/RNG/simulator state, and the usual model files are overwritten by later
saves. Seeded profiles enable exact wind replay; that is not a guarantee of exact
mid-run training resumption or bitwise cross-platform optimizer reproducibility.
Keep/copy run folders and selected checkpoints when preserving experiments.

## 6. Commands

Use Python 3.11+ with `pip install -r requirements-mappo.txt` (FLORIS 4.7).

```sh
# Default square, constant 8 m/s wind from 270 degrees
python train_mappo.py --wind-mode off --experiment_name square4_fixed

# Dynamic speed only
python train_mappo.py --wind-mode speed --wind-speed-min 6 --wind-speed-max 10 --wind-seed 123 --no-video

# Dynamic direction only, centered on 270 degrees
python train_mappo.py --wind-mode direction --wind-direction 270 --wind-direction-half-width 15 --wind-seed 123 --no-video

# Both vary: 30 training episodes, separate seeded held-out wind, moderate changes
python train_mappo.py --wind-mode both --wind-speed-min 6 --wind-speed-max 10 --wind-direction 270 --wind-direction-half-width 15 --wind-knot-interval 120 --seed 2 --wind-seed 123 --eval-wind-seed 456 --num_env_steps 6000 --episode_length 200 --numeric-eval-interval 5 --no-video --experiment_name square4_dynamic

# Short integration check, not evidence of learning
python train_mappo.py --num_env_steps 8 --episode_length 4 --ppo_epoch 1 --wind-mode both --wind-knot-interval 4 --no-video
```

For a custom farm, write a JSON array such as `[[0,0],[5,0]]` and pass its path
using `--layout-file`. N derives from that file; there is no second agent-count
setting to keep synchronized. The default four-turbine square needs no layout file.

Wind replay from a registry record in Python:

```python
from wind_profiles import WindScenario
scenario = WindScenario.from_dict(record["scenario"])
obs = env.reset(scenario=scenario)  # env duration must match the recorded duration
```

## 7. Change map and reasons

| Location | Actual change and reason |
|---|---|
| `wind_profiles.py` | Seeded wind profiles with linear speed ramps, stepped directions, and replay |
| `wind_farm_env.py: WindFarmEnv.__init__/reset/_measure/step` | N-dependent layout/spaces/arrays/measurement checks/rewards; removes two-agent shape assumptions |
| `wind_farm_env.py: make_scenario/_prepare_next_tick/_YawRateController` | Per-episode functions, correct ambient/history timing and absolute-heading control under changing wind |
| `train_mappo.py` | Wind/layout CLI, model and wind seed logging, N-derived runner configuration, explicit held-out replay and numeric state files |
| `shared_critic_runner.py: __init__` | Agent-space validation and environment-provided time index; builds N actors/value models around one farm critic |
| `shared_critic_runner.py: warmup/insert/_register_scenario` | Full scenario archive and terminal-before-reset attribution |
| `shared_critic_runner.py: metric/trace writers, save/restore` | Seed/ambient trace columns and format-6 structural checkpoint validation |
| `evaluate_wind_farm.py` | N-independent results, physical trajectory snapshots and rate-limited dynamic baseline |
| `wind_farm_video.py` | Four-agent display, current ambient/scenario label, explicit scenario replay and frame-matching CSV |
| `plot_training_loss.py` | Greedy evaluation energy for each turbine and shared critic loss |
| `plot_training_diagnostics.py` | Detects all actor and critic fields in the training logs |
| `plot_rewards.py`, `plot_training_energy.py` | Recognize dynamic yaw-tracking references in numeric evaluation plots |
| `plot_evaluation_energy.py` | Derive reward-to-energy normalization from recorded farm rating or agent count; old two-agent configurations imply the original 20 MW rating |
| `select_team_weight.py` | Preserve all per-turbine results and require matched wind scenario/count across candidate runs |
| `README_MAPPO.md` | Updated general operating guide linked to this batch record |

## 8. Verification and interpretation

The tests cover four-turbine geometry and 40 MW reward arithmetic; N-shaped
measurements/actions/dones; 37-feature critic state and time index 12; independent
actors/individual critics with one farm-critic update stream; checkpoint mismatch
rejection; wind-mode toggles; independent and repeatable seeds; global-RNG
isolation; JSON replay; smooth final partial intervals; north crossings; dynamic
wind at exact measurement ticks; zero-action heading preservation; and retained
older ambient history. Reporting tests cover all-agent fields and matched
numeric/video trajectories.

Real FLORIS 4.7 smoke runs exercised dynamic feedforward training, dynamic
recurrent training, optional baseline evaluation, videos and plotting. In the
short video smoke, both initial and post-update numeric/video CSVs matched at
all five measurement ticks and the MP4 had exactly five frames. Training wind
records were also checked against archived functions and the sum of all four
turbine powers. These are integration checks, not a new convergence study.

Validation on this batch completed with **44 tests passing**:

```sh
python -m unittest discover -v
```

Local evidence (under the ignored `rl_runs/mappo/` output directory):

- `square4_dynamic_smoke/20261114_110929_5a4166e7`: two four-step feedforward
  updates, both dynamic dimensions, north-sector directions, separate train/eval
  seeds, optional yaw-tracking reference; all eight training ticks matched
  archived wind functions and four-turbine power sums.
- `square4_video_smoke/20261114_111006_41199bb6`: one four-step update, initial
  and final videos; numeric/video CSVs matched for every measurement at both
  evaluations (absolute comparison tolerance 1e-8), and each MP4 had five frames.
- `square4_recurrent_smoke/20261114_111058_f4f388a3`: one twenty-step recurrent
  update with changing wind direction, including evaluation and all reports.

The checks used the temporary Python 3.11 environment at
`/private/tmp/windfarm-pinned-venv/bin/python`; each run records exact package
versions in `runtime_versions.json`. That temporary path is session-specific;
create your own environment from the requirements for continued work.

Known limits: wind is a global boundary function, observations still hide parts
of the wake history, raw circular actor inputs retain their 0/360 discontinuity,
and current-inflow analytic baselines are approximate under changing wind. The
earlier two-turbine reward-weight selection does not establish the best weight
for four turbines or randomized winds. Use matched scenario suites and multiple
model seeds when evaluating learning or selecting new settings.
