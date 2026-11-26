# Wind-farm MAPPO: four-turbine square and seeded wind

## Exponential advantage weighting

New training runs default to `--advantage-weight-schedule exponential` with
`--advantage-decay-beta 0.05`. For zero-based round index r (the number of fully
completed collection/update rounds), the individual weight is `(1-beta)**r`
and the farm weight is `1-(1-beta)**r`. Both separately standardized advantages
are mixed using these weights. Both individual and farm critics continue training
even when their advantage has zero actor weight.

With two workers and beta 0.05, episodes 1-2 use 1.0/0.0, episodes 3-4 use 0.95/0.05,
and episodes 5-6 use 0.9025/0.0975 (individual/farm). The weights remain fixed across
all workers, actors, minibatches and PPO epochs in a round. Decay advances once
after that round's complete update. Beta must be finite and in [0,1]: zero keeps
the individual weight at one; one switches fully to farm after the first round.

```powershell
python train_mappo.py --episode-workers 2 --num_env_steps 10000 --ppo_epoch 1 --advantage-decay-beta 0.05 --no-video
```

To retain fixed 50/50 weighting, use
`--advantage-weight-schedule constant --team-reward-weight 0.5`.
`--team-reward-weight` applies only in constant mode. Saved `config.json` records
the schedule and beta; `training_metrics.csv` and `rollout_steps.csv` record the
individual and farm weights actually used in each round. The change applies to
new training processes; an already-running process retains its loaded behavior.
Loading actor/critic checkpoints starts a fresh training schedule at round zero,
just as the existing launcher starts fresh optimizers and episode counters.

## Parallel episodes with synchronized training

```powershell
python train_mappo.py --episode-workers 5 --episode_length 200 --num_env_steps 20000 --ppo_epoch 1 --no-video
```

This runs 100 total training episodes as 20 rounds of five episodes. Each episode
has its own spawned simulator process. Actors remain in the parent process and
generate actions for all workers using the same frozen policy. All simulators
receive their actions before the parent waits for their results. After every
worker completes its episode, the parent trains on all five trajectories. Workers
remain idle at their terminal states until training finishes, then reset together
for the next round. Simulator processes persist across rounds.

`--episode-workers` is an alias for `--n_rollout_threads`; it defaults to one.
`--ppo_epoch` independently controls the training passes over each round's data.
For example, `--ppo_epoch 10` uses the same collection barriers, followed by ten
training passes. Minibatching can produce multiple optimizer steps within a pass.
Use `--episodes` to set the total number of training episodes directly.
`--episode-workers 4 --episodes 22` collects five rounds of four episodes and
one final round of two. Each episode still runs for `--episode_length` steps.
Use either `--episodes` or `--num_env_steps`. The step-budget flag counts all
worker transitions and requires a multiple of `episode_length * episode-workers`.
Evaluation runs before training, after a round crosses an evaluation interval,
and after the final round. Evaluation rollouts are counted separately.

`--threads-per-worker` defaults to one to avoid nested numerical-library thread
pools. Windows schedules the separate processes across available CPU cores;
workers are not pinned to fixed physical cores. Increasing the worker count
increases memory use. Each worker has an independent reproducible wind stream
identified by its environment index. Policies are updated only in the parent.

`training_metrics.csv` contains one row per individual episode, with
`update_round` and `environment_index`. Losses describe the shared round update;
energy and rewards describe that specific episode. `rollout_steps.csv` and wind
scenario records preserve distinct episode identifiers across workers.

See [README_FOUR_TURBINES_DYNAMIC_WIND.md](README_FOUR_TURBINES_DYNAMIC_WIND.md)
for the structured change record, implementation locations, design decisions,
verification, and information-preservation details for this update.

Uses the sibling `../light_mappo` checkout directly; no submodule or copied MAPPO
code is needed. Pass `--mappo-path C:/path/to/light_mappo` for another location.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-mappo.txt
.venv/Scripts/python.exe train_mappo.py
```

The environment derives a configurable farm from the existing FLORIS case without
changing the original example. The default is four IEA-10MW turbines in a square,
ordered `(0,0)`, `(0,5D)`, `(5D,0)`, `(5D,5D)`, with D=198 m and 990 m between
adjacent turbines. Agent indices 0/1 are the first column, and 2/3 the second.
`--layout-file <json-file>` accepts a JSON array of `[x,y]` or `[x,y,z]` positions
in rotor diameters. Agent count, farm bounds, array sizes, centralized features,
and rated power derive from the layout; all turbines currently use the case's
same 10MW type. With N turbines the normalized farm rating is N*10MW.

The layout-specific launchers accept the same training flags as `train_mappo.py`.
The four-turbine launcher uses the default square. The six-turbine launcher uses
the OFF/FLORIS layout with two rows of three turbines at 5D spacing. Both presets
are built into the launchers and can be overridden with `--layout-file`.

```powershell
python train_mappo_4_turbines.py --episodes 22 --no-video
python train_mappo_6_turbines.py --episodes 22 --no-video
```

Wind is fixed at 8 m/s from 270 degrees by default. `--wind-speed` and
`--wind-direction` set these base conditions (meteorological **from** degrees).
The `--wind-mode` switch enables seeded time-varying wind inside every episode:

| Option | Behavior |
| --- | --- |
| `--wind-mode off` | Fixed base speed and direction; default. |
| `--wind-mode speed` | Random speed knots; fixed base direction. |
| `--wind-mode direction` | Fixed base speed; random direction knots. |
| `--wind-mode both` | Random speed and direction knots. |
| `--wind-seed` | Training-wind seed; defaults to `--seed`. |
| `--eval-wind-seed` | Evaluation-wind seed; defaults to `--seed`, using a separate stream. |
| `--wind-speed-min`, `--wind-speed-max` | Random speed knot bounds; defaults 6 and 10 m/s. |
| `--wind-direction-half-width` | Direction knot range around the base direction; default +/-15 degrees. |
| `--wind-knot-interval` | Seconds between wind knots. Randomized direction holds for this long. Default 120. |

```powershell
# Four turbines with fixed wind and numeric evaluations
.venv/Scripts/python.exe train_mappo.py --wind-mode off --no-video

# Four turbines with reproducible dynamic speed and direction
.venv/Scripts/python.exe train_mappo.py --wind-mode both --wind-seed 123 --eval-wind-seed 456 --wind-speed-min 6 --wind-speed-max 10 --wind-direction 270 --wind-direction-half-width 15 --wind-knot-interval 120 --no-video
```

`wind_profiles.py` samples knots using explicit master seed, stream, environment,
and episode IDs. Training uses the global
episode number minus one as the wind episode ID, so a given logged episode
gets the same profile with any worker count when the wind settings match.
It uses independent speed and
direction RNG substreams; toggling one quantity preserves the other's sampled
realization. Speed ramps linearly between knots. Direction holds constant until
the next knot and switches immediately there. A partial final interval keeps the
last direction and follows the ongoing speed ramp until the episode ends. Saved
scenarios replay using their recorded direction interpolation.
No global NumPy randomness is consumed by profile generation.

Dynamic wind is applied to the new ambient-history head at the next measurement
tick, while existing wake history is retained. Heading commands remain absolute
rates, so zero command holds the rotor heading when the wind changes. Each
episode starts aligned with that profile's initial ambient direction; wind and
yaw histories are initialized consistently at that starting condition.

Each turbine has its own actor and individual value critic; one additional shared
centralized critic estimates future farm reward. Each rollout trains each actor
using a mixture of its individual GAE advantage and the farm GAE advantage,
then trains the individual critics and the shared farm critic on their respective
targets. The farm critic has one optimizer. This is
implemented in `shared_critic_runner.py`; light_mappo is unchanged.
Each actor receives exactly `[local wind speed in m/s, local wind direction in
degrees, its own wind-relative yaw in degrees]`. Yaw is `wind direction - rotor
heading`, wrapped to `[-180, 180)`, so turbines aligned with local inflow observe
zero yaw. For fixed local wind direction, positive heading-rate commands decrease
this yaw. The actor sees the actual yaw, including
backward-facing angles, rather than the substitute angle used by FLORIS. No power,
position, turbine identity, or other turbine's information is included. The FLORIS
adapter now returns rotor-area-averaged horizontal velocity components including
modeled transverse flow. Components are averaged linearly on the rotor grid (with
cubature weights when configured), kept in original turbine order, and rotated
from the wind-aligned FLORIS frame into world coordinates. The observation speed
is the magnitude of this mean horizontal vector; direction is its meteorological
wind-from angle. Yaw is relative to that local direction, not ambient direction.
This is modeled engineering-wake inflow, not blade-resolved CFD; vertical flow
does not enter yaw. FLORIS's power calculation retains its own cubic-mean speed
and aerodynamic yaw conventions. Wake-history propagation still uses ambient
velocity. Retrain policies learned with the previous ambient-direction inputs.

Each action is a yaw/heading rate in degrees per second. The Gaussian policy
head passes its sample through a scaled `tanh`, bounding its output to [-6, 6].
PPO log probabilities include the transformation's Jacobian, and the initial
Gaussian noise is in degrees per second. The environment also clips its input as
an actuator safety bound. `bounded_action_head.py` implements the transformation.
Positive means increasing absolute heading. There is no yaw travel limit;
turbines can make full turns. Power is zero at wind-relative yaw magnitudes of
90 degrees or greater, using the shortest signed angle (so 360 degrees is aligned
again). OFF retains continuous, unwrapped yaw history for wake interpolation.
To avoid invalid FLORIS calculations, backward-facing wakes are approximated
using a sideways rotor (just inside +/-90 degrees) while their measured power is
explicitly zero. This is a numerical modeling convention, not a physical model
of reverse-flow operation. The controller records the applied rate in `info`.

Every 4-second control interval (`--dt`), actor `i` gets its own turbine's
trapezoidal interval energy divided by `10e6 * dt`. The team reward is farm
interval energy divided by `total_rated_power_w * dt` (`40e6 * dt` for the
default square). It is the mean of the individual rewards for these identical
turbines, trains the shared critic, and contributes to every actor update.
There is no terminal energy bonus. `info` contains both individual and farm
power and energy, as well as `team_reward`. Episodes are finite-horizon tasks,
default 200 steps/800 s, starting with aligned turbines. `gamma=1` makes both
reward-to-go objectives undiscounted.

For each turbine and timestep, the individual value estimate starts from an
analytic aligned reference and adds the discounted remaining-step count times a
learned per-step residual from that turbine's individual critic. The analytic
reference reads the installed FLORIS turbine
power curve and evaluates it at that turbine's current local wind speed. It
divides the interpolated aligned power by 10 MW and multiplies by the discounted
number of steps remaining. The curve is the same for all turbines, but their
reference values can differ because their local wind speeds differ in a wake.
`--wind-speed` and the dynamic wind profile change ambient wind and the local speeds
seen in a run. The curve is read directly; computing this reference does not run
an additional simulation. It assumes the current local speed persists, and the
current wake can depend on earlier actions, so it is an approximation, not the
simulated outcome of an aligned farm or the true optimal value.

At the end of each episode, the runner computes each actor's individual GAE
advantage using its own normalized energy reward and its reference-plus-residual
value, alongside the farm's GAE advantage using the farm critic. It standardizes those
components separately and mixes them as `(1 - w) * individual + w * team`,
where `w` follows the exponential schedule above by default. In constant mode,
`w = --team-reward-weight` defaults to `0.5` and is constrained to [0, 1].
The combined advantage is normalized again by the MAPPO trainer. A weight of
zero removes the team advantage from actor updates; one removes the individual
signal. The critic is trained on the farm target, not an individual target.
Because the two components are separately standardized each rollout, this is a
heuristic blend rather than an exact fixed weighted sum of the raw objectives.
The individual critics receive the time-aware centralized critic state, are
feedforward even when recurrent actors are enabled, and learn per-step residuals
from the analytic references. Scaling by the discounted remaining horizon makes
the individual value go to zero at the episode end. All critics are updated after the actor passes, so their
new estimates affect subsequent rollouts rather than revising actor updates that
have already happened. Initial residual outputs are zero; the analytic reference
is therefore the initial individual value estimate. The estimates remain
imperfect, especially when the wake or wind changes.
This mixture is a design tradeoff: maximizing an individual turbine's output can
conflict with sacrificing upstream power to increase downstream production, so
it does not guarantee the policy that maximizes total farm energy.

The small OFF change exposes its original simulation loop as `iter_sim()`, pausing
after measurement so the adapter can supply the next action. `run_sim()` still
consumes the complete loop for existing scripts. Wake history persists between
actions, and action/reward timing follows FDynSim's existing convention.

The launcher defaults to actor and critic learning rates of `0.00035` and
1 PPO epoch per rollout, so each episode's data is used for one training pass.
Override these settings with `--lr`, `--critic_lr`, and `--ppo_epoch`.

```powershell
# Short integration check (not enough training to learn useful control)
.venv/Scripts/python.exe train_mappo.py --num_env_steps 8 --episode_length 4 --ppo_epoch 1 --no-video

# Optional memory for the partially observed wake dynamics
.venv/Scripts/python.exe train_mappo.py --algorithm_name rmappo --use_recurrent_policy
```

Yaw now lets an actor distinguish aligned and misaligned rotor orientations.
The full wake history remains hidden, so observations are still partial. Independent
actors let the turbines learn distinct roles without adding identity to
their observations. Input LayerNorm remains disabled to preserve the independent
speed and angle measurements; hidden layers still use LayerNorm. The centralized
critic receives `9*N+1` inputs: every actor observation with speed/10,
direction/360, and yaw/180, the fraction
of the episode remaining, every current turbine power divided by rated power, and
five historical yaw samples for each turbine at lags 5, 10, 20, 30 and 40
steps. History angles are wrapped and scaled to [-1, 1]; when a lag exceeds the
available OFF state chain, the oldest available state is used. These samples are
useful wake context, not the full simulator state. The default N=4 gives 37
inputs, with the remaining-time feature at index 12. `WindFarmEnv` exposes
`remaining_time_index=3*N`, and the runner passes that index to the horizon
wrapper rather than assuming the former two-turbine index 6. The critic estimates expected
remaining team reward under the current policy and its training targets; it is
not an oracle for the optimal achievable future energy. Its network predicts
average team reward per discounted remaining step; `HorizonScaledCritic`
multiplies that output by the discounted remaining-step count to produce the
return-valued prediction used by GAE. The farm critic is fitted against GAE
targets divided by that same count, which keeps the regression in per-step units
and makes the value zero at the terminal horizon. ValueNorm and PopArt are
disabled and rejected for this integration.

Start a new training run after this change. Older checkpoints have incompatible
critic inputs, action distributions, or individual-value architecture and are
rejected. Current checkpoints use format 6 in `checkpoint_metadata.json`, checking
agent count, layout, rating, observation shape, and remaining-time index.

Implementation hooks:

- `WindFarmEnv._measure`, `get_centralized_obs`, `get_individual_baseline_values`,
  and `step` in `wind_farm_env.py` retrieve observations, build the critic input,
  evaluate the analytic aligned reference, and compute energy rewards.
- `WindProfileConfig`, `WindScenario`, and `make_wind_scenario` in
  `wind_profiles.py` define configuration, seeded generation, serialization,
  interpolation, and explicit replay without a new random draw.
- `TanhBoundedActionHead` in `bounded_action_head.py` transforms Gaussian actions
  and calculates their corrected PPO log probabilities.
- `HorizonScaledCritic` in `horizon_critic.py` maps the farm network's per-step
  prediction to a remaining-return value for GAE.
- `_individual_gae`, `SharedCriticRunner.warmup`, `insert`, `train`, and
  `_train_individual_critics` in `shared_critic_runner.py` collect states and
  rewards, compute separate individual and farm advantages, and fit their critics.
- `evaluate_episode` in `evaluate_wind_farm.py` runs the aligned baseline or a
  greedy policy without adding its trajectory to the training data.

Artifacts go to `rl_runs/mappo/<experiment>/<timestamp>/`, including configuration
and `runtime_versions.json` with the Python and key package versions,
one continuously updated `training_metrics.csv`,
optional `wind_farm_episodeN.mp4` evaluation videos, `evaluation_metrics.csv`,
TensorBoard logs, N actor checkpoints (`actor_agent{i}.pt`), N individual critic
checkpoints (`individual_critic_agent{i}.pt`), and one farm `critic.pt`.
The default square creates four actors and four individual value critics.
`--model_dir <models-folder>` loads the current checkpoint format
for further training; optimizer state is not restored. The launcher supports
isolated parallel training environments plus a separate evaluation environment.
Evaluate learned policies against aligned operation with identical initial states
and episode durations before claiming an energy improvement.

`wind_scenarios.jsonl` records training-profile `reset` and `completed` events.
Each event stores the full knot times, speeds, unwrapped directions, interpolation
description, master and episode seeds, stream/environment/episode IDs, and stable
scenario ID. `DummyVecEnv` automatically resets an ended episode before returning,
so a final reset event can describe the next, unused episode; completed events
identify profiles that actually supplied a full training rollout. Training metrics
and `rollout_steps.csv` retain the completed transition's scenario metadata from
`info`, rather than accidentally reporting the next reset's wind.
`config.json` preserves both wind configurations, layout, count, time index, and
farm rating; `runtime_versions.json` preserves key software versions. Exact resumed
training is not promised by the saved network weights alone: optimizer and all
RNG/counter states are not restored by the current checkpoint workflow.

`training_metrics.csv` is created at run initialization and gets one row per
episode after its round's network update, before any scheduled evaluation.
Each row is immediately written and the file closed, so it can be inspected while
training continues. Columns contain the episode number, total environment steps,
elapsed wall time, each actor's PPO policy loss, the shared critic's value loss,
farm energy for that episode (joules and MWh), cumulative farm energy summed over
completed training episodes, the team return, and each actor's individual return.
Each actor also has an `individual_value_loss` and `individual_value_rmse`;
the latter compares its pre-update value estimate with realized individual
discounted return-to-go from the same exploratory rollout.
The pre-update `critic_return_rmse` and `critic_explained_variance` compare the
rollout-time return-valued predictions against realized discounted team
returns. They are in-sample diagnostics, not held-out measures of generalization;
explained variance is undefined if the sampled returns have no variance.
`critic_value_loss` measures the farm critic's fit in per-discounted-step units;
it is not the raw squared error in total-return units. Losses are averages over the PPO
epochs/minibatches; actor policy losses exclude the entropy bonus. There is no
single combined actor/critic loss because these are optimized separately.
Energy comes from the simulator's per-episode integral, counted once for the farm,
rather than once per agent. Evaluation episodes are excluded from this training
CSV and from cumulative training energy. Elapsed wall time includes evaluations.

At the end of a successful run, `cumulative_energy_by_episode.png` is saved next
to the CSV. It plots training episode energy from `episode_energy_mwh`, numeric
evaluation energy from `evaluation_metrics.csv`, and the simulated aligned
baseline if present. Energy resets at the start of every episode; the plot does
not sum energy across training episodes. Its tightly fitted y-axis includes all
visible series and the baseline, with 12% padding around the observed range and
a fallback for constant or single-episode data.
Regenerate the plot for an existing CSV using:
`python plot_training_energy.py <run-folder>`.

The run also saves `training_loss_by_episode.png`, with a greedy evaluation
energy panel for each turbine and a shared critic value-loss panel. The turbine
panels show MWh above or below the wind-aligned benchmark when it uses the same
wind scenario. In that case, the dashed zero line marks the benchmark. With no
matching benchmark, the panel shows absolute MWh. Actor losses remain recorded
in `training_metrics.csv`. Regenerate this plot using:
`python plot_training_loss.py <run-folder>`.

After generating both plots, the run saves `training_plots.html`. Open it in a
browser to see the farm energy plot above the turbine and critic panels, each with its original
aspect ratio preserved at any window width. The PNGs are embedded unchanged, so
this single HTML file works offline and can be copied without its run folder.
Regenerate it for a folder that already contains both plots using:
`python training_report.py <run-folder>`.

Evaluation episode 0 uses the untrained actors' greedy policy. The same
deterministic action rule (the `tanh`-transformed Gaussian mean) evaluates the
actors after every `--numeric-eval-interval` completed episodes (default 10) and
at the final episode.

By default, episode 0 also runs an ideal wind-aligned benchmark on the same wind
profile. At each tick it points every turbine into its measured local wind for
0° relative yaw without the usual 6°/s limit. OFF samples inflow with the rotor
in its ambient-aligned position and uses zero yaw loss for the power calculation.
The controller sets the local-wind heading and records it for wake propagation.
Pass `--no-wind-aligned-episode-zero` to omit the benchmark or
`--wind-aligned-episode-zero` to enable it explicitly.

`evaluation_metrics.csv` records the policy (`wind_aligned` or `greedy`), farm
and per-turbine energies and returns, and wind-scenario identifiers.
The benchmark's per-tick trajectory is saved to `wind_aligned_steps.csv`.
The evaluation profile is generated once using the evaluation seed and stream 1,
saved in `evaluation_wind_scenario.json`, and replayed unchanged at every
checkpoint. Numeric evaluations, the optional baseline, and videos explicitly
reset to this same scenario; they do not sample unrelated winds on each reset.
Pass `--no-video` to retain numeric evaluations and per-tick CSVs without
generating video or video-based state/reward plots.

With `--evaluate-aligned-baseline`, an extra physical benchmark is evaluated at
episode 0 on the same profile. Fixed-wind mode uses initially aligned headings
and zero yaw-rate commands, labeled `aligned_baseline`. Dynamic modes use a
local-wind yaw-tracking controller with the same actuator bounds, labeled
`yaw_tracking_baseline`. This is an operational controller, not a perfect
foreknowledge benchmark or the analytic reference used by GAE. A fixed-wind
baseline measured in an earlier run is not a valid matched benchmark for an
arbitrary dynamic profile.

`evaluation_episodeN_steps.csv` records all evaluation ticks, including the
initial state: actual turbine positions/headings, actor observations, commanded
and applied rates, turbine/farm power, energy, and ambient wind. The video saves
`wind_farm_episodeN_steps.csv` from its own rendered rollout, so frame k can be
checked directly against tick k. Both contain scenario ID and seed metadata.
For the wind-aligned policy, rates describe the heading change over each tick.

Videos use the number of completed training episodes:
`wind_farm_episode0.mp4`, `wind_farm_episode10.mp4`, etc. The default
100-episode run produces 11 videos ending with `wind_farm_episode100.mp4`.
For a final episode that is not a multiple of the evaluation interval, that
episode is evaluated as well; for example, a 12-episode run at the default
interval evaluates episodes 0, 10, and 12. The final evaluation is recorded once
even when it coincides with the interval. Evaluation uses a separate simulator,
preserves training random-number states, and never enters training buffers or
counts against `num_env_steps`.

Each rendered evaluation also saves one time-history figure per turbine:
`turbine1_states_episodeN.png` through `turbine{N_agents}_states_episodeN.png`, using the same
episode number as the corresponding video. Each figure has three panels showing
that actor's observed local wind speed (m/s), wind-from direction (degrees), and
wind-relative yaw (degrees in [-180, 180)), against simulation time in seconds.
The recorded observations come from the same evaluation as the MP4, including
the initial state and final state, with an automatically fitted y-axis for each
panel. These are the actual actor inputs, including backward-facing yaw angles,
not the substitute angles used internally by the FLORIS wake model. Creating
the figures requires no extra simulation or training episode.

`evaluation_states.html` collects these state-history figures into one standalone
page, grouped by evaluation episode in numerical order and turbine index order.
An episode navigation bar jumps to each section. Original aspect ratios
are preserved, and all PNGs are embedded for offline viewing. The report is
updated after every rendered evaluation, starting at episode 0. Regenerate it from existing
plots with `python training_report.py <run-folder> --evaluation-states`.

Each rendered evaluation also saves `reward_history_episodeN.png`, showing the
team farm reward per control step against simulation time. Rewards are aligned to
interval end times (first sample at `dt`, not an invented reward at time zero).
This is the farm signal used by the critic and actor mixture, not an individual
turbine reward. The corresponding raw times and
rewards are saved in `evaluation_rewards_episodeN.npz` for reproducibility. The
reward figure is embedded below the turbine states in that episode's HTML section.

The section is named **Farm history** and also contains
`farm_energy_history_episodeN.png`: cumulative farm energy generated during that
evaluation versus simulation time, in MWh, starting from zero at time zero.
This uses the simulator's energy integral recorded in the same rollout as the
video; it is not a sum across evaluation episodes. Raw energy histories are saved
as `evaluation_energy_episodeN.npz`. Existing reward-only evaluations can be
backfilled with `python plot_evaluation_energy.py <run-folder>`, then rebuilding
the evaluation HTML. This inverts the farm rating recorded in the run configuration
(`40e6 * dt` for the default square, historical `20e6 * dt` for two turbines) and
therefore retains the small rounding error of the recorded rewards. If a legacy
run records neither farm rating nor agent count, the reconstruction fails rather
than assuming two turbines. New evaluations record simulator energy directly.

Training CSV rows record `episode_return` (the sum of team rewards within that
episode), `episode_end_reward` (its final team reward), and each actor's
individual return. When training finishes, `reward_by_episode.png` plots team
episode reward for training and greedy evaluation, with an automatically fitted y-axis.
The **Farm energy within each episode** plot and **Greedy farm energy** subplot
show the wind-aligned benchmark's final energy as a labeled dashed grey line.
The optional aligned baseline also appears when recorded. The energy and reward
summaries prefer numeric evaluation CSV values and use legacy
video NPZ histories only for episodes missing from the CSV, avoiding duplicate
points. This compares all recorded episodes within the current run, not a running
sum across episodes or across different run folders. The summary is embedded in both
`evaluation_states.html` and `training_plots.html`. Regenerate it from recorded
reward data with `python plot_rewards.py <run-folder>` and rebuild the HTML with
`training_report.py`. Historical files without recorded rewards are not fabricated.

Each video shows a top view of
the actual layout, rotor bars and front-facing orientation arrows, ambient/local
wind arrows and speeds, headings, relative yaw angles, applied yaw rates, farm
power history, and total energy. No synthetic wake field is drawn. Backward-facing
rotors are shown at their real angles. The initial and final episode states are
included.
The power subplot automatically zooms to the power values observed so far, with
12% padding and a small minimum range for constant output, instead of using the
full farm rated-power range. Y-axis labels remain absolute MW values without an offset.
Default playback is 10x simulation speed, one frame per control step:
each frame lasts `dt / 10` seconds. With a 4 s step this is 0.4 s per frame
(2.5 FPS). A 200-step episode produces about 80 seconds of video. Use
`--video-fps` to explicitly override this rate. Videos replay the policy used for
the numeric evaluation at each checkpoint with the same duration and held-out
wind profile.

`rollout_steps.csv` records every exploratory training transition, one row per
zero-based tick and environment, after each completed rollout. `episode` is
one-based. `time_start_s` and `time_end_s` bound the control interval. Each
row includes `scenario_id`, `episode_seed`, `wind_mode`, and measured endpoint
`ambient_wind_speed_m_s` and `ambient_wind_direction_deg`. These are distinct from
the agents' local inflow observations. Each
actor's `wind_speed_m_s`, `wind_direction_deg`, `yaw_deg`, and
`sampled_rate_deg_s` are the observation and bounded Gaussian sample at the
start of that interval; `applied_rate_deg_s`, `power_w`, and `farm_power_w`
come from the interval's ending simulator information. Per-actor `reward` and
`team_reward` are normalized interval energy. Per-actor `return_to_go` and
`team_return_to_go` are the realized discounted rewards from that tick to the
end of the episode. `team_value` is the horizon-scaled critic prediction in
return units before the optimizer changes it; it is not a future measurement.
`team_gae_advantage` is the unstandardized team GAE signal.
Per-actor `mixed_advantage` combines the separately standardized individual and
team signals; `ppo_advantage` is that mixed signal after the final normalization
used by the actor optimizer. The returns and advantages are estimates from the
sampled training episode, not greedy-evaluation results.

Per actor, `aligned_reference_value` is the analytic power-curve reference,
`individual_value` adds the horizon-scaled learned residual, and `individual_gae_advantage`
is the pre-standardization individual GAE computed with that value and the
turbine's own rewards. These are rollout-time quantities, before critic updates.

`training_metrics.csv` additionally contains `actor{i}_ppo_clip_fraction` for
every actor: the fraction of sampled importance ratios outside
the configured PPO clipping interval, averaged across each actor's optimizer
minibatches and epochs. The fraction does not itself say whether clipping was
active in the minimum surrogate for every one of those samples.

After training, generate a compact diagnostic figure and machine-readable
summary with:

```powershell
.venv/Scripts/python.exe plot_training_diagnostics.py <run-folder>
```

Training also writes these diagnostics automatically after a successful run.
The command writes `training_diagnostics.png` and `training_diagnostics.json` beside
the run logs. Panels show team and individual critic return RMSE, PPO clip
fractions, each turbine's fraction of exploratory yaw samples at |yaw| ≥ 90°,
and evaluation farm energy with the aligned benchmark if recorded. Missing or
nonfinite samples are omitted. Critic and yaw data describe exploratory training
rollouts. Evaluation energy comes from deterministic policy rollouts.

To screen farm-advantage weights, run matched pilots with the same environment,
seed, rollout length and optimizer settings, use `--advantage-weight-schedule constant`,
and only change `--team-reward-weight`.
For example, run weights `0`, `0.5`, and `1` under distinct experiment names,
then pass their completed run folders to:

```powershell
.venv/Scripts/python.exe select_team_weight.py <run-weight-0> <run-weight-0.5> <run-weight-1> --output weight_selection.json
```

The selection utility requires an initial evaluation and a final greedy evaluation, plus
complete training metrics for each candidate. It checks that configurations
match aside from weight, seed and experiment label; for multiple seeds, each
candidate must use the same seed set. It selects the highest mean final greedy
farm energy, records per-turbine outcomes and sample sizes, and breaks exact
ties toward the lowest weight. An aligned-baseline evaluation is not required
for selection. Initial/final evaluations and candidate weights must use the same
wind scenario for each matched seed. Short pilots screen the tested weights only: they do not establish
a global optimum, and choosing and assessing a weight on the same pilots is
optimistic. Confirm any selected weight using new seeds and wind conditions.
