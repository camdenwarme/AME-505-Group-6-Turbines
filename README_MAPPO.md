# Two-turbine MAPPO

Uses the sibling `../light_mappo` checkout directly; no submodule or copied MAPPO
code is needed. Pass `--mappo-path C:/path/to/light_mappo` for another location.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-mappo.txt
.venv/Scripts/python.exe train_mappo.py
```

The environment derives a two-turbine case from the existing FLORIS case without
changing the original example. The turbines are at (0, 0) and (5D, 0), with D=198 m.
Ambient wind is constant: 8 m/s from 270 degrees by default. Change it with
`--wind-speed` and `--wind-direction` (meteorological degrees, **from** direction).

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
interval energy divided by `20e6 * dt` (the average of the two individual
rewards); it trains the shared critic and contributes to both actor updates.
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
number of steps remaining. The curve is the same for both turbines, but their
reference values can differ because their local wind speeds differ in a wake.
`--wind-speed` changes the configured ambient wind and therefore the local speeds
seen in a run. The curve is read directly; computing this reference does not run
an additional simulation. It assumes the current local speed persists, and the
current wake can depend on earlier actions, so it is an approximation, not the
simulated outcome of an aligned farm or the true optimal value.

At the end of each episode, the runner computes each actor's individual GAE
advantage using its own normalized energy reward and its reference-plus-residual
value, alongside the farm's GAE advantage using the farm critic. It standardizes those
components separately and mixes them as `(1 - w) * individual + w * team`,
where `w = --team-reward-weight` defaults to `0.5` and is constrained to [0, 1].
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
10 PPO epochs per rollout. This medium update size is a starting point, not a
proven optimum. Override it with `--lr`, `--critic_lr`, and `--ppo_epoch`.

```powershell
# Short integration check (not enough training to learn useful control)
.venv/Scripts/python.exe train_mappo.py --num_env_steps 8 --episode_length 4 --ppo_epoch 1 --no-video

# Optional memory for the partially observed wake dynamics
.venv/Scripts/python.exe train_mappo.py --algorithm_name rmappo --use_recurrent_policy
```

Yaw now lets an actor distinguish aligned and misaligned rotor orientations.
The full wake history remains hidden, so observations are still partial. Two
separate actors let the turbines learn distinct roles without adding identity to
their observations. Input LayerNorm remains disabled to preserve the independent
speed and angle measurements; hidden layers still use LayerNorm. The centralized
critic receives 19 inputs: both actor observations with speed/10, direction/360,
and yaw/180, the fraction
of the episode remaining, both current turbine powers divided by rated power, and
five historical yaw samples for each turbine at lags 5, 10, 20, 30 and 40
steps. History angles are wrapped and scaled to [-1, 1]; when a lag exceeds the
available OFF state chain, the oldest available state is used. These samples are
useful wake context, not the full simulator state. The critic estimates expected
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
rejected. Current checkpoints use format 5 in `checkpoint_metadata.json`.

Implementation hooks:

- `WindFarmEnv._measure`, `get_centralized_obs`, `get_individual_baseline_values`,
  and `step` in `wind_farm_env.py` retrieve observations, build the critic input,
  evaluate the analytic aligned reference, and compute energy rewards.
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
TensorBoard logs, two actor checkpoints (`actor_agent0.pt`, `actor_agent1.pt`),
two individual critic checkpoints (`individual_critic_agent0.pt`,
`individual_critic_agent1.pt`), and one farm `critic.pt`.
`--model_dir <models-folder>` loads the current checkpoint format
for further training; optimizer state is not restored. The minimal launcher uses
one training environment plus a separate evaluation environment; process isolation
is required for future concurrent OFF workers.
Evaluate learned policies against aligned operation with identical initial states
and episode durations before claiming an energy improvement.

`training_metrics.csv` is created at run initialization and gets one row after
every training episode's network update, before any scheduled evaluation.
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
not twice for the two agents. Evaluation episodes are excluded from this training
CSV and from cumulative training energy. Elapsed wall time includes evaluations.

At the end of a successful run, `cumulative_energy_by_episode.png` is saved next
to the CSV. It plots training episode energy from `episode_energy_mwh`, numeric
greedy evaluation energy from `evaluation_metrics.csv`, and the simulated aligned
baseline if present. Energy resets at the start of every episode; the plot does
not sum energy across training episodes. Its tightly fitted y-axis includes all
visible series and the baseline, with 12% padding around the observed range and
a fallback for constant or single-episode data.
Regenerate the plot for an existing CSV using:
`python plot_training_energy.py <run-folder>`.

The run also saves `training_loss_by_episode.png`, with separate panels for
turbine 1's actor policy loss, turbine 2's actor policy loss, and the shared
critic's value loss. Each panel has an independently fitted y-axis with 12%
padding so losses on different scales remain readable. Points are episode-level
averages over PPO epochs/minibatches, as recorded in the CSV. Actor losses may be
negative and exclude the entropy bonus. Regenerate this plot using:
`python plot_training_loss.py <run-folder>`.

After generating both plots, the run saves `training_plots.html`. Open it in a
browser to see the energy plot above the training-loss plot, each with its original
aspect ratio preserved at any window width. The PNGs are embedded unchanged, so
this single HTML file works offline and can be copied without its run folder.
Regenerate it for a folder that already contains both plots using:
`python training_report.py <run-folder>`.

Before training (`episode0`), every `--numeric-eval-interval` completed training
episodes (default 10), and after the final episode, an evaluation episode runs
using the actors' greedy policy (the `tanh`-transformed Gaussian mean), with no
exploration or learning. `evaluation_metrics.csv` records farm and per-turbine
energy and returns. With `--evaluate-aligned-baseline`, a separate zero-yaw-rate
simulation is also evaluated at episode 0 under the same initial conditions. It
is an optional physical benchmark, not the analytic reference used by GAE or an
optimized wind-farm controller. Pass `--no-video` to retain numeric evaluations
without generating video or video-based state/reward plots.

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
`turbine1_states_episodeN.png` and `turbine2_states_episodeN.png`, using the same
episode number as the corresponding video. Each figure has three panels showing
that actor's observed local wind speed (m/s), wind-from direction (degrees), and
wind-relative yaw (degrees in [-180, 180)), against simulation time in seconds.
The recorded observations come from the same evaluation as the MP4, including
the initial state and final state, with an automatically fitted y-axis for each
panel. These are the actual actor inputs, including backward-facing yaw angles,
not the substitute angles used internally by the FLORIS wake model. Creating
the figures requires no extra simulation or training episode.

`evaluation_states.html` collects these state-history figures into one standalone
page, grouped by evaluation episode in numerical order, with turbine 1 above
turbine 2. An episode navigation bar jumps to each section. Original aspect ratios
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
the evaluation HTML. This inverts the fixed `20e6 * dt` reward normalization and
therefore retains the small rounding error of the recorded rewards; new
evaluations record simulator energy directly.

Training CSV rows record `episode_return` (the sum of team rewards within that
episode), `episode_end_reward` (its final team reward), and each actor's
individual return. When training finishes, `reward_by_episode.png` plots team
episode reward against episode number, with an automatically fitted y-axis.
Training and greedy numeric evaluations are separate labeled series whether or
not videos are enabled; an aligned baseline is included when recorded. Both
energy and reward plots prefer numeric evaluation CSV values and use legacy
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
full 0-20 MW range. Y-axis labels remain absolute MW values without an offset.
Default playback is 10x simulation speed, one frame per control step:
each frame lasts `dt / 10` seconds. With a 4 s step this is 0.4 s per frame
(2.5 FPS). A 200-step episode produces about 80 seconds of video. Use
`--video-fps` to explicitly override this rate. These are separate evaluation
episodes using a greedy policy, not recordings of the exploratory training episodes. Each
uses the same wind and time settings as training.

`rollout_steps.csv` records every exploratory training transition, one row per
zero-based tick and environment, after each completed rollout. `episode` is
one-based. `time_start_s` and `time_end_s` bound the control interval. Each
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

`training_metrics.csv` additionally contains `actor0_ppo_clip_fraction` and
`actor1_ppo_clip_fraction`: the fraction of sampled importance ratios outside
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
and greedy farm energy with the aligned baseline if recorded. Missing or
nonfinite samples are omitted. Critic and yaw data describe exploratory training
rollouts; greedy energy comes from separate evaluation episodes.

To screen farm-advantage weights, run matched pilots with the same environment,
seed, rollout length and optimizer settings and only change `--team-reward-weight`.
For example, run weights `0`, `0.5`, and `1` under distinct experiment names,
then pass their completed run folders to:

```powershell
.venv/Scripts/python.exe select_team_weight.py <run-weight-0> <run-weight-0.5> <run-weight-1> --output weight_selection.json
```

The selection utility requires an initial and final greedy evaluation and
complete training metrics for each candidate. It checks that configurations
match aside from weight, seed and experiment label; for multiple seeds, each
candidate must use the same seed set. It selects the highest mean final greedy
farm energy, records per-turbine outcomes and sample sizes, and breaks exact
ties toward the lowest weight. An aligned-baseline evaluation is not required
for selection. Short pilots screen the tested weights only: they do not establish
a global optimum, and choosing and assessing a weight on the same pilots is
optimistic. Confirm any selected weight using new seeds and wind conditions.
