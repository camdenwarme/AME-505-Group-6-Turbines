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

Each turbine has its own actor; one shared centralized critic sees both observations.
The shared critic has one optimizer and one value normalizer. Each rollout trains
both actors using the same value estimates, then trains the critic on one copy of
the farm trajectory (once per PPO epoch/minibatch, rather than once per turbine).
This behavior is implemented in `shared_critic_runner.py`; light_mappo is unchanged.
Each actor receives exactly `[local wind speed in m/s, local wind direction in
degrees, its own wind-relative yaw in degrees]`. Yaw is `wind direction - rotor
heading`, wrapped to `[-180, 180)`, so aligned turbines observe zero yaw. Positive
heading-rate commands decrease this yaw. The actor sees the actual yaw, including
backward-facing angles, rather than the substitute angle used by FLORIS. No power,
position, turbine identity, or other turbine's information is included. OFF's wake model supplies
an effective local speed and reconstructs its direction using the ambient state;
this is not a resolved CFD measurement of local wake crossflow.

Each action is a yaw/heading rate in degrees per second, saturated to [-6, 6].
Positive means increasing absolute heading. There is no yaw travel limit;
turbines can make full turns. Power is zero at wind-relative yaw magnitudes of
90 degrees or greater, using the shortest signed angle (so 360 degrees is aligned
again). OFF retains continuous, unwrapped yaw history for wake interpolation.
To avoid invalid FLORIS calculations, backward-facing wakes are approximated
using a sideways rotor (just inside +/-90 degrees) while their measured power is
explicitly zero. This is a numerical modeling convention, not a physical model
of reverse-flow operation. The controller records the applied rate in `info`.

Every 4-second control interval (`--dt`), both agents get the same trapezoidal
farm-energy reward, divided by the fixed scale `20e6 * dt`. There is no terminal
energy bonus. `info` includes physical interval/episode energy in joules and
total power in watts. Episodes are finite-horizon tasks, default 200 steps/800 s,
starting with aligned turbines. `gamma=1` targets undiscounted total energy.

The small OFF change exposes its original simulation loop as `iter_sim()`, pausing
after measurement so the adapter can supply the next action. `run_sim()` still
consumes the complete loop for existing scripts. Wake history persists between
actions, and action/reward timing follows FDynSim's existing convention.

```powershell
# Short integration check (not enough training to learn useful control)
.venv/Scripts/python.exe train_mappo.py --num_env_steps 8 --episode_length 4 --ppo_epoch 1

# Optional memory for the partially observed wake dynamics
.venv/Scripts/python.exe train_mappo.py --algorithm_name rmappo --use_recurrent_policy
```

Yaw now lets an actor distinguish aligned and misaligned rotor orientations.
The full wake history remains hidden, so observations are still partial. Two
separate actors let the turbines learn distinct roles without adding identity to
their observations. Input LayerNorm remains disabled to preserve the independent
speed and angle measurements. The centralized critic receives six inputs: the
three observations from each turbine. Start a new training run when moving from
the older separate-critic architecture or older observation sizes.

Artifacts go to `rl_runs/mappo/<experiment>/<timestamp>/`, including configuration,
one continuously updated `training_metrics.csv`,
automatically generated `wind_farm_episodeN.mp4` evaluation videos,
TensorBoard logs, two actor checkpoints (`actor_agent0.pt`, `actor_agent1.pt`), one
`critic.pt`, and shared value-normalization statistics (`value_normalizer.pt` when
ValueNorm is enabled). `--model_dir <models-folder>` loads this checkpoint format
for further training; optimizer state is not restored. The minimal launcher uses
one training environment plus a separate evaluation environment; process isolation
is required for future concurrent OFF workers.
Evaluate learned policies against aligned operation with identical initial states
and episode durations before claiming an energy improvement.

`training_metrics.csv` is created at run initialization and gets one row after
every training episode's network update, before any scheduled video evaluation.
Each row is immediately written and the file closed, so it can be inspected while
training continues. Columns contain the episode number, total environment steps,
elapsed wall time, each actor's PPO policy loss, the shared critic's value loss,
farm energy for that episode (joules and MWh), and cumulative energy summed over
completed training episodes (joules and MWh). Losses are averages over the PPO
epochs/minibatches; actor policy losses exclude the entropy bonus. There is no
single combined actor/critic loss because these are optimized separately.
Energy comes from the simulator's per-episode integral, counted once for the farm,
not twice for the two agents. Evaluation episodes are excluded from this training
CSV and from cumulative training energy. Elapsed wall time includes evaluations.

At the end of a successful run, `cumulative_energy_by_episode.png` is saved next
to the CSV. Its top panel plots the total energy generated within each episode,
with a tightly fitted y-axis to reveal changes between episodes. Its bottom panel
plots the running cumulative energy across all training episodes. Both use MWh
and episode number. This distinguishes fluctuations in episode performance from
the steadily increasing total across training. Axis limits include 12% padding
around the observed range, with a fallback for constant or single-episode data.
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

Before training (`episode0`), after every 10 completed training episodes, and after
the final episode, a fresh episode runs using the actors' deterministic mean
actions with no exploration or learning. Videos use the number of completed
training episodes: `wind_farm_episode0.mp4`, `wind_farm_episode10.mp4`, etc. The
default 100-episode run produces 11 videos ending with `wind_farm_episode100.mp4`.
For a final episode that is not a multiple of 10, that episode also gets a video;
for example, a 12-episode run produces videos for episodes 0, 10, and 12. The
final evaluation is recorded once even when it coincides with the 10-episode
interval. Evaluation uses a separate simulator, preserves training random-number
states, and never enters training buffers or counts against `num_env_steps`.
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
`--video-fps` to explicitly override this rate. These are separate deterministic
evaluation episodes, not recordings of the exploratory training episodes. Each
uses the same wind and time settings as training.
