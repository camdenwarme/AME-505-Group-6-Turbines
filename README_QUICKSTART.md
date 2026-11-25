# Quickstart: local setup, integration check, and training

This is the condensed version of setup, checks, and training commands. For the
full design, see [README_MAPPO.md](README_MAPPO.md) (how the MAPPO setup works)
and [README_FOUR_TURBINES_DYNAMIC_WIND.md](README_FOUR_TURBINES_DYNAMIC_WIND.md)
(four-turbine layout and dynamic wind), and [README_GRAPH_LAYER.md](README_GRAPH_LAYER.md) (optional neighbour graph layer and layout files). Commands below are for Windows PowerShell,
run from the repository folder. On macOS/Linux, replace `.venv/Scripts/python.exe`
with `.venv/bin/python`.

## 1. Folder layout

`train_mappo.py` looks for `light_mappo` **next to** this repository. Do not
nest it inside the repository:

```text
AME505Project/
├── AME-505-Group-6-Turbines/   <- this repository (run commands here)
└── light_mappo/                <- https://github.com/tinyzqh/light_mappo, unmodified
```

```powershell
git clone https://github.com/camdenwarme/AME-505-Group-6-Turbines.git
git clone https://github.com/tinyzqh/light_mappo.git
```

If `light_mappo` is somewhere else, add `--mappo-path C:/path/to/light_mappo` to every
`train_mappo.py` command.

## 2. One-time setup (Python 3.11)

Use Python 3.11. That is the version these instructions were tested with
(FLORIS 4.7, torch 2.15 CPU, gym 0.26.2, NumPy 2.4).

```powershell
cd AME-505-Group-6-Turbines
py -3.11 -m venv .venv                                    # or: uv venv --seed --python 3.11 .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements-mappo.txt
```

- If `python`/`py` says *"Python was not found ... Microsoft Store"*, Python is not
  installed. Install 3.11 from python.org, or run `uv python install 3.11` and
  use the `uv venv` line above.
- The first install downloads roughly 1 GB (mostly torch).
- Training runs on the CPU. A CUDA build of torch is optional and only speeds up
  the small networks; most of the time is spent in the FLORIS simulation.

## 3. Check local integration (about 2 minutes)

Run these in order. Each one should match the expected result.

| # | Command | Expected result |
|---|---------|-----------------|
| 1 | `.venv/Scripts/python.exe -c "import floris, torch, gym; print(torch.__version__)"` | A version string. A Gym "please upgrade to Gymnasium" notice is harmless. |
| 2 | `.venv/Scripts/python.exe -m unittest discover -v` | `Ran 58 tests ... OK` (about 60 s) |
| 3 | `.venv/Scripts/python.exe train_mappo.py --num_env_steps 8 --episode_length 4 --ppo_epoch 1 --no-video --experiment_name smoke` | `Evaluation episode 0 greedy: ... MWh`, then `Evaluation episode 2 ...`, then `Checkpoints and logs: ...` (about 20 s) |
| 4 | `.venv/Scripts/python.exe train_mappo.py --num_env_steps 8 --episode_length 4 --ppo_epoch 1 --wind-mode both --wind-knot-interval 4 --evaluate-aligned-baseline --experiment_name smoke_video` | Same as step 3, plus a `yaw_tracking_baseline` line and `wind_farm_episode0.mp4` / `wind_farm_episode2.mp4` (about 30 s) |

Step 3 tests the simulator, the agents, and the reports. Step 4 adds dynamic wind,
the baseline controller, and MP4 rendering. Smoke runs are too short to learn
anything useful. Their output goes to `rl_runs/mappo/<experiment_name>/<timestamp>/`
(git-ignored), which you can delete.

| Symptom | Fix |
|---------|-----|
| `Set --mappo-path to your light_mappo checkout.` | `light_mappo` is not next to this repository. Clone it there or pass `--mappo-path`. |
| `ModuleNotFoundError: floris` / `torch` | You are using the system Python instead of `.venv/Scripts/python.exe`. |
| `Incompatible checkpoint; format 6 requires ...` | The `--model_dir` checkpoint has a different layout or format. Start a fresh run. |

## 4. Start a training session

**Number of training episodes = `--num_env_steps` ÷ `--episode_length`** (rounded
down). Each episode lasts `--episode_length` × `--dt` simulated seconds (default
200 × 4 s = 800 s). Choose the number of episodes, then set
`--num_env_steps = episodes × episode_length`:

```powershell
# Template: E episodes of 200 steps -> --num_env_steps (E*200)
.venv/Scripts/python.exe train_mappo.py --num_env_steps 20000 --episode_length 200 --experiment_name my_run
```

| Episodes | `--num_env_steps` (with `--episode_length 200`) | Greedy evaluations (default interval 10) |
|---------:|------------------:|---|
| 10  | 2000  | episodes 0, 10 |
| 50  | 10000 | 0, 10, ..., 50 |
| 100 | 20000 *(default)* | 0, 10, ..., 100 |
| 300 | 60000 | 0, 10, ..., 300 |

**Measured runtime** (Ryzen 9 3900X, CPU only, `--wind-mode both`, `--no-video`):
about **65 s per 200-step training episode** and **60 s per greedy evaluation**.
Estimate: `episodes × 65 s + (episodes/10 + 1) × 60 s`. That is about **2 hours** for
the default 100 episodes and about 1 hour for 50. With video on, each evaluation
simulates the episode a second time and renders about 200 frames, so it costs at
least another minute (not measured). Use `--no-video` for long runs.
`training_metrics.csv` gets one row per finished episode, so you can check
progress while training runs. Ctrl+C stops the run. The checkpoint in `models/`
is overwritten after every episode, so the latest one survives a stop.
`models_best/` keeps a separate copy of the policy with the highest greedy-evaluation
farm energy so far (`best_evaluation.json` says which episode). Training often
peaks before its last episode, so use `--model_dir <run>/models_best` to continue
from, or evaluate, the best policy instead of the last one.

> **Recommended for every real run: add `--use_value_active_masks`.** This is an
> on/off switch whose default is ON, so passing it turns value active-masking OFF.
> That works around the farm-critic loss weighting bug described in §9.

## 5. Toggles

### Farm and wind options (defined in `train_mappo.py`)

| Flag | Default | Effect |
|------|---------|--------|
| `--wind-mode {off,speed,direction,both}` | `off` | `off`: constant wind. `speed`, `direction`, `both`: those quantities change over time within each episode, following a seeded random profile. |
| `--wind-speed` | `8.0` | Constant speed (m/s), used when speed is not randomized. |
| `--wind-direction` | `270.0` | Constant direction, or the center of randomized directions (meteorological "from" degrees). |
| `--wind-speed-min` / `--wind-speed-max` | `6.0` / `10.0` | Range for random speed knots (`speed`/`both`). |
| `--wind-direction-half-width` | `15.0` | Random direction knots fall within ± this many degrees of `--wind-direction`. |
| `--wind-knot-interval` | `120.0` | Seconds between random knots, with linear interpolation between them. Smaller values give faster changes. |
| `--wind-seed` | `--seed` | Seed for training wind. The same seed reproduces the same profiles. |
| `--eval-wind-seed` | `--seed` | Seed for the single held-out evaluation profile, which is replayed at every evaluation. |
| `--layout-file <json>` | 4-turbine square | JSON array of `[x,y]` (or `[x,y,z]`) positions in rotor diameters (D = 198 m). The number of agents comes from this file. |
| `--dt` | `4.0` | Control interval in seconds. |
| `--team-reward-weight` | `0.5` | Each actor's advantage = (1−w)·own-turbine + w·farm. `0` = individual only, `1` = farm only. Each turbine's own energy is still part of the farm reward. At `1` the individual critics are still trained and logged, but they no longer affect the actors. |
| `--episodes-per-update` | `1` | Complete episodes collected before each PPO update. Each runs in its own simulator, stepped one after another, so the run takes as long as before but makes 1/E as many updates. `1` is the original behaviour. Requires `--num_env_steps` to be a multiple of `episode_length × E` and `--numeric-eval-interval` to be a multiple of E. See `CHANGELOG.md`. |
| `--numeric-eval-interval` | `10` | Run a greedy evaluation every N training episodes (also at episode 0 and the final episode). |
| `--evaluate-aligned-baseline` | off | At episode 0, also simulate a reference controller: aligned turbines for fixed wind, yaw tracking for dynamic wind. |
| `--no-video` | off | Skip MP4s and their state plots. Numeric evaluations and CSVs are still written. |
| `--final-video` | off | No videos during training. After the last episode, render the final policy (`wind_farm_final_episodeN.mp4`) and the best greedy policy (`wind_farm_best_episodeN.mp4`, from `models_best/`), plus `evaluation_states.html`. Use it with `--no-video` for long runs. |
| `--video-fps` | `10/dt` | Video playback rate. The default plays at 10× simulation speed. |
| `--mappo-path` | `../light_mappo` | Location of the light_mappo checkout. |

### Useful light_mappo options (passed through)

| Flag | Launcher default | Effect |
|------|-----------------|--------|
| `--num_env_steps` | `20000` | Total training steps. Sets the number of episodes (§4). |
| `--episode_length` | `200` | Steps per episode. |
| `--experiment_name` | `square4` | Output subfolder name. |
| `--seed` | `1` | Seed for network initialization and exploration. Also the default wind seed. |
| `--lr` / `--critic_lr` | `3.5e-4` | Actor and critic learning rates. |
| `--ppo_epoch` | `10` | PPO passes over each update's batch (one episode by default, E with `--episodes-per-update E`). |
| `--num_mini_batch` | `1` | Minibatches per PPO pass. |
| `--entropy_coef` | `0.01` | Exploration bonus. |
| `--clip_param` | `0.2` | PPO clip range. |
| `--gamma` / `--gae_lambda` | `1.0` / `0.95` | Discount factor and GAE λ. |
| `--hidden_size` / `--layer_N` | `64` / `1` | Network width and depth. |
| `--algorithm_name rmappo --use_recurrent_policy` | `mappo` | Recurrent (GRU) actors. Requires `episode_length` to be divisible by `--data_chunk_length` (10) and at least 2 chunks long. |
| `--use_linear_lr_decay` | off | Decrease the learning rate linearly to 0 over the run. |
| `--model_dir <run>/models` | none | Continue from a saved checkpoint (weights only; optimizer state is not restored). |
| `--n_training_threads` | `2` | Number of torch CPU threads. |

**On/off switches whose default is ON:** passing these flags turns the feature
**off**: `--cuda`, `--use_gae`, `--use_huber_loss`, `--use_clipped_value_loss`,
`--use_value_active_masks`, `--use_max_grad_norm`, `--use_orthogonal`,
`--use_ReLU`.

**The launcher rejects:** `--n_rollout_threads` other than 1 (use
`--episodes-per-update` instead, which sets it), `--use_eval`,
`--use_popart`, `--use_valuenorm`, and `--algorithm_name` without its matching
recurrent flag (and vice versa).

## 6. Copy-paste recipes

```powershell
# 100 episodes, constant 8 m/s from 270°, with reference controller, no video
.venv/Scripts/python.exe train_mappo.py --use_value_active_masks --wind-mode off --evaluate-aligned-baseline --no-video --experiment_name fixed_100

# 50 episodes, speed varies 6-10 m/s, direction fixed
.venv/Scripts/python.exe train_mappo.py --use_value_active_masks --num_env_steps 10000 --wind-mode speed --wind-speed-min 6 --wind-speed-max 10 --wind-seed 123 --eval-wind-seed 456 --no-video --experiment_name speed_50

# 50 episodes, direction varies ±15° around 270°, speed fixed at 8 m/s
.venv/Scripts/python.exe train_mappo.py --use_value_active_masks --num_env_steps 10000 --wind-mode direction --wind-direction 270 --wind-direction-half-width 15 --wind-seed 123 --eval-wind-seed 456 --no-video --experiment_name direction_50

# 30 episodes, both vary, evaluation every 5 episodes
.venv/Scripts/python.exe train_mappo.py --use_value_active_masks --num_env_steps 6000 --wind-mode both --wind-seed 123 --eval-wind-seed 456 --numeric-eval-interval 5 --no-video --experiment_name both_30

# Recurrent actors
.venv/Scripts/python.exe train_mappo.py --use_value_active_masks --algorithm_name rmappo --use_recurrent_policy --experiment_name recurrent

# Custom layout: write a JSON file such as [[0,0],[5,0],[10,0]], then
.venv/Scripts/python.exe train_mappo.py --use_value_active_masks --layout-file row3.json --experiment_name row3

# 60 episodes, two complete episodes per PPO update (30 updates). With constant
# wind this finished below the default 1 episode per update; meant for dynamic wind.
.venv/Scripts/python.exe train_mappo.py --use_value_active_masks --num_env_steps 12000 --episodes-per-update 2 --no-video --final-video --experiment_name batch2_60

# Continue a previous run (same layout and options)
.venv/Scripts/python.exe train_mappo.py --use_value_active_masks --model_dir rl_runs/mappo/fixed_100/<timestamp>/models --experiment_name fixed_100_more

# Team-weight comparison: identical runs except the weight, then rank them
.venv/Scripts/python.exe train_mappo.py --use_value_active_masks --team-reward-weight 0   --no-video --experiment_name w0
.venv/Scripts/python.exe train_mappo.py --use_value_active_masks --team-reward-weight 0.5 --no-video --experiment_name w05
.venv/Scripts/python.exe train_mappo.py --use_value_active_masks --team-reward-weight 1   --no-video --experiment_name w1
.venv/Scripts/python.exe select_team_weight.py <w0-run> <w05-run> <w1-run> --output weight_selection.json
# Chart: farm energy vs baseline and upstream-turbine yaw, one line per weight
.venv/Scripts/python.exe plot_team_weight_comparison.py <w0-run> <w05-run> <w1-run> --output team_weight_comparison.png
```

Add `--evaluate-aligned-baseline` to the three training commands if you want the baseline line on the comparison chart.

## 7. Where results go

Each run creates `rl_runs/mappo/<experiment_name>/<timestamp>/`. Open these first:

| File | What it shows |
|------|---------------|
| `training_plots.html` | All training charts in one offline page. Rebuilt after every greedy evaluation and again at the end of the run. |
| `evaluation_turbines.png` | Per-turbine greedy evaluations: farm energy vs baseline, each turbine's energy change, each turbine's mean yaw (§8). |
| `evaluation_states.html` | Per-turbine wind, yaw, and power histories for each evaluation (requires video to be on). |
| `training_metrics.csv` | One row per PPO update, which is one per episode by default. With `--episodes-per-update E`, rows fall on episodes E, 2E, …, and the energy and return columns are means over those E episodes. Updated live. |
| `evaluation_metrics.csv` | Greedy and baseline energy at each evaluation. |
| `wind_farm_episodeN.mp4` | Top-down video of the greedy evaluation after N episodes. With `--final-video`: `wind_farm_final_episodeN.mp4` and `wind_farm_best_episodeN.mp4` instead. |
| `models/` | Latest checkpoint (`actor_agent*.pt`, `individual_critic_agent*.pt`, `critic.pt`). |
| `models_best/` | Checkpoint with the highest greedy farm energy so far, plus `best_evaluation.json`. |
| `config.json`, `wind_scenarios.jsonl`, `evaluation_wind_scenario.json` | Every option and wind profile used, for reproducibility. |

## 8. Plotting a run

Training rebuilds every chart below, and `training_plots.html`, after each greedy
evaluation (every `--numeric-eval-interval` episodes, default 10). Open
`training_plots.html` in the run folder and refresh it to follow a live run.

The commands below are for older runs, or for a snapshot between evaluations.
The scripts read the CSVs written so far, so they also work while a run is
still training. Run them from the repository folder.

```powershell
# 1. Pick the run: the newest one (folder names start with the timestamp)...
$run = (Get-ChildItem rl_runs/mappo/*/* -Directory | Sort-Object Name | Select-Object -Last 1).FullName
# ...or a specific one:
# $run = "rl_runs/mappo/<experiment>/<timestamp>"

# 2. Build every plot, then the HTML page that collects them (about 10 s)
foreach ($s in "plot_training_energy","plot_training_loss","plot_rewards","plot_training_diagnostics","plot_evaluation_turbines","training_report") {
  .venv/Scripts/python.exe "$s.py" $run
}

# 3. Open the report
start "$run/training_plots.html"
```

The plots are written to the run folder:

| File | Shows | Healthy run |
|------|-------|-------------|
| `cumulative_energy_by_episode.png` | Farm energy (MWh) per training episode (blue) and per greedy evaluation (orange). | Noisy at first, then rises and levels off. Greedy should be at or above the training line. |
| `reward_by_episode.png` | Episode return (sum of rewards) for training and evaluation. | Same shape as the energy plot, because reward is normalized power. |
| `training_loss_by_episode.png` | PPO policy loss for each actor, and the shared critic's value loss. | Policy loss stays small and noisy near 0 (this is normal for PPO). Critic loss drops quickly toward 0. |
| `training_diagnostics.png` + `.json` | Four panels, listed below. The JSON holds first, last, mean, min and max of each quantity. | See below. |
| `evaluation_turbines.png` | **Usually the most informative chart.** Top: farm energy per greedy evaluation, with the baseline (dashed) and the best evaluation circled. Middle: each turbine's energy change vs the baseline (or vs the episode-0 evaluation if no baseline was recorded). Bottom: each turbine's mean wind-relative yaw. | Farm line above the baseline. Wake steering shows up as an upstream turbine yawed and slightly negative while the turbine behind it is clearly positive (in the default layout with wind from 270°, T1 shadows T3 and T2 shadows T4). |
| `training_plots.html` | All of the above on one offline page. | — |

Each turbine keeps the same color and marker in every chart (T1 blue circle,
T2 orange square, T3 green triangle, T4 yellow diamond). The colors are defined
once in `plot_style.py`.

Panels in `training_diagnostics.png`:

- **Critic prediction error.** RMSE of the farm critic and each turbine's critic.
  It should fall and flatten. If one turbine stays clearly higher than the
  others, its critic struggles to predict that turbine's return (usually a
  waked turbine).
- **PPO ratios outside clip range.** Clip fraction per actor. About 0–0.2 is
  typical. If it stays above about 0.3, the updates are too large: lower
  `--lr` or `--ppo_epoch`.
- **Exploratory yaw magnitude ≥ 90°.** Fraction of training steps in which a
  turbine faced 90° or more away from the wind. It should drop to 0 early. If
  it stays above 0, the policy is spinning turbines off the wind.
- **Greedy farm energy.** Energy from each greedy evaluation. With
  `--evaluate-aligned-baseline`, a dashed line shows the reference
  controller. The goal is to finish above that line.

Video-only extras. These need a run **without** `--no-video`:

```powershell
.venv/Scripts/python.exe plot_evaluation_energy.py $run          # energy within each evaluation episode
.venv/Scripts/python.exe training_report.py $run --evaluation-states   # evaluation_states.html
```

Comparing runs: `evaluation_metrics.csv` (greedy energy per evaluation) is the
simplest file to put side by side. For runs that differ only in
`--team-reward-weight`, use `select_team_weight.py` (§6).

## 9. Known issues from the 2026-11-14 integration check

1. **Farm-critic loss uses the wrong mask (real bug, not fixed in code).** In
   `shared_critic_runner.py` (`SharedCriticRunner.train`, the shared-critic loop),
   the minibatch tuple is indexed `sample[8]` for RNN masks and `sample[9]` for
   `active_masks`. In light_mappo's buffer, index 7 is `masks`, 8 is
   `active_masks`, and 9 is the **old action log-probabilities**. Because
   `use_value_active_masks` defaults to ON, the farm critic's loss becomes
   Σ(loss·logπ)/Σ(logπ). This is harmless early in training, when all log-probs
   are negative. Once a policy's standard deviation shrinks below about
   0.3 deg/s, log-probs take both signs. Some samples then get reversed critic
   gradients, and the denominator can approach zero. **Workaround without code
   changes:** pass `--use_value_active_masks` (the loss becomes a plain mean, which
   is correct because every turbine is always active). **Code fix:** use
   `sample[7]` in `per_step_values(...)` and `(5, 6, 8)` in the tuple below it.
   The `sample[8]`→`[7]` part only matters for recurrent critics, and is harmless
   with the current one-episode-per-rollout setup.
2. `evaluate_wind_farm.py` reads `env.farm_rated_power_w`, but the environment
   defines `total_rated_power_w`. The fallback (N × 10 MW) gives the same value for
   the current all-10 MW farms, so results are unaffected.
3. Gym 0.26.2 prints a "please upgrade to Gymnasium" notice on import. This is
   expected; only `gym.spaces.Box` is used.
4. The repository tracks some `__pycache__/*.cpython-313.pyc` files. Other Python
   versions add untracked `.pyc` files next to them. A `.venv` created with
   `py -3.11 -m venv` also appears as untracked. Neither should be committed.
