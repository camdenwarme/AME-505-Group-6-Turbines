# Changelog

## 2026-11-24: Rate-limited wind-aligned benchmark (`wind_aligned`)

Ported from Jack Brower's `mappo_4_turbines_jack` commit `e8bf437` (only the
baseline; nothing else from that commit). Every turbine yaws toward its
measured local wind direction, like Jack's ideal controller, but rate-limited
to the actuator bounds of [-6, 6] deg/s (`evaluate_wind_farm.wind_aligned_rates`).
It runs once at episode 0 on the evaluation scenario and is recorded in
`evaluation_metrics.csv` as policy `wind_aligned`, with its trajectory in
`wind_aligned_steps.csv`. `--no-wind-aligned-episode-zero` skips it. Under
dynamic wind it matches the existing `yaw_tracking_baseline`. With constant
wind (`--wind-mode off`) it still tracks local inflow, where the old aligned
baseline holds its initial headings. Training is unaffected: training,
rollout and greedy-evaluation CSVs were unchanged in a seeded check. The
charts do not draw this benchmark yet.

## 2026-11-24: Optional neighbour graph layer for the actors (`--graph-layer`)

`--graph-layer on` puts a graph-attention encoder in front of each turbine's
actor MLP. The encoder works on a fixed neighbour graph read from a layout file,
and its edge features (downstream distance, crosswind offset, wake proxy) are
recomputed in the wind frame every step. Actors stay separate per turbine, and
the farm critic and per-turbine value models are unchanged. The default, `off`,
reproduces the previous code value for value. The new layout file format lists
turbine coordinates and neighbour links, and any turbine count works.
`turbine_graph.py make/check` generates and validates these files. See
[README_GRAPH_LAYER.md](README_GRAPH_LAYER.md) for usage, design reasoning (from
scratch rather than InforMARL), exact changes and verification. No performance
comparison has been run yet.

## 2026-11-15: Multi-episode PPO batches (`--episodes-per-update`)

### Personal Note
Aaron here - This is a changelog and experiment summary of looking at expanding the amount of episodes present in each rollout buffer. Having multiple episodes and having advantage estimates be a function of both of them when training should even out the updates, and reduce the significant PPO clip ratio of samples that I was seeing from having 10 training epochs with old policies per episode. Alternatives to this approach are: **Decreasing epochs per episode** although you then have to run more episodes to get the same total number of updates and advantage estimates, and **Increasing episode length** which I have not tried yet. Increasing episode length could provide more transitions and experience for the critic to be updated and learn, and with dynamic, changing wind directions, a longer episode could achieve the same goal while retaining the current default structure of the trainer, although the tradeoff is increased simulation time. 

### Summary

The new flag `--episodes-per-update E` collects **E complete episodes before each
PPO update**, instead of one. The default, `E = 1`, reproduces the previous code
exactly; this was checked value for value against an earlier run (see
Verification).

**Result:** in a 60-episode, constant-wind test, two episodes per update did
**not** beat one (+3.1% and +4.4% against +6.7% over the baseline; see
Experiment). The flag is kept, off by default, for the dynamic-wind runs, where
it is more likely to help.

| | Before (and `E = 1`) | `E = 2` |
|---|---|---|
| Samples per update, per actor | 200 (one episode) | 400 (two episodes) |
| Updates in a 60-episode run | 60 | 30 |
| Simulation time per episode | unchanged | unchanged (simulators step one after another) |
| Gradient steps per update | `ppo_epoch × num_mini_batch` | same formula, over the larger batch |

### Why

Each update used to train on a single 200-step episode: consecutive,
strongly related time steps with one random draw of exploration noise. Reusing
that batch for 10 PPO epochs pushes the policy toward whatever that one episode
happened to show. In the 2026-11-14 runs the PPO clip fraction peaked at
0.25–0.31, meaning that in some updates about a quarter of the samples reached
the clip limit.

Standard MAPPO setups train on thousands of samples per update. A larger batch
should give steadier updates, at the cost of fewer of them for the same
simulation time. This also matters for the planned dynamic-wind runs, where each
episode's wind differs and a single episode is an even noisier sample.

### Design choice: one simulator per episode, not a longer buffer

light_mappo already stores one buffer column per rollout environment
("thread"). GAE, terminal masks, advantage standardization and minibatching all
work across those columns. So E episodes per update is simply **E training
environments, stepped one after another by light_mappo's `DummyVecEnv` in this
process**.
- The training math is untouched; no light_mappo or OFF code was modified.
- Each column is a complete 200-step episode, so the time-remaining critic
  input, the horizon-scaled farm critic and the individual reference values keep
  their meaning.

The alternative was one environment and a buffer E × 200 steps long with episode
boundaries inside it. It was rejected because many parts of the runner assume
buffer length = episode length (horizons, references, logging).

### Changes, with reasons

Every code change is marked with a comment containing `Multi-episode batches`
or `--episodes-per-update`, so they are easy to find.

1. **New flag `--episodes-per-update`** (`train_mappo.py:47`).
   - **Default 1**, so existing commands and results are unaffected.
   - It sets light_mappo's `n_rollout_threads` (`train_mappo.py:103`). The
     launcher still rejects `--n_rollout_threads` directly, so there is one
     documented way in.
   - Both values are saved in `config.json`.
2. **Input checks for E > 1** (`train_mappo.py:94-102`).
   - **`num_env_steps` must be a multiple of `episode_length × E`.** light_mappo
     rounds the number of updates down, which would otherwise silently drop the
     leftover episodes.
   - **`--numeric-eval-interval` must be a multiple of E.** Updates happen only
     every E episodes, so other intervals can't be honoured exactly.
   - **Not applied when E = 1,** so behaviour for every existing command line is
     unchanged.
3. **One simulator per episode, each with its own wind stream**
   (`train_mappo.py:114-120`).
   - `environment_id = index` gives each simulator a distinct seeded wind
     profile in the `speed`, `direction` and `both` modes. With constant wind
     they are identical.
   - Environment 0 is exactly the previous single environment.
4. **Action-space seeds** (`train_mappo.py:156`). Each environment's gym
   action-space sampler gets a distinct seed; environment 0 keeps the old one.
   These samplers aren't used in training (actions come from the policy). This
   only avoids identical seeds.
5. **Environment list built without the late-binding bug**
   (`train_mappo.py:158`). `lambda train_env=train_env: train_env` binds each
   environment when the list is built. A plain `lambda: train_env` inside a loop
   would make every slot return the last environment.
6. **Episode numbering in the runner** (`shared_critic_runner.py`).
   - `_completed_episodes` now counts **environment episodes**, increasing by E
     per update (`:568`).
   - **Why:** evaluation labels, `models_best/`, video names and chart x-axes
     then stay in episodes, so runs with different E line up against each other.
   - Environment e runs episode `completed + e + 1` of the current update. This
     is used for wind-scenario records (`:281`, `:318`) and `rollout_steps.csv`
     (`:406`). It keeps `(episode, tick)` unique, which
     `plot_training_diagnostics.py` relies on.
7. **Training metrics** (`shared_critic_runner.py:339`).
   - Still **one row per update**. Row "episode" values are E, 2E, …
   - Energy and return columns are **means over the update's E episodes**;
     `cumulative_energy` adds all E. `environment_steps` = episodes × 200.
   - **Why one row per update:** the loss, clip-fraction and critic columns
     exist once per update. Repeating them on E rows would suggest E separate
     measurements.
8. **Evaluation schedule** (`shared_critic_runner.py:568`).
   - Evaluate when the episode count **reaches or passes** a multiple of the
     interval, and after the final update.
   - For E = 1 this is the same as the old `episode % interval == 0` rule.
   - For E > 1 it never skips an evaluation, even when the runner is used
     directly without the launcher's input checks.
9. **`select_team_weight.py:36-42`.** The completeness check now expects one
   row per update and a final episode of `updates × E`. For E = 1 the expected
   values are the same as before.

### What deliberately did not change

- **PPO and critic training** (`train()`). With E columns, individual and farm
  advantages are standardized over all E × 200 samples, which is the intended
  effect. `--ppo_epoch` and `--num_mini_batch` keep their meanings, applied to
  the larger batch.
- **Checkpoint format** (still 6). Model shapes don't depend on E, so you can
  resume an E = 2 run from an E = 1 checkpoint and vice versa.
- **Evaluation and videos.** These use a separate evaluation simulator, as
  before.
- **light_mappo and the bundled OFF source:** unmodified.

### Known limitations

- **No speed-up.** The simulators step one after another in one process, so the
  wall-clock time per episode is unchanged. True parallel simulators would need
  separate processes (see `README_MAPPO.md`).
- **Memory.** E OFF/FLORIS instances are alive at once. That's small for 4
  turbines.
- **Shared OFF logger.** OFF attaches its handlers to one global logger, so each
  simulator's log file also receives the others' lines. These are temporary
  files deleted at every reset. A real-simulator test confirms there is no
  effect on results.
- **Wind columns in `training_metrics.csv`** (`scenario_id`, `episode_seed`,
  end wind) describe the last environment's episode in each update. Every
  episode's full scenario is in `wind_scenarios.jsonl`.
- **light_mappo's console line** "updates k/K" counts updates, not episodes.

### How to turn it off or remove it

- **Off:** omit the flag or pass `--episodes-per-update 1`. That is verified to
  be identical to the previous code.
- **Remove:**
  1. Delete the flag and its checks in `train_mappo.py`, and return to a single
     `env` and `DummyVecEnv([lambda: env])`.
  2. In `shared_critic_runner.py`, change the marked expressions back:
     `+ self.n_rollout_threads` → `+ 1`, drop `* self.n_rollout_threads` from
     the cumulative energy, drop `+ environment_index` / `+ env_index` from
     episode numbers, and restore the `% interval == 0` evaluation rule.
  3. Revert `select_team_weight.py:36-42`.
  4. Delete the tests listed below.

### Verification

- **Unit tests** (all 47 pass):
  - `test_shared_critic_runner.py::test_multi_episode_updates_number_episodes_and_schedule_evaluations`
    (new) uses fake environments with E = 2. It checks:
    - three updates over six episodes, each simulator stepped 12 times;
    - critic optimizer steps = updates × epochs × minibatches;
    - training rows at episodes 2/4/6 with per-episode energy and the
      cumulative total;
    - unique per-step episode numbers;
    - evaluations at 0/2/4/6, and at 0/4/6 for an interval of 3.
  - `test_wind_farm_env.py::test_interleaved_simulators_do_not_interact` (new)
    uses the real OFF simulator. Simulator A gives **exactly identical**
    observations, rewards and critic inputs whether it runs alone or interleaved
    with simulator B, including while B finishes and a fresh B starts.
  - `test_four_turbine_reporting.py` (extended) covers the
    `select_team_weight.py` completeness check for E = 2.
- **Regression run:** `rl_runs/mappo/check_e1_regression`, 2 episodes at E = 1.
  It reproduced the first two episodes of `rl_runs/mappo/hyp1_w1` **value for
  value**: all 37 training-metric columns (except wall time), the episode-0
  evaluation, and all 400 per-step rollout rows.
- **Smoke run:** `rl_runs/mappo/smoke_e2`, 4 episodes at E = 2.
  - Training rows at episodes 2 and 4; cumulative energy adds all four episodes.
  - Per-step rows alternate between simulators 0 and 1; each episode has its own
    wind-scenario record.
  - Evaluations at 0, 2 and 4; `models_best/`, both final videos, all charts and
    HTML reports were produced.
  - `select_team_weight.py` accepted the run.
  - No errors or skipped charts.

### Experiment: does a two-episode batch help? (60 episodes)

**Short answer: not in this setting.** With constant wind and 60 training
episodes, the original one-episode updates finished best: +6.7% over the aligned
baseline, against +3.1% and +4.4% for the two-episode runs. **The default stays
`E = 1`.** Two-episode batches are worth re-testing once the wind changes during
training (see Recommendation).

#### Design

All three runs share these settings:
- team weight 1, seed 1, constant 8 m/s wind from 270°, `--use_value_active_masks`;
- one greedy evaluation every 10 episodes;
- aligned baseline of 2.3825 MWh per episode.

| Run | E | `--ppo_epoch` | Updates | Gradient steps per network | Folder |
|---|---|---|---|---|---|
| **Control** | 1 | 10 | 60 | 600 | `rl_runs/mappo/hyp1_w1/20261114_180157_9000a8e5` (first 60 episodes) |
| **E2 / P10** | 2 | 10 | 30 | 300 | `rl_runs/mappo/batch2_epoch10/20261115_224850_e2fbd49a` |
| **E2 / P20** | 2 | 20 | 30 | 600 | `rl_runs/mappo/batch2_epoch20/20261115_224850_dfa17160` |

- **Control.** This is the first 60 episodes of the existing 100-episode
  team-weight-1 run, used instead of a fresh run. That is valid for two reasons:
  - Training is deterministic for a given seed (the regression run above
    reproduced it value for value).
  - The learning rate doesn't decay, so episodes 1–60 of a longer run are the
    same as a 60-episode run.
- **E2 / P10** changes only the batch size. It has the same simulation budget,
  but half the updates and half the gradient steps.
- **E2 / P20** also doubles the PPO epochs, so its gradient steps match the
  control. That separates "fewer updates" from "less optimization".

#### Results

| | Control (E1, P10) | E2, P10 | E2, P20 |
|---|---|---|---|
| Greedy evaluation at episode 60 | **2.543 MWh (+6.7%)** | 2.457 (+3.1%) | 2.486 (+4.4%) |
| Best greedy evaluation | +6.7% (ep. 60) | +4.9% (ep. 40) | +4.4% (ep. 50) |
| Mean of evaluations, episodes 30–60 | **+4.5%** | +3.3% | +3.0% |
| Training (exploration) energy, episodes 41–60 | **+5.4%** | +2.3% | +4.5% |
| Upstream yaw at episode 60, T1 / T2 | +11° / −15° | −1° / −13° | −15° / +4° |
| Wakes steered at episode 60 | **both** | T2 → T4 only | T1 → T3 only |
| PPO clip fraction, mean / max | 0.052 / 0.246 | 0.032 / 0.237 | 0.049 / **0.153** |

Greedy evaluations every 10 episodes, % vs aligned baseline:

| Episode | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| Control | +0.4 | −3.6 | −0.3 | +2.7 | +1.7 | +6.6 | +6.7 |
| E2, P10 | +0.4 | −8.6 | −6.4 | +2.2 | +4.9 | +2.9 | +3.1 |
| E2, P20 | +0.4 | −9.3 | +2.7 | −0.2 | +3.3 | +4.4 | +4.4 |

Each run's gain at episode 60, split by turbine pair (% of the baseline farm
energy):

| | T1 → T3 pair | T2 → T4 pair | Farm |
|---|---|---|---|
| Control | +2.7 | +4.0 | +6.7 |
| E2, P10 | −0.3 | +3.4 | +3.1 |
| E2, P20 | +4.2 | +0.2 | +4.4 |

A chart of the evaluations and upstream yaw angles is at
`rl_runs/mappo/batch2_comparison.png`. It is local only, because `rl_runs/` is
git-ignored.

#### Interpretation

1. **The number of updates mattered more than the batch size.**
   - With constant wind, every episode sees the same wind. The two episodes in a
     batch differ only in exploration noise, so the second one adds little new
     information.
   - At the same simulation budget, E = 2 halves the number of times the policy
     improves and then collects fresh data with the improved policy.
2. **Doubling the epochs recovered part of the gap, not all of it.** E2/P20
   finished at +4.4%, against +3.1% for E2/P10 and +6.7% for the control. More
   passes over the same samples is not the same as new samples from an updated
   policy.
   - E2/P20 had the lowest maximum clip fraction (0.153, against 0.246 for the
     control), even with 20 epochs. So the larger batch did make individual
     updates more moderate, which is the effect it was meant to have.
   - Within 60 episodes, that steadiness did not turn into more energy.
3. **The gap is almost entirely the second wake.**
   - At episode 60 the control steers both upstream turbines.
   - Each E = 2 run steers only one: P10 steers T2 → T4 and P20 steers T1 → T3.
   - Each steered pair is worth about 3–4% on its own. With a farm-only reward,
     learning to steer the second pair is a harder credit-assignment problem,
     and fewer updates means fewer chances to discover it.
4. **Greedy evaluations were not steadier with E = 2.**
   - All three runs dipped at episode 10.
   - E2/P10's Turbine 1 swung to about +40° mean yaw and back before settling
     near 0°.

#### Caveats

- **One seed per arm.** A different seed changes the exploration noise and could
  reorder close results. The 1–3 point differences between arms are about the
  size of the jumps between one run's consecutive evaluations.
  - Robust: the control steers both wakes and the E = 2 runs steer one.
  - Not robust: the exact percentages.
- **60 episodes is short.** The E = 2 runs had only 30 updates, and longer
  training might bring all three to the same place.
- **Constant wind is the least favourable case for larger batches.** When each
  episode has different wind, a single episode is a much noisier sample of the
  task. That is where averaging over several episodes per update is expected to
  pay off.
- **Wall-clock times are not comparable.** The control shared the computer with
  two other training runs; each E = 2 run shared it with one. The change itself
  does not make episodes faster (see Known limitations).

#### Recommendation

- **Keep `E = 1` (the default) for constant-wind training.**
- **Re-test when dynamic-wind training starts.**
  - Compare E = 1 against E = 2 (and possibly E = 4) at the same number of
    simulated episodes.
  - Use `--ppo_epoch 10` and at least two seeds per arm.
  - Judge by the mean of the late greedy evaluations rather than the single best
    one.
