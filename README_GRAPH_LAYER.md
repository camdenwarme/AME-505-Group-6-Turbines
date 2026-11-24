# Neighbour graph layer for the turbine actors

This branch (`graph_layer`, from `mappo_4_turbines`) adds an optional graph
neural network (GNN) layer in front of each turbine's actor. With it on, each
turbine's policy sees its neighbours' measurements as well as its own. The edges
between turbines are weighted by where the wind is coming from right now.

- **Off by default.** `--graph-layer off` reproduces `mappo_4_turbines` value
  for value. Training, evaluation and rollout CSVs were identical for both
  feedforward and recurrent runs (see [Verification](#verification)).
- **It adds a layer and replaces nothing.** PPO, GAE, the tanh action head, the
  GRU option, the farm critic and the per-turbine value models do not change.
  Only the actor's feature extractor gains a graph encoder in front of the
  existing MLP.
- **Separate actors, critic unchanged.** These are Aaron's choices from
  2026-11-24. Each turbine keeps its own actor, so each turbine has its own
  graph encoder.
- **Any turbine count.** Pass a layout file with turbine coordinates and an
  explicit neighbour list. Nothing else needs to change.

---

## 1. Quick use

```powershell
# Default 2 x 2 farm with the graph layer on (uses layouts/square4.json)
.venv/Scripts/python.exe train_mappo.py --graph-layer on --num_env_steps 20000 --experiment_name graph_sq4

# Six turbines, two-hop neighbourhoods, upwind-only messages, with a GRU for wake delay
.venv/Scripts/python.exe train_mappo.py --graph-layer on --layout-file layouts/six_turbines.json `
    --graph-hops 2 --graph-messages upwind `
    --algorithm_name rmappo --use_recurrent_policy --data_chunk_length 10 `
    --wind-mode both --num_env_steps 40000 --experiment_name graph_six

# Same farm without the graph, for the ablation (layout files work with the graph off too)
.venv/Scripts/python.exe train_mappo.py --layout-file layouts/six_turbines.json --wind-mode both `
    --num_env_steps 40000 --experiment_name nograph_six
```

Smoke test (about 30 s):

```powershell
.venv/Scripts/python.exe train_mappo.py --graph-layer on --num_env_steps 8 --episode_length 4 --ppo_epoch 1 --no-video --experiment_name smoke_graph
```

### New flags

| Flag | Default | Meaning |
|---|---|---|
| `--graph-layer {off,on}` | `off` | Toggle the graph encoder. `off` gives the original actors exactly. |
| `--graph-hops K` | `1` | Each actor sees turbines up to K neighbour links away (its *ego graph*). |
| `--graph-layers L` | `K` | Rounds of message passing. With `L ≥ 2`, information from 2 links upwind passes through the turbine in between. |
| `--graph-hidden H` | `64` | Width of the graph embedding. Must be divisible by `--graph-heads`. |
| `--graph-heads` | `4` | Number of attention heads. |
| `--graph-messages {both,upwind}` | `both` | `both`: every link sends messages both ways, and attention learns from the wind-frame edge features which ones matter. `upwind`: also hard-masks messages from turbines downstream of the receiver. Side-by-side neighbours, within 0.25 D, are kept. |

`--layout-file` now accepts the new layout format below or the old bare
`[[x, y], ...]` array. The graph layer needs the new format, because only it
contains neighbours.

### All existing toggles still work

Each row was run end to end through `train_mappo.py` with `--graph-layer on`, or
is covered by `test_turbine_graph.py`.

| Toggle | With graph on |
|---|---|
| `--wind-mode off/speed/direction/both` | ✓ Edge features follow the measured direction every step. |
| `--algorithm_name rmappo --use_recurrent_policy` | ✓ The GRU sits after the graph encoder and MLP. |
| `--episodes-per-update E` | ✓ |
| `--team-reward-weight`, `--gamma`, `--ppo_epoch`, `--num_mini_batch` | ✓ Unchanged. |
| `--evaluate-aligned-baseline`, `--final-video`, videos, plots, reports | ✓ They read the first 3 observation columns, which are always the local features. |
| `--model_dir` (resume) | ✓ A graph checkpoint records its graph in `checkpoint_metadata.json`. Loading it into a mismatched run, such as a different layout, links, hops or graph off, is refused with a clear message. Graph-off checkpoints have the same format as before. |

---

## 2. Layout files

Positions are in rotor diameters, with x pointing east and y north (the same as
before). The order of `turbines` is the agent order: turbine 1 is `actor_agent0`.
`neighbors` lists **undirected** physical links, which are the grid lines you
would draw on a map. Which end of a link is upwind is decided each step from
the wind, so you never list directions.

```json
{
  "description": "3 x 2 grid, 5 D spacing",
  "turbines": [
    {"name": "T1", "x": 0, "y": 0},
    {"name": "T2", "x": 0, "y": 5},
    {"name": "T3", "x": 5, "y": 0},
    {"name": "T4", "x": 5, "y": 5},
    {"name": "T5", "x": 10, "y": 0},
    {"name": "T6", "x": 10, "y": 5}
  ],
  "neighbors": [
    ["T1", "T2"], ["T1", "T3"], ["T2", "T4"],
    ["T3", "T4"], ["T3", "T5"], ["T4", "T6"], ["T5", "T6"]
  ]
}
```

- `name` is optional (the default is `T1`, `T2`, …). Neighbour entries may use
  names or 0-based indices. Duplicates and reversed pairs are merged.
- `z` is optional (default 0). `description` and any other keys are ignored, so
  you can add your own notes, for example `"_note": "..."`.
- Shipped examples: [layouts/square4.json](layouts/square4.json) (the default
  farm) and [layouts/six_turbines.json](layouts/six_turbines.json).

**You don't have to write the links by hand.** This command suggests grid-line
links: each turbine is linked to the turbines at its nearest spacing.

```powershell
.venv/Scripts/python.exe turbine_graph.py make six_turbines.json layouts/six_turbines.json
.venv/Scripts/python.exe turbine_graph.py make my_positions.json layouts/my_farm.json --diagonals
.venv/Scripts/python.exe turbine_graph.py check layouts/my_farm.json --hops 2
```

`make` takes any layout file, including an old bare array, and writes the
new format. Edit its `neighbors` list as you like. `check` prints each turbine's
links, which turbines each actor will see, and the observation size.

---

## 3. How it works

### Where it sits in MAPPO

```
                       ┌──────────── per turbine i (own weights) ─────────────┐
env.step ─► local obs ─┤ TurbineGraph.observe: ego graph of i, wind-frame edges│
 (N × 3)               │   │                                                   │
                       │   ▼                                                   │
                       │ node embed ─► L × EdgeGAT ─► ego embedding (slot 0)  │
                       │                                   │                   │
                       │ own [speed, dir, yaw] ────────────┴► concat           │
                       │                                        ▼              │
                       │                         light_mappo MLPBase (unchanged)│
                       │                                        ▼              │
                       │                         GRU (if rmappo, unchanged)    │
                       │                                        ▼              │
                       │                         TanhBoundedActionHead         │
                       └───────────────────────────────────────────────────────┘
farm critic, per-turbine value models, GAE, PPO update: unchanged
```

PPO's gradients flow through the graph encoder, so it trains end to end with
the actor. light_mappo's buffer stores the larger observation without any
change to its code.

### Actor observation with the graph on

For turbine *i*, the observation is one flat, fixed-size vector:

```
[ speed, direction, yaw | node features (M × 6) | edge features (M × M × 4) | adjacency (M × M) ]
```

The first 3 values are exactly the old observation. *M* is the largest ego graph
size in the farm, and smaller ego graphs are zero-padded. Slot 0 is always
turbine *i*. For the 2 × 2 square with 1 hop, M = 3 and the size is 66. For six
turbines with 2 hops, M = 6 and the size is 219.

**Node features** (for each turbine in the ego graph):

| # | Feature |
|---|---|
| 0 | local wind speed / 10 m/s |
| 1 | wind-relative yaw / 180° |
| 2, 3 | sin and cos of (its measured direction − neighbourhood direction) |
| 4, 5 | its downstream and crosswind offset from turbine *i*, / 10 D |

**Edge features** (message from sender *b* to receiver *a*, for linked pairs only):

| # | Feature |
|---|---|
| 0 | downstream distance of *a* from *b*, / 10 D. Positive means *b* is upwind of *a*. |
| 1 | crosswind offset, / 10 D |
| 2 | wake proxy in [0, 1]: a Jensen-style deficit shape, `exp(-½(c/r)²) / (1 + 2kx)²` with `r = 0.5 + kx` and `k = 0.05`. It is 0 when *a* is not downstream of *b*. |
| 3 | 1 for the self-loop, otherwise 0 |

The **wind frame** is the circular mean of the measured local directions in the
ego graph. It is recomputed every step, so the actor uses only measurements
from itself and its neighbours (decentralized execution with neighbour
communication). This is how a wind shift changes the graph: the topology stays
fixed, and the edge features (and in `upwind` mode, the mask) change. An
upwind neighbour gets a positive downstream distance and a nonzero wake proxy.
Attention can learn to weight it highly and ignore downwind neighbours.

### The graph layer

`graph_actor.EdgeGATLayer` is a dense, masked, multi-head attention layer
(GATv2-style) that uses edge features:

```
score(b→a) = att · LeakyReLU(W_dst h_a + W_src h_b + W_edge e_ba)     (per head)
α          = softmax over the senders b allowed by the adjacency (self-loop always allowed)
h_a'       = LayerNorm(h_a + ReLU(W_out Σ_b α_ab (W_val h_b + W_eval e_ba)))
```

Padded slots are masked out and cannot affect the result (a unit test checks
this). After L layers, the ego turbine's embedding is concatenated with its own
raw 3 features and passed to light_mappo's `MLPBase`. That means `--layer_N`,
`--hidden_size`, `--use_ReLU`, `--use_orthogonal` and
`--use_feature_normalization` keep their meaning.

### Wake delay

Wakes take longer than one 4 s step to travel 5 D, so a downwind turbine needs
memory to act on what its upwind neighbour did several steps ago. The existing
`--algorithm_name rmappo --use_recurrent_policy` GRU runs after the graph
encoder and provides that memory. I recommend it with the graph layer under
dynamic wind.

---

## 4. Why from scratch rather than InforMARL

I considered [InforMARL](https://github.com/nsidn98/InforMARL) (MIT license) as a
base. It is MAPPO with a graph layer, built on the same on-policy codebase
light_mappo comes from. I wrote the layer from scratch (about 100 lines in
`graph_actor.py`) and kept InforMARL's design ideas:

| | InforMARL as a base | From scratch (chosen) |
|---|---|---|
| Dependencies | Needs `torch_geometric` (not installed, and a heavy install on Windows with CPU torch 2.15) | Plain torch |
| Integration | Has its own graph runner and buffer (separate `node_obs`, `adj`, `agent_id` arrays), its own MPE graph env, and a shared-policy runner. Your `SharedCriticRunner`, individual critics, multi-episode batches and reporting would need porting onto its runner, or its runner onto yours. | Graph data travels inside the normal observation, so light_mappo's buffer, recurrent chunking and your runner are untouched |
| Edge semantics | Entity graph (agents, obstacles, goals) with distance-based edges | Fixed physical links with wind-frame, wake-aware edge features |
| Graph size | Sparse message passing for large dynamic graphs | Dense attention over M ≤ ~15 nodes per ego graph. Cheap, and the FLORIS step dominates run time anyway |
| Ideas reused | — | Graph passed through the observation, attention with edge features, ego-node readout, an option to pool for the critic later |

In short, InforMARL's value is its runner and env plumbing, and that is exactly
the part that conflicts with your existing setup. The graph layer itself is
small.

---

## 5. Exact changes

| File | Change |
|---|---|
| `turbine_graph.py` (new) | Layout file loader (new and legacy formats), grid-link suggestion, human-readable writer, `TurbineGraph` (ego graphs, wind-frame node and edge features, upwind mask), `GraphObsSpec`, `make`/`check` CLI |
| `graph_actor.py` (new) | `EdgeGATLayer`, `GraphActorBase` (graph encoder plus the upstream `MLPBase`), `install_graph_actor` (swaps `actor.base` and rebuilds that actor's Adam) |
| `layouts/square4.json`, `layouts/six_turbines.json` (new) | Example layout files with neighbour lists |
| `test_turbine_graph.py` (new) | 9 tests: file formats, shipped files, edge geometry under two wind directions, upwind masking, padding invariance, gradients, runner training plus checkpoint round trip (feedforward and GRU) |
| `wind_farm_env.py` | Optional `graph=` argument. When set, `observation_space` widens and `reset`/`step` return `graph.observe(local_obs)`. `_measure`, rewards and critic observations are unchanged. |
| `shared_critic_runner.py` | If the envs have a graph, install the graph encoder on each actor after the bounded action head. Add `actor_graph` to checkpoint metadata (graph runs only). Refuse to load graph checkpoints into non-graph runs. The rollout CSV reads `obs[:3]`. |
| `train_mappo.py` | `--graph-*` flags. `--layout-file` goes through `load_layout`. The graph is passed to training and evaluation envs. The graph description is saved in `config.json`. |
| `evaluate_wind_farm.py`, `wind_farm_video.py` | Log and plot `obs[..., :3]` (a no-op with the graph off) |

The full diff is `git diff mappo_4_turbines graph_layer`.

---

## 6. Verification

- **Graph off is identical to `mappo_4_turbines`.** I exported
  `mappo_4_turbines` and ran both branches with
  `--num_env_steps 60 --episode_length 10 --ppo_epoch 2 --wind-mode both --seed 5`,
  once feedforward and once with `rmappo --use_recurrent_policy`.
  `training_metrics.csv`, `evaluation_metrics.csv` and `rollout_steps.csv` were
  identical in every column except wall-clock time.
- **Graph on runs end to end** on the real OFF/FLORIS simulator: square4 with 1
  hop; six turbines with 2 hops, upwind messages and a GRU; and square4 with
  `--episodes-per-update 2 --evaluate-aligned-baseline --final-video` (both
  videos rendered, and `models_best` reloaded into the graph actors).
- **Unit tests:** `.venv/Scripts/python.exe -m unittest` gives `Ran 56 tests ... OK`.
  That is the previous 47 plus the 9 new ones.

These short runs check the plumbing only. They are far too short to show any
performance difference. The next step is a matched comparison with the same
seed, wind mode and episode count, graph off against graph on (ideally with the
GRU).

## 7. Limitations and possible next steps

- **One trained policy per layout.** With separate actors, each actor's weights
  belong to one turbine. Moving to a new layout means training again. Sharing
  actor weights would allow train-small, run-large, but that was declined for
  now.
- **The critic doesn't use the graph.** The farm critic already sees every
  turbine. A pooled graph critic is the natural follow-up if critic input size
  becomes a problem at larger N.
- **The wake proxy is a cue, not a model.** It ignores yaw-induced wake
  deflection and added turbulence. The attention and the node yaw features let
  the network learn the corrections.
- **Neighbours' current measurements only.** Wake-delay memory comes from the
  GRU, not from lagged neighbour features. If the GRU isn't enough, adding the
  neighbours' recent yaw history as node features is a small change in
  `TurbineGraph._node_features`.
