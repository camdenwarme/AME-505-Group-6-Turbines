"""Neighbour-graph layouts, observations, actor encoder and runner wiring (no simulator)."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from gym import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "light_mappo"))
from config import get_config
from envs.env_wrappers import DummyVecEnv
from graph_actor import GraphActorBase
from shared_critic_runner import SharedCriticRunner
from turbine_graph import TurbineGraph, grid_neighbors, layout_to_json, load_layout

ROOT = Path(__file__).resolve().parent
SQUARE = [[0, 0], [0, 5], [5, 0], [5, 5]]          # T1 T2 T3 T4, x east, y north


def _square_graph(**kwargs):
    return TurbineGraph(SQUARE, [(0, 1), (0, 2), (1, 3), (2, 3)], **kwargs)


def _local(direction=270.0, speed=8.0):
    return np.array([[speed, direction, 0.0]] * 4, dtype=np.float32)


def _edge(graph, obs, receiver, sender):
    """Edge features sender -> receiver from the receiver's own observation."""
    _, _, edges, adjacency = graph.spec.split(obs[receiver])
    nodes = graph.ego_nodes[receiver]
    a, b = nodes.index(receiver), nodes.index(sender)
    return edges[a, b], adjacency[a, b]


class LayoutFileTest(unittest.TestCase):
    def test_object_and_legacy_formats(self):
        legacy = load_layout(SQUARE)
        self.assertIsNone(legacy.edges)
        self.assertEqual(legacy.names, ["T1", "T2", "T3", "T4"])
        named = load_layout({"turbines": [{"name": n, "x": x, "y": y}
                                          for n, (x, y) in zip("ABCD", SQUARE)],
                             "neighbors": [["A", "B"], [2, 0], ["D", "B"], ["C", "D"], ["B", "A"]]})
        self.assertEqual(named.edges, [(0, 1), (0, 2), (1, 3), (2, 3)])
        np.testing.assert_array_equal(named.positions_d[:, :2], SQUARE)
        with self.assertRaises(ValueError):
            load_layout({"turbines": [{"x": 0, "y": 0}], "neighbors": [["T1", "T9"]]})

    def test_grid_suggestion_roundtrip_and_shipped_files(self):
        self.assertEqual(grid_neighbors(SQUARE), [(0, 1), (0, 2), (1, 3), (2, 3)])
        self.assertEqual(len(grid_neighbors(SQUARE, diagonals=True)), 6)
        layout = load_layout(SQUARE)
        layout.edges = grid_neighbors(SQUARE)
        again = load_layout(json.loads(layout_to_json(layout)))
        self.assertEqual(again.edges, layout.edges)
        shipped = load_layout(ROOT / "layouts" / "square4.json")
        np.testing.assert_array_equal(shipped.positions_d[:, :2], SQUARE)
        self.assertEqual(shipped.edges, layout.edges)

    def test_graph_requires_neighbors(self):
        with self.assertRaisesRegex(ValueError, "neighbors"):
            TurbineGraph(SQUARE, None)


class GraphObservationTest(unittest.TestCase):
    def test_shape_and_local_prefix(self):
        graph = _square_graph()
        self.assertEqual(graph.ego_nodes[0], [0, 1, 2])
        self.assertEqual(graph.spec.slots, 3)
        local = _local()
        obs = graph.observe(local)
        self.assertEqual(obs.shape, (4, graph.spec.size))
        np.testing.assert_array_equal(obs[:, :3], local)
        two_hop = _square_graph(hops=2)
        self.assertEqual(two_hop.ego_nodes[0], [0, 1, 2, 3])

    def test_edges_follow_wind_direction(self):
        graph = _square_graph()
        # Wind from the west: T1 (x=0) is upwind of T3 (x=5).
        obs = graph.observe(_local(270.0))
        downstream_edge, allowed = _edge(graph, obs, 2, 0)
        self.assertTrue(allowed)
        self.assertAlmostEqual(downstream_edge[0], 0.5, places=5)   # 5 D downstream
        self.assertAlmostEqual(downstream_edge[1], 0.0, places=5)
        self.assertAlmostEqual(downstream_edge[2], 1 / 1.5 ** 2, places=5)  # Jensen, k = 0.05
        upstream_edge, _ = _edge(graph, obs, 0, 2)
        self.assertAlmostEqual(upstream_edge[0], -0.5, places=5)
        self.assertEqual(upstream_edge[2], 0.0)
        side_edge, _ = _edge(graph, obs, 1, 0)                       # T1 -> T2, crosswind only
        self.assertAlmostEqual(side_edge[0], 0.0, places=5)
        self.assertEqual(side_edge[2], 0.0)
        # Wind from the north: now T2 (y=5) is upwind of T1, and T1 -> T3 is crosswind.
        obs = graph.observe(_local(0.0))
        self.assertGreater(_edge(graph, obs, 0, 1)[0][2], 0.4)
        self.assertEqual(_edge(graph, obs, 2, 0)[0][2], 0.0)

    def test_upwind_mode_drops_downstream_senders(self):
        graph = _square_graph(messages="upwind")
        obs = graph.observe(_local(270.0))
        self.assertTrue(_edge(graph, obs, 2, 0)[1])       # upwind T1 -> T3 kept
        self.assertFalse(_edge(graph, obs, 0, 2)[1])      # downstream T3 -> T1 dropped
        self.assertTrue(_edge(graph, obs, 1, 0)[1])       # side-by-side kept
        self.assertFalse(_edge(graph, obs, 0, 2)[0].any())


class GraphActorTest(unittest.TestCase):
    def _base(self, graph, layers=1):
        args = get_config().parse_args([])
        args.use_feature_normalization = False
        torch.manual_seed(0)
        return GraphActorBase(args, graph.spec, graph_hidden=16, heads=2, layers=layers)

    def test_forward_gradients_and_padding(self):
        graph = TurbineGraph([[0, 0], [5, 0], [10, 0]], [(0, 1), (1, 2)])   # T1 has 1 neighbour, M = 3
        base = self._base(graph)
        obs = torch.as_tensor(graph.observe(np.array([[8, 270, 0], [6, 270, 10], [5, 270, -5]],
                                                     dtype=np.float32)))
        features = base(obs)
        self.assertEqual(tuple(features.shape), (3, base.hidden_size))
        features.sum().backward()
        self.assertTrue(all(p.grad is not None for p in base.layers.parameters()))
        # Values written into T1's padded slot must not change T1's features.
        padded = obs.clone()
        _, nodes, edges, _ = graph.spec.split(padded[0])
        nodes[2] = 5.0
        edges[2] = 5.0
        torch.testing.assert_close(base(padded[:1]), base(obs[:1]))

    def test_upwind_neighbour_changes_receiver(self):
        graph = TurbineGraph([[0, 0], [5, 0]], [(0, 1)], messages="upwind")
        base = self._base(graph)
        calm = graph.observe(np.array([[8, 270, 0], [6, 270, 0]], dtype=np.float32))
        yawed = graph.observe(np.array([[8, 270, 25], [6, 270, 0]], dtype=np.float32))
        out_calm, out_yawed = base(torch.as_tensor(calm)), base(torch.as_tensor(yawed))
        # Downstream T2 sees upstream T1's yaw; T1 cannot see T2 in upwind mode.
        self.assertFalse(torch.allclose(out_calm[1], out_yawed[1]))


class _GraphFarm:
    """Two turbines in a row with a neighbour graph and the real observation layout."""

    dt = 4.0
    num_agents = 2
    graph = TurbineGraph([[0, 0], [5, 0]], [(0, 1)])
    observation_space = [spaces.Box(-np.inf, np.inf, (graph.spec.size,), np.float32)] * 2
    share_observation_space = [spaces.Box(-np.inf, np.inf, (7,), np.float32) for _ in range(2)]
    action_space = [spaces.Box(-6, 6, (1,), np.float32) for _ in range(2)]

    def _obs(self):
        return self.graph.observe(np.array([[8, 270, -self.t], [6, 270, self.t]], dtype=np.float32))

    def reset(self):
        self.t = 0
        return self._obs()

    def get_centralized_obs(self):
        return np.array([8, 270, -self.t, 6, 270, self.t, 1 - self.t / 4], dtype=np.float32)

    def get_individual_baseline_values(self, gamma):
        return np.array([0.2, 0.8], np.float32) * sum(gamma ** n for n in range(4 - self.t))

    def step(self, actions):
        self.t += 1
        infos = [{"episode_energy_j": self.t * 3.6e6, "team_reward": 0.5, "farm_power_w": 10,
                  "turbine_power_w": 4 + 2 * i, "yaw_rate_deg_s": float(actions[i][0])}
                 for i in range(2)]
        return self._obs(), np.array([[0.2], [0.8]], np.float32), np.full(2, self.t == 4), infos

    def close(self):
        pass


class GraphRunnerTest(unittest.TestCase):
    def test_graph_actors_train_and_checkpoint(self):
        torch.set_num_threads(1)
        for mode in ("feedforward", "recurrent"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(3)
                args = get_config().parse_args([])
                args.use_valuenorm = False
                args.episode_length, args.n_rollout_threads, args.ppo_epoch = 4, 1, 2
                args.num_mini_batch = 2 if mode == "feedforward" else 1
                args.use_recurrent_policy = mode == "recurrent"
                args.data_chunk_length = 2
                args.use_feature_normalization = False
                args.graph_hidden, args.graph_heads = 16, 2
                envs = DummyVecEnv([_GraphFarm])
                config = {"all_args": args, "envs": envs, "eval_envs": None, "num_agents": 2,
                          "device": torch.device("cpu"), "run_dir": Path(directory)}
                runner = SharedCriticRunner(config)
                try:
                    self.assertIsNot(runner.policy[0].actor, runner.policy[1].actor)
                    self.assertIsInstance(runner.policy[0].actor.base, GraphActorBase)
                    optimized = {id(p) for group in runner.policy[0].actor_optimizer.param_groups
                                 for p in group["params"]}
                    self.assertEqual(optimized, {id(p) for p in runner.policy[0].actor.parameters()})
                    runner.warmup()
                    for step in range(4):
                        values, actions, log_probs, states, critic_states, commands = runner.collect(step)
                        obs, rewards, dones, infos = envs.step(commands)
                        runner.insert((obs, rewards, dones, infos, values, actions,
                                       log_probs, states, critic_states))
                    runner.individual_rewards[0, 0, :, 0] += 0.3
                    runner.compute()
                    before = [p.detach().clone() for p in runner.policy[1].actor.base.layers.parameters()]
                    runner.train()
                    self.assertTrue(any(not torch.equal(a, b) for a, b in
                                        zip(before, runner.policy[1].actor.base.layers.parameters())))
                    runner.save()
                    models = Path(runner.save_dir)
                    metadata = json.loads((models / "checkpoint_metadata.json").read_text())
                    self.assertEqual(metadata["actor_graph"]["edges"], [[0, 1]])
                    saved = {k: v.clone() for k, v in runner.policy[0].actor.state_dict().items()}
                    args.model_dir = models
                    restored = SharedCriticRunner(config)
                    try:
                        for key, value in restored.policy[0].actor.state_dict().items():
                            torch.testing.assert_close(value, saved[key])
                    finally:
                        restored.writter.close()
                finally:
                    runner.writter.close()
                    envs.close()


if __name__ == "__main__":
    unittest.main()
