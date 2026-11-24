"""Turbine layout files and wind-conditioned neighbour graphs for the actors.

Layout file (JSON, positions in rotor diameters, see layouts/README in
README_GRAPH_LAYER.md):

    {
      "description": "2 x 2 square, 5 D spacing",
      "turbines": [
        {"name": "T1", "x": 0, "y": 0},
        {"name": "T2", "x": 0, "y": 5},
        ...
      ],
      "neighbors": [["T1", "T2"], ["T1", "T3"], ...]
    }

The list order of "turbines" is the agent order. "neighbors" lists undirected
physical links (the grid lines); names or 0-based indices both work. Which end
of a link is upwind is decided every step from the measured wind direction, so
a wind shift reweights edges instead of rewiring the graph. The older format, a
bare JSON array of [x, y] or [x, y, z] rows, still loads (positions only).

When the graph layer is on, each actor's observation becomes

    [own speed, own direction, own yaw  |  node features  |  edge features  |  adjacency]

for the ego graph within `hops` links of that turbine. The first three
entries are exactly the original observation, so logging code can keep reading
obs[..., :3]. Everything after them is fixed-size and zero-padded, so a light_mappo
buffer can store it unchanged.

Run this file directly to create or check a layout file:

    python turbine_graph.py check layouts/square4.json
    python turbine_graph.py make six_turbines.json layouts/six_turbines.json
"""

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

LOCAL_FEATURES = 3          # speed (m/s), direction (deg, from), yaw (deg)
NODE_FEATURES = 6           # see TurbineGraph._node_features
EDGE_FEATURES = 4           # see TurbineGraph._edge_features
# Lengths are divided by this many rotor diameters to keep features near unit scale.
LENGTH_SCALE_D = 10.0
# Simple top-hat (Jensen) style wake proxy: wake radius grows by WAKE_K per D.
WAKE_K = 0.05
# In "upwind" mode a message is dropped when its source is more than this many
# rotor diameters downstream of the receiver (side-by-side neighbours are kept).
UPWIND_TOLERANCE_D = 0.25


# --------------------------------------------------------------------------- #
# Layout files
# --------------------------------------------------------------------------- #

@dataclass
class Layout:
    positions_d: np.ndarray            # (N, 3) rotor diameters
    names: list
    edges: list                        # undirected (i, j) pairs with i < j, or None
    description: str = ""


def load_layout(path_or_data):
    """Read a layout file (object or legacy array format) into a Layout."""
    data = path_or_data
    if isinstance(path_or_data, (str, Path)):
        data = json.loads(Path(path_or_data).read_text())
    if isinstance(data, list):                       # legacy: [[x, y], ...]
        positions = _positions(data)
        return Layout(positions, [f"T{i + 1}" for i in range(len(positions))], None)
    if not isinstance(data, dict) or "turbines" not in data:
        raise ValueError('A layout file must be a JSON array of [x, y] rows or an object with "turbines".')
    turbines = data["turbines"]
    if not isinstance(turbines, list) or not turbines:
        raise ValueError('"turbines" must be a non-empty list.')
    rows, names = [], []
    for index, turbine in enumerate(turbines):
        if isinstance(turbine, dict):
            if "x" not in turbine or "y" not in turbine:
                raise ValueError(f'Turbine {index} needs "x" and "y" (rotor diameters).')
            rows.append([turbine["x"], turbine["y"], turbine.get("z", 0.0)])
            names.append(str(turbine.get("name", f"T{index + 1}")))
        else:
            rows.append(list(turbine) + [0.0] * (3 - len(turbine)))
            names.append(f"T{index + 1}")
    if len(set(names)) != len(names):
        raise ValueError("Turbine names must be unique.")
    positions = _positions(rows)
    edges = None
    if data.get("neighbors") is not None:
        lookup = {name: i for i, name in enumerate(names)}
        edges = set()
        for pair in data["neighbors"]:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Each neighbor entry must be a pair, got {pair!r}.")
            a, b = (_turbine_index(item, lookup, len(names)) for item in pair)
            if a == b:
                raise ValueError(f"Neighbor pair {pair!r} links a turbine to itself.")
            edges.add((min(a, b), max(a, b)))
        edges = sorted(edges)
    return Layout(positions, names, edges, str(data.get("description", "")))


def _positions(rows):
    positions = np.asarray(rows, dtype=float)
    if positions.ndim != 2 or positions.shape[1] not in (2, 3) or not len(positions):
        raise ValueError("Turbine positions must be [x, y] or [x, y, z] rows in rotor diameters.")
    if positions.shape[1] == 2:
        positions = np.column_stack((positions, np.zeros(len(positions))))
    if not np.isfinite(positions).all():
        raise ValueError("Turbine positions must be finite.")
    return positions


def _turbine_index(item, lookup, count):
    if isinstance(item, str):
        if item not in lookup:
            raise ValueError(f"Neighbor names unknown turbine {item!r}; known: {sorted(lookup)}.")
        return lookup[item]
    if isinstance(item, int) and 0 <= item < count:
        return item
    raise ValueError(f"Neighbor entry {item!r} is not a turbine name or 0-based index below {count}.")


def grid_neighbors(positions_d, diagonals=False, tolerance=0.05):
    """Suggest grid-line links: each turbine to its nearest-spacing neighbours.

    A pair is linked when its distance is within (1 + tolerance) of either
    turbine's nearest-neighbour spacing (times sqrt(2) with diagonals).
    """
    xy = np.asarray(positions_d, dtype=float)[:, :2]
    if len(xy) < 2:
        return []
    distance = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(distance, np.inf)
    nearest = distance.min(axis=1)
    reach = (np.sqrt(2.0) if diagonals else 1.0) * (1 + tolerance)
    edges = []
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            if distance[i, j] <= reach * max(nearest[i], nearest[j]):
                edges.append((i, j))
    return edges


def layout_to_json(layout):
    """Human-readable layout text: one turbine and one link per line."""
    lines = ["{"]
    if layout.description:
        lines.append(f'  "description": {json.dumps(layout.description)},')
    lines.append('  "turbines": [')
    for k, (name, (x, y, z)) in enumerate(zip(layout.names, layout.positions_d)):
        entry = f'{{"name": {json.dumps(name)}, "x": {_number(x)}, "y": {_number(y)}'
        entry += (f', "z": {_number(z)}}}' if z else "}")
        lines.append("    " + entry + ("," if k < len(layout.names) - 1 else ""))
    lines.append("  ],")
    lines.append('  "neighbors": [')
    edges = layout.edges or []
    for k, (a, b) in enumerate(edges):
        pair = f"[{json.dumps(layout.names[a])}, {json.dumps(layout.names[b])}]"
        lines.append("    " + pair + ("," if k < len(edges) - 1 else ""))
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _number(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


# --------------------------------------------------------------------------- #
# Graph observation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GraphObsSpec:
    """Sizes the actor needs to split a flat graph observation."""
    slots: int                      # M: padded nodes per ego graph (ego is slot 0)
    node_features: int = NODE_FEATURES
    edge_features: int = EDGE_FEATURES
    local_features: int = LOCAL_FEATURES

    @property
    def size(self):
        m = self.slots
        return (self.local_features + m * self.node_features
                + m * m * self.edge_features + m * m)

    def split(self, obs):
        """Split [..., size] (numpy or torch) into local, nodes, edges, adjacency."""
        m, f, e, l = self.slots, self.node_features, self.edge_features, self.local_features
        if obs.shape[-1] != self.size:
            raise ValueError(f"Graph observation has {obs.shape[-1]} values, expected {self.size}.")
        lead = obs.shape[:-1]
        a = l + m * f
        b = a + m * m * e
        return (obs[..., :l],
                obs[..., l:a].reshape(*lead, m, f),
                obs[..., a:b].reshape(*lead, m, m, e),
                obs[..., b:].reshape(*lead, m, m))


class TurbineGraph:
    """Fixed neighbour topology; edge features recomputed from wind every step.

    Message direction: adjacency[a, b] = 1 means slot b sends to slot a. Edge
    features for b -> a are measured in the wind frame of the receiver's
    neighbourhood:
      0  downstream distance of a from b / 10 D (positive: b is upwind of a)
      1  crosswind offset of a from b / 10 D
      2  wake proxy in [0, 1]: Jensen-style deficit shape of b's wake at a
      3  1 for the self-loop, else 0
    """

    def __init__(self, positions_d, edges, hops=1, messages="both"):
        self.positions_d = np.asarray(positions_d, dtype=float)
        if self.positions_d.ndim != 2 or self.positions_d.shape[1] < 2:
            raise ValueError("positions_d must be (N, 2) or (N, 3).")
        self.num_turbines = n = len(self.positions_d)
        if edges is None:
            raise ValueError(
                "The graph layer needs a layout file with a \"neighbors\" list. Create one with: "
                "python turbine_graph.py make <positions.json> <layout.json>")
        self.edges = sorted({(min(a, b), max(a, b)) for a, b in edges})
        if any(not (0 <= a < n and 0 <= b < n) or a == b for a, b in self.edges):
            raise ValueError("Neighbor links must join two different existing turbines.")
        if int(hops) != hops or hops < 1:
            raise ValueError("graph hops must be a positive integer.")
        if messages not in ("both", "upwind"):
            raise ValueError('graph messages must be "both" or "upwind".')
        self.hops, self.messages = int(hops), messages
        self.adjacency = np.zeros((n, n), dtype=bool)
        for a, b in self.edges:
            self.adjacency[a, b] = self.adjacency[b, a] = True
        # Ego graph of each turbine: itself first, then by hop count and index.
        self.ego_nodes = [self._ego(i) for i in range(n)]
        self.spec = GraphObsSpec(max(len(nodes) for nodes in self.ego_nodes))
        m = self.spec.slots
        self._slots = np.zeros((n, m), dtype=int)
        self._valid = np.zeros((n, m), dtype=bool)
        self._static_adjacency = np.zeros((n, m, m), dtype=bool)
        for i, nodes in enumerate(self.ego_nodes):
            k = len(nodes)
            self._slots[i, :k] = nodes
            self._valid[i, :k] = True
            induced = self.adjacency[np.ix_(nodes, nodes)].copy()
            np.fill_diagonal(induced, True)
            self._static_adjacency[i, :k, :k] = induced

    @classmethod
    def from_layout(cls, layout, hops=1, messages="both"):
        return cls(layout.positions_d, layout.edges, hops=hops, messages=messages)

    def _ego(self, start):
        depth = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if depth[node] == self.hops:
                continue
            for other in np.flatnonzero(self.adjacency[node]):
                if other not in depth:
                    depth[int(other)] = depth[node] + 1
                    queue.append(int(other))
        return sorted(depth, key=lambda node: (depth[node], node))

    def describe(self):
        """JSON-friendly summary stored in checkpoints and run configs."""
        return {"edges": [list(edge) for edge in self.edges], "hops": self.hops,
                "messages": self.messages, "slots": self.spec.slots,
                "node_features": self.spec.node_features,
                "edge_features": self.spec.edge_features}

    # ------------------------------------------------------------------ #

    def observe(self, local_obs):
        """(N, 3) raw local observations -> (N, spec.size) actor observations."""
        local_obs = np.asarray(local_obs, dtype=np.float64)
        n, m = self.num_turbines, self.spec.slots
        if local_obs.shape != (n, LOCAL_FEATURES):
            raise ValueError(f"Expected local observations with shape ({n}, 3).")
        speed, direction, yaw = local_obs.T
        out = np.zeros((n, self.spec.size), dtype=np.float32)
        for i in range(n):
            slots, valid = self._slots[i], self._valid[i]
            nodes = slots[valid]
            # Neighbourhood wind direction: circular mean of measured directions.
            radians = np.deg2rad(direction[nodes])
            frame = np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
            flow, cross = self._wind_axes(frame)
            xy = self.positions_d[slots, :2]
            nodes_f = np.zeros((m, NODE_FEATURES))
            nodes_f[valid] = self._node_features(speed[nodes], direction[nodes], yaw[nodes],
                                                 xy[valid] - xy[0], frame, flow, cross)
            # delta[a, b] = position of receiver a relative to sender b.
            delta = xy[:, None, :] - xy[None, :, :]
            downstream = delta @ flow
            crosswind = delta @ cross
            adjacency = self._static_adjacency[i].copy()
            if self.messages == "upwind":
                adjacency &= (downstream > -UPWIND_TOLERANCE_D) | np.eye(m, dtype=bool)
            edges_f = self._edge_features(downstream, crosswind)
            edges_f[~adjacency] = 0.0
            out[i] = np.concatenate((local_obs[i], nodes_f.ravel(), edges_f.ravel(),
                                     adjacency.ravel().astype(float)))
        return out

    @staticmethod
    def _wind_axes(direction_deg):
        # Meteorological "from" direction -> unit vector the flow travels along.
        # 270 deg (from west) flows toward +x, matching wind_farm_env._measure.
        angle = np.deg2rad(270.0 - direction_deg)
        flow = np.array([np.cos(angle), np.sin(angle)])
        return flow, np.array([-flow[1], flow[0]])

    @staticmethod
    def _node_features(speed, direction, yaw, offset_d, frame, flow, cross):
        """speed/10, yaw/180, sin/cos of direction relative to the neighbourhood
        frame, and the node's downstream/crosswind offset from the ego turbine."""
        relative = np.deg2rad(direction - frame)
        return np.column_stack((speed / 10.0, yaw / 180.0, np.sin(relative), np.cos(relative),
                                offset_d @ flow / LENGTH_SCALE_D, offset_d @ cross / LENGTH_SCALE_D))

    @staticmethod
    def _edge_features(downstream, crosswind):
        m = downstream.shape[0]
        positive = np.maximum(downstream, 0.0)
        # Wake radius in D grows linearly from the rotor radius (0.5 D).
        radius = 0.5 + WAKE_K * positive
        wake = np.where(downstream > 1e-6,
                        np.exp(-0.5 * (crosswind / radius) ** 2) / (1.0 + 2.0 * WAKE_K * positive) ** 2,
                        0.0)
        features = np.stack((downstream / LENGTH_SCALE_D, crosswind / LENGTH_SCALE_D, wake,
                             np.eye(m)), axis=-1)
        return features


# --------------------------------------------------------------------------- #
# Command line helper
# --------------------------------------------------------------------------- #

def _ascii_map(layout):
    xy = layout.positions_d[:, :2]
    lines = []
    for name, (x, y) in zip(layout.names, xy):
        linked = [layout.names[b if a == i else a] for i in [layout.names.index(name)]
                  for a, b in (layout.edges or []) if i in (a, b)]
        lines.append(f"  {name:>6}  x={x:7.2f} D  y={y:7.2f} D  neighbours: {', '.join(linked) or '(none)'}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make", help="Write a layout file with suggested grid-line neighbours.")
    make.add_argument("source", type=Path, help="Any layout file, including a bare [[x, y], ...] array.")
    make.add_argument("output", type=Path)
    make.add_argument("--diagonals", action="store_true", help="Also link diagonal neighbours.")
    make.add_argument("--description", default="")
    check = commands.add_parser("check", help="Validate a layout file and print its graph.")
    check.add_argument("layout", type=Path)
    check.add_argument("--hops", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "make":
        layout = load_layout(args.source)
        layout.edges = grid_neighbors(layout.positions_d, diagonals=args.diagonals)
        layout.description = args.description or layout.description
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(layout_to_json(layout))
        print(f"Wrote {args.output} with {len(layout.edges)} links. Edit the \"neighbors\" list as needed.")
        print(_ascii_map(layout))
        return
    layout = load_layout(args.layout)
    print(f"{args.layout}: {len(layout.names)} turbines, "
          f"{'no' if layout.edges is None else len(layout.edges)} neighbour links")
    print(_ascii_map(layout))
    if layout.edges is not None:
        graph = TurbineGraph.from_layout(layout, hops=args.hops)
        print(f"hops={args.hops}: {graph.spec.slots} nodes per ego graph, "
              f"actor observation size {graph.spec.size}")
        for name, nodes in zip(layout.names, graph.ego_nodes):
            print(f"  {name:>6} sees {', '.join(layout.names[k] for k in nodes[1:]) or '(nothing)'}")


if __name__ == "__main__":
    main()
