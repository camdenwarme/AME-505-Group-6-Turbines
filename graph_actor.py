"""Graph-attention encoder in front of light_mappo's actor MLP.

Only the actor's feature extractor (R_Actor.base) changes. The recurrent layer
(--use_recurrent_policy), the tanh-bounded action head, PPO, GAE and both
critics are untouched. Per turbine:

    flat graph observation (turbine_graph.TurbineGraph.observe)
      -> node embedding -> L x EdgeGATLayer (dense, masked, edge features)
      -> ego embedding (slot 0)
      -> concat with the original 3 local features
      -> light_mappo MLPBase (same layer_N / hidden_size / feature norm options)
      -> [GRU if rmappo] -> TanhBoundedActionHead

The attention is GATv2-style: score(b -> a) = att . LeakyReLU(W_dst h_a + W_src h_b
+ W_edge e_ba), softmax over the senders allowed by the adjacency. Because the
edge features carry downstream distance and a wake proxy, the layer can learn
to weight upwind neighbours highly and ignore downwind ones, and that weighting
changes automatically as the wind direction changes.
"""

import torch
from torch import nn
from torch.nn import functional as F

from algorithms.utils.mlp import MLPBase

from turbine_graph import GraphObsSpec


class EdgeGATLayer(nn.Module):
    """One dense multi-head attention message-passing layer with edge features."""

    def __init__(self, dim, edge_dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("graph hidden size must be divisible by the number of heads.")
        self.heads, self.head_dim = heads, dim // heads
        self.dst = nn.Linear(dim, dim)
        self.src = nn.Linear(dim, dim, bias=False)
        self.edge = nn.Linear(edge_dim, dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.att)
        self.value = nn.Linear(dim, dim)
        self.edge_value = nn.Linear(edge_dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edges, adjacency):
        # h: [B, M, D], edges: [B, M(dst), M(src), E], adjacency: [B, M, M] (dst, src)
        batch, m, _ = h.shape
        split = (batch, m, m, self.heads, self.head_dim)
        hidden = (self.dst(h)[:, :, None] + self.src(h)[:, None, :] + self.edge(edges))
        scores = (F.leaky_relu(hidden.view(split), 0.2) * self.att).sum(-1)       # [B, M, M, H]
        # Every row keeps its self-loop, so padded slots never produce empty softmaxes.
        allowed = (adjacency > 0.5) | torch.eye(m, dtype=torch.bool, device=h.device)
        scores = scores.masked_fill(~allowed[..., None], float("-inf"))
        alpha = torch.softmax(scores, dim=2)
        messages = (self.value(h)[:, None, :] + self.edge_value(edges)).view(split)
        aggregated = (alpha[..., None] * messages).sum(dim=2).reshape(batch, m, -1)
        return self.norm(h + F.relu(self.out(aggregated)))


class GraphActorBase(nn.Module):
    """Drop-in replacement for R_Actor.base; returns [B, hidden_size] features."""

    def __init__(self, args, spec, graph_hidden=64, heads=4, layers=1):
        super().__init__()
        if not isinstance(spec, GraphObsSpec):
            raise TypeError("spec must be a GraphObsSpec.")
        if int(layers) != layers or layers < 1:
            raise ValueError("graph layers must be a positive integer.")
        self.spec = spec
        self.hidden_size = args.hidden_size
        self.embed = nn.Sequential(nn.Linear(spec.node_features, graph_hidden), nn.ReLU(),
                                   nn.LayerNorm(graph_hidden))
        self.layers = nn.ModuleList(EdgeGATLayer(graph_hidden, spec.edge_features, heads)
                                    for _ in range(int(layers)))
        # Reuse the upstream MLP so --layer_N, --hidden_size, --use_ReLU,
        # --use_orthogonal and --use_feature_normalization keep their meaning.
        self.mlp = MLPBase(args, (spec.local_features + graph_hidden,))

    def encode(self, obs):
        """Return (local features, ego graph embedding)."""
        local, nodes, edges, adjacency = self.spec.split(obs)
        valid = adjacency.diagonal(dim1=-2, dim2=-1)[..., None]        # self-loop marks real nodes
        h = self.embed(nodes) * valid
        for layer in self.layers:
            h = layer(h, edges, adjacency) * valid
        return local, h[:, 0]

    def forward(self, obs):
        local, ego = self.encode(obs)
        return self.mlp(torch.cat((local, ego), dim=-1))


def install_graph_actor(policy, spec, args, graph_hidden=64, heads=4, layers=1):
    """Swap the actor's MLPBase for GraphActorBase and rebuild its optimizer.

    The RNN and action head keep their parameters. The optimizer is rebuilt
    because it was created over the replaced base's parameters.
    """
    device = next(policy.actor.parameters()).device
    policy.actor.base = GraphActorBase(args, spec, graph_hidden, heads, layers).to(device)
    policy.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=policy.lr,
                                              eps=policy.opti_eps, weight_decay=policy.weight_decay)
