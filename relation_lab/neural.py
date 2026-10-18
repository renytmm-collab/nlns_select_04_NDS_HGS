"""Independent PyTorch model; no upstream source copied, no runtime compilation."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import require
from .vrp import action_mask


class Encoder(nn.Module):
    def __init__(self, width, heads, layers):
        super().__init__()
        self.input = nn.Linear(6, width)
        self.neighbor = nn.Linear(2 * width, width)
        self.route = nn.Linear(2 * width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.attention = nn.ModuleList([nn.MultiheadAttention(width, heads, dropout=0, batch_first=True) for _ in range(layers)])
        self.feedforward = nn.ModuleList([nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2 * layers)])

    def forward(self, state):
        require(set(state) == {"coordinates", "demands", "capacity", "routes"}, "policy input must contain ordinary fields only")
        device = self.input.weight.device
        xy, demands, capacity, routes = (state[k] for k in ("coordinates", "demands", "capacity", "routes"))
        n = len(demands)
        loads = [0.] * n
        neighbors = [[0, 0] for _ in range(n)]
        membership = list(range(n))
        for ri, route in enumerate(routes):
            load = sum(demands[i] for i in route) / capacity
            for j, node in enumerate(route):
                loads[node] = load
                neighbors[node] = [route[j - 1] if j else 0, route[j + 1] if j + 1 < len(route) else 0]
                membership[node] = ri + 1
        membership[0] = 0
        features = [[xy[i][0], xy[i][1], demands[i] / capacity, loads[i], math.dist(xy[i], xy[0]), float(i == 0)] for i in range(n)]
        h = self.input(torch.tensor(features, dtype=self.input.weight.dtype, device=device))
        neighbors = torch.tensor(neighbors, dtype=torch.long, device=device)
        h = self.norm1(h + F.relu(self.neighbor(torch.cat((h, h[neighbors].mean(1)), -1))))
        # Membership equality avoids route-number embeddings and preserves route permutation symmetry.
        membership = torch.tensor(membership, device=device)
        same = (membership[:, None] == membership[None, :]).to(h.dtype)
        pooled = (same @ h) / same.sum(-1, keepdim=True)
        h = self.norm2(h + F.relu(self.route(torch.cat((h, pooled), -1))))
        for i, (attention, ff) in enumerate(zip(self.attention, self.feedforward)):
            a, _ = attention(h[None], h[None], h[None], need_weights=False)
            h = self.norms[2*i](h + a[0])
            h = self.norms[2*i+1](h + ff(h))
        return h


class Policy(nn.Module):
    def __init__(self, width=64, heads=4, layers=2):
        super().__init__()
        self.spec = {"width": width, "heads": heads, "layers": layers}
        self.encoder = Encoder(width, heads, layers)
        self.initial = nn.Linear(width, width)
        self.gru = nn.GRUCell(width, width)
        self.key = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)

    def decode(self, h, k, generator=None, greedy=False, forced=None):
        require(type(k) is int and 1 <= k < len(h), "destroy size must be in 1..N")
        require(forced is None or len(forced) == k, "forced sequence length")
        hidden = torch.tanh(self.initial(h[1:].mean(0)))
        previous = h[0]
        keys = self.key(h)
        selected, logprobs, distributions = [], [], []
        for step in range(k):
            hidden = self.gru(previous[None], hidden[None])[0]
            logits = 10 * torch.tanh((keys @ self.query(hidden)) / math.sqrt(h.shape[-1]))
            mask = torch.tensor(action_mask(len(h)-1, selected), device=h.device)
            logp = torch.log_softmax(logits.masked_fill(mask, -torch.inf), dim=-1)
            if forced is not None:
                node = forced[step]
                require(type(node) is int and 0 <= node < len(h) and not bool(mask[node]), "illegal forced action")
            elif greedy:
                node = int(logp.argmax().item())
            else:
                require(generator is not None, "explicit independent policy generator required")
                node = int(torch.multinomial(logp.detach().exp().cpu(), 1, generator=generator).item())
            selected.append(node)
            logprobs.append(logp[node])
            distributions.append(logp)
            previous = h[node]
        return selected, torch.stack(logprobs).sum(), distributions

    def forward(self, state, k, generator=None, greedy=False):
        return self.decode(self.encoder(state), k, generator, greedy)


class Model(Policy):
    def __init__(self, **spec):
        super().__init__(**spec)
        # Constructed after the policy for every training group, including lambda=0.
        width = self.spec["width"]
        self.relation = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 1))

    def pair_logits(self, h):
        customers = h[1:]
        a, b = customers[:, None], customers[None, :]
        return self.relation(torch.cat((a+b, a*b), dim=-1)).squeeze(-1)


def relation_objective(model, h, target):
    logits = model.pair_logits(h)
    target = torch.tensor(target, dtype=logits.dtype, device=logits.device)
    index = torch.triu_indices(len(target), len(target), offset=1, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[index[0], index[1]], target[index[0], index[1]])


def create_model(spec, seed, device="cpu"):
    # Initialization does not consume the caller's policy/data/global RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = Model(**spec)
    return model.to(device)


def export_policy(model):
    with torch.random.fork_rng(devices=[]):
        policy = Policy(**model.spec)
    policy.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("relation.")})
    return policy.eval()
