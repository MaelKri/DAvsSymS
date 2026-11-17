"""Réseaux : actor gaussien (BC, DAgger, SAC, RLPD) et ensemble de critiques avec LayerNorm (SAC, RLPD)."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common.hparams as hp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clamp_min(0.1))

    def forward(self, x):
        return (x - self.mean) / self.std


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, mean, std):
        super().__init__()
        self.norm = Normalizer(mean, std)
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hp.HIDDEN),
            nn.ReLU(),
            nn.Linear(hp.HIDDEN, hp.HIDDEN),
            nn.ReLU(),
            nn.Linear(hp.HIDDEN, 2 * act_dim),
        )
        with torch.no_grad():
            self.body[-1].weight.mul_(0.01)
            self.body[-1].bias.zero_()

    def forward(self, obs):
        mean, log_std = self.body(self.norm(obs)).chunk(2, dim=-1)
        return mean, log_std.clamp(-5, 2)

    def deterministic(self, obs):
        return torch.tanh(self(obs)[0])

    def sample(self, obs):
        mean, log_std = self(obs)
        u = mean + log_std.exp() * torch.randn_like(mean)
        logp = torch.distributions.Normal(mean, log_std.exp()).log_prob(u)
        logp = logp - 2 * (math.log(2) - u - F.softplus(-2 * u))
        return torch.tanh(u), logp.sum(-1)

    @torch.no_grad()
    def act(self, x):
        return self.deterministic(torch.as_tensor(x, device=DEVICE)[None])[0].cpu().numpy()


class EnsembleLinear(nn.Module):
    def __init__(self, n, inp, out):
        super().__init__()
        b = 1 / math.sqrt(inp)
        self.weight = nn.Parameter(torch.empty(n, inp, out).uniform_(-b, b))
        self.bias = nn.Parameter(torch.empty(n, 1, out).uniform_(-b, b))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class Critics(nn.Module):
    def __init__(self, obs_dim, act_dim, mean, std):
        super().__init__()
        self.norm, n, h = Normalizer(mean, std), hp.N_CRITICS, hp.HIDDEN
        self.l1 = EnsembleLinear(n, obs_dim + act_dim, h)
        self.l2 = EnsembleLinear(n, h, h)
        self.l3 = EnsembleLinear(n, h, 1)
        self.ln = nn.LayerNorm(h, elementwise_affine=False)

    def forward(self, obs, act):
        x = torch.cat([self.norm(obs), act], dim=-1).expand(hp.N_CRITICS, -1, -1)
        h = F.relu(self.ln(self.l1(x)))
        h = F.relu(self.ln(self.l2(h)))
        return self.l3(h).squeeze(-1)


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)
