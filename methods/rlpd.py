"""SAC, RLPD-init (buffer pré-rempli) et RLPD (symmetric sampling), avec le même agent."""

import copy

import numpy as np
import torch

import common.hparams as hp
from common.nets import tensor
from common.stk import ACT_DIM, env_action, features, progress, reward


class Buffer:
    def __init__(self, obs_dim, act_dim, capacity):
        self.o, self.o2 = np.zeros((2, capacity, obs_dim), np.float32)
        self.a = np.zeros((capacity, act_dim), np.float32)
        self.r, self.d = np.zeros((2, capacity), np.float32)
        self.n, self.ptr, self.cap = 0, 0, capacity

    def add(self, o, a, r, o2, d):
        i = (self.ptr + np.arange(len(r))) % self.cap
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr, self.n = (self.ptr + len(r)) % self.cap, min(self.n + len(r), self.cap)

    def sample(self, k):
        i = np.random.randint(self.n, size=k)
        return self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i]


class SAC:
    def __init__(self, actor, critics):
        self.actor, self.critics = actor, critics
        self.target = copy.deepcopy(critics).requires_grad_(False)
        self.log_alpha = torch.zeros(1, device=tensor(0).device, requires_grad=True)
        self.target_entropy = -float(ACT_DIM)
        self.opt_actor = torch.optim.Adam(actor.parameters(), lr=hp.LR)
        self.opt_critics = torch.optim.Adam(critics.parameters(), lr=hp.LR)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=hp.LR)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def update_critics(self, o, a, r, o2, d):
        with torch.no_grad():
            a2, logp2 = self.actor.sample(o2)
            q2 = self.target(o2, a2)
            q2 = q2[torch.randperm(hp.N_CRITICS, device=q2.device)[: hp.N_MIN]].min(0).values
            y = r + hp.GAMMA * (1 - d) * (q2 - self.alpha * logp2)
        loss = ((self.critics(o, a) - y) ** 2).mean(1).sum()
        self.opt_critics.zero_grad()
        loss.backward()
        self.opt_critics.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(self.target.parameters()), list(self.critics.parameters()), hp.TAU)

    def update_actor(self, o):
        a, logp = self.actor.sample(o)
        loss = (self.alpha * logp - self.critics(o, a).mean(0)).mean()
        self.opt_actor.zero_grad()
        loss.backward()
        self.opt_actor.step()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

    @torch.no_grad()
    def q_vs_mc(self, lap, n_samples=8):
        o, a = tensor(lap["obs"]), tensor(lap["actions"])
        q = self.critics(o, a).mean(0).cpu().numpy()
        ent = -torch.stack([self.actor.sample(o)[1] for _ in range(n_samples)]).mean(0).cpu().numpy()
        T = len(q)
        term = np.zeros(T, bool)
        term[-1] = lap["finished"]
        r = reward(lap["dist"][:-1], lap["dist"][1:], term)
        G, g, alpha = np.zeros(T, np.float32), 0.0, self.alpha.item()
        for t in range(T - 1, -1, -1):
            g = r[t] if (t == T - 1) else r[t] + hp.GAMMA * (alpha * ent[t + 1] + g)
            G[t] = g
        keep = slice(None) if lap["finished"] else slice(0, max(T - 300, 1))
        return dict(
            q_mean=float(q[keep].mean()),
            mc_return=float(G[keep].mean()),
            q_minus_mc=float((q - G)[keep].mean()),
            alpha=alpha,
        )


def train(method, actor, critics, data, env, evaluate, log):
    agent = SAC(actor, critics)
    obs_dim, act_dim = data["obs"].shape[1], ACT_DIM
    online = Buffer(obs_dim, act_dim, hp.BUFFER_SIZE)
    offline = None
    expert_data = (data["obs"], data["actions"], data["rewards"], data["next_obs"], data["terminated"])
    if method == "rlpd_init":
        online.add(*expert_data)
    elif method == "rlpd":
        offline = Buffer(obs_dim, act_dim, len(data["obs"]))
        offline.add(*expert_data)
    n_labels = 0 if method == "sac" else len(data["obs"])
    start_steps = hp.SAC_START_STEPS if method == "sac" else 0

    def batch():
        if offline is None:
            b = online.sample(hp.BATCH_SIZE)
        else:
            k = int(hp.BATCH_SIZE * hp.OFFLINE_RATIO)
            b = [np.concatenate(x) for x in zip(offline.sample(k), online.sample(hp.BATCH_SIZE - k))]
        return [tensor(x) for x in b]

    def evaluate_and_log(env_steps, grad_steps):
        lap = evaluate()
        log(lap, env_steps=env_steps, expert_labels=n_labels, grad_steps=grad_steps, **agent.q_vs_mc(lap))

    env_steps = grad_steps = 0
    next_eval = 0
    while env_steps < hp.TOTAL_STEPS:
        if env_steps >= next_eval:
            evaluate_and_log(env_steps, grad_steps)
            next_eval = env_steps + hp.EVAL_EVERY
        obs, _ = env.reset(seed=0)
        for _ in range(hp.MAX_STEPS):
            x = features(obs)
            if env_steps < start_steps:
                a = np.random.uniform(-1, 1, act_dim).astype(np.float32)
            else:
                with torch.no_grad():
                    a = agent.actor.sample(tensor(x)[None])[0][0].cpu().numpy()
            obs2, _, terminated, truncated, _ = env.step(env_action(a))
            r = reward(progress(obs), progress(obs2), terminated)
            online.add(x[None], a[None], np.array([r]), features(obs2)[None], np.array([float(terminated)]))
            obs, env_steps = obs2, env_steps + 1
            if online.n >= hp.BATCH_SIZE and env_steps > start_steps:
                for _ in range(hp.UTD):
                    agent.update_critics(*batch())
                    grad_steps += 1
                agent.update_actor(batch()[0])
            if terminated or truncated or env_steps >= hp.TOTAL_STEPS:
                break
    evaluate_and_log(env_steps, grad_steps)
