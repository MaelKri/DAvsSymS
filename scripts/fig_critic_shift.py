"""Figure du critic shift : Q du critique et Q mesuré par rejeu, en fonction du braquage."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import common.hparams as hp
from common.analysis import COLOR, COND, FIG_SEED, LABEL, RUNS, T_STATE, save
from common.nets import DEVICE, Actor, Critics, tensor
from common.stk import ACT_DIM, env_action, features, load_dataset, make_env, obs_stats, progress, reward

CONDS = ("sac", "rlpd_s0", "rlpd")
DIMS = {1: "braquage", 2: "frein"}
VALUES = np.linspace(-1, 1, 11)
FINE = np.linspace(-1, 1, 81)
FLAT_STATES = (150, T_STATE, 450)
CACHE = RUNS / "eval/critic_profiles.npz"
mean, std = obs_stats()


def load(c):
    run = RUNS / COND[c][0] / f"seed{FIG_SEED}"
    actor = Actor(len(mean), ACT_DIM, mean, std).to(DEVICE)
    actor.load_state_dict(torch.load(run / "actor.pt", map_location=DEVICE))
    critics = Critics(len(mean), ACT_DIM, mean, std).to(DEVICE)
    critics.load_state_dict(torch.load(run / "critics.pt", map_location=DEVICE))
    return actor, critics, pd.read_csv(run / "log.csv")["alpha"].iloc[-1]


def measured_q(env, actor, alpha, t_state, dim=None, value=None):
    obs, _ = env.reset(seed=0)
    rewards, states = [], []
    for t in range(hp.MAX_STEPS):
        x = features(obs)
        a = actor.act(x)
        if t == t_state:
            s_t, a_t, d_t = x, a.copy(), progress(obs)
            if dim is not None:
                a[dim] = value
        obs2, _, terminated, truncated, _ = env.step(env_action(a))
        if t >= t_state:
            rewards.append(float(reward(progress(obs), progress(obs2), terminated)))
            states.append(features(obs2))
        obs = obs2
        if terminated or truncated:
            break
    torch.manual_seed(0)
    with torch.no_grad():
        ent = -torch.stack([actor.sample(tensor(np.array(states)))[1] for _ in range(32)]).mean(0).cpu().numpy()
    g = 0.0
    for k in range(len(rewards) - 1, -1, -1):
        g = rewards[k] + (0.0 if k == len(rewards) - 1 else hp.GAMMA * (alpha * ent[k] + g))
    return g, s_t, a_t, d_t


# --- 1. rejeux (mis en cache)
if CACHE.exists():
    cache = dict(np.load(CACHE))
else:
    env, cache = make_env(), {}
    for c in CONDS:
        actor, _, alpha = load(c)
        for t in FLAT_STATES:
            cache[f"{c}_flat_{t}"] = np.array([measured_q(env, actor, alpha, t, 1, v)[0] for v in VALUES])
        g0, s_t, a_t, d_t = measured_q(env, actor, alpha, T_STATE)
        cache.update({f"{c}_g0": g0, f"{c}_s": s_t, f"{c}_a": a_t, f"{c}_d": d_t})
        for dim in DIMS:
            cache[f"{c}_true_{dim}"] = np.array([measured_q(env, actor, alpha, T_STATE, dim, v)[0] for v in VALUES])
        print(f"replays {c} done")
    np.savez(CACHE, **cache)

# --- 2. platitude et déterminisme
for c in CONDS:
    for t in FLAT_STATES:
        print(f"flatness {c} t={t} = {np.ptp(cache[f'{c}_flat_{t}']):.3f}")
for c in CONDS:
    print(f"determinism {c} = {np.ptp(cache[f'{c}_true_2'][VALUES <= 0]):.3g}")

# --- 3. profils à T_STATE
fig, axes = plt.subplots(1, len(CONDS), figsize=(12, 3.6))
for ax, c in zip(axes, CONDS):
    _, critics, _ = load(c)
    s_t, a_t, d_t = cache[f"{c}_s"], cache[f"{c}_a"], float(cache[f"{c}_d"])
    A = np.repeat(a_t[None], len(FINE), 0)
    A[:, 1] = FINE
    with torch.no_grad():
        q = critics(tensor(np.repeat(s_t[None], len(FINE), 0)), tensor(A)).cpu().numpy()
    ax.fill_between(FINE, q.min(0), q.max(0), color=COLOR[c], alpha=0.2, label="ensemble (min–max)")
    ax.plot(FINE, q.mean(0), color=COLOR[c], lw=2, label="critic Q")
    ax.plot(VALUES, cache[f"{c}_true_1"], "k.-", label="measured Q")
    ax.axvline(a_t[1], color="0.4", ls="--", lw=1, label="actor action")
    if COND[c][3]:
        data = load_dataset(COND[c][3])[0]
        near = data["actions"][np.abs(data["dist"] - d_t) < 3, 1]
        ax.plot(
            near,
            np.full(len(near), 0.03),
            "|",
            color="#2E8B3E",
            ms=12,
            mew=2,
            transform=ax.get_xaxis_transform(),
            label="steering in the data (± 3 m)",
        )
    ax.set(xlabel=r"steering imposed at $s_t$", title=LABEL[c])
axes[0].set_ylabel(r"$Q(s_t, a)$")
axes[-1].legend(fontsize=7)
save(fig, "fig_critic_profiles")
