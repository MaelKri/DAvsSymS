"""Fait rouler l'expert (avec bruit DART optionnel sur le braquage) et sauvegarde ses transitions dans data/."""

import argparse
import json

import numpy as np

import common.hparams as hp
from common.stk import env_action, expert_action, features, make_env, progress
from experts import Team4Expert

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=1)
p.add_argument("--noise", type=float, default=0.0)
args = p.parse_args()

env, expert = make_env(), Team4Expert()
keys = ("obs", "actions", "labels", "next_obs", "dist", "next_dist", "terminated", "episode")
buf = {k: [] for k in keys}
for ep in range(args.episodes):
    rng = np.random.default_rng(ep)
    obs, _ = env.reset(seed=0)
    expert.reset()
    for t in range(hp.MAX_STEPS):
        label = expert_action(expert(obs))
        a = label.copy()
        a[1] = np.clip(a[1] + args.noise * rng.normal(), -1, 1)
        obs2, _, terminated, truncated, _ = env.step(env_action(a))
        for k, v in zip(keys, (features(obs), a, label, features(obs2), progress(obs), progress(obs2), terminated, ep)):
            buf[k].append(v)
        obs = obs2
        if terminated or truncated:
            break
    print(f"épisode {ep} : {t + 1} pas, tour fini = {terminated}")

data = {k: np.asarray(v) for k, v in buf.items()}
meta = dict(expert="team4", track=hp.TRACK, episodes=args.episodes, noise=args.noise)
out = f"data/team4_{hp.TRACK}_ep{args.episodes}_noise{args.noise:g}.npz"
np.savez_compressed(out, meta=json.dumps(meta), **data)
print(f"{len(data['obs'])} transitions -> {out}")
