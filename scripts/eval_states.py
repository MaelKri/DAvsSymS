"""Fait conduire les agents d'imitation avec l'expert en labelliseur, pour la figure du policy shift."""

import numpy as np
import torch

from common.analysis import DART_DATASET, RUNS, run_dirs
from common.nets import DEVICE, Actor, tensor
from common.stk import ACT_DIM, features, heading_error, load_dataset, make_env, obs_stats, rollout
from experts import Team4Expert

mean, std = obs_stats()
env, expert = make_env(), Team4Expert()


def norm(x):
    return (x - mean) / np.maximum(std, 0.1)


def load_actor(run):
    actor = Actor(len(mean), ACT_DIM, mean, std).to(DEVICE)
    actor.load_state_dict(torch.load(run / "actor.pt", map_location=DEVICE))
    return actor


out = RUNS / "eval/policy"
out.mkdir(parents=True, exist_ok=True)
ref = tensor(norm(load_dataset(DART_DATASET)[0]["obs"]))
for c in ("bc_s0", "dart", "dagger"):
    for run in run_dirs(c):
        actor = load_actor(run)
        lap = rollout(env, lambda obs: actor.act(features(obs)), labeller=expert)
        nn_dist = torch.cdist(tensor(norm(lap["obs"])), ref).min(1).values.cpu().numpy()
        np.savez(
            out / f"{c}_{run.name}.npz",
            obs=lap["obs"],
            actions=lap["actions"],
            labels=lap["labels"],
            offset=lap["offset"],
            heading=heading_error(lap["obs"]),
            dist=lap["dist"],
            nn_dist=nn_dist,
            steps=lap["steps"],
            finished=lap["finished"],
        )
        print(f"{c} {run.name}: {lap['steps']} steps, finished = {lap['finished']}")
