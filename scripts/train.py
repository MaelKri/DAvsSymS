"""Entraîne un agent (méthode, seed, dataset) et écrit ses logs, ses poids et son dernier tour dans runs/."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import common.hparams as hp
from common.nets import DEVICE, Actor, Critics
from common.stk import ACT_DIM, expert_action, features, load_dataset, make_env, obs_stats, rollout
from experts import Team4Expert
from methods import imitation, rlpd

p = argparse.ArgumentParser()
p.add_argument("--method", required=True, choices=["expert", "bc", "dagger", "sac", "rlpd", "rlpd_init"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--dataset", default="data/team4_abyss_ep1_noise0.npz")
p.add_argument("--out", default="runs")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
out = Path(args.out) / args.method / f"seed{args.seed}"
out.mkdir(parents=True, exist_ok=True)
env, expert = make_env(), Team4Expert()


def save_lap(lap):
    np.savez(out / "final_lap.npz", **{k: lap[k] for k in ("obs", "actions", "dist", "offset", "steps", "finished")})


if args.method == "expert":
    lap = rollout(env, lambda obs: expert_action(expert(obs)))
    save_lap(lap)
    print(f"expert : {lap['steps']} pas, tour fini = {lap['finished']}")
    raise SystemExit

data, meta = load_dataset(args.dataset)
mean, std = obs_stats()
obs_dim = data["obs"].shape[1]
actor = Actor(obs_dim, ACT_DIM, mean, std).to(DEVICE)
(out / "config.json").write_text(json.dumps(dict(vars(args), dataset_meta=meta, hparams=hp.as_dict()), indent=2))

last_lap = {}


def evaluate():
    last_lap["lap"] = rollout(env, lambda obs: actor.act(features(obs)))
    return last_lap["lap"]


t0, writer = time.time(), None
log_file = open(out / "log.csv", "w", newline="")


def log(lap, **values):
    global writer
    row = dict(
        wall=round(time.time() - t0, 1),
        **values,
        lap_steps=lap["steps"],
        finished=int(lap["finished"]),
        final_dist=np.nan if lap["finished"] else lap["dist"][-1],
    )
    if writer is None:
        writer = csv.DictWriter(log_file, fieldnames=list(row), extrasaction="ignore")
        writer.writeheader()
    writer.writerow(row)
    log_file.flush()
    print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))


if args.method == "bc":
    imitation.bc(actor, data, evaluate, log)
elif args.method == "dagger":
    imitation.dagger(actor, data, env, expert, evaluate, log)
else:
    critics = Critics(obs_dim, ACT_DIM, mean, std).to(DEVICE)
    rlpd.train(args.method, actor, critics, data, env, evaluate, log)
    torch.save(critics.state_dict(), out / "critics.pt")

torch.save(actor.state_dict(), out / "actor.pt")
save_lap(last_lap["lap"])
print(f"terminé en {time.time() - t0:.0f} s -> {out}")
