"""Interface avec SuperTuxKart : environnement, features, actions, récompense, tour de piste et datasets."""

import json
import warnings

import gymnasium as gym
import numpy as np
from pystk2_gymnasium import AgentSpec

import common.hparams as hp

warnings.filterwarnings("ignore", message=".*share an object.*")

N_PATHS = 5
ACT_DIM = 3

I_START0 = slice(0, 3)
I_END0 = slice(15, 18)
I_OFFSET = 41


def make_env(render=False, name="agent"):
    return gym.make(
        "supertuxkart/full-v0",
        agent=AgentSpec(use_ai=False, name=name),
        track=hp.TRACK,
        num_kart=1,
        laps=1,
        render_mode="human" if render else None,
    )


def _first_k(points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < N_PATHS:
        points = np.concatenate([points, np.repeat(points[-1:], N_PATHS - len(points), axis=0)])
    return points[:N_PATHS]


def features(obs):
    return np.concatenate(
        [
            _first_k(obs["paths_start"]).ravel(),
            _first_k(obs["paths_end"]).ravel(),
            _first_k(obs["paths_width"]).ravel(),
            obs["velocity"],
            obs["center_path"],
            np.ravel(obs["center_path_distance"]),
            np.ravel(obs["max_steer_angle"]),
        ]
    ).astype(np.float32)


def progress(obs):
    return float(np.ravel(obs["distance_down_track"])[0])


def expert_action(action):
    acc = float(np.ravel(action["acceleration"])[0])
    steer = float(np.ravel(action["steer"])[0])
    return np.array([2 * acc - 1, steer, 2 * float(action["brake"]) - 1], dtype=np.float32)


def env_action(u):
    return {
        "acceleration": np.array([np.clip((u[0] + 1) / 2, 0, 1)], dtype=np.float32),
        "steer": np.array([np.clip(u[1], -1, 1)], dtype=np.float32),
        "brake": int(u[2] > 0),
        "nitro": 0,
        "drift": 0,
        "rescue": 0,
        "fire": 0,
    }


def reward(dist, next_dist, terminated):
    delta = np.asarray(next_dist, np.float32) - np.asarray(dist, np.float32)
    delta = np.where(np.asarray(terminated, bool) | (np.abs(delta) > 50), 0.0, delta)
    return delta - hp.TIME_PENALTY


def rollout(env, act, labeller=None):
    obs, _ = env.reset(seed=0)
    if labeller is not None:
        labeller.reset()
    out = {"obs": [], "actions": [], "labels": [], "dist": [], "offset": []}
    finished = False
    for t in range(hp.MAX_STEPS):
        if labeller is not None:
            out["labels"].append(expert_action(labeller(obs)))
        a = act(obs)
        out["obs"].append(features(obs))
        out["actions"].append(a)
        out["dist"].append(progress(obs))
        out["offset"].append(float(np.ravel(obs["center_path_distance"])[0]))
        obs, _, terminated, truncated, _ = env.step(env_action(a))
        if terminated or truncated:
            finished = bool(terminated)
            break
    out["dist"].append(progress(obs))
    res = {k: np.asarray(v, dtype=np.float32) for k, v in out.items() if len(v)}
    res["steps"], res["finished"] = t + 1, finished
    return res


def heading_error(x):
    d = x[..., I_END0] - x[..., I_START0]
    return np.arctan2(d[..., 0], d[..., 2])


def load_dataset(path):
    d = np.load(path)
    data = {k: d[k] for k in d.files if k != "meta"}
    data["rewards"] = reward(data["dist"], data["next_dist"], data["terminated"]).astype(np.float32)
    return data, json.loads(str(d["meta"]))


def obs_stats():
    d = np.load(hp.NORM_DATASET)
    return d["obs"].mean(0), d["obs"].std(0)
