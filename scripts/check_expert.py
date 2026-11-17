"""Fait rouler l'expert seul sur une piste et affiche le nombre de pas pour un tour."""

import argparse

import gymnasium as gym
from pystk2_gymnasium import AgentSpec

from experts import Team4Expert

p = argparse.ArgumentParser()
p.add_argument("--track", default="abyss")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max-steps", type=int, default=3000)
args = p.parse_args()

env = gym.make(
    "supertuxkart/full-v0", agent=AgentSpec(use_ai=False, name="team4"), track=args.track, num_kart=1, laps=1
)
expert = Team4Expert()
obs, _ = env.reset(seed=args.seed)
expert.reset()
for t in range(args.max_steps):
    obs, r, terminated, truncated, info = env.step(expert(obs))
    if terminated or truncated:
        break
print(f"{args.track} seed={args.seed}: {t + 1} pas, terminated={terminated}")
env.close()
