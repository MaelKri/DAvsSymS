"""Behavior cloning et DAgger sur l'actor partagé."""

import numpy as np
import torch
import torch.nn.functional as F

import common.hparams as hp
from common.nets import tensor
from common.stk import features, rollout


def fit(actor, opt, X, Y, n_steps):
    X, Y = tensor(X), tensor(Y)
    for _ in range(n_steps):
        i = torch.randint(len(X), (hp.BATCH_SIZE,), device=X.device)
        loss = F.mse_loss(actor.deterministic(X[i]), Y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def bc(actor, data, evaluate, log):
    opt = torch.optim.Adam(actor.parameters(), lr=hp.LR)
    for k in range(hp.BC_GRAD_STEPS // hp.BC_EVAL_EVERY):
        loss = fit(actor, opt, data["obs"], data["labels"], hp.BC_EVAL_EVERY)
        log(evaluate(), env_steps=0, expert_labels=len(data["obs"]), grad_steps=(k + 1) * hp.BC_EVAL_EVERY, loss=loss)


def dagger(actor, data, env, expert, evaluate, log):
    opt = torch.optim.Adam(actor.parameters(), lr=hp.LR)
    X, Y, env_steps = [data["obs"]], [data["labels"]], 0
    for it in range(hp.DAGGER_ITERS + 1):
        if it > 0:
            lap = rollout(env, lambda obs: actor.act(features(obs)), labeller=expert)
            X.append(lap["obs"])
            Y.append(lap["labels"])
            env_steps += lap["steps"]
        loss = fit(actor, opt, np.concatenate(X), np.concatenate(Y), hp.DAGGER_GRAD_STEPS)
        log(
            evaluate(),
            env_steps=env_steps,
            expert_labels=sum(map(len, X)),
            grad_steps=(it + 1) * hp.DAGGER_GRAD_STEPS,
            loss=loss,
            iteration=it,
        )
