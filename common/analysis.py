"""Outils communs aux scripts d'analyse : conditions, chargement des runs, métriques, tests et tracés."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common.hparams as hp

RUNS = Path("runs")
FIG = Path("figures")
CLEAN_DATASET = "data/team4_abyss_ep1_noise0.npz"
DART_DATASET = "data/team4_abyss_ep20_noise0.3.npz"

THRESHOLD_RATIO = 1.05
DELTA_EQ = 15
FIG_SEED = 0
T_STATE = 290

COND = {
    "bc_s0": ("s0/bc", r"BC ($D_1$)", "#888888", CLEAN_DATASET),
    "dart": ("s0.3/bc", r"BC-DART ($D_{20}$)", "#6BAED6", DART_DATASET),
    "dagger": ("s0/dagger", "DAgger", "#1D9E75", CLEAN_DATASET),
    "sac": ("none/sac", "SAC (no data)", "#7F77DD", None),
    "init": ("s0.3/rlpd_init", r"RLPD-init ($D_{20}$)", "#E3A33B", DART_DATASET),
    "rlpd": ("s0.3/rlpd", r"RLPD ($D_{20}$)", "#D85A30", DART_DATASET),
    "rlpd_s0": ("s0/rlpd", r"RLPD ($D_1$)", "#8E2B0E", CLEAN_DATASET),
}
LABEL = {k: v[1] for k, v in COND.items()}
COLOR = {k: v[2] for k, v in COND.items()}

# --- 1. chargement et métriques


def run_dirs(cond):
    return sorted((RUNS / COND[cond][0]).glob("seed*"), key=lambda p: int(p.name[4:]))


def load(cond):
    return {int(d.name[4:]): pd.read_csv(d / "log.csv") for d in run_dirs(cond) if (d / "log.csv").exists()}


def expert_steps():
    return int(np.load(RUNS / "expert/seed0/final_lap.npz")["steps"])


def threshold():
    return THRESHOLD_RATIO * expert_steps()


def perf(df):
    return np.where(df["finished"] == 1, df["lap_steps"], hp.MAX_STEPS)


def summary(df, thr):
    p = perf(df)
    hit = np.flatnonzero(p <= thr)
    return dict(
        time=df["env_steps"].iloc[hit[0]] if len(hit) else hp.TOTAL_STEPS + 1,
        reached=len(hit) > 0,
        final=np.median(p[-3:]),
        best=p.min(),
        labels=df["expert_labels"].iloc[-1],
        env=df["env_steps"].iloc[-1],
        grad=df["grad_steps"].iloc[-1],
        minutes=df["wall"].iloc[-1] / 60,
    )


def summaries(cond, thr):
    return pd.DataFrame([dict(seed=s, **summary(df, thr)) for s, df in load(cond).items()])


def on_grid(logs, values, grid):
    out = []
    for df in logs.values():
        x, v = df["env_steps"].values, np.asarray(values(df), float)
        i = np.searchsorted(x, grid, side="right") - 1
        out.append(np.where((i >= 0) & (grid <= x[-1]), v[np.clip(i, 0, None)], np.nan))
    return np.array(out)


# --- 2. tests


def fisher(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return stats.fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])[1]


def mannwhitney(a, b):
    return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue


def holm(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def median_diff_ci(a, b, n=10_000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = np.median(rng.choice(a, (n, len(a))), 1) - np.median(rng.choice(b, (n, len(b))), 1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return np.median(a) - np.median(b), lo, hi


def equivalence(lo, hi, delta=DELTA_EQ):
    if -delta <= lo and hi <= delta:
        return "equivalent"
    if lo > delta or hi < -delta:
        return "different"
    return "inconclusive"


# --- 3. tracés


def binned(x, y, bins, min_count=20):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_count:
            out.append((np.median(x[m]), *np.percentile(y[m], [25, 50, 75])))
    return np.array(out).reshape(-1, 4)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"-> {FIG / name}.png")
