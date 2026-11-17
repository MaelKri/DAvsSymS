"""Tableau 1, tests statistiques, figure du seuil et figure de la surestimation du critique."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common.hparams as hp
from common.analysis import (
    COLOR,
    COND,
    DELTA_EQ,
    FIG,
    LABEL,
    RUNS,
    equivalence,
    expert_steps,
    fisher,
    holm,
    load,
    mannwhitney,
    median_diff_ci,
    on_grid,
    perf,
    save,
    summaries,
    threshold,
)
from common.stk import reward

thr, expert = threshold(), expert_steps()
S = {c: summaries(c, thr) for c in COND}
S = {c: s for c, s in S.items() if len(s)}
FIG.mkdir(exist_ok=True)
INTERACTIVE = ["dagger", "sac", "init", "rlpd", "rlpd_s0"]

# --- 1. tableau 1
rows = []
for c, s in S.items():
    n = len(s)
    time = "—" if c in ("bc_s0", "dart") else f"{np.median(s.time):.0f}" if s.reached.mean() > 0.5 else "> budget"
    rows.append(
        [
            LABEL[c],
            n,
            f"{s.reached.sum()}/{n}",
            time,
            f"{np.median(s.final):.0f}",
            f"{(s.final <= thr).sum()}/{n}",
            f"{(s.best < expert).sum()}/{n}",
            f"{s.best.min():.0f}",
            f"{np.median(s.labels):.0f}",
            f"{np.median(s.env):.0f}",
            f"{np.median(s.grad):.0f}",
            f"{np.median(s.minutes):.1f}",
        ]
    )
cols = [
    "method",
    "seeds",
    "threshold reached",
    "steps to threshold",
    "final lap",
    "final under threshold",
    "beats expert",
    "best lap",
    "expert labels",
    "env steps",
    "grad steps",
    "minutes",
]
widths = [max(len(str(r[i])) for r in rows + [cols]) for i in range(len(cols))]
print("\n".join("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(cols))) for r in [cols] + rows))
(FIG / "table1.csv").write_text("\n".join(";".join(map(str, r)) for r in [cols] + rows))
print(f"\nexpert = {expert} steps, threshold = {thr:.0f} steps\n")

# --- 2. tests
H = [
    ("H1", "dagger", "dart", "final lap under threshold", lambda a, b: fisher(a.final <= thr, b.final <= thr)),
    ("H2", "rlpd", "init", "steps to threshold", lambda a, b: mannwhitney(a.time, b.time)),
    ("H3", "rlpd", "sac", "steps to threshold", lambda a, b: mannwhitney(a.time, b.time)),
    ("H4", "dagger", "rlpd", "steps to threshold", lambda a, b: mannwhitney(a.time, b.time)),
]
p = [test(S[a], S[b]) for _, a, b, _, test in H]
for (h, a, b, crit, _), pi, pa in zip(H, p, holm(p)):
    print(f"{h} {a} vs {b} ({crit}): p = {pi:.4f}, p_holm = {pa:.4f}")
for _, a, b, _, _ in H:
    d, lo, hi = median_diff_ci(S[a].final, S[b].final)
    print(f"final lap {a} - {b} = {d:+.1f} [{lo:+.1f}, {hi:+.1f}] ({equivalence(lo, hi)}, margin {DELTA_EQ})")
for c, s in S.items():
    print(f"beats expert {c} = {(s.best < expert).sum()}/{len(s)}")
print(f"beats expert dagger vs rlpd: p = {fisher(S['dagger'].best < expert, S['rlpd'].best < expert):.4f}")

# --- 3. figure 1 : seuil et fiabilité
grid = np.linspace(0, hp.TOTAL_STEPS, 301)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
for c in INTERACTIVE:
    if c not in S:
        continue
    ax1.step(grid, [(S[c].time <= x).mean() for x in grid], where="post", color=COLOR[c], label=LABEL[c])
    ok = np.nanmean(on_grid(load(c), lambda df: perf(df) <= thr, grid), 0)
    ok = pd.Series(ok).rolling(21, center=True, min_periods=1).mean().where(~np.isnan(ok))
    ax2.plot(grid, ok, color=COLOR[c], label=LABEL[c])
ax1.set(
    xlabel="environment steps",
    ylabel="fraction of seeds",
    title=f"(a) threshold reached at least once (≤ {thr:.0f} steps)",
    ylim=(-0.03, 1.03),
)
ax2.set(
    xlabel="environment steps",
    ylabel="fraction of seeds",
    title="(b) lap under the threshold (moving average, 2,000 steps)",
    ylim=(-0.03, 1.03),
)
ax1.legend(fontsize=8, loc="lower right")
save(fig, "fig1_seuil")

# --- 4. figure Q : surestimation du critique
lap = np.load(RUNS / "expert/seed0/final_lap.npz")
term = np.zeros(len(lap["dist"]) - 1, bool)
term[-1] = True
r = reward(lap["dist"][:-1], lap["dist"][1:], term)
G = np.zeros(len(r))
for t in range(len(r) - 1, -1, -1):
    G[t] = r[t] + (hp.GAMMA * G[t + 1] if t < len(r) - 1 else 0.0)
ref = G.max()
rl = [c for c in ("sac", "init", "rlpd", "rlpd_s0") if c in S]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
alphas, qs = [], []
for c in rl:
    q = on_grid(load(c), lambda df: df["q_mean"], grid)
    ax.plot(grid, np.nanmedian(q, 0), color=COLOR[c], label=LABEL[c])
    ax.fill_between(grid, *np.nanpercentile(q, [25, 75], 0), color=COLOR[c], alpha=0.15)
    late = [df[df["env_steps"] >= hp.TOTAL_STEPS - 5_000] for df in load(c).values()]
    alphas += [d["alpha"].median() for d in late]
    qs += [d["q_mean"].median() for d in late]
    ax2.scatter(alphas[-len(late) :], qs[-len(late) :], color=COLOR[c], s=25)
ax.axhline(ref, color="k", ls="--", lw=1, label="max. return of an expert lap")
ax2.axhline(ref, color="k", ls="--", lw=1)
ax.set(xlabel="environment steps", ylabel="mean Q on the evaluation lap", title="(a) critic value during training")
ax2.set(xlabel="α (median over the last 5,000 steps)", ylabel="mean Q (same window)", title="(b) one dot per run")
ax.legend(fontsize=8)
save(fig, "fig_q_reference")
rho, p_rho = spearmanr(alphas, qs)
print(f"max return of an expert lap = {ref:.1f}")
print(f"spearman alpha / Q ({len(qs)} RL runs) = {rho:.2f}, p = {p_rho:.2g}")
