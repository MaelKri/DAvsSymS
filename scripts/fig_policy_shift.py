"""Figure du policy shift : plans de phase et erreur d'imitation selon la distance aux données."""

import matplotlib.pyplot as plt
import numpy as np

from common.analysis import COLOR, DART_DATASET, FIG_SEED, LABEL, RUNS, binned, save
from common.stk import I_OFFSET, heading_error, load_dataset

CONDS = ("bc_s0", "dart", "dagger")
X, Y = (-10, 6), (-1, 1)


def phase(offset, heading):
    return np.clip(offset, *X), np.clip(heading, *Y)


data = load_dataset(DART_DATASET)[0]["obs"]
expert = np.load(RUNS / "expert/seed0/final_lap.npz")["obs"]
laps = {c: np.load(RUNS / f"eval/policy/{c}_seed{FIG_SEED}.npz") for c in CONDS}
err = {c: np.linalg.norm(lap["actions"] - lap["labels"], axis=1) for c, lap in laps.items()}
vmax = np.percentile(np.concatenate(list(err.values())), 98)

fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2]))
for k, (ax, c) in enumerate(zip(axes, CONDS)):
    lap = laps[c]
    ax.scatter(*phase(data[:, I_OFFSET], heading_error(data)), s=2, color="0.82", label=r"data $D_{20}(0.3)$")
    ax.plot(*phase(expert[:, I_OFFSET], heading_error(expert)), color="k", lw=0.8, label="expert")
    sc = ax.scatter(*phase(lap["offset"], lap["heading"]), c=err[c], s=6, cmap="viridis", vmin=0, vmax=vmax, zorder=3)
    ax.set(
        title=f"({'abc'[k]}) {LABEL[c]} ({int(lap['steps'])} steps{'' if lap['finished'] else ', not finished'})",
        xlabel="lateral offset to the centre (m)",
        xlim=(X[0] - 0.3, X[1] + 0.3),
        ylim=(Y[0] - 0.05, Y[1] + 0.05),
    )
    if k:
        ax.set_yticklabels([])
axes[0].set_ylabel("heading error (rad)")
axes[0].legend(fontsize=8, loc="upper left", markerscale=4)
fig.colorbar(sc, ax=axes[:3], label="imitation error", shrink=0.9)

ax = axes[3]
for c in CONDS:
    runs = [np.load(f) for f in sorted((RUNS / "eval/policy").glob(f"{c}_seed*.npz"))]
    x = np.concatenate([r["nn_dist"] for r in runs])
    y = np.concatenate([np.linalg.norm(r["actions"] - r["labels"], axis=1) for r in runs])
    bins = np.quantile(x, np.linspace(0, 1, 11))
    bins[-1] += 1e-6
    b = binned(x, y, bins)
    ax.plot(b[:, 0], b[:, 2], "o-", ms=3, color=COLOR[c], label=LABEL[c])
    ax.fill_between(b[:, 0], b[:, 1], b[:, 3], color=COLOR[c], alpha=0.15)
ax.set(
    xscale="log",
    xlabel="distance of the state to the data (nearest neighbour)",
    ylabel="imitation error",
    title="(d) all seeds",
)
ax.legend(fontsize=8)
save(fig, "fig_policy_shift")
