# DAgger versus symmetric sampling (RLPD) on SuperTuxKart

Mini-projet 2 de Reinforcement Learning. Avec le même budget de données
expertes, on compare DAgger, behavior cloning, SAC et RLPD (avec ou sans symmetric sampling) sur un tour de la
piste `abyss` de SuperTuxKart, et on mesure les deux distribution shifts (politique et critique).

## Installation

```sh
uv sync
```

Les hyperparamètres sont dans `common/hparams.py`.

## 1. Collecte des données de l'expert

```sh
uv run python scripts/collect_expert.py --episodes 1               # D_1 : un tour propre
uv run python scripts/collect_expert.py --episodes 20 --noise 0.3  # D_20(0.3) : 20 tours, bruit DART
```

Les datasets sont écrits dans `data/`.

## 2. Entraînements

Tour de référence de l'expert :

```sh
uv run python scripts/train.py --method expert --out runs
```

Puis un run par condition et par seed. Un run d'imitation prend environ 2 minutes, un run RL environ 30 minutes
sur GPU.

```sh
D20=data/team4_abyss_ep20_noise0.3.npz
for s in 0 1 2 3 4 5 6 7 8 9; do
  uv run python scripts/train.py --method dagger    --seed $s --out runs/s0
  uv run python scripts/train.py --method bc        --seed $s --out runs/s0.3 --dataset $D20
  uv run python scripts/train.py --method sac       --seed $s --out runs/none
  uv run python scripts/train.py --method rlpd_init --seed $s --out runs/s0.3 --dataset $D20
  uv run python scripts/train.py --method rlpd      --seed $s --out runs/s0.3 --dataset $D20
done
for s in 0 1 2 3 4; do
  uv run python scripts/train.py --method bc   --seed $s --out runs/s0
  uv run python scripts/train.py --method rlpd --seed $s --out runs/s0
done
```

Chaque run écrit dans `runs/<dossier>/<méthode>/seed<k>/` : `log.csv`, `config.json`, `actor.pt`, `critics.pt`
(RL) et `final_lap.npz`.

## 3. Analyses

```sh
uv run python scripts/analyze_main.py      # tableau 1, tests, fig1_seuil.png, fig_q_reference.png
uv run python scripts/eval_states.py       # tours des agents d'imitation labellisés par l'expert
uv run python scripts/fig_policy_shift.py  # fig_policy_shift.png
uv run python scripts/fig_critic_shift.py  # fig_critic_profiles.png, contrôles de platitude et de déterminisme
```

Les résultats sont écrits dans `figures/` ou affichés dans la console.
`fig_policy_shift.py` a besoin des sorties de `eval_states.py`.

## Structure

```
common/    hyperparamètres, réseaux, interface SuperTuxKart, outils d'analyse
methods/   behavior cloning, DAgger, SAC, RLPD-init, RLPD
experts/   expert scripté de l'équipe 4 (osigaud/pystk_project)
scripts/   collecte, entraînement, analyses et figures
stk_actor/ template du cours (soumission), non utilisé par le projet
```

Le code a été formaté avec [ruff](https://docs.astral.sh/ruff/) :

```sh
uv run ruff check --fix .
uv run ruff format .
```