"""Hyperparamètres du projet, partagés par tous les scripts."""

# --- 1. environnement et récompense
TRACK = "abyss"
MAX_STEPS = 1500
TIME_PENALTY = 1.0
NORM_DATASET = "data/team4_abyss_ep20_noise0.3.npz"

# --- 2. réseaux et optimisation
HIDDEN = 256
LR = 3e-4
BATCH_SIZE = 256

# --- 3. imitation
BC_GRAD_STEPS = 20_000
BC_EVAL_EVERY = 2_000
DAGGER_ITERS = 20
DAGGER_GRAD_STEPS = 2_000

# --- 4. SAC / RLPD
TOTAL_STEPS = 30_000
EVAL_EVERY = 1_000
UTD = 10
N_CRITICS = 10
N_MIN = 2
OFFLINE_RATIO = 0.5
SAC_START_STEPS = 5_000
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 1_000_000


def as_dict():
    return {k: v for k, v in globals().items() if k.isupper()}
