"""Central configuration for ObsCodec Phase 2a."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
ASSETS_DIR = ROOT / "assets"

for d in [DATA_DIR, CHECKPOINT_DIR, ASSETS_DIR]:
    d.mkdir(exist_ok=True)

# ── MPE scenario definitions ──────────────────────────────────────
SCENARIOS = {
    "simple_spread": {
        "n_agents": 5,
        "n_landmarks": 5,
        "obs_dim": 30,           # 4(self)+16(others)+10(landmarks)
        "agent_dims": [6, 6, 6, 6, 6],  # ~equal per-agent dims
    },
    "simple_tag": {
        "n_agents": 6,           # 3 predators + 3 prey
        "n_landmarks": 0,
        "obs_dim": 24,           # 4(self)+20(others)
        "agent_dims": [4, 4, 4, 4, 4, 4],
    },
    "simple_world_comm": {
        "n_agents": 6,           # 4 leaders + 2 followers
        "n_landmarks": 6,        # 4 food + 2 forests
        "obs_dim": 36,           # 4(self)+20(others)+12(landmarks)
        "agent_dims": [6, 6, 6, 6, 6, 6],
    },
}

# ── Agent observation structure (per scenario) ────────────────────
# Each scenario has a per-agent dim count list for per-agent channel tests
AGENT_SPECS = {k: v["agent_dims"] for k, v in SCENARIOS.items()}

# ── Data collection ────────────────────────────────────────────────
TOTAL_SAMPLES = 100_000       # ~100K total obs vectors across all scenarios
SAMPLES_PER_SCENARIO = TOTAL_SAMPLES // len(SCENARIOS)  # ~33K per scenario
RANDOM_SEED = 42

# ── Training defaults ──────────────────────────────────────────────
BATCH_SIZE = 256
DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-3
EARLY_STOP_PATIENCE = 30

# ── β-VAE best-practice defaults (from Phase 1 pilots) ─────────────
# no free_bits, warmup=150, lr=5e-5, decoder_hidden = encoder_hidden
VAE_WARMUP = 150
VAE_LR = 5e-5
VAE_FREE_BITS = 0.0       # nats/dim — per-dimension KL floor (Kingma et al. 2016)
VAE_EPOCHS = 250
VAE_DECODER_HIDDEN_MULT = 1  # multiplier on encoder hidden_dim (1 = symmetric)

# ── Collapse-barrier experiments (Phase 2c) ─────────────────────────
FREE_BITS_SWEEP = [0.0, 0.5, 1.0, 2.0, 4.0]       # nats/dim
DECODER_MULT_SWEEP = [1, 2, 4]                       # × encoder hidden_dim
ANTI_COLLAPSE_BETAS = [0.1, 0.5, 1.0, 2.0, 4.0, 10.0]  # focused on transition region

# ── VQ-VAE defaults ────────────────────────────────────────────────
VQ_COMMITMENT_COST = 0.25
VQ_CODEBOOK_SIZE = 512

# ── Channel evaluation ─────────────────────────────────────────────
AWGN_SNR_RANGE = [-10, -5, 0, 5, 10, 15, 20, 25, 30]  # dB
PACKET_LOSS_RATES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
FADING_MODES = ["block", "agent_blocks", "iid"]
HETERO_SNR_RANGE = (0, 20)  # (min, max) per-agent SNR range in dB

# ── Adaptive rate allocation ─────────────────────────────────────
DEFAULT_TOTAL_BIT_BUDGET = 240     # total bits per observation
MIN_BITS_PER_DIM = 2
MAX_BITS_PER_DIM = 16
ADAPTIVE_STRATEGIES = ["water_filling", "proportional", "uniform"]

# ── Coordination test ─────────────────────────────────────────────
COORDINATION_LOSS_RATES = [0.0, 0.1, 0.2, 0.3, 0.5]

# ── Constants ──────────────────────────────────────────────────────
NATS_TO_BITS = 1.442695
COLLAPSE_KL_THRESHOLD = 0.05
