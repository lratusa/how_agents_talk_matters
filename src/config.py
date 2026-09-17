"""Central configuration: paths, model, decoding params (preregistration §1-§2)."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BENCH_DIR = os.path.join(DATA_DIR, "benchmarks")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

BENCHMARK_FILES = {
    "mmlu_pro": os.path.join(BENCH_DIR, "mmlu_pro_test.parquet"),
    "supergpqa": os.path.join(BENCH_DIR, "supergpqa_all.jsonl"),
    "gsm8k": os.path.join(BENCH_DIR, "gsm8k_test.parquet"),
    "math500": os.path.join(BENCH_DIR, "math500_test.jsonl"),
}

# API endpoint/key (prereg §1 Amendment 4): DeepSeek balance exhausted
# mid-pilot (-11.49 CNY, HTTP 402) and a BigModel resource pack was about
# to expire (user instruction 2026-09-17); primary provider switched to
# Zhipu BigModel glm-4-plus. Constraint-driven, not results-driven.
ENV_KEY_FILE = r"D:/pc-project/Jinshang_LLM/new_implementation/.env.local"
ENV_KEY_NAME = os.environ.get("FPRR_KEY_NAME", "BIGMODEL_API_KEY")

API_BASE = os.environ.get(
    "FPRR_API_BASE", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
# Primary model: glm-4-plus (probe: MMLU-Pro 68.3%, SuperGPQA 41.7%,
# unparsed 1/60, stochastic at T=0.8). Overridable via --model.
MODEL = os.environ.get("DEEPSEEK_MODEL", "glm-4-plus")

# Decoding (prereg §1 Amendments 1+4): identical for all agents/conditions.
# glm-4-plus is non-reasoning; modest token budgets suffice (probe mean
# ~500 completion tokens/call, truncation negligible).
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS_SOLVE = 1500   # solve + review + debate update
MAX_TOKENS_SYNTH = 2500   # synthesis
# No seed parameter: provider does not support it (prereg §1).

# Embeddings (prereg §2): bge-m3 via local Ollama, 1024-dim.
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024
EMBED_CACHE = os.path.join(CACHE_DIR, "embeddings.jsonl")

# Distance ablations (prereg §8): D1 conclusion+justification cosine (primary),
# D2 justification-only cosine, D3 hybrid lambda*d_sem + (1-lambda)*disagree,
# D4 NLI contradiction (roberta-large-mnli, symmetric mean).
DISTANCE_ID = os.environ.get("DISTANCE_ID", "D1")
LAMBDA_HYBRID = float(os.environ.get("LAMBDA_HYBRID", "0.5"))
NLI_MODEL = "roberta-large-mnli"

N_AGENTS_DEFAULT = 4  # prereg §5 (N=4 primary; {2,4,6,8} ablation)

# All conditions defined by prereg §5.
ALL_CONDITIONS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9"]
# Conditions that consume a pairing (and hence a pairing_seed).
PAIRING_CONDITIONS = {"C3": "random", "C4": "nearest", "C5": "rgfm", "C6": "maxweight"}
REVIEW_CONDITIONS = ["C3", "C4", "C5", "C6", "C9"]  # stage D
SYNTH_CONDITIONS = ["C2", "C3", "C4", "C5", "C6", "C7", "C9"]  # stage E

MCQ_BENCHMARKS = {"mmlu_pro", "supergpqa"}

for _d in (RUNS_DIR, CACHE_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)
