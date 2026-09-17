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

# API key file (read-only, outside the project; has a UTF-8 BOM).
ENV_KEY_FILE = r"D:/pc-project/Jinshang_LLM/new_implementation/.env.local"
ENV_KEY_NAME = "DEEPSEEK_API_KEY"

API_BASE = "https://api.deepseek.com/v1/chat/completions"
# Primary model: default deepseek-flash, overridable via env DEEPSEEK_MODEL
# or run_experiment.py --model. (Prereg §1 leaves the slot to be filled by
# the probe; the default here is the cheaper candidate.)
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

# Decoding (prereg §1 as amended 2026-09-17 pre-pilot): identical for all
# agents and all conditions. deepseek-flash is a reasoning model whose
# hidden reasoning tokens count against max_tokens; 1000 caused truncation
# (empty content) in >30% of probe calls, so budgets were raised pre-pilot.
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS_SOLVE = 4000   # solve + review + debate update
MAX_TOKENS_SYNTH = 6000   # synthesis
# No seed parameter: provider does not support it (prereg §1).

# Embeddings (prereg §2): bge-m3 via local Ollama, 1024-dim.
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024
EMBED_CACHE = os.path.join(CACHE_DIR, "embeddings.jsonl")

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
