"""CLI entry point.

Example:
  python run_experiment.py --benchmark gsm8k --n-questions 4 --seed 0 \
      --conditions C0,C1,C2,C3,C4,C5,C6,C9 --mock

Resumable: re-running the same command reuses all cached JSONL records.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config, datasets
from src.client import make_clients
from src.evaluate import evaluate
from src.pipeline import Pipeline, make_run_id


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FPRR multi-agent experiment runner")
    p.add_argument("--benchmark", required=True,
                   choices=["mmlu_pro", "supergpqa", "gsm8k", "math500"])
    p.add_argument("--n-questions", type=int, required=True)
    p.add_argument("--seed", type=int, default=0,
                   help="experimental seed (question sampling + generation replicate)")
    p.add_argument("--n-agents", type=int, default=config.N_AGENTS_DEFAULT)
    p.add_argument("--conditions", default=",".join(config.ALL_CONDITIONS),
                   help="comma-separated subset of %s" % ",".join(config.ALL_CONDITIONS))
    p.add_argument("--pairing-seed", type=int, default=0)
    p.add_argument("--model", default=config.MODEL)
    p.add_argument("--mock", action="store_true",
                   help="deterministic offline mock; no API/Ollama calls")
    p.add_argument("--distance-id", default=config.DISTANCE_ID,
                   choices=["D1", "D2", "D3", "D4"],
                   help="distance ablation (prereg §8); non-D1 uses its own run dir")
    p.add_argument("--lambda-hybrid", type=float, default=config.LAMBDA_HYBRID,
                   help="lambda for D3 hybrid distance")
    p.add_argument("--reuse-initials-from", default=None,
                   help="path to a run dir whose initial.jsonl is copied in, "
                        "so ablation runs share the exact same cached initial "
                        "responses (prereg §18)")
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--no-evaluate", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Distance ablation wiring (must happen before make_run_id).
    config.DISTANCE_ID = args.distance_id
    config.LAMBDA_HYBRID = args.lambda_hybrid
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(conditions) - set(config.ALL_CONDITIONS)
    if unknown:
        raise SystemExit("unknown conditions: %s" % sorted(unknown))

    questions = datasets.sample_questions(args.benchmark, args.n_questions, args.seed)
    print("sampled %d questions from %s (seed=%d)"
          % (len(questions), args.benchmark, args.seed))

    model = "mock-model" if args.mock else args.model
    client, embedder = make_clients(args.mock, model=model,
                                    max_concurrency=args.max_concurrency)
    run_id = make_run_id(args.benchmark, len(questions), args.seed,
                         args.pairing_seed, args.n_agents, model, args.mock)
    run_dir = os.path.join(config.RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    if args.reuse_initials_from:
        src = os.path.join(args.reuse_initials_from, "initial.jsonl")
        dst = os.path.join(run_dir, "initial.jsonl")
        if not os.path.exists(dst):
            import shutil
            shutil.copyfile(src, dst)
            print("reused cached initials from %s" % src)
    print("run dir: %s" % run_dir)

    pipe = Pipeline(questions, client, embedder, run_dir,
                    n_agents=args.n_agents, conditions=conditions,
                    seed=args.seed, pairing_seed=args.pairing_seed)
    counts = pipe.run()
    print("new records per stage: %s" % json.dumps(counts))
    print("API calls made this run: %d" % client.calls_made)

    if not args.no_evaluate:
        out, json_path, md_path = evaluate(run_dir, args.benchmark)
        print("results: %s" % json_path)
        print("results: %s" % md_path)
        for cond in sorted(out["conditions"]):
            d = out["conditions"][cond]
            acc = "—" if d["accuracy"] is None else "%.3f" % d["accuracy"]
            print("  %s accuracy=%s (n=%d)" % (cond, acc, d["n"]))


if __name__ == "__main__":
    main()
