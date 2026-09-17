"""Unified benchmark loaders + seeded stratified sampler.

Unified record: dict(qid, benchmark, question, options: list[str]|None,
gold: str, stratum: str|None).

Gold normalization:
- MCQ benchmarks: gold is the option letter (A-J).
- gsm8k: number after '####' -> normalized numeric string.
- math500: raw answer string, whitespace-collapsed.
"""
import json
import random
import re

import pandas as pd

from . import config


def normalize_value(s):
    """Normalize a free-answer value for exact-match scoring.

    Strips whitespace, $, commas, latex \boxed{...} wrapping, \left/\right,
    and trailing '.'. If the result parses as a number, returns a canonical
    numeric string ('18' not '18.0'); else the collapsed raw string.
    """
    if s is None:
        return None
    s = str(s).strip()
    m = re.fullmatch(r"\\boxed\{(.*)\}", s, flags=re.DOTALL)
    if m:
        s = m.group(1).strip()
    s = s.replace("$", "").replace(",", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return s


def _norm_gold_free(s):
    return normalize_value(s)


def load_benchmark(name):
    """Load the full benchmark as a list of unified records."""
    path = config.BENCHMARK_FILES[name]
    recs = []
    if name == "mmlu_pro":
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            recs.append(dict(
                qid=str(r["question_id"]), benchmark=name,
                question=str(r["question"]),
                options=[str(o) for o in r["options"]],
                gold=str(r["answer"]).strip().upper(),
                stratum=str(r["category"])))
    elif name == "supergpqa":
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                recs.append(dict(
                    qid=str(r["uuid"]), benchmark=name,
                    question=str(r["question"]),
                    options=[str(o) for o in r["options"]],
                    gold=str(r["answer_letter"]).strip().upper(),
                    stratum=str(r.get("discipline"))))
    elif name == "gsm8k":
        df = pd.read_parquet(path)
        for i, (_, r) in enumerate(df.iterrows()):
            gold = str(r["answer"]).split("####")[-1].strip()
            recs.append(dict(
                qid="gsm8k-%d" % i, benchmark=name,
                question=str(r["question"]), options=None,
                gold=_norm_gold_free(gold), stratum=None))
    elif name == "math500":
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                recs.append(dict(
                    qid=str(r.get("unique_id") or r["problem"][:40]),
                    benchmark=name,
                    question=str(r["problem"]), options=None,
                    gold=_norm_gold_free(r["answer"]),
                    stratum=str(r.get("subject"))))
    else:
        raise ValueError("unknown benchmark: %s" % name)
    return recs


def is_mcq(benchmark):
    return benchmark in config.MCQ_BENCHMARKS


SAMPLING_SEED = 42  # fixed question-sampling seed (prereg §9 Amendment 3):
# question IDs are identical across experimental seeds; experimental seeds
# only re-replicate stochastic generation / pairing / shuffles.


def sample_questions(benchmark, n, seed=None):
    """Stratified sample (stratified by category/discipline/subject
    where available), returning a fixed, deterministic list of records.
    The `seed` argument is IGNORED for sampling: question selection always
    uses SAMPLING_SEED so the same IDs are reused across all conditions
    AND all experimental seeds (prereg §4, §9 Amendment 3).
    """
    recs = load_benchmark(benchmark)
    if n >= len(recs):
        return recs
    rng = random.Random(SAMPLING_SEED)
    strata = {}
    for r in recs:
        strata.setdefault(r["stratum"], []).append(r)
    if len(strata) == 1:
        chosen = rng.sample(recs, n)
    else:
        # proportional allocation, largest-remainder rounding
        total = len(recs)
        alloc = {}
        fracs = []
        for k, group in strata.items():
            exact = n * len(group) / total
            alloc[k] = min(int(exact), len(group))
            fracs.append((exact - int(exact), k))
        short = n - sum(alloc.values())
        for _, k in sorted(fracs, reverse=True):
            if short <= 0:
                break
            if alloc[k] < len(strata[k]):
                alloc[k] += 1
                short -= 1
        chosen = []
        for k, group in sorted(strata.items()):
            k = str(k)
            take = min(alloc.get(k, 0), len(group))
            chosen.extend(rng.sample(group, take))
        rng.shuffle(chosen)
    return chosen[:n]
