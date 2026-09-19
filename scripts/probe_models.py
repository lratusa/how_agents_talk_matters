#!/usr/bin/env python
"""Single-agent capability probe: choose primary model for FPRR experiments.

Samples stratified subsets of MMLU-Pro and SuperGPQA, queries candidate
DeepSeek models at T=0.8, parses FINAL ANSWER, reports accuracy.
Pilot-only tool; results guide benchmark/model selection (documented in
preregistration.md). Never used for final results.
"""
import asyncio, json, os, random, re, sys, time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "benchmarks"
OUT = ROOT / "data" / "pilot"
OUT.mkdir(parents=True, exist_ok=True)

# Keys: PROBE_KEY env var, or a dotenv file (FPRR_KEY_FILE, default .env.local).
def load_key(name):
    env = Path(os.environ.get("FPRR_KEY_FILE", ".env.local"))
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "%s not found. Export it, or set FPRR_KEY_FILE to a dotenv file "
        "containing %s=<your-key>." % (name, name))

API = os.environ.get("PROBE_API", "https://api.deepseek.com/v1/chat/completions")
KEY = os.environ.get("PROBE_KEY") or load_key(
    os.environ.get("PROBE_KEY_NAME", "DEEPSEEK_API_KEY"))
EXTRA = json.loads(os.environ.get("PROBE_EXTRA", "{}"))  # e.g. thinking toggle

SOLVE_PROMPT = """Solve the following problem. Think step by step, then give a concise justification (at most 120 words) of the key reasoning. End your response with a single line exactly of the form:

FINAL ANSWER: <letter>

Problem:
{question}

Options:
{options}"""

ANS_RE = re.compile(r"FINAL ANSWER:\s*\(?([A-J])\)?", re.IGNORECASE)

def fmt_options(opts):
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))

def parse_letter(text, n_opts):
    m = None
    for m in ANS_RE.finditer(text or ""):
        pass
    if m:
        L = m.group(1).upper()
        if L in "ABCDEFGHIJ"[:n_opts]:
            return L
    return None

async def call(client, model, prompt, sem, max_retries=5):
    async with sem:
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8, "top_p": 0.95,
                    "max_tokens": int(os.environ.get("PROBE_MAX_TOKENS", "4000")),
                }
                payload.update(EXTRA)
                r = await client.post(API, json=payload, headers={"Authorization": f"Bearer {KEY}"}, timeout=180)
                if r.status_code == 200:
                    d = r.json()
                    return d["choices"][0]["message"]["content"], d.get("usage", {})
                if r.status_code in (429, 500, 502, 503):
                    await asyncio.sleep(2 ** attempt + random.random())
                    continue
                return None, {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
            except Exception as e:
                await asyncio.sleep(2 ** attempt + random.random())
        return None, {"error": "retries exhausted"}

async def run(model, questions, n_conc=16):
    sem = asyncio.Semaphore(n_conc)
    async with httpx.AsyncClient() as client:
        tasks = [call(client, model, q["prompt"], sem) for q in questions]
        results = await asyncio.gather(*tasks)
    rows = []
    for q, (text, usage) in zip(questions, results):
        pred = parse_letter(text or "", q["n_opts"])
        rows.append({"qid": q["qid"], "benchmark": q["benchmark"], "model": model,
                     "gold": q["gold"], "pred": pred, "correct": pred == q["gold"],
                     "text": text, "usage": usage})
    return rows

def main():
    rng = random.Random(42)
    questions = []
    # MMLU-Pro: stratified 60 (6 per category x 10 random categories)
    mp = pd.read_parquet(DATA / "mmlu_pro_test.parquet")
    cats = rng.sample(sorted(mp["category"].unique().tolist()), 10)
    for c in cats:
        sub = mp[mp["category"] == c].sample(n=6, random_state=42)
        for _, r in sub.iterrows():
            opts = list(r["options"])
            questions.append({"qid": f"mmlupro_{r['question_id']}", "benchmark": "mmlu_pro",
                              "prompt": SOLVE_PROMPT.format(question=r["question"], options=fmt_options(opts)),
                              "gold": r["answer"], "n_opts": len(opts)})
    # SuperGPQA: random 60
    sg = [json.loads(l) for l in open(DATA / "supergpqa_all.jsonl", encoding="utf-8")]
    sg = rng.sample(sg, 60)
    for r in sg:
        opts = list(r["options"])
        gold = "ABCDEFGHIJ"[r["answer_letter"] and ord(r["answer_letter"]) - 65] if isinstance(r.get("answer_letter"), str) else None
        gold = r.get("answer_letter") or gold
        questions.append({"qid": f"supergpqa_{r['uuid']}", "benchmark": "supergpqa",
                          "prompt": SOLVE_PROMPT.format(question=r["question"], options=fmt_options(opts)),
                          "gold": gold, "n_opts": len(opts)})

    models = sys.argv[1:] or ["deepseek-flash", "deepseek-v4-pro"]
    all_rows = []
    for model in models:
        t0 = time.time()
        rows = asyncio.run(run(model, questions))
        dt = time.time() - t0
        all_rows += rows
        for bm in ("mmlu_pro", "supergpqa"):
            sub = [r for r in rows if r["benchmark"] == bm]
            acc = sum(r["correct"] for r in sub) / max(1, len(sub))
            unparsed = sum(1 for r in sub if r["pred"] is None)
            toks = sum(r["usage"].get("total_tokens", 0) for r in sub)
            print(f"{model} | {bm}: acc={acc:.3f} n={len(sub)} unparsed={unparsed} tokens={toks} time={dt:.0f}s")
    with open(OUT / "probe_results.jsonl", "a", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
