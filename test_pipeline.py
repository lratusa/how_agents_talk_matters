"""Offline smoke test — NO API calls, NO Ollama. Run: python test_pipeline.py

Covers:
  1. frozen parsers (last-occurrence rule, unparsed -> None, normalization)
  2. matching algorithms (perfect matchings, determinism, maxweight optimality)
  3. full mock pipeline on 8 toy questions, all conditions
  4. cache correctness + resumability (second run makes 0 chat calls)
  5. gold answers never appear in any logged prompt
  6. evaluate.py produces results/*.json + *.md
"""
import itertools
import json
import os
import random
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config, matching, prompts
from src.client import MockChatClient, MockEmbedder
from src.evaluate import evaluate, majority_vote
from src.matching import (cosine_distance_matrix, greedy_matching,
                          is_perfect_matching, maxweight_matching,
                          random_matching)
from src.parsing import (parse_answer, parse_review_verdict,
                         parse_updated_answer, split_justification)
from src.pipeline import Pipeline

PASS = []


def ok(name, cond):
    assert cond, "FAILED: %s" % name
    PASS.append(name)
    print("ok - %s" % name)


# ---------------------------------------------------------------------------
# 1. Parsers
# ---------------------------------------------------------------------------

def test_parsers():
    ok("parser: last FINAL ANSWER wins",
       parse_answer("reasoning\nFINAL ANSWER: A\nmore\nFINAL ANSWER: C", True) == "C")
    ok("parser: lowercase/parens letter",
       parse_answer("FINAL ANSWER: (b)", True) == "B")
    ok("parser: unparsed -> None", parse_answer("no marker here", True) is None)
    ok("parser: free value normalize $1,234.0 -> 1234",
       parse_answer("FINAL ANSWER: $1,234.0", False) == "1234")
    ok("parser: boxed value", parse_answer("FINAL ANSWER: \\boxed{18}", False) == "18")
    ok("parser: free unparsed -> None", parse_answer("nothing", False) is None)
    ok("parser: verdict last occurrence",
       parse_review_verdict("REVIEW VERDICT: PEER CORRECT\n...\nREVIEW VERDICT: PEER INCORRECT")
       == "PEER INCORRECT")
    ok("parser: verdict missing -> None", parse_review_verdict("blah") is None)
    ok("parser: updated answer last occurrence",
       parse_updated_answer("YOUR UPDATED ANSWER: A\nYOUR UPDATED ANSWER: D", True) == "D")
    ok("parser: justification = text before last FINAL ANSWER",
       split_justification("part1\nFINAL ANSWER: A\npart2\nFINAL ANSWER: B")
       == "part1\nFINAL ANSWER: A\npart2")
    # majority vote: tie -> lowest agent index; tie flag
    fake = [{"agent_idx": 0, "parsed_answer": "B"},
            {"agent_idx": 1, "parsed_answer": "A"},
            {"agent_idx": 2, "parsed_answer": "A"},
            {"agent_idx": 3, "parsed_answer": "B"}]
    ok("majority: tie broken by lowest agent index",
       majority_vote(fake) == ("B", True))  # A: {1,2}, B: {0,3}; lowest index holds B
    fake2 = [{"agent_idx": 0, "parsed_answer": "A"},
             {"agent_idx": 1, "parsed_answer": "A"},
             {"agent_idx": 2, "parsed_answer": "B"},
             {"agent_idx": 3, "parsed_answer": "B"}]
    ok("majority: tie, lowest index holds A",
       majority_vote(fake2) == ("A", True))
    fake2[3]["parsed_answer"] = "A"  # A: agents 0,1,3 -> clear winner
    ok("majority: clear winner", majority_vote(fake2) == ("A", False))
    ok("majority: all unparsed -> (None, False)",
       majority_vote([{"agent_idx": 0, "parsed_answer": None}]) == (None, False))


# ---------------------------------------------------------------------------
# 2. Matching
# ---------------------------------------------------------------------------

def _brute_force_maxweight(dist):
    n = dist.shape[0]
    best = -np.inf
    for perm in itertools.permutations(range(n)):
        w = sum(dist[perm[i], perm[i + 1]] for i in range(0, n, 2))
        best = max(best, w)
    return best


def test_matching():
    rng_np = np.random.RandomState(7)
    for n in (2, 4, 6, 8):
        X = rng_np.randn(n, 16)
        dist = cosine_distance_matrix(X)
        ok("distance matrix symmetric, zero diagonal (n=%d)" % n,
           np.allclose(dist, dist.T) and np.allclose(np.diag(dist), 0))
        rng = random.Random(42)
        for name, pairs in [
            ("random", random_matching(n, random.Random(42))),
            ("rgfm", greedy_matching(dist, random.Random(42), True)),
            ("nearest", greedy_matching(dist, random.Random(42), False)),
            ("maxweight", maxweight_matching(dist)),
        ]:
            ok("%s matching is perfect (n=%d)" % (name, n),
               is_perfect_matching(pairs, n))
        # determinism
        ok("rgfm deterministic given seed",
           greedy_matching(dist, random.Random(42), True)
           == greedy_matching(dist, random.Random(42), True))
        ok("random matching deterministic given seed",
           random_matching(n, random.Random(9)) == random_matching(n, random.Random(9)))
        # maxweight optimality vs independent brute force
        mw_pairs = maxweight_matching(dist)
        mw_weight = sum(dist[i, j] for i, j in mw_pairs)
        ok("maxweight optimal (n=%d)" % n,
           abs(mw_weight - _brute_force_maxweight(dist)) < 1e-9)
    # odd n rejected
    try:
        random_matching(3, random.Random(0))
        ok("odd n rejected", False)
    except ValueError:
        ok("odd n rejected", True)


# ---------------------------------------------------------------------------
# 3. Prompt templates (verbatim anchors from prereg §6)
# ---------------------------------------------------------------------------

def test_prompt_templates():
    p = prompts.solve_prompt("Q?", ["x", "y"])
    ok("solve prompt MCQ shape",
       "FINAL ANSWER: <letter>" in p and "Options:\nA. x\nB. y" in p)
    p = prompts.solve_prompt("Q?", None)
    ok("solve prompt free-answer shape",
       "FINAL ANSWER: <value>" in p and "Options:" not in p)
    r = prompts.review_prompt("Q?", "A", "mine", "B", "theirs", True)
    ok("review prompt verbatim anchors",
       "You are reviewing another independent solution to the same problem." in r
       and "Judge the reasoning, not writing style or confidence." in r
       and "REVIEW VERDICT: PEER CORRECT or PEER INCORRECT" in r
       and "YOUR UPDATED ANSWER: <letter>" in r
       and "Your own solution:\nAnswer: A\nJustification: mine" in r
       and "Peer solution:\nAnswer: B\nJustification: theirs" in r)
    s1 = prompts.synth_prompt("Q?", "M", with_reviews=True, mcq=True)
    s2 = prompts.synth_prompt("Q?", "M", with_reviews=False, mcq=True)
    ok("synth prompt reviews clause toggle",
       ", and pairwise peer reviews of them," in s1
       and ", and pairwise peer reviews of them," not in s2)
    ok("synth prompt free-answer marker",
       "FINAL ANSWER: <value>" in prompts.synth_prompt("Q?", "M", False, False))


# ---------------------------------------------------------------------------
# 4. Full mock pipeline
# ---------------------------------------------------------------------------

def _toy_questions_free(n=8):
    # benchmark name 'gsm8k' so the pipeline treats them as free-answer
    return [dict(qid="toy-f-%d" % i, benchmark="gsm8k",
                 question="Toy question %d: compute something." % i,
                 options=None, gold=str(100000 + i), stratum=None)
            for i in range(n)]


def _toy_questions_mcq(n=4):
    return [dict(qid="toy-m-%d" % i, benchmark="mmlu_pro",
                 question="Toy MCQ %d?" % i,
                 options=["option-%d-%d" % (i, j) for j in range(4)],
                 gold="ABCD"[i % 4], stratum="toy") for i in range(n)]


def _all_logged_prompts(run_dir):
    out = []
    for fn in ("initial.jsonl", "reviews.jsonl", "debate.jsonl", "synth.jsonl"):
        path = os.path.join(run_dir, fn)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.append(json.loads(line)["prompt"])
    return out


def _run_pipeline(tmp, questions, conditions, tag, mcq):
    run_dir = os.path.join(tmp, tag)
    client, embedder = MockChatClient(), MockEmbedder()
    pipe = Pipeline(questions, client, embedder, run_dir, n_agents=4,
                    conditions=conditions, seed=0, pairing_seed=0)
    counts = pipe.run()
    return pipe, client, counts


def test_pipeline_mock(tmp):
    conditions = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9"]
    questions = _toy_questions_free(8)
    pipe, client, counts = _run_pipeline(tmp, questions, conditions,
                                         "run_free", mcq=False)

    # cache files exist
    for fn in ("initial.jsonl", "distances.jsonl", "matchings.jsonl",
               "reviews.jsonl", "debate.jsonl", "synth.jsonl"):
        ok("cache file %s written" % fn,
           os.path.exists(os.path.join(pipe.run_dir, fn)))

    ok("initial records = 8q x 4 agents", len(pipe.initial.records) == 32)
    ok("distance matrices = 8", len(pipe.distances.records) == 8)
    ok("matchings = 8q x 4 pairing conditions", len(pipe.matchings.records) == 32)
    # reviews: C3-C6 -> 4 ordered reviews per q each; C9 -> 4 self-reviews
    ok("reviews = 8q x (4 conds x 4 + 4 self)", len(pipe.reviews.records) == 160)
    ok("debate updates = 8q x 4", len(pipe.debate.records) == 32)
    ok("synth = 8q x 7 conditions", len(pipe.synth.records) == 56)

    # every initial record carries the full audit schema
    rec = next(iter(pipe.initial.records.values()))
    for field in ("qid", "benchmark", "condition", "seed", "pairing_seed",
                  "model", "temperature", "top_p", "prompt", "raw_output",
                  "parsed_answer", "justification", "usage", "latency_s",
                  "timestamp"):
        ok("initial record has field %s" % field, field in rec)
    ok("usage has token triple",
       set(rec["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"})

    # matchings from the run are valid perfect matchings on the same matrix
    for mrec in pipe.matchings.records.values():
        ok("run matching perfect (%s %s)" % (mrec["qid"], mrec["condition"]),
           is_perfect_matching([tuple(p) for p in mrec["pairs"]], 4))
    # all four pairing conditions saw the SAME distance matrix per question
    dmat = {d["qid"]: d["matrix"] for d in pipe.distances.records.values()}
    ok("distance matrix stored per question", len(dmat) == 8)

    # reviews are reciprocal for C3-C6 and self for C9
    r34 = [r for r in pipe.reviews.records.values() if r["condition"] == "C3"]
    ok("C3 reviews reciprocal",
       all(any(r2["reviewer_idx"] == r["reviewee_idx"]
               and r2["reviewee_idx"] == r["reviewer_idx"]
               and r2["qid"] == r["qid"] for r2 in r34) for r in r34))
    r9 = [r for r in pipe.reviews.records.values() if r["condition"] == "C9"]
    ok("C9 reviews are self-reviews",
       all(r["reviewer_idx"] == r["reviewee_idx"] for r in r9))

    # gold never appears in any logged prompt (distinctive toy golds)
    prompts_all = _all_logged_prompts(pipe.run_dir)
    ok("prompts logged: %d" % len(prompts_all), len(prompts_all) > 0)
    for q in questions:
        ok("gold %s never in any prompt" % q["gold"],
           all(q["gold"] not in p for p in prompts_all))
    # no condition names / pairing info leak into review or synth prompts
    review_prompts = [r["prompt"] for r in pipe.reviews.records.values()]
    ok("no condition/distance leakage in review prompts",
       all("C3" not in p and "C5" not in p and "distance" not in p.lower()
           for p in review_prompts))
    synth_prompts = [r["prompt"] for r in pipe.synth.records.values()]
    ok("synth materials anonymized (S-labels, no agent indices)",
       all("S1" in p and "agent_idx" not in p and "agent 0" not in p.lower()
           for p in synth_prompts))

    # resumability: second run must make ZERO new chat calls
    client2, embedder2 = MockChatClient(), MockEmbedder()
    pipe2 = Pipeline(questions, client2, embedder2, pipe.run_dir, n_agents=4,
                     conditions=conditions, seed=0, pairing_seed=0)
    counts2 = pipe2.run()
    ok("second run: 0 new API calls", client2.calls_made == 0)
    ok("second run: all stages cache-hit", all(v == 0 for v in counts2.values()))

    # determinism: mock outputs identical across fresh runs in a new dir
    pipe3, client3, _ = _run_pipeline(tmp, questions, conditions,
                                      "run_free_again", mcq=False)
    same = all(pipe3.initial.get(qid=q["qid"], seed=0, agent_idx=i)["raw_output"]
               == pipe.initial.get(qid=q["qid"], seed=0, agent_idx=i)["raw_output"]
               for q in questions for i in range(4))
    ok("mock run reproducible (identical raw outputs)", same)
    m1 = pipe.matchings.get(qid="toy-f-0", seed=0, condition="C5", pairing_seed=0)
    m3 = pipe3.matchings.get(qid="toy-f-0", seed=0, condition="C5", pairing_seed=0)
    ok("matchings reproducible across runs", m1["pairs"] == m3["pairs"])

    # evaluate runs and writes outputs
    out, json_path, md_path = evaluate(pipe.run_dir, "gsm8k", out_dir=tmp,
                                       questions=questions, mcq=False)
    ok("evaluate wrote json", os.path.exists(json_path))
    ok("evaluate wrote md", os.path.exists(md_path))
    ok("evaluate covers all conditions",
       set(out["conditions"]) == set(conditions))
    ok("evaluate accuracies in [0,1]",
       all(0.0 <= d["accuracy"] <= 1.0 for d in out["conditions"].values()))
    ok("evaluate has paired diffs + mcnemar + holm",
       "C5-C3" in out["paired_differences"]
       and "C5_vs_C3" in out["mcnemar"]
       and "C5_vs_C3" in out["mcnemar_holm_adjusted_p"])
    ok("evaluate has secondary/diversity/token blocks",
       bool(out["secondary"]) and bool(out["diversity"]) and bool(out["tokens"]))

    # MCQ path
    q_mcq = _toy_questions_mcq(4)
    pipe_m, _, _ = _run_pipeline(tmp, q_mcq, conditions, "run_mcq", mcq=True)
    letters = {r["parsed_answer"] for r in pipe_m.initial.records.values()}
    ok("MCQ mock answers parse to letters",
       letters <= set("ABCD") and len(letters) > 0)
    out_m, _, _ = evaluate(pipe_m.run_dir, "mmlu_pro", out_dir=tmp,
                           questions=q_mcq, mcq=True)
    ok("MCQ evaluate covers all conditions",
       set(out_m["conditions"]) == set(conditions))


def main():
    test_parsers()
    test_matching()
    test_prompt_templates()
    tmp = tempfile.mkdtemp(prefix="fprr_smoke_")
    try:
        test_pipeline_mock(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL %d CHECKS PASSED" % len(PASS))


if __name__ == "__main__":
    main()
