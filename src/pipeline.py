"""Experiment orchestration with JSONL caching.

Stages (prereg §5, §13):
  A. initial generation  -> initial.jsonl    key (qid, seed, agent_idx)
  B. embeddings/distances-> distances.jsonl  key (qid, seed)
  C. matchings           -> matchings.jsonl  key (qid, seed, condition, pairing_seed)
  D. reviews             -> reviews.jsonl    key (qid, seed, condition, pairing_seed,
                                                 reviewer_idx, reviewee_idx)
  C7 debate updates      -> debate.jsonl     key (qid, seed, agent_idx)
  E. synthesis           -> synth.jsonl      key (qid, seed, condition, pairing_seed)

Resumability: a record whose cache key already exists is never regenerated.
Stage A is shared by ALL conditions (C2-C7/C9 consume the exact same cache).
No ground truth is ever placed in any prompt.
"""
import asyncio
import json
import os
import random
from datetime import datetime, timezone

import numpy as np

from . import config, matching, prompts
from .client import _sha256
from .datasets import is_mcq
from .parsing import (parse_answer, parse_review_verdict, parse_updated_answer,
                      split_justification)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class JsonlCache:
    """Append-only JSONL store with an in-memory key index."""

    def __init__(self, path, key_fields):
        self.path = path
        self.key_fields = key_fields
        self.records = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.records[self._key(rec)] = rec

    def _key(self, rec):
        return tuple(rec.get(k) for k in self.key_fields)

    def get(self, **key):
        return self.records.get(tuple(key.get(k) for k in self.key_fields))

    def add(self, rec):
        key = self._key(rec)
        if key in self.records:
            return False
        self.records[key] = rec
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True

    def filter(self, **kw):
        return [r for r in self.records.values()
                if all(r.get(k) == v for k, v in kw.items())]


def base_record(qid, benchmark, condition, seed, pairing_seed, stage,
                agent_ids, prompt, call, parsed):
    """Uniform record schema (prereg §13): every call logs ids, model,
    decoding params, full prompt, raw output, parsed fields, usage, latency,
    timestamp."""
    rec = {
        "qid": qid, "benchmark": benchmark, "condition": condition,
        "seed": seed, "pairing_seed": pairing_seed, "stage": stage,
        "model": call["model"],
        "temperature": config.TEMPERATURE, "top_p": config.TOP_P,
        "prompt": prompt, "raw_output": call["content"],
        "usage": call["usage"], "latency_s": call["latency_s"],
        "timestamp": utcnow(),
    }
    rec.update(agent_ids)
    rec.update(parsed)
    return rec


class Pipeline:
    def __init__(self, questions, client, embedder, run_dir,
                 n_agents=config.N_AGENTS_DEFAULT,
                 conditions=config.ALL_CONDITIONS,
                 seed=0, pairing_seed=0):
        self.questions = questions
        self.client = client
        self.embedder = embedder
        self.run_dir = run_dir
        self.n = n_agents
        self.conditions = list(conditions)
        self.seed = seed
        self.pairing_seed = pairing_seed
        os.makedirs(run_dir, exist_ok=True)
        self.initial = JsonlCache(os.path.join(run_dir, "initial.jsonl"),
                                  ("qid", "seed", "agent_idx"))
        self.distances = JsonlCache(os.path.join(run_dir, "distances.jsonl"),
                                    ("qid", "seed"))
        self.matchings = JsonlCache(
            os.path.join(run_dir, "matchings.jsonl"),
            ("qid", "seed", "condition", "pairing_seed"))
        self.reviews = JsonlCache(
            os.path.join(run_dir, "reviews.jsonl"),
            ("qid", "seed", "condition", "pairing_seed",
             "reviewer_idx", "reviewee_idx"))
        self.debate = JsonlCache(os.path.join(run_dir, "debate.jsonl"),
                                 ("qid", "seed", "agent_idx"))
        self.synth = JsonlCache(
            os.path.join(run_dir, "synth.jsonl"),
            ("qid", "seed", "condition", "pairing_seed"))

    # -- helpers -----------------------------------------------------------

    def _initials(self, qid):
        """Cached initial responses for (qid, self.seed), ordered by agent."""
        out = []
        for i in range(self.n):
            rec = self.initial.get(qid=qid, seed=self.seed, agent_idx=i)
            if rec is None:
                raise RuntimeError("missing initial for %s agent %d" % (qid, i))
            out.append(rec)
        return out

    @staticmethod
    def _answer_or_unparsed(rec):
        return rec["parsed_answer"] if rec["parsed_answer"] is not None else "UNPARSED"

    # -- Stage A: initial generation ---------------------------------------

    async def stage_a(self):
        tasks = []  # (q, agent_idx, prompt) for missing records
        for q in self.questions:
            mcq = is_mcq(q["benchmark"])
            for i in range(self.n):
                if self.initial.get(qid=q["qid"], seed=self.seed, agent_idx=i):
                    continue
                prompt = prompts.solve_prompt(q["question"], q["options"])
                tasks.append((q, i, prompt, mcq))

        async def one(q, i, prompt, mcq):
            call = await self.client.chat(prompt, config.MAX_TOKENS_SOLVE)
            return q, i, prompt, mcq, call

        results = await asyncio.gather(*[one(*t) for t in tasks])
        for q, i, prompt, mcq, call in results:
            rec = base_record(
                q["qid"], q["benchmark"], "shared", self.seed, None,
                "initial", {"agent_idx": i}, prompt, call,
                {"parsed_answer": parse_answer(call["content"], mcq),
                 "justification": split_justification(call["content"])})
            rec["max_tokens"] = config.MAX_TOKENS_SOLVE
            self.initial.add(rec)
        return len(tasks)

    # -- Stage B: embeddings + distance matrices -----------------------------

    def stage_b(self):
        done = 0
        for q in self.questions:
            if self.distances.get(qid=q["qid"], seed=self.seed):
                continue
            recs = self._initials(q["qid"])
            texts = [
                "CONCLUSION: %s\nJUSTIFICATION: %s"
                % (self._answer_or_unparsed(r), r["justification"])
                for r in recs
            ]
            vecs = self.embedder.embed_texts(texts)
            dist = matching.cosine_distance_matrix(vecs)
            self.distances.add({
                "qid": q["qid"], "benchmark": q["benchmark"],
                "condition": "shared", "seed": self.seed, "pairing_seed": None,
                "stage": "distance", "n_agents": self.n,
                "metric": "cosine", "distance_id": "D1",
                "embedding_model": getattr(self.embedder, "model_name",
                                           "unknown"),
                "embedding_keys": [_sha256(t) for t in texts],
                "matrix": [[round(float(x), 6) for x in row] for row in dist],
                "timestamp": utcnow()})
            done += 1
        return done

    # -- Stage C: matchings ---------------------------------------------------

    def stage_c(self):
        done = 0
        wanted = [c for c in self.conditions if c in config.PAIRING_CONDITIONS]
        for q in self.questions:
            drec = self.distances.get(qid=q["qid"], seed=self.seed)
            if drec is None:
                raise RuntimeError("missing distance matrix for %s" % q["qid"])
            dist = np.asarray(drec["matrix"], dtype=float)
            for cond in wanted:
                if self.matchings.get(qid=q["qid"], seed=self.seed,
                                      condition=cond,
                                      pairing_seed=self.pairing_seed):
                    continue
                algo = config.PAIRING_CONDITIONS[cond]
                # per-question RNG derived from the recorded pairing seed
                rng = random.Random("%s|%s|%s|%s" % (self.pairing_seed,
                                                     self.seed, q["qid"], algo))
                pairs = matching.compute_matching(algo, dist, rng)
                if not matching.is_perfect_matching(pairs, self.n):
                    raise RuntimeError("invalid matching for %s %s" % (q["qid"], cond))
                self.matchings.add({
                    "qid": q["qid"], "benchmark": q["benchmark"],
                    "condition": cond, "seed": self.seed,
                    "pairing_seed": self.pairing_seed, "stage": "matching",
                    "algorithm": algo, "pairs": [list(p) for p in pairs],
                    "diagnostics": matching.matching_diagnostics(dist, pairs),
                    "timestamp": utcnow()})
                done += 1
        return done

    # -- Stage D: reviews ------------------------------------------------------

    async def stage_d(self):
        tasks = []
        for q in self.questions:
            mcq = is_mcq(q["benchmark"])
            initials = self._initials(q["qid"])
            for cond in self.conditions:
                if cond not in config.REVIEW_CONDITIONS:
                    continue
                if cond == "C9":
                    pseed = None
                    pairs = matching.self_pairs(self.n)
                else:
                    pseed = self.pairing_seed
                    mrec = self.matchings.get(qid=q["qid"], seed=self.seed,
                                              condition=cond, pairing_seed=pseed)
                    if mrec is None:
                        raise RuntimeError("missing matching %s %s" % (q["qid"], cond))
                    pairs = [tuple(p) for p in mrec["pairs"]]
                # reciprocal: both directions of each pair (for C9, (i,i) once)
                ordered = []
                for i, j in pairs:
                    ordered.append((i, j))
                    if i != j:
                        ordered.append((j, i))
                for reviewer, reviewee in ordered:
                    if self.reviews.get(qid=q["qid"], seed=self.seed,
                                        condition=cond, pairing_seed=pseed,
                                        reviewer_idx=reviewer,
                                        reviewee_idx=reviewee):
                        continue
                    own = initials[reviewer]
                    peer = initials[reviewee]
                    prompt = prompts.review_prompt(
                        q["question"],
                        self._answer_or_unparsed(own), own["justification"],
                        self._answer_or_unparsed(peer), peer["justification"],
                        mcq)
                    tasks.append((q, cond, pseed, reviewer, reviewee, prompt, mcq))

        async def one(q, cond, pseed, reviewer, reviewee, prompt, mcq):
            call = await self.client.chat(prompt, config.MAX_TOKENS_SOLVE)
            return q, cond, pseed, reviewer, reviewee, prompt, mcq, call

        results = await asyncio.gather(*[one(*t) for t in tasks])
        for q, cond, pseed, reviewer, reviewee, prompt, mcq, call in results:
            rec = base_record(
                q["qid"], q["benchmark"], cond, self.seed, pseed, "review",
                {"reviewer_idx": reviewer, "reviewee_idx": reviewee},
                prompt, call,
                {"review_verdict": parse_review_verdict(call["content"]),
                 "updated_answer": parse_updated_answer(call["content"], mcq)})
            rec["max_tokens"] = config.MAX_TOKENS_SOLVE
            self.reviews.add(rec)
        return len(tasks)

    # -- C7: debate update -------------------------------------------------------

    async def stage_debate(self):
        tasks = []
        for q in self.questions:
            mcq = is_mcq(q["benchmark"])
            initials = self._initials(q["qid"])
            # all agents receive the SAME anonymized shuffled view per question
            order = list(range(self.n))
            random.Random("debate|%s|%s" % (q["qid"], self.seed)).shuffle(order)
            materials = "\n\n".join(
                prompts.candidate_block(
                    "S%d" % (k + 1),
                    self._answer_or_unparsed(initials[idx]),
                    initials[idx]["justification"])
                for k, idx in enumerate(order))
            for i in range(self.n):
                if self.debate.get(qid=q["qid"], seed=self.seed, agent_idx=i):
                    continue
                prompt = prompts.debate_prompt(q["question"], materials, mcq)
                tasks.append((q, i, prompt, mcq))

        async def one(q, i, prompt, mcq):
            call = await self.client.chat(prompt, config.MAX_TOKENS_SOLVE)
            return q, i, prompt, mcq, call

        results = await asyncio.gather(*[one(*t) for t in tasks])
        for q, i, prompt, mcq, call in results:
            rec = base_record(
                q["qid"], q["benchmark"], "C7", self.seed, None, "debate",
                {"agent_idx": i}, prompt, call,
                {"parsed_answer": parse_answer(call["content"], mcq),
                 "justification": split_justification(call["content"])})
            rec["max_tokens"] = config.MAX_TOKENS_SOLVE
            self.debate.add(rec)
        return len(tasks)

    # -- Stage E: synthesis -------------------------------------------------------

    def _synth_materials(self, q, cond):
        """Anonymized, seeded-shuffled candidate materials per question.

        Labels S1..Sk are assigned after a seeded shuffle; review blocks use
        the same label mapping (no agent indices leak into the prompt).
        Returns (materials_text, with_reviews).
        """
        mcq = is_mcq(q["benchmark"])
        pseed = self.pairing_seed if cond in config.PAIRING_CONDITIONS else None
        initials = self._initials(q["qid"])
        order = list(range(self.n))
        random.Random("synth|%s|%s|%s|%s" % (q["qid"], self.seed, cond, pseed)
                      ).shuffle(order)
        label = {idx: "S%d" % (k + 1) for k, idx in enumerate(order)}

        if cond == "C7":
            blocks = []
            for idx in order:
                drec = self.debate.get(qid=q["qid"], seed=self.seed, agent_idx=idx)
                blocks.append(prompts.candidate_block(
                    label[idx], self._answer_or_unparsed(drec),
                    drec["justification"]))
            return "\n\n".join(blocks), False

        blocks = [
            prompts.candidate_block(label[idx], self._answer_or_unparsed(initials[idx]),
                                    initials[idx]["justification"])
            for idx in order
        ]
        with_reviews = cond in config.REVIEW_CONDITIONS
        if with_reviews:
            rrecs = self.reviews.filter(qid=q["qid"], seed=self.seed,
                                        condition=cond, pairing_seed=pseed)
            rrecs.sort(key=lambda r: (label[r["reviewee_idx"]],
                                      label[r["reviewer_idx"]]))
            rblocks = [
                prompts.review_block(label[r["reviewer_idx"]],
                                     label[r["reviewee_idx"]],
                                     r["raw_output"])
                for r in rrecs
            ]
            if rblocks:
                blocks.append("Peer reviews:\n\n" + "\n\n".join(rblocks))
        return "\n\n".join(blocks), with_reviews

    async def stage_e(self):
        tasks = []
        for q in self.questions:
            mcq = is_mcq(q["benchmark"])
            for cond in self.conditions:
                if cond not in config.SYNTH_CONDITIONS:
                    continue
                pseed = self.pairing_seed if cond in config.PAIRING_CONDITIONS else None
                if self.synth.get(qid=q["qid"], seed=self.seed, condition=cond,
                                  pairing_seed=pseed):
                    continue
                materials, with_reviews = self._synth_materials(q, cond)
                prompt = prompts.synth_prompt(q["question"], materials,
                                              with_reviews, mcq)
                tasks.append((q, cond, pseed, prompt, mcq))

        async def one(q, cond, pseed, prompt, mcq):
            call = await self.client.chat(prompt, config.MAX_TOKENS_SYNTH)
            return q, cond, pseed, prompt, mcq, call

        results = await asyncio.gather(*[one(*t) for t in tasks])
        for q, cond, pseed, prompt, mcq, call in results:
            rec = base_record(
                q["qid"], q["benchmark"], cond, self.seed, pseed, "synth",
                {}, prompt, call,
                {"parsed_answer": parse_answer(call["content"], mcq)})
            rec["max_tokens"] = config.MAX_TOKENS_SYNTH
            self.synth.add(rec)
        return len(tasks)

    # -- orchestration -------------------------------------------------------------

    def run(self):
        """Run the stages needed for the requested conditions. Returns counts
        of newly created records per stage (0 => fully cache-hit)."""
        counts = {}
        counts["A_initial"] = asyncio.run(self.stage_a())
        try:
            if any(c in config.PAIRING_CONDITIONS for c in self.conditions):
                counts["B_distances"] = self.stage_b()
                counts["C_matchings"] = self.stage_c()
            if any(c in config.REVIEW_CONDITIONS for c in self.conditions):
                counts["D_reviews"] = asyncio.run(self.stage_d())
            if "C7" in self.conditions:
                counts["C7_debate"] = asyncio.run(self.stage_debate())
            if any(c in config.SYNTH_CONDITIONS for c in self.conditions):
                counts["E_synth"] = asyncio.run(self.stage_e())
        finally:
            asyncio.run(self.client.close())
        return counts


def make_run_id(benchmark, n_questions, seed, pairing_seed, n_agents, model, mock):
    tag = "%s__n%d__seed%d__pseed%d__N%d__%s" % (
        benchmark, n_questions, seed, pairing_seed, n_agents, model)
    return tag + ("__mock" if mock else "")
