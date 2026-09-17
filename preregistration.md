# Pre-registration — Response-Conditioned Disassortative Peer Review (FPRR)

**Frozen:** 2026-09-17, before any full-test evaluation. Pilot/probe data (~120 questions, single-agent only, plus pipeline smoke tests) were used solely for model/benchmark selection and bug detection, per the staged design. Any deviation after this date is labeled exploratory.

## 1. Primary model

Selection rule (executed before freezing): single-agent probe of `deepseek-flash` and `deepseek-v4-pro` (DeepSeek API, T=0.8, top_p=0.95) on 60 stratified MMLU-Pro + 60 random SuperGPQA questions; choose the model whose mean single-agent accuracy falls in the 40–85% band, preferring the cheaper one if both qualify.

**Chosen primary model:** ~~deepseek-flash~~ → superseded by **`glm-4-plus`** per Amendment 4 (§9), a resource-constraint-driven change made during the pilot phase; probe data for both models recorded below and in `data/pilot/probe_results.jsonl`. Original probe (2026-09-17, 60 MMLU-Pro + 60 SuperGPQA, T=0.8, max_tokens=1000): deepseek-flash — MMLU-Pro 58.3%, SuperGPQA 33.3%, 36 s/120 calls; deepseek-v4-pro — MMLU-Pro 51.7%, SuperGPQA 18.3%, 161 s/120 calls. glm-4-plus probe (max_tokens=4000): MMLU-Pro 68.3%, SuperGPQA 41.7%, unparsed 1/60.

## 2. Embedding model (pinned)

`bge-m3:latest` via local Ollama 0.32.13, digest `790764642607`, 1024-dim, F16. Embedded text = `"CONCLUSION: <final answer text>\nJUSTIFICATION: <justification>"` (D1, primary). The question text is NOT embedded.

## 3. Hypotheses (H1–H6)

As specified in the research brief §13. Primary confirmatory tests: H1 (FPRR > random pairing), H2 (FPRR > nearest pairing), H3 (FPRR > majority vote / self-consistency). H4–H6 are secondary/mechanistic.

## 4. Benchmarks

| Benchmark | Split | Role | Access |
|---|---|---|---|
| MMLU-Pro (TIGER-Lab, 2024-06) | test (12,032) | primary | CC-BY-4.0; HF parquet snapshot 2026-09-17 |
| SuperGPQA (m-a-p, 2025-01) | all (26,526; we sample) | primary (hard, recent) | ODC-BY; HF snapshot 2026-09-17 |
| GSM8K | test (1,319) | sanity check only | MIT |
| MATH-500 | test (500) | secondary, if unsaturated | MIT |

GPQA-Diamond and LiveBench were considered but are access-gated from this environment (no HF token); SuperGPQA replaces them as the recent graduate-level benchmark. Documented deviation from the initial candidate list.

Question sampling: fixed IDs, stratified by category/discipline, same IDs across all conditions.

## 5. Conditions

| ID | Name | Initial answers | Pairing | Review | Aggregation |
|---|---|---|---|---|---|
| C0 | Single | 1 | — | — | — |
| C1 | Majority vote (self-consistency) | N=4 | — | — | deterministic vote |
| C2 | Ensemble + synthesizer | N=4 (cached) | — | — | synthesizer |
| C3 | Random pair review | N=4 (cached) | random perfect matching | reciprocal | synthesizer |
| C4 | Nearest pair review | N=4 (cached) | greedy nearest | reciprocal | synthesizer |
| C5 | **FPRR (RGFM)** | N=4 (cached) | randomized greedy farthest | reciprocal | synthesizer |
| C6 | Max-weight matching | N=4 (cached) | global max-weight perfect matching | reciprocal | synthesizer |
| C9 | Self-review matched (compute control) | N=4 (cached) | self | self-review ×N | synthesizer |
| C7 | Vanilla all-to-all debate (1 round) | N=4 (cached) | broadcast | debate update | synthesizer |

C2–C7/C9 use the EXACT SAME cached initial responses per (question, experimental seed). Call budget per question: C0=1; C1=4; C2=5; C3–C6=9 each; C7=9; C9=9.

## 6. Frozen prompts

### 6.1 Solve prompt (initial generation)

```
Solve the following problem. Think step by step, then give a concise justification (at most 120 words) of the key reasoning. End your response with a single line exactly of the form:

FINAL ANSWER: <letter>

Problem:
{question}

Options:
{options}
```

(Free-answer benchmarks replace `<letter>` with `<value>` and omit Options.)

### 6.2 Peer-review prompt (identical in C3–C6)

```
You are reviewing another independent solution to the same problem.

Do not assume that your own answer is correct.
Do not assume that the other answer is wrong.
Check the peer solution independently.

Identify:
1. claims or steps you believe are correct;
2. specific factual, mathematical, or logical errors;
3. assumptions that require verification;
4. whether the peer's conclusion should be retained or changed;
5. your proposed corrected conclusion.

Judge the reasoning, not writing style or confidence.

Problem:
{question}

Your own solution:
Answer: {own_answer}
Justification: {own_justification}

Peer solution:
Answer: {peer_answer}
Justification: {peer_justification}

End your review with a single line exactly of the form:
REVIEW VERDICT: PEER CORRECT or PEER INCORRECT
And then a line:
YOUR UPDATED ANSWER: <letter>
```

### 6.3 Synthesizer prompt (identical in C2–C7/C9)

```
You are given several independent candidate solutions to the same problem{, and pairwise peer reviews of them,} with anonymized randomized labels. Produce the single best final answer.

Rules:
- Do not count repeated claims as independent evidence; the same reasoning may appear multiple times.
- Evaluate argument quality, not rhetorical confidence or frequency.
- Use the criticisms to locate specific errors.
- Recover a correct minority solution if its reasoning is sound.
- Ignore style and confidence.

Problem:
{question}

{materials}

End with a single line exactly of the form:
FINAL ANSWER: <letter>
```

No ground truth, condition names, distances, or pairing information is ever shown to any agent. Candidate order is shuffled with a seeded RNG per question; labels (S1, S2, …) are randomized.

**Amendment 2 (2026-09-17, pre-pilot):** the C7 debate-update prompt was not frozen in the original §6; it is frozen here verbatim (as implemented in `src/prompts.py`):

```
You are given several independent candidate solutions to the same problem with anonymized randomized labels. Verify each candidate's reasoning independently, then give your own updated answer.

Rules:
- Do not assume any candidate is correct, including the one matching your own prior answer.
- Check each candidate's reasoning step by step.
- Ignore style and confidence.

Problem:
{question}

{candidates}

End with a single line exactly of the form:
FINAL ANSWER: <letter>
```

## 7. Pairing algorithms

- **RGFM (primary):** seeded PRNG; repeatedly pick a random unpaired agent i, pair with argmax_k d(i,k); record seed (order-dependent).
- **Nearest:** same loop with argmin.
- **Random:** seeded random perfect matching.
- **Max-weight:** exact maximum-weight perfect matching on the complete graph (exhaustive over the (N−1)!! matchings; N≤8 ⇒ ≤105 matchings).

## 8. Distance metrics

- **D1 (primary):** cosine distance over embeddings of "CONCLUSION + JUSTIFICATION".
- **D2:** cosine over justification only.
- **D3:** hybrid `λ·d_semantic + (1−λ)·1(a_i ≠ a_j)`, λ ∈ {0.25, 0.5, 0.75} tuned on pilot only.
- **D4:** NLI contradiction probability (roberta-large-mnli, CPU inference) between justification pairs, if practical; otherwise labeled not run.

## 9. Sample sizes and seeds

- **Pilot:** ~100 questions × {MMLU-Pro, SuperGPQA, GSM8K}, single experimental seed 0. Used for bug detection, baseline estimation, λ tuning, and benchmark selection. Pilot results are not reported as final evidence.

**Amendment 3 (2026-09-17, pre-pilot):** question sampling uses a FIXED sampling seed (42) independent of the experimental seed, so all experimental seeds evaluate the SAME question IDs (enabling paired analysis across seeds); experimental seeds re-replicate only stochastic generation, pairing, and shuffles (the provider exposes no seed parameter).

**Amendment 4 (2026-09-17, during pilot, constraint-driven):** primary model changed from `deepseek-flash` to **`glm-4-plus`** (Zhipu BigModel API, `https://open.bigmodel.cn/api/paas/v4`). Cause: the DeepSeek account balance was exhausted after the first pilot benchmark (HTTP 402, balance −11.49 CNY) and a BigModel resource pack was about to expire (user directive). This change was driven by an external resource constraint, NOT by pilot results; condition-level pilot rankings played no role. Probe of glm-4-plus under the frozen solve prompt (T=0.8, same 60+60 questions): MMLU-Pro 68.3%, SuperGPQA 41.7%, unparsed 1/60 — within the 40–85% band. Stochasticity verified (distinct surfaces on repeated identical calls at T=0.8); the model is non-reasoning, so max_tokens revert to 1500 (solve/review/debate) / 2500 (synthesis). glm-5.3-flash (73.3%/55.0%) was rejected for a 25% parse-failure rate on SuperGPQA. The MMLU-Pro deepseek-flash pilot is retained as a pipeline-validation artifact only and is never mixed into glm-4-plus results. The pilot is re-run in full under glm-4-plus.
- **Full:** MMLU-Pro 500q (stratified), SuperGPQA 500q (stratified), GSM8K 300q. Experimental seeds: 3 full seeds on MMLU-Pro; 1 seed (seed 0) on SuperGPQA/GSM8K unless pilot variance indicates otherwise. RGFM pairing-seed variance: 3 pairing seeds on a 200q MMLU-Pro subset. N ablation {2,4,6,8} on the same 200q subset (conditions C1–C6 + C9).
- Power note: n=500 paired questions gives ~80% power to detect a ~4–5pp paired accuracy difference given ~25% discordance (McNemar approximation). Smaller effects will be reported with CIs without claiming significance.

## 10. Statistics

- Primary endpoint: exact-answer accuracy (deterministic parser: last `FINAL ANSWER:` line; unparsed ⇒ incorrect; parse rule frozen here).
- Paired bootstrap over questions, 10,000 resamples, 95% CIs for accuracy differences.
- McNemar's test (continuity-corrected) for C5 vs {C3, C4, C1, C9}; Holm–Bonferroni across these four primary comparisons.
- Effect sizes: absolute pp differences; odds ratios where informative.
- Secondary metrics: majority-error recovery rate, correct-majority corruption rate, minority rescue rate, consensus transitions, pair-review correction rate, diversity statistics (mean/max/min pairwise cosine distance, unique answers, answer entropy).

## 11. Compute matching & reporting

C9 (self-review matched) is the primary compute-matched control (same call count as C5). Token usage (prompt/completion/total) and wall-clock recorded per call; report accuracy per million tokens.

## 12. Failure-mode analysis (pre-registered)

Outlier amplification; correct-majority disruption; false opposition (style ≠ epistemic disagreement); shared misconception; synthesizer judge failure. Quantified via pair-composition categories (correct–correct, correct–incorrect, incorrect–incorrect, etc.) and review outcomes, using ground truth only post hoc.

## 13. Reproducibility

JSONL raw records for every call (question ID, prompts, raw outputs, parsed outputs, usage, latency, timestamps, condition, seed). Cached initial responses reused across conditions. Embedding vectors and distance matrices saved. Code versioned in git. No manual answer edits; parser fixes after freezing apply uniformly and are logged.
