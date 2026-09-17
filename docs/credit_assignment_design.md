# Minimal Causal Experiment Design: Epistemic Credit Assignment under Disagreement

Status: **design frozen, not yet run** (paid compute reserved for SuperGPQA replication first).
Date: 2026-09-17. Depends on paper 1 (FPRR) results; positioning per `literature/drp_novelty_gate.md` (GO-WITH-REPOSITIONING).

## 1. Research question

Paper 1 established a pathway: semantic dissimilarity between paired responses →
higher false-opposition rate (correct answers judged wrong) and destabilization of
correct reviewers, which cancels the detection gain. The proposed DRP mechanism
assumes that **information isolation (blindness)** breaks this pathway while
preserving error recognition.

This experiment tests that assumption causally, in isolation, before any repair
pipeline is built:

> **Does hiding candidate answers from the verifier preserve true-error recognition
> while eliminating distance-driven false opposition?**

Core positioning (post-repositioning): *how to extract reliable epistemic credit
from disagreement when no deterministic checker exists* — not the gate itself.

## 2. Design: single-factor visibility manipulation

Fixed across all conditions (same control philosophy as paper 1):

- **Candidates**: reused from cached paper-1 runs (`data/runs/*__n500__*`), never
  regenerated. Each stimulus = a pair (r_A, r_B) from the same question whose
  parsed answers differ.
- **Crux**: one decisive-claim extraction per question, generated ONCE by a
  neutral agent (sees the question and that two independent solvers disagree,
  does NOT see the candidate answers), cached, reused for all visibility
  conditions.
- **Model**: deepseek-flash, thinking disabled, T=0.8, top_p=0.95 — identical to
  paper 1.
- **Budget**: one verifier call per unit (below); 2 judgment repetitions per
  unit to estimate verifier stochasticity.

Only manipulated variable: **what the verifier sees**.

| Condition | Verifier input | Recognition derived by |
|---|---|---|
| **V1 both-visible** (status quo, = paper-1 review setting minus ownership) | question + both candidates (answer + justification), labels randomized | per-candidate verdict |
| **V2 blind** | question + crux only; no candidate content | verifier answers the crux independently; each candidate scored by whether its reasoning is consistent with the verified crux (deterministic comparison call, candidates shown only at this scoring step, one at a time) |
| **V3 one-sided** | question + exactly one candidate (each side judged in a separate call; balanced which side first) | per-candidate absolute verdict, no contrast available |
| **V4 anonymized-evidence** | question + both reasoning chains with conclusions and identities stripped; verifier evaluates evidence quality, conclusions revealed only in a second scoring step | per-evidence verdict mapped back to candidates |

V2 operationalizes "blindness as anti-anchoring". V4 separates *conclusion/identity
anchoring* from *evidence comparison*: if V4 ≈ V2, anchoring on conclusions drives
false opposition; if V4 ≈ V1, the damage comes from exposure to the opposing
reasoning itself.

## 3. Materials and sampling

Source pools (cached initials, gold used post hoc only):

- MMLU-Pro n=1500 (3 seeds × 500): 223 disagreement questions; SuperGPQA n=500
  (seed 0): 178 disagreement questions.
- Pair types (MCQ: different answers ⇒ at most one correct):
  - **mixed** (correct × incorrect): the recognition test bed;
  - **wrong–wrong**: measures hallucinated recognition (verifier should find both
    wrong; a "picked a winner" response is a failure).
- Stratified sample per benchmark: all mixed pairs up to 200, sampled uniformly
  across pair-distance deciles (distance matrix already cached), plus 100
  wrong–wrong pairs. Random A/B label and presentation order per call; a subset
  (~20%) repeated with reversed order to quantify position bias.

## 4. Endpoints

Per visibility condition:

- **Recognition accuracy**: P(identifies the correct side | mixed pair).
- **True correction recognition (TCR)**: P(judges wrong side wrong | wrong side).
- **False opposition (FO)**: P(judges correct side wrong | correct side).
- **Credit quality**: P(correct side judged correct ∧ wrong side judged wrong |
  mixed pair).
- **Distance moderation (key)**: slope of FO vs pair semantic distance per
  condition (logistic regression, FO ~ d × condition). Paper 1 supplies the
  reference slope under both-visible review: FO rises from ~1% (lowest decile) to
  ~5–13% (top decile).

## 5. Hypotheses (to pre-register before any run)

- **HC1 (blindness blocks the pathway)**: FO slope on distance ≈ 0 under V2,
  positive under V1; the interaction is significant. *This is the experiment.*
- **HC2**: recognition accuracy(V2) ≥ recognition accuracy(V1) on mixed pairs.
- **HC3**: V3 has lower FO but lower TCR than V1 (contrast aids detection but
  breeds doubt) — tests whether "seeing the opponent" is net information or net
  pollution, quantifying paper 1's C9-vs-C3 observation at pair level.
- **HC4**: V1 < V4 < V2 ordering on FO (conclusion/identity anchoring accounts
  for part, not all, of the pathway).

Falsification: if HC1 fails (FO slope persists under V2), blindness does not
block the pathway and DRP's information-isolation premise is wrong — stop the
line and report. If HC1 holds but HC2 fails (blind verifier cannot recognize the
correct side), credit extraction itself is infeasible without candidate access —
redirect to anonymized-evidence (V4) as the operative mechanism.

## 6. Statistics

- Unit of analysis: pair (mixed) or side (TCR/FO). Logistic regression with
  condition × distance interaction; cluster-robust SE by question (multiple
  pairs per question).
- Paired bootstrap over questions for condition differences on shared pairs
  (same candidates judged under all four conditions).
- Holm–Bonferroni across the four hypotheses.
- 2 repetitions per call → report inter-repetition agreement; pooled verdicts
  via majority.

## 7. Budget estimate

Per benchmark (~300 questions): 300 crux extractions + mixed pairs (≤200 ×
(V1:1 + V2:2 + V3:2 + V4:2 calls)) + wrong–wrong (100 × same) ≈ 2400 calls ×
2 repetitions ≈ 5k calls. Both benchmarks ≈ 10k calls ≈ ¥20 at observed rates.
Half-scale pilot (MMLU only, 1 repetition) ≈ ¥5–6.

## 8. Relation to prior art (from novelty gate v2, to be updated by v3)

- V2's independent crux answering descends from CoVe (2309.11495) — cite as
  component prior; our contribution is the *causal isolation* of visibility,
  not the component.
- Recognition-under-visibility relates to PVD's verifier (2605.25133) and DR's
  crux (2607.01251), but neither manipulates verifier information access with
  fixed candidates/crux.
- Downstream (only if HC1+HC2 pass): repair + certificate + guarded replacement,
  with GuardedRepair (2605.24613), DISC (2606.21724), PVD as repair baselines.

## 9. Integrity constraints (carried from paper 1 protocol)

No ground truth in any verifier prompt; gold used only post hoc; deterministic
verdict parsing defined before running; raw JSONL logging; no manual edits of
outputs; cached candidates never regenerated; all conditions share identical
stimuli.
