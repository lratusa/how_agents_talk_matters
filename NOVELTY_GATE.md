# Novelty Gate Report — Farthest-Pair Reciprocal Review (FPRR)

**Date:** 2026-09-17
**Literature cutoff:** 2026-09-17 (nothing after this date considered)
**Decision:** **GO** — no substantially identical prior art found.

---

## 1. Method under evaluation (working name FPRR)

1. **Stage A:** N (even) homogeneous instances of the SAME LLM checkpoint, identical system prompt / problem / decoding config, generate answers independently. Stochastic sampling is the only diversity source.
2. **Stage B:** Each response (final answer + concise justification) is embedded with a pinned sentence-embedding model; pairwise cosine-distance matrix computed.
3. **Stage C:** Agents paired by farthest semantic distance — primary: Randomized Greedy Farthest Matching (RGFM); secondary: global maximum-weight perfect matching.
4. **Stage D:** Reciprocal pairwise peer review — within each pair, i reviews j and j reviews i; each reviewer sees only {problem, own answer, partner answer}; no ground truth, no distance info, anonymized labels.
5. **Stage E:** One final synthesizer (same LLM) aggregates all original answers + all reviews into one final answer.

No training, no fine-tuning, no retrieval, no external knowledge, no tools.

## 2. Search protocol

Six parallel search streams were executed, each with full-text reading of Method/Algorithm sections (arXiv PDF/HTML, OpenReview, ACL Anthology), not just abstracts:

1. Seed papers 1–4 (Du et al.; ReConcile; More Agents; Estornell & Liu) + cited-by snowballing on arXiv:2305.14325 (all 2,235 Semantic Scholar citers screened at abstract level, hits deep-read).
2. Seed papers 5–10 (arXiv 2410.12853, 2505.22960, 2508.17536, 2511.07784, 2601.19921, 2603.20640 — all IDs verified to resolve).
3. Embedding/distance-based pairing & routing (12+ query formulations incl. "farthest pairing", "disassortative", "heterophily", "maximum distance matching", "contrastive peer critique").
4. Adaptive/learned/optimized communication topology (GPTSwarm, DyLAN, AgentPrune, G-Designer, MacNet, Graph-GRPO, Guided Topology Diffusion, Codebook Agent, + open queries).
5. LLM peer-review / mutual-critique systems + scientific reviewer-assignment lineage (TPMS etc.).
6. Diversity-based sampling/pruning/selection + GitHub code search (site:github.com embedding-diversity debate pairing).

Databases: Google Scholar, Semantic Scholar (incl. API), arXiv, OpenReview, ACL Anthology, NeurIPS/ICML/ICLR proceedings, GitHub.
Detailed per-paper A–K tables: `literature/*.md` (6 reports, 40+ papers deep-read).

## 3. Novelty criteria matrix

Legend: ✓ = present as specified; ~ = partial/different realization; ✗ = absent.

| Work | A same model | B same-prompt indep. | C resp. embed. | D pairwise dist. | E dist→topology | F farthest select | G 1-to-1 match | H reciprocal review | I pair-limited | J aggregation | K no training |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Du et al. 2305.14325 (ICML 2024) | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ (broadcast) | ✗ | ~ | ✓ |
| ReConcile 2309.13007 (ACL 2024) | ✗ | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| More Agents Is All You Need 2402.05120 | ✓ | ✓ | ✗ | ~ (BLEU/freq, most-SIMILAR) | ✗ | ✗ | ✗ | ✗ | ✗ | ~ (vote) | ✓ |
| Estornell & Liu, NeurIPS 2024 (OpenReview sy7eSEXdPC) | ✗ (multi-LLM) | ✓ | ✓ (ADA-2) | ✓ | ✗ (prunes broadcast pool) | ~ (max-diversity subset) | ✗ | ✗ | ✗ | ✗ | ✓ |
| Diversity of Thought 2410.12853 | ~ (multi-model) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| MAD as Test-Time Scaling 2505.22960 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✓ |
| Debate or Vote 2508.17536 (NeurIPS 2025) | ✓ | ✓ | ✗ (theory aside only) | ✗ | ✗ | ✗ | ✗ | ✗ | ~ (Sparse MAD) | ✓ | ✓ |
| Can LLM Agents Really Debate? 2511.07784 | ~ | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✓ |
| Demystifying MAD 2601.19921 | ✓ | ✓ | ✗ (unique-answer counts) | ✗ | ✗ | ~ (greedy init pool) | ✗ | ✗ | ✗ | ~ | ✗ (GRPO/LoRA) |
| DAR / Hear Both Sides 2603.20640 | ✓ | ~ | ✗ (metric only) | ✗ | ✗ | ~ (LLM-judged subset→broadcast) | ✗ | ✗ | ✗ | ~ | ✓ |
| **CortexDebate 2507.03928 (ACL 2025 Findings)** | ✗ (hetero.) | ✓ | ✓ | ✓ | ✓ (dissimilarity in edge weight) | ~ | ✗ | ✗ | ✗ | ~ (vote) | ✓ |
| **DySCo 2606.01828** | ? | ✓ | ~ | ~ | ✓ (1 of 5 edge signals) | ~ | ✗ | ✗ (one-way) | ✗ | ✓ | ✓ |
| **RMoA (ACL 2025 Findings)** | ✗ | ✗ | ✓ | ✓ | ~ | ~ (farthest-point subset for aggregation) | ✗ | ✗ | ✗ | ✓ | ✓ |
| S2-MAD 2502.04790 (NAACL 2025) | ✓ | ✓ | ✓ | ✓ | ~ (suppresses similar; inverted) | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| DyTopo 2602.06039 | ✗ | ✗ | ~ (need/offer descriptors) | ~ | ~ (routes by SIMILARITY) | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| MACE 2607.11250 | ✓ | ✓ | ✗ (n-gram bandit) | ~ | ✓ (LinUCB peer selection) | ~ | ✗ | ✗ | ✗ | ✓ | ✓ |
| GPTSwarm 2402.16823 (ICML 2024) | ✓ | ✓ | ✗ | ✗ | ✗ (learned REINFORCE edges) | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| DyLAN 2310.02170 (COLM 2024) | ✓ | ✓ | ✗ | ✗ | ✗ (LLM-ranker pruning) | ✗ | ✗ | ~ | ✗ | ~ | ✓ |
| AgentPrune 2410.02506 (ICLR 2025) | ✓ | ✓ | ✗ | ✗ | ✗ (trained masks) | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| G-Designer 2410.11782 | ✓ | ✗ | ✗ (profile/query embeds) | ✗ | ✗ (trained VGAE) | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| MacNet 2406.07155 | ✓ | ✓ | ✗ | ✗ | ✗ (static DAGs) | ✗ | ✗ | ✗ (asymmetric) | ~ | ✓ | ✓ |
| Graph-GRPO 2603.02701 | ✓ | ✗ | ✗ (role⊕query) | ✗ | ✗ (trained GNN policy) | ✗ | ✗ (DAG) | ✗ | ✗ | ✓ | ✗ |
| Guided Topology Diffusion 2510.07799 (ACL 2026) | ✓ | ✗ | ✗ (query embeds) | ✗ | ✗ (trained diffusion) | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Codebook Agent 2609.02264 | ✓ | ✗ | ✗ (query embeds) | ✗ | ✗ (trained codebook) | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Xu et al. Peer Review 2311.08152 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (all-to-all) | ✗ | ~ (vote) | ✓ |
| PiCO 2402.01830 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ (explicitly RANDOM) | ✓ | ✓ | ✓ | ~ | ✓ |
| Sparse MAD 2406.11776 (EMNLP 2024 Findings) | ✓ | ✓ | ✗ | ✗ | ✗ (fixed topologies) | ✗ | ✗ | ~ | ~ | ✓ | ✓ |
| QD Debate Tournament 2510.05909 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ (paired by skill standing) | ~ (Swiss) | ✗ (adversarial) | ~ | ~ | ✓ |
| DIVSE 2310.07088 | ✓ | ✗ (prompt-level diversity) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ (vote) | ✓ |
| Slim-SC 2509.13990 / SSDP 2511.08595 / DeepPrune 2510.08483 | ✓ | ✓ | ~ (trace similarity) | ~ | ✗ (single-agent pruning) | ✗ | ✗ | ✗ | ✗ | ~ | ✓ |

## 4. Closest prior art (must differentiate in the paper)

1. **CortexDebate (arXiv:2507.03928)** — the single closest work. Edge weights include output cosine dissimilarity, so response-level distance does shape a sparse debate graph (E ✓). But: heterogeneous models; dissimilarity is one additive term in a trust formula; result is a thresholded sparse broadcast graph, NOT a perfect matching (G ✗); communication is debate rounds, not reciprocal pair-limited review (H, I ✗); aggregation is majority vote, not a synthesizer. FPRR differs in mechanism (matching + reciprocal review + synthesis) and in the scientific question (who reviews whom, causally isolated against random pairing on identical cached responses).
2. **Estornell & Liu (NeurIPS 2024)** — embedding-based max-diversity pruning + echo-chamber/misconception theory. Diversity selects which agents stay in a broadcast pool; it never builds pairwise topology, no matching, no reciprocal review.
3. **DySCo (arXiv:2606.01828)** — "cosine distance between reasoning embeddings" is one of five optional edge-scoring signals; per-receiver top-k senders, one-way critique; no matching, no reciprocity.
4. **RMoA (ACL 2025 Findings)** — farthest-point-style greedy diversity selection, but over which responses enter an aggregation mixture, not which agent reviews whom.
5. **S2-MAD (arXiv:2502.04790)** — uses response cosine similarity with the OPPOSITE polarity (filters out redundant similar viewpoints in fixed groups).
6. **Xu et al. (arXiv:2311.08152)** — reciprocal peer review exists, but on an all-to-all graph with no distance signal and full visibility.

Negative corroboration: the 141-study MAD survey (arXiv:2607.26212) reports 88.7% static topologies and contains no embedding-distance pairing anywhere in its taxonomy. GitHub search found no repository implementing embedding-based review-pair matching.

## 5. Stop-condition assessment

The stop condition requires a single paper implementing:
same-model independent generation → semantic-distance **farthest pairing** → **reciprocal pairwise review** → final aggregation.

- No examined paper satisfies E+F+G+H together; **none satisfies even F+G** (farthest selection as a one-to-one matching).
- G (perfect matching / one-to-one pairing) appears only in PiCO (pairing explicitly random) and the QD tournament (paired by skill, adversarial).
- H (reciprocal i↔j review) appears only in Xu et al. (all-to-all, no distance) and PiCO (random pairing).
- The combination is absent. **STOP condition NOT triggered.**

## 6. Verdict and conservative novelty claim

**GO.**

Conservative novelty statement (to be used in the paper):

> A training-free, response-conditioned peer-review topology that converts semantic disagreement among homogeneous same-prompt LLM samples into disassortative reciprocal review pairs.

We explicitly do NOT claim: first multi-agent debate system; first diversity-aware debate; first semantic-distance-based multi-agent method; first adaptive multi-agent topology. CortexDebate, Estornell & Liu, DySCo, RMoA, and S2-MAD are cited as the nearest neighbors and differentiated as above.

## 7. Caveats

- Estornell & Liu has no arXiv version; analysis is from the NeurIPS 2024 proceedings PDF.
- Two very recent items (Ha & Mitchell ICML 2026; one anonymous ACL 2026 submission; AgentDropout-ECNCT) were assessed via abstracts/snippets due to access walls; none shows signs of distance-based reciprocal pairing.
- September-2026 indexing may lag; the survey-level negative evidence (2607.26212, through ~July 2026) and six independent search streams make a missed exact duplicate unlikely.
