# Novelty-Gate Literature Search: FPRR (Farthest-Pair Reciprocal Review)

**Date:** 2026-09-17 (literature cutoff: 2026-09-17; anything later ignored)
**Scope:** Verification and deep read of the four mandatory seed papers + cited-by snowballing on arXiv:2305.14325 (Semantic Scholar, 2,235 citing papers scanned) + supplementary web search for response-dissimilarity-based agent pairing/routing.

## The method under evaluation (FPRR)

N homogeneous instances of the SAME LLM checkpoint, same system prompt, same problem, same decoding config, generate answers independently (stochastic sampling is the only diversity source) → each response (final answer + concise justification) is embedded with a sentence-embedding model → pairwise cosine-distance matrix → agents paired by farthest semantic distance (randomized greedy farthest matching; separately global maximum-weight perfect matching) → within each pair, reciprocal peer review (i reviews j and j reviews i; each sees only problem + own answer + partner's answer; no ground truth, no distance info) → a single final synthesizer (same LLM) aggregates all original answers + all reviews. No training, no fine-tuning, no retrieval, no external knowledge, no tools.

## Criteria legend

A. homogeneous instances of the same LLM · B. same-prompt independent initial solving · C. response-level semantic embeddings · D. pairwise semantic distance between responses · E. distance used to construct communication/review topology · F. farthest/dissimilar peer selection · G. one-to-one pairing / perfect matching · H. reciprocal critique (i reviews j AND j reviews i) · I. pair-limited information exchange · J. final aggregation/synthesis stage · K. no training / no weight modification

---

## Seed paper 1 — Du et al., Multiagent Debate

**Citation:** Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch. "Improving Factuality and Reasoning in Language Models through Multiagent Debate." arXiv:2305.14325 (v1 23 May 2023; later ICML 2024). https://arxiv.org/abs/2305.14325 — full text read via ar5iv.

**Method summary:** Multiple instances (typically 3; ablations to 5+) of the same LLM (gpt-3.5-turbo-0301) independently generate answers with the same starting prompt. For ~2 rounds, every agent receives a "consensus prompt" containing the concatenated responses of ALL other agents and updates its answer; for large N, other agents' responses are first summarized by the LLM. The population converges to a consensus answer, extracted from final-round responses. Side experiments mix ChatGPT/Bard and persona prompts.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | **Yes** (primary setup) | "multiple agents represented as copies of a large language model"; "We use chatGPT-based language model in all our experiments except those in Figure 11..." |
| B | **Yes** | "we first prompt each agent to independently solve the given problem or task"; "In our experiments we use the same prompts for all agents." |
| C | **No** | No embeddings anywhere; "Individual responses from other agents are concatenated and given as context to each agent." |
| D | **No** | No distance computation; consensus is LLM-judged textually. |
| E | **No** | Fixed fully-connected topology; summarization for large N, not distance-based routing. |
| F | **No** | Diversity observed but never measured/used: "individual model instances propose a diverse range of answers despite being the same model class." |
| G | **No** | N-way broadcast every round; no pairing/matching. |
| H | **Partial** | Implicit all-to-all mutual critique ("verifying the collection of responses given by other agents, and refining its own response"), not pairwise reciprocal review. |
| I | **No** | Each agent sees ALL other agents' responses (or a summary of all). |
| J | **Partial** | Converged consensus extracted from last-round responses; no dedicated synthesizer over answers+reviews. |
| K | **Yes** | "we require only black-box access to language model generations – no model-internal information such as likelihoods or gradients is needed." |

**Semantic-distance pairing/routing/topology?** None whatsoever.
**Verdict: strong overlap** (canonical seed for homogeneous multi-instance debate: A, B, K yes; H, J partial) — not prior art for FPRR's distinguishing machinery (C–G, I absent).

---

## Seed paper 2 — Chen et al., ReConcile

**Citation:** Justin Chih-Yao Chen, Swarnadeep Saha, Mohit Bansal. "ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs." arXiv:2309.13007 (v1 22 Sep 2023; ACL 2024). https://arxiv.org/abs/2309.13007 — full text read via ar5iv.

**Method summary:** Three DIFFERENT LLMs (ChatGPT, Bard, Claude2; GPT-4 variant) each independently produce an answer, CoT explanation, and verbal confidence. Up to 3 rounds of fully-connected discussion: each agent sees all agents' grouped answers/explanations/confidences plus few-shot "convincing samples." Stops on consensus or round limit; final answer by confidence-weighted vote. Ablation shows model diversity is critical.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | **No** (opposite design) | "Each agent is a distinct LLM, potentially trained with different pre-training data and model architectures." Same-model appears only as a weaker ablation ("w/o Multiple Models"). |
| B | **Partial** | Independent initial responses via zero-shot CoT prompt, but each agent gets its own convincing-sample demonstrations. |
| C | **No** | No sentence embeddings; responses grouped by discrete answer label. |
| D | **No** | No distance matrix or similarity computation between responses. |
| E | **No** | Fixed fully-connected round table; no topology construction. |
| F | **No** | No peer selection; "convincing samples" selected from training data by human-explanation rectification. |
| G | **No** | "all agents generating a revised explanation and answer based on all other agents' explanations and answers from the previous round." |
| H | **Partial** | Group-level mutual revision, no structured pairwise reciprocal review artifact. |
| I | **No** (explicitly opposite) | "𝒟ᵢ⁽ʳ⁾ consists of the answers ... and explanations ... of all agents from the previous round." |
| J | **Yes** (but voting, not LLM synthesis) | "we employ these confidences to compute a weighted vote as the final answer." |
| K | **Yes** | Pure inference-time prompting; no learned recalibration. |

**Semantic-distance pairing/routing/topology?** None. Diversity is exogenous (different model families).
**Verdict: related only.** Architecturally opposite to FPRR on A, I, J; C–G all absent. Mandatory related-work citation, no novelty threat.

---

## Seed paper 3 — Li et al., More Agents Is All You Need

**Citation:** Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, Deheng Ye. "More Agents Is All You Need." arXiv:2402.05120 (v1 3 Feb 2024; v2 11 Oct 2024). https://arxiv.org/abs/2402.05120 — full text read via ar5iv (v2).

**Method summary:** Agent Forest = sampling-and-voting. The same query is fed to a single LLM N times (diversity purely from stochastic decoding; temperature/nucleus ablated in §5.4). Voting: each sample scored by cumulative similarity to all others, V(s_i) = Σ_{j≠i} sim(s_i, s_j); answer = arg max. Similarity is occurrence frequency / math-equivalence for closed tasks, BLEU for code. No inter-agent communication, no review, no pairing, no synthesizer. Step-wise/Hierarchical variants recurse over subtasks.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | **Yes** | "the query of the task ... is iteratively fed into a single LLM ... to generate multiple outputs." |
| B | **Yes** | "we generate N samples by solely querying the LLM M N times with each sample represented as s = M(x)"; "each input remains the same when we increase the number of agents." |
| C | **No** | No embedding model; similarity is occurrence frequency or BLEU (surface n-gram overlap). |
| D | **Partial** | Pairwise similarity computed (V(s_i) = Σ sim(s_i, s_j)), but BLEU/frequency, not semantic embeddings. |
| E | **No** | No communication graph; similarity feeds only the vote. |
| F | **No** (opposite direction) | Selects the MOST similar (medoid-like) sample: arg max cumulative similarity. |
| G | **No** | No pairing/matching anywhere in Algorithm 1 or the paper. |
| H | **No** | No critique step; debate/reflection appear only as external baselines it can wrap. |
| I | **No** | No information exchange between samples at all. |
| J | **Partial** | Aggregation exists but is an argmax vote, not an LLM call. |
| K | **Yes** | "our method is unsupervised, without the need for additional training data." |

**Semantic-distance pairing/routing/topology?** Explicitly none; deliberately avoids interaction structures.
**Verdict: related only** (foundational baseline for FPRR's front end: A, B, K; D, J partial). Mandatory baseline citation.

---

## Seed paper 4 — Estornell & Liu, Multi-LLM Debate

**Citation:** Andrew Estornell (ByteDance Research), Yang Liu (UC Santa Cruz). "Multi-LLM Debate: Framework, Principals, and Interventions." NeurIPS 2024 (poster), NeurIPS 37:28938–28964. **No arXiv preprint exists** (verified via arXiv API search by title and author); OpenReview id sy7eSEXdPC; DOI 10.52202/079017-0911. Full text read from the official NeurIPS PDF (incl. appendices, Algorithm 1).

**Method summary:** Theoretical analysis of standard broadcast debate (echo-chamber/tyranny-of-majority theorems) plus three interventions on the shared response pool: (i) Diversity Pruning — keep k of n responses maximizing pairwise KL divergence, using sentence-embedding distance (ADA-2) as proxy; (ii) Quality Pruning — keep k closest to the question; (iii) Misconception Refutation — an LLM lists errors in each kept response and minimally corrects it. All agents see the SAME pruned pool. Experiments: 6 agents × 10 rounds, homogeneous (6× GPT-3.5, 6× Llama-3, etc.) and heterogeneous configurations. Final answer = last-round responses.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | **Partial** | Homogeneous debates run in experiments ("6× GPT-3.5, 6× Llama-3...") and theory (Thm 5.1, "n copies of the same model"), but homogeneity is framed as the echo-chamber PATHOLOGY; headline setup is heterogeneous "Multi-LLM." |
| B | **Yes** | Algorithm 1: "get response z_j^(0) by prompting LLM_j ... // Get an initial set of responses from each LLM." |
| C | **Yes** | "For sentence embeddings (which serve as a proxy of the latent concepts Θ), we use sentence embeddings from ADA-2." |
| D | **Yes** | Diversity pruning maximizes "∑_{zi,zj ∈ Z} KL(D(θ\|zi), D(θ\|zj))"; "we use the distance between sentence embeddings of each string"; Fig. 2 uses "pairwise cosine similarity." |
| E | **No** (closest partial) | Distance selects a response SUBSET for the shared pool; all agents see the same pruned set. No per-agent routing/topology. |
| F | **Partial** | Dissimilarity-based selection of responses (max-diversity subset), not dissimilar PEER pairing. |
| G | **No** | No pairing or matching anywhere. |
| H | **No** | Misconception Refutation is centralized one-directional LLM editing, not reciprocal review. |
| I | **No** | Every agent sees the whole pool: "Several other models have provided responses ... Model 1: ... Model n: ..." |
| J | **No** | "Return Zall[-n:] // Each model's response on the last round of debate." No synthesizer. |
| K | **Yes** | Pure inference-time; off-the-shelf API/vLLM models. |

**Semantic-distance pairing/routing/topology?** No — embeddings + pairwise distance exist but only for pool pruning and analysis; topology stays all-to-all broadcast.
**Verdict: strong overlap on diagnosis and ingredients (C, D, partial F; echo chambers, diversity pruning, misconception refutation all addressed), related only on mechanism.** Must-cite motivational prior; not prior art on FPRR's pairing mechanism.

---

## Snowball search (cited-by on arXiv:2305.14325 + web search)

**Method:** Semantic Scholar API `/graph/v1/paper/arXiv:2305.14325/citations`, all 2,235 citing papers paged; keyword-filtered titles/abstracts for semantic/embedding/cosine/dissimilar/farthest/matching/pairing/misconception/echo-chamber/diversity/topology/peer-review/routing/distance (548 hits), ~60 triaged, method sections full-text-read for every plausible candidate. **Zero hits in all 2,235 abstracts for "farthest", "perfect matching", "maximum-weight matching", "bipartite matching" (pairing sense), or "agents are paired".**

### Candidate 1 — CortexDebate (STRONGEST OVERLAP found)

**Citation:** Sun, Zhao, Wan, Gong. "CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate." Findings of ACL 2025. arXiv:2507.03928 (v1 5 Jul 2025). https://arxiv.org/abs/2507.03928 — full text read (HTML v1).

**Method summary:** Heterogeneous LLM agents (Qwen/Mistral/Typhoon/Llama/Gemma) debate over a sparse directed graph. Edge weight W(i→j) = C×R×I/S (McKinsey Trust Formula), where "intimacy" I = 1 − mean pairwise cosine similarity of the two agents' past outputs — response dissimilarity RAISES edge weight. Edges below each node's mean incoming weight are pruned each round. Agents read connected neighbors and regenerate; final answer by majority vote.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | **No** | "The backbone models involved in the debating system... are Qwen-2.5-7B..., Mistral-7B..., Typhoon-1.5-8B..., Llama-3.1-8B..., and Gemma-2-9B." |
| B | **Partial** | Independent initial answers but different models/prompts. |
| C | **Partial** | "MDM first uses cosine similarity to calculate the textual similarity between Oᵢ and Oⱼ" (embedding model unspecified). |
| D | **Yes** | Pairwise response cosine distance computed. |
| E | **Yes** | "Intimacy represents the average degree of difference in viewpoints between Aᵢ and Aⱼ in history debates... I_d = 1 − Sim̄_d"; "the edges with weights below W̄ are removed, resulting in a sparse debating graph." |
| F | **Partial** | Dissimilarity raises edge weight, but mixed with credibility/reliability/self-orientation; per-node threshold pruning, not farthest selection. |
| G | **No** | Sparse directed graph, arbitrary in-degree; no one-to-one pairing or perfect matching. |
| H | **Partial** | Agents "scrutinize the outputs of the LLM agents connected to it"; directed edges, no explicit reciprocal i↔j review protocol. |
| I | **Partial/Yes-ish** | "each LLM agent only debates with the ones that are helpful to it" — neighborhoods, not pairs. |
| J | **Partial** | Majority vote, not an LLM synthesizer. |
| K | **Yes** | Inference-time. |

**Verdict: strong overlap** — the ONLY paper found that wires a debate graph using pairwise response cosine dissimilarity. But heterogeneous models, trust-formula weighting, no pairing/perfect matching, no reciprocal pair-review stage, no synthesizer. **Must-cite / must-differentiate paper for FPRR.**

### Candidate 2 — DyTopo (related only)

Lu, Zhao, Cao. "DyTopo: Dynamic Topology Routing for Multi-Agent Reasoning via Semantic Matching." arXiv:2602.06039 (v1 5 Feb 2026). Agents emit natural-language query/key descriptors embedded with all-MiniLM-L6-v2; edge j→i active iff cosine(qᵢ, kⱼ) > τ. Embeddings + cosine + topology all present **but on agent-advertised descriptors matched by SIMILARITY** ("a communication link should exist from agent j to agent i if the semantic capacity offered by j aligns with the need of i"), not response dissimilarity. A no; G/H/I/J no; K yes. Opposite direction (similarity-matching).

### Candidate 3 — PEAR (related only)

He et al. "PEAR: Permutation-Equivariant Adaptive Routing Multi-Agent Debate." arXiv:2606.20621 (v1 26 May 2026, v2 31 Aug 2026). Router reassigns agents to roles of a k-regular graph each round; Targeted Diversity T(s,t) = 1[y_s ≠ y_t]·1[c_s ≥ τ]·1[c_t ≤ τ] — binary answer mismatch + confidence, NO embeddings ("routing scores based only on label-invariant features"). k-regular directed graph (k=2), not perfect matching; critiques one-directional; majority vote. K yes ("inference-time train-free").

### Candidate 4 — DAR / "Hear Both Sides" (related only)

Nguyen, Nguyen, Nguyen, Venkatesh, Le. arXiv:2603.20640 (v1 21 Mar 2026). A filter LLM retains indices of "agents whose opinions differ the most from each other and from the majority vote" — LLM-prompt-based selection; embeddings appear only as an EVALUATION metric ("We measure diversity as the average pairwise embedding distance (1 − cosine similarity) among retained responses... all-MiniLM-L6-v2"). Topology unchanged; majority vote. No distance computation in the selection loop.

### Candidate 5 — Nexa (related only)

Tastan et al. "Response-Conditioned Parallel-to-Sequential Orchestration for Multi-Agent Systems." arXiv:2605.15573 (v1 15 May 2026). "embeds the resulting responses into a shared semantic space, and then predicts a sparse directed acyclic communication graph" via a TRAINED lightweight transformer (policy-gradient). Response-conditioned topology exists, but graph is policy-predicted, not distance-constructed; K violated (trainable policy).

### Candidate 6 — Zhu et al., "Demystifying Multi-Agent Debate" (related only)

arXiv:2601.19921 (v1 9 Jan 2026; ACL 2026). Diversity-aware init: sample 10, greedily select N maximizing div(S) = |unique(S)| — distinct-answer-string counting, no embeddings. Notably homogeneous: "all agents share identical parameters (θᵢ=θ), and stochasticity arises solely from sampling" — matches FPRR's setup assumption; worth citing for that. Confidence module uses RL/GRPO+LoRA (K no for that component).

### Candidate 7 — ARMOR-MAD (related only)

Niu et al. arXiv:2606.13197 (v1 11 Jun 2026). Training-free heterogeneous MAD: Pre-debate Agreement Routing (skip debate on round-0 agreement), early stopping, Semantic Outlier Detection down-weighting abnormal final answers at aggregation. No pairing, no reciprocal review.

### Candidate 8 — AgentDropout, ECNCT 2026 version (related only; UNVERIFIED beyond abstract)

Xiao, Guo, Wang, Wang, Lu. "AgentDropout: Dynamic Redundancy Elimination for Multi-Agent Collaboration Efficiency." IEEE ECNCT 2026, DOI 10.1109/ECNCT70535.2026.11661404. No arXiv ID; NOT the same as AgentDropout arXiv:2503.18891. Full text paywalled — abstract only: "computes a semantic novelty score for every agent by measuring the divergence of its output relative to the current group consensus. Agents whose novelty score falls below an adaptive threshold are temporarily deactivated." Diversity-based participant pruning, not pairing. Flagged as unverified.

### Candidate 9 — The Geometry of Dialogue (related only)

arXiv:2510.26352 (Oct 2025). Builds a "language model graph" from "semantic coherence of pairwise conversations," clusters models into teams by SIMILARITY (community detection). Between-model teaming, opposite sign.

### Checked and rejected at abstract level (no embedding-distance pairing)

- TriAgent (arXiv:2607.19794) — Semantic Divergence Index routes QUERIES across heterogeneous tiers.
- HCP-MAD (arXiv:2604.09679) — "lightweight pair-agent debates," but pairs are heterogeneous models for consensus verification/early stopping, not chosen by semantic distance.
- AceMAD / "Breaking the Martingale Curse" (arXiv:2603.06801) — peer prediction of belief distributions + proper scoring rules; no embeddings, no pairing.
- "Disentangling Topology and Diversity" (arXiv:2609.14570, v1 13 Sep 2026 — within cutoff) — controlled topology × diversity study; proposes no distance-based pairing.
- "The Cost of Consensus" (arXiv:2605.00914) — homogeneous-debate failure analysis only.

---

## Cross-paper criteria matrix

| Paper | A | B | C | D | E | F | G | H | I | J | K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Du et al. 2305.14325 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ◐ | ❌ | ◐ | ✅ |
| ReConcile 2309.13007 | ❌ | ◐ | ❌ | ❌ | ❌ | ❌ | ❌ | ◐ | ❌ | ✅(vote) | ✅ |
| More Agents 2402.05120 | ✅ | ✅ | ❌ | ◐ | ❌ | ❌ | ❌ | ❌ | ❌ | ◐ | ✅ |
| Estornell & Liu NeurIPS'24 | ◐ | ✅ | ✅ | ✅ | ❌ | ◐ | ❌ | ❌ | ❌ | ❌ | ✅ |
| CortexDebate 2507.03928 | ❌ | ◐ | ◐ | ✅ | ✅ | ◐ | ❌ | ◐ | ◐ | ◐ | ✅ |
| DyTopo 2602.06039 | ❌ | ◐ | ✅ | ✅ | ✅(similarity) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| PEAR 2606.20621 | – | ◐ | ❌ | ◐ | ✅(labels) | ◐ | ❌ | ◐ | ◐ | ❌ | ✅ |
| DAR 2603.20640 | ❌ | ✅ | ❌ | ❌ | ❌ | ◐ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Nexa 2605.15573 | – | – | ✅ | ◐ | ◐(learned) | ❌ | ❌ | ❌ | ❌ | ◐ | ❌ |
| Demystifying MAD 2601.19921 | ✅ | ✅ | ❌ | ❌ | ❌ | ◐ | ❌ | ❌ | ❌ | ❌ | ❌/◐ |

✅ yes · ◐ partial · ❌ no · – not determinable from fetched text

## Bottom-line verdict

**No paper examined (four seed papers read in full + 2,235 citing papers screened + ~14 candidates deep-checked, all ≤ 2026-09-17) implements the FPRR chain:** same-checkpoint independent generation → response-embedding pairwise cosine-distance matrix → farthest-distance one-to-one pairing (greedy or maximum-weight perfect matching) → reciprocal pairwise peer review with pair-limited visibility → single-LLM synthesis of answers + reviews.

- **No paper hits E+F+G+H together.** E (distance→topology) appears only in CortexDebate (with F partial, G/H absent) and DyTopo/PEAR (different signals/directions). G (one-to-one pairing/perfect matching) appears in NONE of the examined papers. H (reciprocal i↔j review) appears in none as an explicit protocol. I (pair-limited visibility) appears in none.
- Closest neighbors, each covering only a fragment: **CortexDebate 2507.03928** (dissimilarity-weighted sparse debate graph — must-cite/must-differentiate); **Estornell & Liu NeurIPS 2024** (response embeddings + pairwise divergence + diversity pruning + misconception refutation for broadcast debate); DyTopo (semantic routing by similarity); PEAR (discrete-disagreement routing); DAR (LLM-based diversity retention); Nexa (learned response-conditioned topology).
- **Caveats:** (1) Estornell & Liu has no arXiv version; quotes from the official NeurIPS PDF. (2) AgentDropout-ECNCT full text paywalled — abstract-only verdict. (3) Semantic Scholar indexing of very recent (Sep 2026) papers may be incomplete. (4) The screen was keyword-based over titles/abstracts; a paper using unusual terminology for pairing-by-dissimilarity could have been missed.
