# Novelty-Gate Literature Search — Seed Papers (5)–(10)

**Proposed method under evaluation (FPRR):** N homogeneous instances of the SAME LLM checkpoint, same system prompt, same problem, same decoding config, generate answers independently → responses embedded with a sentence-embedding model → pairwise cosine-distance matrix → agents PAIRED by farthest semantic distance (randomized greedy farthest matching; separately global maximum-weight perfect matching) → reciprocal peer review within each pair (i reviews j AND j reviews i; each sees only problem + own answer + partner's answer; no ground truth, no distance info) → single synthesizer LLM aggregates all answers + reviews into one final answer. No training, no retrieval, no tools.

**Literature cutoff:** 2026-09-17. **Search date:** 2026-09-17.

**Verification status:** All six assigned arXiv IDs **resolve and match their stated titles** (verified via arXiv abs pages and the arXiv export API). Metadata (authors, dates) pulled from `export.arxiv.org/api/query`.

**Criteria key:**
- A. homogeneous instances of the same LLM
- B. same-prompt independent initial solving
- C. response-level semantic embeddings
- D. pairwise semantic distance between responses
- E. distance used to construct communication/review topology
- F. farthest/dissimilar peer selection
- G. one-to-one pairing / perfect matching
- H. reciprocal critique (i reviews j AND j reviews i)
- I. pair-limited information exchange
- J. final aggregation/synthesis stage
- K. no training / no model-weight modification

---

## (5) arXiv:2410.12853 — Diversity of Thought Elicits Stronger Reasoning Capabilities in Multi-Agent Debate Frameworks

- **Authors:** Mahmood Hegazy (University of Montreal; Mila)
- **Venue:** J Robot Auto Res, 2024, Vol. 5, Issue 3
- **Year:** 2024 · **URL:** https://arxiv.org/abs/2410.12853

**Method summary.** Extends Du et al. (2023) multi-agent debate to heterogeneous model families. Three debating LLMs (e.g., Gemini-Pro, Mixtral 7B×8, PaLM 2-M; homogeneous control: 3× Gemini-Pro) each answer a math question independently ("round 0"). Each round, all three responses go to a fourth model (Response Summarizer) whose summary is fed back to all debating models; after n rounds the final summary (plus answer mode) is the output. The only "diversity" mechanism is choosing different model architectures a priori. No embeddings, distances, pairing, matching, or reciprocal critique anywhere (verified by full-PDF grep: zero occurrences of `embed|cosine|semantic|pair|matching|distance|reciprocal`).

**A–K table:**

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Partial | Homogeneous 3× Gemini-Pro exists only as a control; proposed method uses "diverse model families" |
| B | Yes | "we begin by asking each agent to directly generate responses to the given prompts without engaging in debate... round 0"; "identical starting prompts" |
| C | No | Not found (zero occurrences of embed/cosine in full text) |
| D | No | Not found |
| E | No | Topology is fixed all-to-all-via-summarizer |
| F | No | Diversity is model-family-level, fixed a priori: "diverse model families" |
| G | No | No pairing; 3 agents interact jointly through summarizer |
| H | No | Models "critically examine and build upon each other's arguments" only via a shared summary |
| I | No | Opposite: "the responses from all three debating models are passed through a fourth model... fed back as input to the debating models" |
| J | Yes | "the final summarized response from the summarizer model (Model 4) is considered as the output" |
| K | Yes | Pure inference-time debate; no fine-tuning |

**Verdict: related only.** Shares the generic MAD frame (B, J, K; partial A). None of C–I. No counterpart to E+F+G+H.

---

## (6) arXiv:2505.22960 — Revisiting Multi-Agent Debate as Test-Time Scaling: A Systematic Study of Conditional Effectiveness

- **Authors:** Yongjin Yang, Euiin Yi, Jongwoo Ko, Kimin Lee, Zhijing Jin, Se-Young Yun (KAIST AI; MPI-IS; U. Toronto; Vector Institute)
- **Venue:** arXiv preprint (v1 2025-05-29, v2 2025-06-20) · **URL:** https://arxiv.org/abs/2505.22960 · Code: github.com/euiin/MAD_as_TTS

**Method summary.** Empirical analysis paper, not a new method. Studies vanilla MAD (Du et al.) as test-time scaling vs. self-agent baselines (self-consistency, self-refinement) under matched budgets. Homogeneous MAD: M instances of the same LLM answer independently in round 1, then each round **every agent sees ALL agents' previous responses** (full broadcast) and refines. Heterogeneous variants use different model families or personas. Final answer: majority vote (math) or judge (safety). No embeddings, distances, pairing, or pairwise review anywhere.

**A–K table:**

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Yes | "In homogeneous MAD, all participating agents … are instances of the same underlying language model p" (§2.2) |
| B | Yes | `o_{j,1} = p(q, I_j)` — independent round-1 answers (§2.2, Eq. 2) |
| C | No | No sentence-embedding model anywhere |
| D | No | No distance computation; agreement used only for majority vote / early stopping |
| E | No | Fixed all-to-all: "each agent in MAD considers all previous outputs O_{t-1}" (§2.3) |
| F | No | Diversity via "different model families" or "personas", never most-dissimilar selection |
| G | No | 2/4/8 agents debate jointly; no pairing |
| H | No | Implicit broadcast refinement, no structured i↔j review |
| I | No | "the full context from previous rounds is shared among all agents" (Fig. 1) |
| J | Partial | Majority voting / judge, not an LLM synthesizer (§3) |
| K | Yes | Pure test-time inference |

**Verdict: related only.** Background/baseline citation (MAD as test-time scaling; diversity analysis). Entire FPRR core (C–I) absent; future work explicitly lists "message-passing structures" as unexplored.

---

## (7) arXiv:2508.17536 — Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?

- **Authors:** Hyeong Kyu Choi, Xiaojin Zhu, Sharon Li (U. Wisconsin–Madison)
- **Venue:** arXiv preprint (v1 2025-08-24, v2 2025-10-23); reported as NeurIPS 2025 · **URL:** https://arxiv.org/abs/2508.17536 · Code: github.com/deeplearning-wisc/debate-or-vote

**Method summary.** Analysis paper disentangling MAD into majority voting + inter-agent debate over a communication graph 𝒢. Evaluates Decentralized (fully connected), Sparse, and Centralized MAD topologies with N=5 homogeneous agents; diversity from sampling stochasticity or persona prompts. Builds a Dirichlet-Compound-Multinomial model, proves debate induces a martingale over belief (debate alone doesn't improve expected correctness), proposes belief-biasing interventions (MAD-oracle/-Conformist/-Follower). No embedding-based pairing, no farthest matching, no peer-review stage.

**A–K table:**

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Yes | "we focus on homogeneous agent settings, i.e., all agents share the same underlying model architecture or behavior" |
| B | Yes | "each agent independently generates an initial response" |
| C | No | Only a theoretical aside: count vector "can be viewed more broadly, e.g., … an embedding-based semantic agreement measure" — never implemented |
| D | No | Not found |
| E | No | Topology fixed a priori: "We formalize the communication structure of debate as an undirected graph 𝒢"; Sparse MAD sparsity is for efficiency, not semantic distance |
| F | No | Not found (cites PRD [29] re "agent pairs for debate" — external work, not theirs) |
| G | No | Neighborhoods are arbitrary-size sets |
| H | No | Agents observe neighbor answers and update; no review artifact, no i↔j exchange |
| I | Partial | Sparse MAD limits observation to graph neighbors, but not pairs and not distance-chosen |
| J | Yes | "the final answer is typically derived through an aggregation mechanism, such as majority voting" — vote, not synthesizer |
| K | Yes | Inference-time prompting only |

**Verdict: related only.** Generic outer frame shared (A, B, J, K; partial I). E+F+G+H entirely absent; communication is broadcast over a fixed graph.

---

## (8) arXiv:2511.07784 — Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Debate in Logical Reasoning

- **Authors:** Haolun Wu (McGill/Mila), Zhenkun Li (U. South Florida), Lingyao Li (U. South Florida)
- **Venue:** arXiv preprint (v1 2025-11-11) · **URL:** https://arxiv.org/abs/2511.07784

**Method summary.** Controlled study of broadcast MAD on Knight–Knave–Spy puzzles: (1) independent initial proposals with confidence; (2) player-by-player debate loop where agents take turns presenting arguments to the whole group, then self-adjust; (3) per-player majority vote with a supervisor tie-breaker. Six factors varied (team size, composition, confidence visibility, order, depth, difficulty). Diversity = team composition on an accuracy–confidence grid mixing different models. No embeddings, no distances, no pairing, no pairwise review; communication is full broadcast.

**A–K table:**

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Partial | Homogeneous teams only as baselines ("Hom-Mix Strong/Weak... baselines without model diversity"); anchor config is heterogeneous |
| B | Partial | "Each agent independently assigns roles to all players and reports a confidence score" — independence explicit, prompt identity not stated |
| C | No | No embedding model in §4–5 or appendices |
| D | No | Diversity measured at model level (accuracy–confidence bins), not responses |
| E | No | Fixed broadcast debate loop |
| F | No | "Het-Mix C maximizes diversity across both dimensions" — model-mixing, not dissimilar-response selection |
| G | No | Full-team turn-taking; no pairing |
| H | No | Group critique: "Agents take turns presenting arguments... They may agree or disagree with peers" |
| I | No | "After hearing all arguments about player i, each agent reviews the discussion" — everyone hears everyone |
| J | Partial | Per-player majority vote + supervisor tie-break; not a synthesizer |
| K | Yes | Pure prompting/inference |

**Verdict: related only.** Controlled evaluation of standard broadcast debate; diametrically opposite communication structure to FPRR. No element of E+F+G+H.

---

## (9) arXiv:2601.19921 — Demystifying Multi-Agent Debate: The Role of Confidence and Diversity

- **Authors:** Xiaochen Zhu, Caiqi Zhang (equal), Yizhou Chi, Tom Stafford, Nigel Collier, Andreas Vlachos (Cambridge; Sheffield)
- **Venue:** arXiv preprint (v1 2026-01-09; v3 2026-06-03 — within cutoff) · **URL:** https://arxiv.org/abs/2601.19921 · Code: github.com/SpaceHunterInf/DMAD

**Method summary.** Diagnoses why vanilla MAD underperforms majority vote, proposes two interventions. (1) **Diversity-aware initialisation:** sample N_cand=10 candidates, greedily select N maximizing diversity defined as **the count of distinct answer strings** `div(S) = |unique(S)|` — exact-match based, no embeddings or distances. (2) **Confidence-modulated debate:** verbalized 0–10 confidence, **trained with RL (GRPO + LoRA)**. Debate itself is standard fully-connected broadcast; final output is majority vote. No pairing, matching, or peer review.

**A–K table:**

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Yes | "we initialise 5 homogeneous agents using the same language model"; "all agents share identical parameters (θ_i=θ), and stochasticity arises solely from sampling" |
| B | Yes | "At round t=1, each agent independently samples an initial answer y_{i,1} ∼ a_i(x)" |
| C | No | "we define its diversity as the number of distinct answers: div(S) = \|unique(S)\|" — a count, not embeddings |
| D | No | No distance matrix; set cardinality of unique answer strings |
| E | No | "our method preserves the standard MAD protocol"; agents "homogeneous and fully connected" — diversity picks pool members, not who talks to whom |
| F | No | Greedy subset selection maximizing distinct-answer count for the initial pool, not dissimilar pairing of agents |
| G | No | Fully connected graph; no pairing/matching |
| H | No | No review step: "each agent then revises its answer by applying an answer-update operator y_{i,t} = D(x, R_t)" |
| I | No | "every agent observes all the answers produced by other agents and itself in the previous round" |
| J | Partial | "the ensemble-level MAD output is obtained by majority vote over the terminal responses" — vote, not synthesizer |
| K | Partial | Diversity init is training-free, but the confidence component "train[s] the model... using reinforcement learning (RL)... GRPO... LoRA" |

**Verdict: related only** (closest of the six on motivation). Shares A, B and the premise "initial diversity helps homogeneous-agent debate," but operationalizes diversity as unique-answer counting for pool initialisation, keeps broadcast topology, uses RL training for its headline component, and contains none of E+F+G+H.

---

## (10) arXiv:2603.20640 — Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention

- **Authors:** Manh Nguyen, (Tien) Anh Nguyen, Dung Nguyen, Svetha Venkatesh, (Thai) Hung Le (Deakin University)
- **Venue:** arXiv preprint (v1 2026-03-21; v2 2026-04-14 — within cutoff) · **URL:** https://arxiv.org/abs/2603.20640 · Code: github.com/DA2I2-SLM/DAR

**Method summary.** DAR: broadcast MAD where each round an LLM **filter agent** ℱ receives all N previous-round responses plus the last majority vote and selects a subset of agent indices "whose opinions differ the most from each other and from the majority vote"; retained messages are broadcast unchanged to all agents next round. Final answer is majority vote. Disagreement is judged **by the LLM filter reading texts**, not embedding geometry; all-MiniLM-L6-v2 (1−cosine) appears **only in a §4.3 evaluation analysis** of retained-set diversity. No pairing, matching, or review step; explicitly "requires no topological changes."

**A–K table:**

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Yes | Homogeneous setting; one backbone per debate (Qwen2.5, Falcon3, Llama3.1) |
| B | Partial | Algorithm 1 input `G₀ = {g₀,₁,…,g₀,N}` — independent round-0 generation; identical prompts not stated |
| C | No (analysis only) | "We measure diversity as the average pairwise embedding distance (1−cosine similarity)... We use all-MiniLM-L6-v2" (§4.3) — evaluation metric, not the mechanism; selection is "ℱ is prompted to retain maximally diverse (i.e., disagreeing) answers" (§3.4) |
| D | No (analysis only) | Same §4.3 quote; Algorithm 1 has no embedding step |
| E | No | "our filtering module operates purely at the generation selection stage and requires no topological changes" (§3.4); "agnostic to the underlying communication topology" (§1) |
| F | Partial | Does select most-disagreeing responses — but as a broadcast subset chosen by an LLM judge ("choose agents whose opinions differ the most from each other and from the majority vote", Fig. 5), not farthest-distance geometry, and not pairing |
| G | No | Contrasts itself with topology work; keeps broadcast with filtered subsets |
| H | No | No review/critique stage; agents regenerate answers from retained messages |
| I | No | Agents see the retained subset of all peers' messages |
| J | Partial | "Compute final vote: v_R = mode({ans(g_{R,i})})" (Alg. 1, line 19) — vote, not synthesizer |
| K | Yes | "training-free and compatible with existing MAD pipelines" (§3.4) |

**Verdict: related only** (same neighborhood: diversity/disagreement-aware, homogeneous, training-free MAD). Diversity decides **which messages are retained for broadcast**, never **who talks to whom**; embeddings are post-hoc measurement only. No element of E+F+G+H as implemented mechanism.

---

## Cross-paper summary

| Paper | A | B | C | D | E | F | G | H | I | J | K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2410.12853 Diversity of Thought | ~ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| 2505.22960 MAD as Test-Time Scaling | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✓ |
| 2508.17536 Debate or Vote | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✓ | ✓ |
| 2511.07784 Can LLM Agents Really Debate? | ~ | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✓ |
| 2601.19921 Demystifying MAD | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ~ |
| 2603.20640 Hear Both Sides (DAR) | ✓ | ~ | ✗* | ✗* | ✗ | ~ | ✗ | ✗ | ✗ | ~ | ✓ |

\* embeddings/cosine distance appear only as an evaluation metric, never in the algorithm.

**E+F+G+H co-occurrence:** No paper satisfies even two of E, F, G, H as implemented mechanisms. The closest approaches: DAR (F-partial: dissimilar subset selection by LLM judge, broadcast not pairing) and DMAD (diversity-aware subset selection at initialisation, exact-match counts not embeddings).

**Bottom line:** No exact prior art among seed papers (5)–(10). The full FPRR chain — same-model independent generation → sentence-embedding cosine-distance matrix → farthest-distance pairing (greedy or max-weight perfect matching) → reciprocal pairwise peer review → LLM synthesis — is not implemented by any of the six. All six use broadcast or fixed-graph debate topologies, operationalize diversity via model/persona heterogeneity, exact-match answer counts, or LLM-judged disagreement, and aggregate by majority vote rather than review-informed synthesis.
