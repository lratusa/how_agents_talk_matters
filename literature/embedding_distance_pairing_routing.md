# Novelty-Gate Literature Search — Embedding/Distance-Based Pairing & Routing in Multi-Agent LLM Systems

- **Search area**: embedding/distance-based PAIRING or ROUTING of agents in multi-agent LLM systems (debate / peer review / collaboration).
- **Cutoff**: 2026-09-17. Search date: 2026-09-17.
- **Target method (FPRR)**: N homogeneous same-checkpoint LLM instances, same prompt, independent sampling → response embeddings → pairwise cosine distance → farthest-distance pairing (greedy farthest matching AND global maximum-weight perfect matching) → reciprocal within-pair peer review (pair-limited information) → single synthesizer aggregates answers + reviews. No training.
- **Stop condition**: a credible pre-cutoff paper implementing the SAME complete chain (independent same-model generation → semantic-distance farthest pairing → reciprocal pairwise review → final aggregation). Not triggered by diversity/embeddings/debate/peer-review/dynamic-topology used separately.

## Criteria legend

A. homogeneous instances of the same LLM · B. same-prompt independent initial solving · C. response-level semantic embeddings · D. pairwise semantic distance between responses · E. distance used to construct communication/review topology · F. farthest/dissimilar peer selection · G. one-to-one pairing / perfect matching · H. reciprocal critique · I. pair-limited information exchange · J. final aggregation/synthesis stage · K. no training / no weight modification

## Queries run (WebSearch, arXiv/ACL Anthology/OpenReview/Semantic Scholar/Google-indexed sources)

- `multi-agent LLM debate semantic distance pairing agents`
- `LLM agents farthest pairing dissimilar peer review arXiv`
- `multi-agent debate sentence embedding diversity communication topology`
- `heterophily disassortative multi-agent LLM communication network debate`
- `"reciprocal critique" / "peer review" LLM agents pairwise exchange answers`
- `farthest point sampling semantic diversity LLM ensemble answer selection`
- `"perfect matching" / "maximum weight matching" LLM agents debate pairing`
- `DyLAN dynamic LLM agent network agent selection`
- `"response similarity" / "answer similarity" select debate partner multi-agent LLM routing`
- `LLM agent reviewer assignment embedding similarity routing`
- `"contrastive" peer critique LLM agents dissimilar opinions pairing`
- `"Communication Topology Matters" diversity multi-agent LLM`
- `multi-agent debate "most diverse"/"most distant" responses critique partner embedding cosine`
- `"same model" instances "pair" agents "mutual review"/"review each other" embedding similarity`
- `multi-agent debate "maximize diversity" pair agents "cosine" review synthesize final answer`
- `LLM self-consistency cluster responses embedding representative cross-verification`
- plus follow-up crawls of a 141-paper MAD systematic survey (arXiv 2607.26212) taxonomy sections on topology adaptability/structure.

---

## Paper-by-paper findings

### 1. DySCo: Dynamic Trust-Aware Sparse Communication Topology for LLM-Based Multi-Agent Consensus — STRONG OVERLAP (closest hit on E)

- **Citation**: Wanshuang Gou, Zihan Liu. arXiv:2606.01828, v1 2026-06-01 (Chengdu University). https://arxiv.org/abs/2606.01828
- **Method summary**: Each of n LLM agents independently solves the task (round 0). Each round, for every directed edge (j→i) a value score `s_ij = αT_j + βc_j + γD_ij + δH_ij(x) − ηL_j` is computed from historical trust, confidence, **answer/reasoning divergence D_ij**, task dependency, and token cost; each receiver keeps its top-k highest-value senders under budget; senders emit compressed critique messages; receivers revise; final answer by trust-weighted voting with entropy-based early stopping.
- **A–K**:
  - A: partial — "The agents may be different sampling instances of the same LLM, or heterogeneous agents instantiated through different models..." (homogeneity allowed, not required; experiments use one model).
  - B: yes — "each agent independently solves the task x ... Psolve denotes a unified task-solving prompt."
  - C: partial — embeddings listed as one option: "The divergence score D_ij can be obtained from answer agreement, the cosine distance between reasoning embeddings, or judgments produced by a critique prompt." Main experiments (GSM8K/LogiQA/StrategyQA, all MC-style) use the indicator `D_ij = 1[y_i ≠ y_j]`; embedding distance is not the demonstrated mechanism.
  - D: partial — pairwise divergence is central, but instantiated mainly as answer-mismatch indicator.
  - E: yes (partial in spirit) — divergence is one additive term in edge-value scoring: "selecting neighbors who are both reliable and endowed with complementary perspectives is essential" (ablation "w/o Diversity Score" drops accuracy 84.3→82.1). Topology is response-conditioned.
  - F: partial — higher divergence *raises* edge value, but selection maximizes a mixed score (trust + confidence + divergence + relevance − cost); not "farthest" selection.
  - G: no — Top-K neighbor sets per receiver (k=2), directed, asymmetric; no perfect matching / pairing.
  - H: no — edges are directed sender→receiver; no requirement that i reviews j and j reviews i.
  - I: no — each agent receives from up to k neighbors; no isolated pairs.
  - J: partial — aggregation exists but is trust-weighted voting, not an LLM synthesizer over answers + reviews.
  - K: yes for the LLM (no weight updates); note the paper is a low-key preprint (gpt-3.5-turbo experiments, abstract-style analysis) — credibility moderate.
- **Evidence quotes**: see above; method section §"Dynamic Communication Edge Selection", Eq. 4–6.
- **Verdict**: strong overlap on E (response-conditioned topology via divergence) but fails F/G/H/I/J. Does NOT trigger stop condition.

### 2. RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation — STRONG OVERLAP (closest hit on C+D and greedy farthest-style selection)

- **Citation**: Zhentao Xie, Chengcheng Han, Jinxin Shi, Wenjun Cui, Xin Zhao, Xingjiao Wu, Jiabao Zhao. Findings of ACL 2025, pp. 6575–6602. https://aclanthology.org/2025.findings-acl.342/
- **Method summary**: Improves Mixture-of-Agents. At each layer, responses are embedded (BGE-m3; robustness checked with SGPT, multilingual-e5), the full pairwise cosine-similarity matrix is built, and a **greedy max-diversity (min-max-similarity, i.e., farthest-point-style) selection** picks K=3 most diverse responses to be concatenated as reference for the next layer; residual extraction/aggregation agents preserve inter-layer differences; adaptive termination.
- **A–K**:
  - A: partial — same model per layer used in ablations ("using the same small model in each layer for consistency"), but diversity is induced by **distinct role-playing personas** ("each agent is assigned a distinct role-playing persona"), violating FPRR's same-prompt homogeneity.
  - B: no — different role prompts per proposer.
  - C: yes — "we introduce an embedding model to convert responses into vector representations and compute their similarities."
  - D: yes — "computing the cosine similarity matrix S ∈ R^{n×n} for all pairs of responses."
  - E: partial — similarity decides **which responses survive to the next layer / aggregator input**, not a communication/review graph among agents.
  - F: partial — greedy selection explicitly maximizes diversity (selects candidate minimizing max similarity to the selected set — a farthest-point heuristic), but it selects a *subset for aggregation*, not review partners.
  - G: no. H: no — no peer critique at all. I: no. J: yes — aggregator agents produce the final response. K: yes.
- **Evidence quotes**: "A greedy strategy is then applied to select K responses with the highest diversity, ensuring greater information heterogeneity" (§1); algorithm in §3.2.1 (initialization by min average similarity, iterative min-max selection).
- **Verdict**: strong overlap on C/D + farthest-style greedy selection, but selection targets aggregation input, not reciprocal review pairing. Does NOT trigger stop condition.

### 3. S2-MAD: Breaking the Token Barrier to Enhance Multi-Agent Debate Efficiency — STRONG OVERLAP (embeddings gate debate participation; opposite direction to F)

- **Citation**: Yuting Zeng, Weizhe Huang, Lei Jiang, Tongxuan Liu, Xitai Jin, Tianying Tiana Chen, Jing Li, Xiaohua Xu. NAACL 2025, pp. 9393–9408. arXiv:2502.04790. https://aclanthology.org/2025.naacl-long.475/
- **Method summary**: Selective Sparse MAD. Agents (same LLM, temperature-varied) independently generate initial answers, are split into fixed groups; a Decision-Making Mechanism computes similarity of others' outputs to one's own (regex answer match, or **response vectorization + cosine similarity** with BERT-base), **filters out redundant (similar) viewpoints**, and lets an agent debate only when differing viewpoints exist; majority vote at the end.
- **A–K**:
  - A: yes — each agent initialized as the same LLM; diversity via "random decoding strategy by adjusting the temperature."
  - B: yes — "all agents independently produce their respective solutions for the given [question]."
  - C: yes (alternative module) — "we also propose an alternative vectorization-based approach, where the responses are vectorized using an embedding model, and the cosine similarity is computed" (Bert-base-uncased).
  - D: yes — pairwise cosine similarity between own and others' responses.
  - E: partial — similarity conditions **whether** an agent participates and **which messages are discarded** within a fixed group topology; it does not construct the topology itself (groups are pre-set).
  - F: no — opposite polarity: similar/redundant viewpoints are *filtered out*; agents engage when they *differ*, but no farthest selection of a partner.
  - G: no (group-wise, not pairing). H: no (broadcast within groups). I: no. J: partial (majority vote / consensus summaries, not a synthesizer over reviews). K: yes.
- **Evidence quotes**: "agents systematically filter all incoming information to ensure relevance and uniqueness. Outputs that are identified as similar to either their own or previously received viewpoints are promptly discarded" (§3.2 Redundancy Filtering).
- **Verdict**: strong overlap — demonstrates C/D inside MAD, but uses similarity for redundancy suppression in fixed groups; no pairing, no reciprocal review. Does NOT trigger stop condition.

### 4. MACE / Multi-Agent LLMs Fail to Explore Each Other — RELATED (response-diversity-conditioned peer selection, but learned bandit, no embeddings, no pairing)

- **Citation**: Jiatong Li, Wendi Li, Xin Eric Wang, Sharon Li. arXiv:2607.11250, v1 2026-07-13. https://arxiv.org/abs/2607.11250
- **Method summary**: Formalizes multi-agent exploration as a POSG; each agent learns (LinUCB contextual bandit) which peer to query each round. Context features include **response diversity** (n-gram Jaccard distance between own and peer's current answer), peer distinctiveness (centrality in an n-gram-similarity response graph), historical performance, round.
- **A–K**: A: partial (homogeneous Qwen setting exists in one experiment; also heterogeneous); B: yes-ish (initial independent responses); C: **no — explicitly** "we do not directly use high-dimensional response embeddings as contextual features"; diversity is n-gram Jaccard; D: partial (lexical pairwise distance); E: yes (response-conditioned peer selection, learned); F: partial (diversity is one of 9 bandit features, weight learned from reward — not farthest rule); G: no (each agent picks one peer per round, but choices are independent, not a matching); H: no (directed query); I: partial (agent sees only the chosen peer's response that round — dyadic exchange exists, but not fixed pairs); J: no synthesizer (per-agent answers); K: partial (no LLM weight change, but bandit parameters are learned online).
- **Verdict**: related. Closest on "response-difference → who talks to whom," but mechanism (learned bandit, lexical distance, heterogeneous pools, no review stage) differs fundamentally. Does NOT trigger stop condition.

### 5. Beyond Correctness: Distance-Based Social Dynamics of Multi-Agent Debate — RELATED (distance used for analysis, not topology)

- **Citation**: Seungwoong Ha, Melanie Mitchell. ICML 2026 (main track, poster). OpenReview id o4v0sr8Mpy (no arXiv version; OpenReview full text behind Cloudflare challenge at search time — abstract verified via HuggingFace `ai-conferences/ICML2026` dataset; code: github.com/nokpil/socialLLM).
- **Method summary**: Analysis paper. Uses ConceptARC's quantitative solution-space distance to study how a target LLM's revisions depend on peer answers and distance to ground truth: agents revise more when farther from correct; revisions contract toward ground truth; near-correct wrong peers can overturn correct answers.
- **A–K**: A: partial (target model + controlled peer configurations); B: partial; C/D: distance is a **measurement instrument** on grid solutions, not sentence embeddings of responses; E: no (peer configurations are controlled experimental conditions, not distance-constructed topology); F: no; G: no; H: no; I: n/a; J: no; K: yes.
- **Evidence quote (abstract)**: "We study the microscopic dynamics of answer revision ... we analyze how the likelihood and direction of revision depend on both social context and the distance between answers and the ground truth."
- **Verdict**: related only — distance-aware analysis of debate dynamics; no pairing method. Does NOT trigger stop condition.

### 6. GTD: Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models — RELATED (task-conditioned, not response-conditioned topology)

- **Citation**: Eric Hanchen Jiang, Mengting Li, Guancheng Wan, et al. ACL 2026 (long), pp. 38042–38060. https://aclanthology.org/2026.acl-long.1764/
- **Method summary**: Trains a GNN surrogate reward model + conditional discrete graph diffusion generator to synthesize per-task communication topologies optimizing utility/cost/robustness; conditioning vector = **task query embedding** + graph state embeddings.
- **A–K**: C: partial (embeds the *task query*, not responses); E: no (topology conditioned on task, not on inter-response distance); F/G/H/I: no; J: partial; K: no for the pipeline (trains surrogate + diffusion model; LLM weights untouched).
- **Verdict**: related only. Does NOT trigger stop condition.

### 7. SMoA: Sparse Mixture-of-Agents — RELATED

- **Citation**: Dawei Li, Zhen Tan, Peijia Qian, et al. arXiv:2411.03284, 2024-11-05.
- Response selection + early stopping sparsify information flow in MoA; diversity from **assigned role descriptions**; selection via judge/scoring, not semantic distance; no pairing/review. A: no (roles); C/D/E/F/G/H/I: no; J: yes (aggregator); K: yes.
- **Verdict**: related only.

### 8. Improving Multi-Agent Debate with Sparse Communication Topology (S-MAD) — RELATED (canonical fixed-topology baseline)

- **Citation**: Yunxuan Li, Yibing Du, Jiageng Zhang, Le Hou, Peter Grabowski, Yeqing Li, Eugene Ie. Findings of EMNLP 2024, pp. 7281–7294. arXiv:2406.11776.
- Systematically compares fixed sparse topologies (ring, star, tree, random, complete/bipartite variants); topology is **not** response-conditioned; no embeddings. Demonstrates sparsity can preserve/improve accuracy. A: yes; B: yes; C–F: no; G: some bipartite structures but not matching-by-distance; H: no; J: majority vote; K: yes.
- **Verdict**: related only (background for topology-matters claim).

### 9. GroupDebate — RELATED

- **Citation**: Tongxuan Liu, Xingyu Wang, Weizhe Huang, et al. 2024 (arXiv; cited in S2-MAD). Agents clustered into debate groups sharing intermediate summaries to cut token cost; grouping is not distance-based. No C/D/E/F/G/H. Related only.

### 10. Not Birds of a Feather: Personality-Based Partner Selection in LLM Agents — RELATED (dissimilar partner choice, but personality-driven and endogenous)

- **Citation**: Tao Wang, Hsiang-Ling Chiu, Chihang Wei, Yang Xiu, Zhonghao Hou. arXiv:2607.19785, v1 2026-07-22.
- A host LLM chooses among six persona-conditioned candidates; hosts with personalities "chose partners farther from themselves in trait space than random choice" — anti-homophily in *personality* space, not response-embedding space; selection is done by the LLM itself, no algorithmic matching, no reciprocal review, no aggregation stage. E/F: partial in spirit only. Related only.

### 11. Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity — RELATED (embedding-based diversity metric, no pairing)

- **Citation**: Yingxuan Yang, Chengrui Qu, Muning Wen, Laixi Shi, Ying Wen, Weinan Zhang, Adam Wierman, Shangding Gu. arXiv:2602.03794, v1 2026-02-03.
- Information-theoretic bound: homogeneous-agent scaling saturates because outputs are correlated; introduces K* (effective channel count, embedding eigenvalue-entropy-based semantic diversity measure) to quantify diversity without ground truth. Argues for heterogeneity (models/prompts/tools). No distance-based pairing, no review, no matching. Related only — useful as motivation/citation for "sampling-only diversity is limited".

### 12. Multi-Agent Debate Strategies: Survey, Taxonomy, and Challenges (141 primary studies) — NEGATIVE EVIDENCE

- **Citation**: Quim Motger, Marc Oriol, Jordi Marco, Xavier Franch. arXiv:2607.26212, 2026 (forward snowballing completed July 2025).
- §5.1 Topology: 88.7% of surveyed MAD approaches use **static** topologies; dynamic minority (9.9%) adapt edges via agent-importance (DyLAN), node/edge dropout, learned edge weights ("agents interact only with beneficial peers"), or team assembly — none via response-embedding distance. Embedding use in MAD is essentially limited to message *format* (CIPHER latent-space messages) rather than partner selection. Pairing/matching as a debate structure does not appear in the taxonomy at all (structures enumerated: fully connected, divided-into-groups, layered, chain/ring/star/tree, hierarchical).
- **Verdict**: strong negative evidence that embedding-distance-based pairing is absent from the MAD literature through mid-2025.

### 13. Communication Topology Matters: Diversity in Multi-Agent LLMs — RELATED (unverifiable full text)

- Anonymous submission, OpenReview id yOWhKFfhZY (ACL 2026 submission cycle). Search snippets show it studies "agent opinion trajectories in semantic embedding space across network topologies" — i.e., embedding-space diversity as an **outcome** measured across fixed topologies, not distance-based pairing. OpenReview full text blocked by Cloudflare challenge at search time; treated as related-only with partial verification. No indication of farthest pairing or reciprocal review.

### 14. ARGORA: Orchestrated Argumentation for Causally Grounded LLM Reasoning — RELATED (cosine similarity for redundancy control only)

- **Citation**: Youngjin Jin, Hanna Kim, Kwanwoo Kim, Chanhee Lee, Seungwon Shin. arXiv:2601.21533, v1 2026-01-29.
- Multi-expert discussion compiled into QBAF argumentation graphs; sentence-encoder cosine similarity used only to **prune redundant arguments** ("contextual orthogonality"), not to select partners. C/D: yes (argument-level); E/F/G/H: no. Related only.

### 15. Baseline MAD / peer-review references (for the A/B/H/J/K background)

- Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate", arXiv:2305.14325 (ICML 2024): homogeneous instances, same prompt, independent answers, **full broadcast** exchange, majority answer. A/B/J/K yes; C–I no.
- Liang et al., "Encouraging Divergent Thinking in LLMs through Multi-Agent Debate", arXiv:2305.19118 (EMNLP 2024): tit-for-tat debate + judge; no embeddings/pairing.
- Xu et al., "Towards Reasoning in LLMs via Multi-Agent Peer Review Collaboration", arXiv:2311.08152 (2023): independent solutions → peer review (each solution reviewed by other agents) → revision; reviewers see others' solutions but assignment is not distance-based and not reciprocal-pair-limited. A/B/J/K yes; H partial (review is mutual in aggregate, not pairwise); C–G/I no.

---

## Cross-paper summary matrix

| Paper | A | B | C | D | E | F | G | H | I | J | K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DySCo (2606.01828) | ~ | ✓ | ~ | ~ | ✓ | ~ | ✗ | ✗ | ✗ | ~ | ✓ |
| RMoA (ACL25-Find) | ~ | ✗ | ✓ | ✓ | ~ | ~ | ✗ | ✗ | ✗ | ✓ | ✓ |
| S2-MAD (NAACL25) | ✓ | ✓ | ✓ | ✓ | ~ | ✗ | ✗ | ✗ | ✗ | ~ | ✓ |
| MACE (2607.11250) | ~ | ✓ | ✗ | ~ | ✓ | ~ | ✗ | ✗ | ~ | ✗ | ~ |
| Ha & Mitchell (ICML26) | ~ | ~ | ✗ | ~(grid) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| GTD (ACL26) | ~ | – | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✗ |
| SMoA (2411.03284) | ✗ | – | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| S-MAD (EMNLP24-Find) | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Not Birds of a Feather (2607.19785) | ~ | – | ✗ | ✗ | ~ | ~ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Agent Scaling via Diversity (2602.03794) | ✓ | ✓ | ~ | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✓ |
| ARGORA (2601.21533) | ~ | ~ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Du et al. MAD (2305.14325) | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Xu et al. Peer Review (2311.08152) | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✗ | ✓ | ✓ |

(✓ = yes, ~ = partial, ✗ = no, – = not determinable from verified text)

## Bottom line

- **No paper hits E+F+G+H together.** The closest on E (response-conditioned topology) is DySCo, which uses mixed-value Top-K directed edge selection — not farthest pairing, not matching, not reciprocal review, and its divergence term defaults to answer-mismatch indicators. The closest on C+D+farthest-style selection is RMoA, whose greedy max-diversity selection feeds an aggregator — no pairing, no review. S2-MAD uses embeddings/cosine to *suppress* similar viewpoints inside fixed groups — the opposite pairing polarity. MACE conditions peer choice on response diversity but via a learned bandit over n-gram features, with no pairing or review stage.
- **Stop condition NOT triggered** for the embedding/distance-based pairing & routing area. No credible pre-2026-09-17 paper found implementing the FPRR chain (same-model independent generation → semantic-distance farthest pairing/matching → reciprocal pairwise review → synthesis).
- **Caveats**: (1) OpenReview full texts were Cloudflare-blocked; Ha & Mitchell (ICML 2026) and the anonymous "Communication Topology Matters" submission were assessed from verified abstracts/snippets + code repos, not full method sections. Neither shows any sign of distance-based pairing, but a small residual risk remains. (2) DySCo is a single-institution preprint with modest experiments; even if read maximally, it fails G/H/I/J. (3) The 141-paper MAD survey (Motger et al.) independently corroborates the absence of embedding-distance pairing in debate topology design through July 2025.
