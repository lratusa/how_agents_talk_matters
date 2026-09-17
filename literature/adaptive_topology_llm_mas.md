# Novelty Gate Literature Search — Adaptive / Learned / Optimized Communication Topology in LLM Multi-Agent Systems

**Area:** Communication-topology construction in LLM multi-agent systems (learned, adaptive, query-conditioned, or distance-based).
**Literature cutoff:** 2026-09-17 (anything later ignored).
**Search date:** 2026-09-17.

## Method under evaluation (working name FPRR)

N homogeneous instances of the SAME LLM checkpoint (same system prompt, same problem, same decoding config; stochastic sampling is the only diversity source) answer independently → each response (answer + concise justification) is embedded with a sentence-embedding model → pairwise cosine-distance matrix → agents are PAIRED by farthest semantic distance (randomized greedy farthest matching; separately a global maximum-weight perfect matching) → within each pair, reciprocal peer review (i reviews j AND j reviews i; each sees only problem, own answer, partner's answer; no ground truth, no distance info) → a single synthesizer (same LLM) aggregates all original answers + all reviews into one final answer. No training, no fine-tuning, no retrieval, no external knowledge, no tool use.

## Novelty criteria

- **A.** homogeneous instances of the same LLM
- **B.** same-prompt independent initial solving
- **C.** response-level semantic embeddings
- **D.** pairwise semantic distance between responses
- **E.** distance used to construct communication/review topology
- **F.** farthest/dissimilar peer selection
- **G.** one-to-one pairing / perfect matching
- **H.** reciprocal critique (i reviews j AND j reviews i)
- **I.** pair-limited information exchange (agents see only their partner)
- **J.** final aggregation/synthesis stage
- **K.** no training / no model-weight modification

**Stop condition:** a credible paper predating 2026-09-17 implementing substantially the same complete chain (same-model independent generation → semantic-distance-based farthest pairing → reciprocal pairwise review → final aggregation).

**Bottom line up front:** No verified paper hits E+F+G+H together. No verified paper hits even F+G (farthest/dissimilar selection + one-to-one pairing) in any form. **No exact prior art was found in this assigned area.** Details and caveats at the end.

---

## 1. Assigned named papers (all verified to exist)

### 1.1 GPTSwarm — "Language Agents as Optimizable Graphs"

- **Authors:** Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, Jürgen Schmidhuber (KAUST; IDSIA/USI/SUPSI)
- **Venue:** ICML 2024 (Oral; PMLR vol. 235, pp. 62743–62767)
- **arXiv:** 2402.16823 · https://arxiv.org/abs/2402.16823 (full text read at arxiv.org/html/2402.16823v3)

**Method summary.** LLM-agent systems are cast as computational graphs: nodes are operations (LLM inference, tools), edges are communication channels. Two optimization modes: (1) edge optimization relaxes inter-agent connectivity into Bernoulli edge parameters and optimizes expected task utility with REINFORCE over a distribution of DAGs; (2) node optimization rewrites per-node prompts with an LLM-based optimizer over input–output history. Final decisions come from majority voting / self-consistency / "Choose Best". No embeddings, no semantic distances, no pairing.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Partial | "In all experiments, we used GPT-4-Turbo with the token sampling temperature 0.2" — but "we run a set of 7 different IO agents instructed to behave according to various roles." |
| B | Partial | "all IO agents are prompted identically" (adversarial setting; treated as a weakness, not a principled diversity stage) |
| C | No | Only ℝ-valued edge parameters: "assign a real-valued parameter θ_i ∈ ℝ to each potential edge e_i." No sentence embeddings. |
| D | No | Utility is task reward: "a task τ and its associated utility function u_τ that maps the candidate graphs to real numbers." |
| E | No | "we optimize the inference structure by employing RL techniques applied to the potential edges of a given graph." |
| F | No | No dissimilarity selection; "edge optimization effectively filters adversarial agents from a swarm." |
| G | No | Arbitrary DAG over all agents' nodes, 2^d configurations — no matching constraint. |
| H | No | One-directional critic in Reflexion only; cycles excluded: "only consider composite graphs that are DAGs." |
| I | No | "Each node n ∈ N receives as input x and the output z_n from its predecessor nodes." |
| J | Yes | "The collective decision on the final answer is made through majority voting"; "Choose 'Best' refers to the LLM's favorite answer." |
| K | Yes | LLM weights frozen; optimization is over prompts and edge parameters only. |

**Topology from response-level semantic distances?** No — REINFORCE over Bernoulli edge parameters maximizing task utility; embeddings never enter. **Farthest pairing?** No — no pairing mechanism exists.
**Verdict: related only.** Caveat: HTML fetch truncated around references/appendix; an appendix-only mechanism cannot be 100% ruled out, but the method sections define topology construction entirely as REINFORCE edge optimization.

### 1.2 DyLAN — "Dynamic LLM-Agent Network: An LLM-agent Collaboration Framework with Agent Team Optimization"

- **Authors:** Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, Diyi Yang (Tsinghua, Georgia Tech, Stanford)
- **Venue:** COLM 2024 (arXiv v1 Oct 2023)
- **arXiv:** 2310.02170 · https://arxiv.org/abs/2310.02170 (method §3.1–3.3 read via ar5iv)

**Method summary.** Collaboration is a feed-forward network across time steps; each agent sees all previous-layer responses plus the query. Dynamics: (1) an LLM "ranker" selects top-m responses and deactivates low performers; (2) Byzantine-style early stopping when >2/3 of a layer agree. Offline agent-team optimization computes an Agent Importance Score from textual peer ratings propagated backward. Roles typically heterogeneous; fully inference-time.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Partial | Same backbone but role-heterogeneous: "Agents assigned diverse roles in DyLAN might respond from their specific perspectives." |
| B | No/Partial | First layer answers in parallel, but prompts differ by role. |
| C | No | Ranking by "an additional LLM agent, as the 'LLM Ranker'"; scores are textual peer ratings, no embeddings. |
| D | No | Only scalar ratings: "we use w_{t−1,j,i} to refer to the rating score on a_{t−1,j} from a_{t,i}." |
| E | No | "Agents identified as not being useful are deactivated in subsequent layers, so as are the edges connecting them." Pruning, not distance construction. |
| F | No | Opposite direction: "we select the top-m responses to feed forward"; "extract the top-k agents that are most contributory." |
| G | No | "each node receives responses [of] all other nodes from the previous layer" — all-to-all. |
| H | Partial (weak) | Directional layer-to-layer rating: "we first ask each agent to rate its predecessors on their solutions." Not mutual within a pair. |
| I | No | Explicitly contradicted: all-to-all previous-layer visibility. |
| J | Partial | Consensus/early stopping: "terminated when over 2/3 of agents in a single layer have a consistent answer." |
| K | Yes | Fully inference-time: "inference-time agent selection and an early-stopping mechanism." |

**Topology from response-level semantic distances?** No — LLM ranker + importance scores from peer ratings; no embeddings or cosine distances. **Farthest pairing?** No.
**Verdict: related only.**

### 1.3 AgentPrune — "Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems"

- **Authors:** Guibin Zhang*, Yanwei Yue*, Zhixun Li*, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, Tianlong Chen
- **Venue:** ICLR 2025 (arXiv v1 2024-10-03)
- **arXiv:** 2410.02506 · https://arxiv.org/abs/2410.02506 (method §2–3 read via ar5iv)

**Method summary.** MAS modeled as spatial-temporal communication graphs. Observing that randomly dropping 10–30% of edges can help ("communication redundancy"), AgentPrune relaxes adjacency into differentiable graph masks, optimizes them with policy gradient (task utility + nuclear-norm low-rank sparsity), enforces DAG via DAGSampling, then one-shot magnitude-prunes to a fixed sparse topology. A plug-in for AutoGen/GPTSwarm/LLM-Debate etc.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Partial | "five gpt-4-based agents" but "We generate different agent profiles using gpt-4 for individual agents." |
| B | No | "agent v_i responds based on the query q, its current role and state, temporal and spatial messages." |
| C | No | "S^S, S^T ∈ R^{|V|×|V|} are differentiable graph masks" — no answer embeddings. |
| D | No | "one-shot magnitude pruning on the optimized graph masks S." |
| E | No | "By training a low-rank-principle-guided graph mask, AgentPrune efficiently identifies the important graph connectivities… one-shot pruning to derive a sparse yet informative communication graph." |
| F | No | TopK keeps largest mask values: "the largest x% elements in matrix S." |
| G | No | Arbitrary sparse DAG: "we require the interaction topology to be a DAG… 𝒢̂^S ← DAGSampling(𝒢̃^S)." |
| H | No | Directed topological-order message passing: "each node is processed only after all its dependencies have been addressed." |
| I | No | Neighborhood-based: "m^S ← {M_ij | v_j ∈ N^S(v_i)}." |
| J | Yes (generic) | "after K rounds of dialogue, a summarizer agent or an answer aggregation mechanism (e.g., voting) is employed to produce the final solution." |
| K | No | Training loop over graph masks: "we iteratively optimize the spatial-temporal connectivity in conjunction with the multi-agent conversation" (policy gradient). LLM weights frozen, but the pipeline is not training-free. |

**Topology from response-level semantic distances?** No — learned differentiable masks via policy gradient. **Farthest pairing?** No.
**Verdict: related only.**

### 1.4 G-Designer — "G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks"

- **Authors:** Guibin Zhang*, Yanwei Yue*, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng (Shanghai AI Lab, Tongji, CUHK, Emory, USTC)
- **Venue:** arXiv preprint (v1 Oct 2024, v3 Feb 2025)
- **arXiv:** 2410.11782 · https://arxiv.org/abs/2410.11782 (full HTML v2 read)

**Method summary.** Agents get distinct roles/profiles/tools. A text encoder (SentenceBERT/MiniLM) embeds each agent's **profile/backbone/plugins** and a task-specific virtual node (the **query**). A variational graph auto-encoder (GCN encoder + Gumbel-sigmoid decoder) decodes a dense adjacency, sparsified with nuclear-norm regularization. The GNN is trained with REINFORCE on a query subset, then frozen; the graph guides K rounds of neighborhood message passing + aggregation.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Partial | "utilize five gpt-4-based agents" — but "we begin by assigning each agent a unique role and profile." |
| B | No | Distinct system prompts: "𝒫_sys = {Role_i, State_i}"; topology-driven multi-round dialogue. |
| C | No | "we employ a node encoder to transform each agent's unique profile into a fixed-length embedding representation" — embeddings are of **profiles and query**, never responses. |
| D | No | Edge probabilities from learned decoder: "ϖ = FFN_d([𝐡_i, 𝐡_j, 𝐡_task])". |
| E | No | "G-Designer employs a variational graph auto-encoder to encode the nodes (agents) along with task-specific information, and to decode the resulting collaboration network." |
| F | No | Nothing resembling farthest/dissimilar pairing. |
| G | No | General sparse directed graph: "ℰ_com = {(i,j) | S̃_ij ≠ 0}". |
| H | No | Directed message passing; no peer-review protocol. |
| I | No | "𝒫_usr^(t) = {𝒬, ∪_{v_j ∈ 𝒩_in(v_i)} ℛ_j^(t)}" — all in-neighbors. |
| J | Yes | "a^(t) ← Aggregate(ℛ_1^(t), ℛ_2^(t), ⋯, ℛ_N^(t))". |
| K | Partial | LLMs not fine-tuned, but the topology module is trained: "we apply policy gradient (Williams, 1992)." |

**Topology from response-level semantic distances?** No — trained VGAE over profile/query embeddings. **Farthest pairing?** No.
**Verdict: not relevant** (adjacent field; closest to FPRR only in using text embeddings at all — but embedded objects and mechanism are categorically different).

### 1.5 MacNet — "Scaling Large Language Model-based Multi-Agent Collaboration"

- **Authors:** Chen Qian, Zihao Xie, Yifei Wang, Wei Liu, Kunlun Zhu, Hanchen Xia, Yufan Dang, Zhuoyun Du, Weize Chen, Cheng Yang, Zhiyuan Liu, Maosong Sun
- **Venue:** arXiv preprint (v1 Jun 2024, v3 Mar 2025)
- **arXiv:** 2406.07155 · https://arxiv.org/abs/2406.07155

**Method summary.** Agents organized into a DAG: an "assistant" agent on each node, a supervisory "instructor" agent on each edge. Interaction follows topological ordering; along each edge, assistant→instructor→assistant run an instruct/refine loop, and only the refined solution propagates downstream. Topologies (chain, star, tree, mesh, layered, random) are generated from structural hyperparameters, not data. Contributions are empirical: small-world/irregular topologies perform best; logistic scaling law up to >1,000 agents.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Partial | Same backbone ("By default, our method utilizes the GPT-3.5-turbo model") but heterogenized: "a pool of 4,000 profiles for assignment"; "temperatures decrease linearly from 1.0 to 0.0 according to topology depths." |
| B | No | "each interaction round involves two adjacent agents refining a previous solution" — no independent-answer phase. |
| C | No | No embedding model anywhere. |
| D | No | No distance matrix anywhere. |
| E | No | "MacNet addresses this challenge by automatically generating various networks through simple hyperparameters (e.g., topology type and scale)"; irregular variant uses "random edge connections." |
| F | No | Only random shortcuts to "'unacquainted' agents" — random wiring, not dissimilarity selection. |
| G | No | Pairwise interaction per edge, but no matching construction. |
| H | No | Asymmetric: "aᵢⱼ offers optimization suggestions … aⱼ provides the refined solution" — never reverse review. |
| I | Partial | Adjacency-limited, not pair-limited: "solutions propagated only from adjacent agents." |
| J | Yes | "Convergent agents assess the strengths and weaknesses of each solution, synthesizing their strengths and discarding weaknesses." |
| K | Yes | Inference-only. |

**Topology from response-level semantic distances?** No — structural/random DAGs. **Farthest pairing?** No.
**Verdict: related only.**

### 1.6 Graph-GRPO — "Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization"

- **Authors:** Yueyang Cang, Xiaoteng Zhang, Erlu Zhao, Zehua Ji, Yuhang Liu, Yuchen He, Zhiyuan Ning, Yijun Chen, Wenge Que (Donghua University), Li Shi (Tsinghua University). *(Author ordering not fully confirmed from fetched HTML; Yueyang Cang is the submitting author.)*
- **Venue:** arXiv preprint, 2026-03-03
- **arXiv:** 2603.02701 · https://arxiv.org/abs/2603.02701

**Method summary.** RL framework for learning communication topologies. A policy network encodes each agent's role ⊕ query with a frozen all-MiniLM-L6-v2 encoder; a GAT produces a DAG-masked connectivity matrix Pθ via a learnable bilinear score; K=16 graphs are sampled per query, executed by GPT-3.5-Turbo agents, and an edge-level group-relative advantage updates θ with a KL-regularized loss. At inference, thresholding Pθ yields a sparse task-specific graph.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Partial | One backbone ("We employed GPT-3.5-Turbo as the backbone LLM") but role-differentiated agents. |
| B | No | Nodes initialized from role⊕query; no independent same-prompt solving. |
| C | No | "xᵢ = Encoder(Roleᵢ ⊕ 𝒬)" — role/query only, answers never embedded. |
| D | No | Not described. |
| E | No | "(Pθ)ᵢⱼ = σ(hᵢWhⱼᵀ)" — learned affinity, not embedding distance. |
| F | No | Not described. |
| G | No | Not described. |
| H | No | DAG "constraining information to flow strictly from earlier agents to later ones" explicitly forbids reciprocity. |
| I | No | Not described. |
| J | Partial | Flow "typically converging towards the final agent v_N" — de facto aggregation, no synthesizer stage. |
| K | No | Explicitly trained: "Update θ by minimizing ℒ(θ)"; "Adam with a learning rate of 1e-4 on NVIDIA A100 GPUs." |

**Topology from response-level semantic distances?** No — trained GNN policy on role/query features. **Farthest pairing?** No.
**Verdict: not relevant** (opposite paradigm: training-based learned topology).

### 1.7 Guided Topology Diffusion (GTD) — "Dynamic Generation of Multi LLM Agents Communication Topologies with Graph Diffusion Models"

- **Authors:** Eric Hanchen Jiang, Guancheng Wan, Sophia Yin, Mengting Li, Yuchen Wu, Xiao Liang, Xinfeng Li, Yizhou Sun, Wei Wang, Kai-Wei Chang, Ying Nian Wu (UCLA / UW / NTU)
- **Venue:** arXiv preprint (v1 2025-10-09, v2 2026-05-16); an ACL 2026 version exists (2026.acl-long.1764)
- **arXiv:** 2510.07799 · https://arxiv.org/abs/2510.07799

**Method summary.** Topology design as conditional discrete graph diffusion: a condition vector C from task query + agent set conditions a Graph Transformer denoiser that turns a noisy adjacency into a sparse directed graph. A GNN surrogate reward model (trained on simulated baseline topologies) steers denoising via zeroth-order optimization — sample K candidates, score with the proxy, keep the best. Generated topology governs message passing among role-specialized GPT-4o-mini agents.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Partial | "The backbone for all agents is GPT-4o-mini" but "we deploy domain-specific agent teams: four MathSolver agents … four CodeSolver agents." |
| B | No | Topology generated before execution; no independent-answer phase. |
| C | No | Only the task condition is encoded: "a task query q and a set of available agents, which together form a task-specific condition vector C." |
| D | No | Not described. |
| E | No | "we synthesize a topology for a novel task condition C_new by steering the diffusion process with the trained surrogate model." |
| F | No | Selection is reward-maximization over candidate graphs. |
| G | No | Output is a general directed graph. |
| H | No | No review stage exists. |
| I | No | Not described. |
| J | No | No distinct aggregation/synthesis step described. |
| K | No | "We first train these components on a pre-computed dataset"; training data from "evaluating baseline topologies on 50 samples from the GSM8K dataset." |

**Topology from response-level semantic distances?** No — query/agent-conditioned trained diffusion. **Farthest pairing?** No.
**Verdict: related only.**

### 1.8 Codebook Agent — "Codebook Agent: Amortized Topology Design for LLM Multi-Agent Systems"

- **Authors:** reported as Yubei Li, Eric Hanchen Jiang, Zhi Zhang, Kai-Wei Chang, Ying Nian Wu (UCLA; submission by Jinxi Yu). *(Caveat: HTML author block parsed unreliably; treat exact author list as unverified.)*
- **Venue:** arXiv preprint, v1 2026-09-02 (within cutoff by 15 days)
- **arXiv:** 2609.02264 · https://arxiv.org/abs/2609.02264

**Method summary.** Amortizes topology design: offline, fixed topologies are executed on training tasks logging (adjacency, query embedding, utility, cost); a VQ autoencoder compresses reward-surviving topologies into a 16-entry query-independent codebook; a reward-weighted MLP maps the **query embedding** to a code distribution; an MLP proxy reranks top-5 candidates. Test-time generation costs 2.4 ms with no search or message passing. Notably reports that "GNN scorers over agent-profile nodes are adjacency-invariant on homogeneous teams."

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Yes (as setting) | "Homogeneous profiles are the default (six of seven design-axis settings)", e.g. 4× MathSolver on gpt-4o-mini — an evaluation config, not a method mechanism. |
| B | No | Agents "exchange intermediate outputs over a communication topology"; no independent stochastic answering phase. |
| C | No | "profile texts p1,…,pN embedded as xi=E(pi) by a frozen sentence encoder"; "c=E(q) for the query embedding" — answers never embedded. |
| D | No | Not described. |
| E | No | "The selected topology is A⋆(c)=arg max … (û(A,c)−λĉ(A,c))" — learned predictor over query embedding. |
| F | No | Not described. |
| G | No | Not described. |
| H | No | No pairing or review mechanism. |
| I | No | Not described. |
| J | Yes (partial) | "A decision node aggregates the final answers following GDesigner" — standard aggregation node, not a synthesizer over reviews. |
| K | No | "The codebook, code predictor, and reranking proxy are trained on 𝒟" — offline training + LLM-execution data collection. |

**Topology from response-level semantic distances?** No — query-embedding-conditioned learned codebook. **Farthest pairing?** No.
**Verdict: not relevant** (topic-adjacent only; learned, query-conditioned, training-based).

---

## 2. Papers found via open-ended queries (deep-analyzed)

Queries run included: "adaptive communication graph LLM agents", "dynamic debate topology LLM", "query-conditioned communication topology", "learned multi-agent graph LLM", "debate routing", "disagreement-based message selection", "multi-agent topology optimization LLM 2025", "multi-agent topology optimization LLM 2026", plus "semantic similarity multi-agent debate", "embedding-based agent pairing", "diverse answer pairing peer review LLM", "farthest pairing agents", "matching agents by answer diversity", "'perfect matching' / 'bipartite matching' agents debate peer assignment LLM", "maximum diversity sampling farthest point LLM responses ensemble then judge", and others.

### 2.1 DyTopo — "Dynamic Topology Routing for Multi-Agent Reasoning via Semantic Matching"

- **Authors:** Lu, Hu, Zhao, Cao (Peking U / Georgia Tech / Tsinghua / Southeast U)
- **arXiv:** 2602.06039 (Feb 2026) · https://arxiv.org/abs/2602.06039

**Method summary.** Manager-guided multi-round framework; role-specialized heterogeneous workers emit natural-language *query* (need) and *key* (offer) descriptors; these are embedded with a fixed all-MiniLM-L6-v2 encoder; cosine similarity between i's query and j's key is thresholded to induce a sparse directed routing graph per round; a Manager aggregates public messages and decides halting. Inference-time, no training.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | No | "Let A={a₁,…,a_N} denote N **heterogeneous** worker agents… instantiated with a **role description** ρᵢ." |
| B | No | Role-conditioned multi-round routing, not independent same-prompt solving. |
| C | No | "each agent outputs lightweight natural-language **query (need) and key (offer) descriptors**; DyTopo embeds **these descriptors**" — not responses/answers. |
| D | No | Similarity is over descriptors, not pairwise between responses. |
| E | Partial | Embeddings do induce the routing graph — but of descriptors, not response distances. |
| F | No | Opposite direction: edges activated by *alignment* — "a communication link should exist from agent j to agent i if the semantic capacity offered by j **aligns with the need** of i." Selects most similar, not farthest. |
| G | No | Thresholded sparse directed graph, no matching. |
| H | No | Directed routing; no reciprocal review. |
| I | Partial | Sparse neighborhood visibility, not pair-limited. |
| J | Partial | Manager aggregates public messages. |
| K | Yes | Inference-time, fixed encoder. |

**Topology from response-level semantic distances?** No — descriptors, not answers. **Farthest pairing?** No — it selects *most similar* (need↔offer), the exact inverse of F.
**Verdict: related only.** Closest neighbor on the "embedding-induced inference-time topology" axis; must-cite differentiation target.

### 2.2 DySCo — "Dynamic Trust-Aware Sparse Communication Topology for LLM-Based Multi-Agent Consensus"

- **Authors:** Gou, Liu
- **arXiv:** 2606.01828 (Jun 2026) · https://arxiv.org/abs/2606.01828

**Method summary.** N agents (experiments all gpt-3.5-turbo) independently solve with a unified prompt; each round, each receiver scores directed edges with composite s = αT + βc + γD + δH − ηL (trust, confidence, divergence, task dependency, cost) and keeps top-k senders; selected senders send one-way compressed critiques; trust-weighted voting at the end.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Partial | Same backbone in experiments. |
| B | Yes | "At round 0, each agent independently solves the task x… with a unified task-solving prompt." |
| C | Partial | "The divergence score D_ij can be obtained from answer agreement, the **cosine distance between reasoning embeddings**, or judgments produced by a critique prompt" (one option among several; MC experiments use answer-mismatch indicator). |
| D | Partial | Same quote — cosine distance between reasoning embeddings is offered as an option. |
| E | Partial | Divergence is 1 of 5 additive terms in edge scoring. |
| F | Partial | Dissimilarity (divergence) is rewarded in edge scoring, but dominated by trust/confidence. |
| G | No | Top-k sender selection (k=2), no matching/pairing. |
| H | No | One-way compressed critiques from selected senders. |
| I | No | Neighborhood visibility. |
| J | Yes | Trust-weighted voting at the end. |
| K | Yes | Inference-time. |

**Topology from response-level semantic distances?** Partial at best — divergence is one optional signal among five. **Farthest pairing?** No — no pairing, no reciprocal review.
**Verdict: related only.** Nearest prior art for signals C/D/E; cite and differentiate (no pairing, no reciprocity, no pair-limited visibility, divergence not the sole topology driver).

### 2.3 AGP — "Adaptive Graph Pruning for Multi-Agent Communication"

- **Authors:** Zhao, Lee, Wang (no "Li" author despite some secondary citations)
- **arXiv:** 2506.02951 (Jun 2025) · https://arxiv.org/abs/2506.02951

**Method summary.** Trained GNN generates task-specific pruned topologies from Sentence-BERT embeddings of **agent profiles + task** (not responses); heterogeneous role-based agents communicate K=3 rounds over the learned graph; a decision agent aggregates.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | No | "a fixed pool of **heterogeneous** LLM-agents." |
| B | No | Role-based multi-round communication. |
| C | No | "Each **agent profile and the task-specific virtual node**… are embedded… via a lightweight Sentence-BERT encoder" — profiles/task, not responses. |
| D | No | Not described. |
| E | No | Learned GNN pruning, not a distance metric. |
| F | No | Not described. |
| G | No | Not described. |
| H | No | Not described. |
| I | No | Neighborhood communication. |
| J | Yes | Decision agent aggregates. |
| K | No | "our method employs a **two-stage training strategy**." |

**Topology from response-level semantic distances?** No. **Farthest pairing?** No.
**Verdict: related only.**

### 2.4 LLM-PeerReview — "Scoring, Reasoning, and Selecting the Best! Ensembling LLMs via a Peer-Review Process"

- **Authors:** Chen, Ji, Mao, Wu, Song, Cheng, Qin, Li, Li, Sun, Wang, Ban, Sun, Ji, Sun, Huang
- **arXiv:** 2512.23213 (v1 Dec 2025, v4 Aug 2026) · https://arxiv.org/abs/2512.23213

**Method summary.** J heterogeneous open-weight LLMs each answer once; every LLM judges all responses (flipped-triple debiasing); scores aggregated by averaging or Dawid–Skene EM; highest-scoring response selected.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | No | "a set of zero-shot inference responses **from heterogeneous LLMs**." |
| B | Partial | Each LLM answers once independently (but different models). |
| C | No | No embeddings. |
| D | No | No distances. |
| E | No | Complete bipartite judging, no topology. |
| F | No | All-to-all scoring. |
| G | No | "Each LLM ℳ_j′ acts as a judge and assigns a score y(i,j,j′) to **each** response." |
| H | No | Judges score all; not reciprocal pairing. |
| I | No | Every judge sees every response. |
| J | Partial | "the highest-scoring response is selected as the best ensemble output" — selection, not synthesis. |
| K | Yes | Inference-time. |

**Verdict: related only.** Peer-review flavor but all-to-all scoring, no embeddings, no pairing, no synthesis.

### 2.5 Li et al. — "Improving Multi-Agent Debate with Sparse Communication Topology"

- **Authors:** Yunshu Li, Abhimanyu Du, et al. (Google DeepMind) — Li, Du, Zhang, Hou, Grabowski, Li, Ie
- **Venue:** Findings of EMNLP 2024, pp. 7281–7294
- **arXiv:** 2406.11776 · https://arxiv.org/abs/2406.11776

**Method summary.** 6 same-LLM agents independently answer round 1 with random decoding; debate over a **static predefined** graph (full → regular → ring); majority-vote consensus. ProbMAD appendix: probabilistic per-round visibility.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Yes | "agents instantiated by the same LLM." |
| B | Yes | "in round 1, agents… independently generate solutions… a random decoding strategy is applied to diversify." |
| C | No | No embeddings. |
| D | No | No distances. |
| E | No | "we focus on regular graphs… permutation invariant… static graphs" — content-blind. |
| F | No | No selection at all. |
| G | No | Ring/regular/full graphs, no matching by content. |
| H | Partial | Debate neighbors exchange views; not a reciprocal review protocol over answers. |
| I | Partial | Neighborhood visibility (sparse graphs). |
| J | Partial | Majority vote, not synthesizer. |
| K | Yes | Inference-time. |

**Topology from response-level semantic distances?** No — static content-blind graphs. **Farthest pairing?** No.
**Verdict: related only — but the closest structural relative** on A+B+K (homogeneous, same-prompt, sampling-only diversity, sparse communication). FPRR should explicitly differentiate from this paper.

### 2.6 GoAgent — "Group-of-Agents Communication Topology Generation"

- **Authors:** Chen, Jiao, Liu, Zhao, Zheng, Xu, Khalil, Liu, Li, Pan
- **arXiv:** 2603.19677 (Mar 2026) · https://arxiv.org/abs/2603.19677

**Method summary.** Trained autoregressive generator conditioned on task-query embedding selects role-heterogeneous "collaborative groups" and predicts inter-group edges; execution by prompt composition along edges; summarizer agent at the end.

| Crit. | Verdict | Evidence |
|---|---|---|
| A | Partial | Role-heterogeneous groups. |
| B | No | Topology generated before answers exist. |
| C | No | Query embedding only. |
| D | No | Not described. |
| E | No | Learned autoregressive generator. |
| F | No | Not described. |
| G | No | Group-centric, not pair-centric. |
| H | No | No review. |
| I | No | Group/edge-based message flow. |
| J | Yes | Summarizer agent at the end. |
| K | No | Trained generator. |

**Verdict: related only.**

---

## 3. Marginal papers (verified existence; assessed from abstracts/snippets, not full-text reads)

- **MANTA** (arXiv:2607.28527, Jul 2026) — inference-time topology self-evolution from collaboration traces; trace-monitoring heuristics, not response-embedding distances; no pairing/review. *Related only.*
- **AMAS** (arXiv:2510.01617, EMNLP 2025 Industry) — adaptive topology via lightweight LLM adaptation; role-based, no response embeddings. *Related only.*
- **Graph-of-Agents (GoA)** (Yun et al., ICLR 2026) — graph-based multi-agent collaboration framework. *Related only.*
- **Discovering communication topologies via causal inference** (arXiv:2608.12921) — causal edge discovery. *Related only.*
- **LLM-TOPLA** (Findings of EMNLP 2024) — diversity-maximizing LLM ensemble selection (DPP-style over model outputs); diversity used for subset selection, not pairing for reciprocal review. *Related only* — worth citing on "diversity as selection criterion."
- **HCPMAD** (arXiv:2604.09679), **SVR-MAD** (arXiv:2605.23099), **DebUnc** (arXiv:2407.06426) — debate variants with heterogeneous consensus / posterior guidance / uncertainty-weighted trust. *Related only.*
- **Knappe et al. 2024** — embeddings of reasoning traces for weighted voting; response embeddings used for vote weighting, no pairing/review. *Related only.*
- **From Replication to Redesign: Pairwise Comparisons for LLM-Based Peer Review** (arXiv:2506.11343) — pairwise LLM review of manuscripts; different domain, no embeddings/matching. *Not relevant.*

---

## 4. Summary matrix (deep-analyzed papers)

| Paper | A | B | C | D | E | F | G | H | I | J | K | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPTSwarm 2402.16823 | P | P | – | – | – | – | – | – | – | Y | Y | related only |
| DyLAN 2310.02170 | P | P/– | – | – | – | – | – | P | – | P | Y | related only |
| AgentPrune 2410.02506 | P | – | – | – | – | – | – | – | – | Y | – | related only |
| G-Designer 2410.11782 | P | – | – | – | – | – | – | – | – | Y | P | not relevant |
| MacNet 2406.07155 | P | – | – | – | – | – | – | – | P | Y | Y | related only |
| Graph-GRPO 2603.02701 | P | – | – | – | – | – | – | – | – | P | – | not relevant |
| GTD 2510.07799 | P | – | – | – | – | – | – | – | – | – | – | related only |
| Codebook Agent 2609.02264 | Y* | – | – | – | – | – | – | – | – | P | – | not relevant |
| DyTopo 2602.06039 | – | – | – | – | P | – | – | – | P | P | Y | related only |
| DySCo 2606.01828 | P | Y | P | P | P | P | – | – | – | Y | Y | related only |
| AGP 2506.02951 | – | – | – | – | – | – | – | – | – | Y | – | related only |
| LLM-PeerReview 2512.23213 | – | P | – | – | – | – | – | – | – | P | Y | related only |
| Li et al. 2406.11776 | Y | Y | – | – | – | – | – | P | P | P | Y | related only (closest on A+B+K) |
| GoAgent 2603.19677 | P | – | – | – | – | – | – | – | – | Y | – | related only |

(Y = yes, P = partial, – = no; * = as evaluation setting, not mechanism)

## 5. Conclusion

- **Does ANY paper hit E+F+G+H together?** No. None hits even F+G (farthest/dissimilar selection + one-to-one pairing) in any form.
- **Does any paper build topology from response-level semantic distances?** No. Closest: DySCo names "cosine distance between reasoning embeddings" as one of five optional edge-scoring signals; DyTopo embeds need/offer descriptors (not answers) to route by *similarity* (inverse of farthest). All learned-topology papers (GPTSwarm, AgentPrune, G-Designer, Graph-GRPO, GTD, Codebook Agent, AGP, GoAgent) condition on query/role/profile embeddings or task reward, never on distances between generated answers.
- **Bottom-line verdict:** **No exact prior art in the assigned area.** The FPRR chain (same-model independent generation → response-embedding distance matrix → farthest perfect/greedy matching → reciprocal pair review with pair-limited visibility → synthesizer aggregation, training-free) was not found in any verified paper up to 2026-09-17. The stop condition is NOT triggered.
- **Suggested must-cite differentiation set:** DyTopo 2602.06039 (semantic-matching routing, similarity direction), DySCo 2606.01828 (divergence-aware edge scoring), Li et al. 2406.11776 (homogeneous sparse MAD topology), LLM-PeerReview 2512.23213 (peer-review ensemble), LLM-TOPLA (diversity-based ensemble selection), plus the learned-topology line (GPTSwarm / AgentPrune / G-Designer / Graph-GRPO / GTD / Codebook Agent) as the training-based contrast class.

**Caveats:** (1) web-index-based search; very recent preprints not yet indexed could be missed; (2) marginal papers in §3 were assessed from abstracts/snippets only; (3) author lists for Graph-GRPO and Codebook Agent could not be fully confirmed from fetched HTML; (4) GPTSwarm full-text fetch truncated before the appendix.
