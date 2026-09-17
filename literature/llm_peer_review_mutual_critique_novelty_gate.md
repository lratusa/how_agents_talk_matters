# Novelty-Gate Literature Search: LLM Peer-Review / Mutual-Critique Systems

**Assignment area:** LLM peer-review and mutual-critique systems; pairwise/reciprocal critique; cross-examination debate; reviewer-assignment methods (TPMS lineage) adapted to LLM agents; diversity-/topology-based agent pairing.

**Proposed method under evaluation (FPRR):** N homogeneous instances of the same LLM checkpoint, same prompt/config, independent stochastic solving → response embeddings → pairwise cosine-distance matrix → farthest-distance one-to-one pairing (greedy farthest matching / maximum-weight perfect matching) → reciprocal within-pair peer review (partner-only visibility) → single synthesizer aggregates all answers + reviews. No training, retrieval, or tools.

**Literature cutoff:** 2026-09-17. All arXiv IDs below were verified by fetching abstract pages and, for the closest papers, full method sections (arXiv HTML/ar5iv/ACL PDFs). Nothing is from memory alone; items that could not be fully verified are explicitly flagged.

**Novelty criteria legend:** A homogeneous same LLM | B same-prompt independent solving | C response-level semantic embeddings | D pairwise semantic distance between responses | E distance → communication/review topology | F farthest/dissimilar peer selection | G one-to-one pairing / perfect matching | H reciprocal critique | I pair-limited information exchange | J final aggregation/synthesis | K no training / no weight modification.

---

## Bottom line

**No verified paper hits E+F+G+H together. No paper implements the FPRR stop-condition chain (same-model independent generation → semantic-distance-based farthest pairing → reciprocal pairwise review → final aggregation). Exact prior art: NOT FOUND in this assignment area.**

The prior art splits cleanly into disjoint buckets:
- **Reciprocal review exists** (Xu et al. 2311.08152 — all-to-all; AutoBench, LivingArena — exhaustive evaluation settings) but never with any selection of *whom* reviews *whom*.
- **Response-embedding distance exists** (S²-MAD 2502.04790 — cosine similarity for redundancy gating; CortexDebate 2507.03928 — cosine similarity as one factor in edge weights; DAR 2603.20640 and Multiagent Finetuning 2501.05707 — embeddings as post-hoc analysis only) but never to construct a one-to-one review pairing, and never for *farthest* selection of review partners.
- **One-to-one pairing exists** (FORD 2305.11595 — fixed model pair; PiCO — random battle pairs; EoT Debate paradigm — fixed tree leaves) but is fixed, random, or structural — never distance-based.
- **Reviewer-assignment-by-similarity (TPMS lineage)** matches reviewers to *manuscripts* by *high* similarity — the opposite direction from farthest pairing, one-directional, and never adapted to LLM agents mutually critiquing generated answers.

---

## Papers examined

### 1. Xu et al. — "Towards Reasoning in Large Language Models via Multi-Agent Peer Review Collaboration"
- **Citation:** Zhenran Xu, Senbao Shi, Baotian Hu, Jindi Yu, Dongfang Li, Min Zhang, Yuxiang Wu. arXiv:2311.08152 (2023). https://arxiv.org/abs/2311.08152
- **Method summary:** Three stages — create, review, revise. N agents (main experiments: 3× gpt-3.5-turbo-0613, no system messages) independently solve; each agent reviews *every* peer's solution with a confidence score; each agent revises from *all* received reviews; final answer by majority vote.
- **Criteria:** A yes | B yes | C no | D no | E no | F no | G no | H **yes (but all-to-all)** | I no | J partial (majority vote, no synthesizer) | K yes
- **Evidence:** "each agent first independently submits its own solution" (§3, Stage 1); "we feed each agent Aᵢ with the solution of its peers (i.e., Aⱼ, where j≠i), one at a time, to write reviews rᵢⱼ" (§3, Stage 2 — every ordered pair, all-to-all); "we feed each agent Aᵢ with the reviews from its peers (i.e., rⱼᵢ), all at once" (violates I); "The final prediction is determined through a majority vote among the n participating agents" (J). Diversity analysis (§4.5) uses INCON (discrete answer-disagreement sign function), post-hoc only, not embeddings, not used for assignment.
- **Verdict:** STRONG OVERLAP — the strongest reciprocal-critique match; but explicitly all-to-all, no embeddings, no pairing, vote aggregation. Deliberately the opposite of pair-limited topology.

### 2. S²-MAD — Zeng et al., "Breaking the Token Barrier"
- **Citation:** arXiv:2502.04790 (2025). https://arxiv.org/abs/2502.04790
- **Method summary:** Sparse MAD variant with a Decision-Making Mechanism: Similarity Calculation Module + Redundancy Filtering + Conditional Participation. Responses are vectorized with an embedding model (BERT-base) and pairwise cosine similarity computed — but similarity is used to *drop redundant viewpoints and decide whether to speak*. Grouping is random shuffle; final answer by majority vote.
- **Criteria:** A yes | B yes | C **yes** | D **yes** | E partial (similarity influences participation/message flow, no topology construction or partner selection) | F no (selects *similar/redundant* for filtering — opposite direction) | G no | H no (agents update own answers; no reciprocal review) | I no (group-wide visibility) | J partial (vote) | K yes
- **Evidence:** "we also propose an alternative vectorization-based approach, where the responses are vectorized using an embedding model, and the cosine similarity is computed to evaluate the similarity of their viewpoints" (§3.2); "Outputs that are identified as similar to either their own or previously received viewpoints are promptly discarded" (§3.2); "Initialize and shuffle the agents randomly" (Algorithm 1, line 4).
- **Verdict:** STRONG OVERLAP on C+D only — the closest use of response-embedding cosine distance, but as a similarity gate, not farthest pairing; no reciprocal review.

### 3. CortexDebate — Sun et al., "Debating Sparsely and Equally for Multi-Agent Debate"
- **Citation:** Yiliu Sun, Zicheng Zhao, Sheng Wan, Chen Gong. Findings of ACL 2025. arXiv:2507.03928. https://arxiv.org/abs/2507.03928
- **Method summary:** Builds a sparse directed debate graph; edge weights from an "MDM" module via the McKinsey Trust Formula W = C×R×I/S, where the intimacy factor I is derived from cosine similarity between agents' textual outputs (higher viewpoint difference → higher weight); edges below the mean are pruned. Final answer by majority vote.
- **Criteria:** A no (heterogeneous backbones: Qwen, Mistral, Typhoon, Llama, Gemma) | B yes | C **yes*** | D **yes** | E **partial** (dissimilarity is one factor in edge weights shaping topology) | F **partial** (formula favors differing viewpoints, but selection is multi-factor mean-threshold pruning, not farthest-partner selection) | G no (each agent debates multiple opponents; not a matching) | H no (debate-style regeneration, no reciprocal critique reports) | I partial (neighbor-limited) | J partial (vote) | K yes
- **Evidence:** "MDM first uses cosine similarity to calculate the textual similarity between Oᵢ^(d−1) and Oⱼ^(d−1)… I_d = 1 − Sim̄_d" where intimacy "represents the average degree of difference in viewpoints between Aᵢ and Aⱼ … as the collision of different viewpoints can enhance the debating effectiveness" (§3.2, Eqs. 6–7); "the edges with weights below [the mean] are removed, resulting in a sparse debating graph" (§4.2); "the debating opponents for Aⱼ, denoted as Debⱼ = {Aᵢ | W_{i→j} = 1, i≠j}" (not one-to-one).
- **Verdict:** STRONG OVERLAP — the only verified paper where pairwise semantic distance between outputs shapes a communication topology *and* favors dissimilarity. Fails A (heterogeneous), G (graph, not matching), H (no reciprocal review). *Caveat: the encoder for "textual similarity" is not specified in the sections read — could be lexical rather than sentence-embedding; worth a PDF check only if the gate hinges on it (it does not, since G/H fail regardless).

### 4. FORD — Xiong et al., "Examining Inter-Consistency of Large Language Models Collaboration: An In-depth Analysis via Debate"
- **Citation:** Kai Xiong, Xiao Ding, Yixin Cao, Ting Liu, Bing Qin. Findings of EMNLP 2023, pp. 7572–7590. arXiv:2305.11595. https://arxiv.org/abs/2305.11595
- **Method summary:** Two (or three) LLMs independently produce stance+argument; on disagreement samples, alternated debate (each refutes the other, up to N rounds); a judge summarizes to a final conclusion.
- **Criteria:** A no (pairs of different models: ChatGPT/Davinci-003, LLaMA/Vicuna) | B yes | C no | D no | E no | F no (disagreement filtering selects *samples*, not partners) | G yes (fixed pair, pre-assigned by model identity) | H yes (reciprocal alternated refutation) | I yes (each sees only partner's arguments) | J yes (judge) | K yes
- **Evidence:** "we conduct fair debates on three pairs of LLMs: different types of LLMs (ChatGPT & Davinci-003 and LLaMA & Vicuna)…" (§4); "the proponent conducts round 1 debate to first counter the opponent… If no compromise, the opponent would conduct round 2 to do the same to the proponent" (§3.2).
- **Verdict:** STRONG OVERLAP on G+H+I — fixed two-agent reciprocal debate, but pairing is by model identity, no embeddings/distance anywhere.

### 5. Du et al. — "Improving Factuality and Reasoning in Language Models through Multiagent Debate"
- **Citation:** Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch. ICML 2024. arXiv:2305.14325. https://arxiv.org/abs/2305.14325
- **Method summary:** N same-model instances, identical prompt, stochastic sampling, all-to-all broadcast debate over rounds, majority/consensus ending.
- **Criteria:** A yes | B yes | C no | D no | E no | F no | G no | H no (solution exchange, not structured reviews) | I no (full broadcast) | J partial (vote) | K yes
- **Verdict:** RELATED ONLY — canonical MAD baseline; shares A+B+K only.

### 6. Exchange-of-Thought (EoT)
- **Citation:** Zhangyue Yin, Qiushi Sun, Cheng Chang, Qipeng Guo, Junqi Dai, Xuanjing Huang, Xipeng Qiu. EMNLP 2023. arXiv:2312.01823. https://arxiv.org/abs/2312.01823
- **Method summary:** Four fixed communication paradigms mapped to network topologies: Memory (bus), Report (star), Relay (ring), Debate (tree — leaf pairs exchange, parents aggregate). Topology fixed a priori; confidence from answer frequency, not embeddings.
- **Criteria:** A yes (default 3× GPT-3.5-Turbo-0301) | B yes | C no | D no | E no ("we order the models by number and connect them in a circle" — fixed) | F no | G partial (tree-leaf pairing by position, not selection) | H partial (answer-sharing exchange, not reciprocal critique reports) | I partial (paradigm-dependent) | J partial (vote) | K yes
- **Evidence:** "we propose four communication paradigms to determine the counterparts for model communication" (§4.1); "leaf nodes to exchange information with each other, while parent nodes are solely responsible for aggregating information" (§4.1).
- **Verdict:** RELATED ONLY — closest structural analog of paired exchange, but pairing is tree-position-fixed with no distance signal.

### 7. ChatEval
- **Citation:** Chi-Min Chan et al. ICLR 2024. arXiv:2308.07201. https://arxiv.org/abs/2308.07201
- **Method summary:** Multi-agent *evaluator* debate over given texts (not the agents' own problem solutions); homogeneous model groups; three fixed communication strategies (one-by-one, simultaneous-talk, simultaneous-talk-with-summarizer); final = majority vote / averaged score.
- **Criteria:** A yes | B no (evaluates given responses, no independent solving) | C no | D no | E no (fixed strategies) | F no | G no | H no | I no (full chat history visible) | J partial (vote/average) | K yes
- **Verdict:** RELATED ONLY — evaluation debate, no pairing mechanism.

### 8. ReConcile
- **Citation:** Justin Chih-Yao Chen, Swarnadeep Saha, Mohit Bansal. ACL 2024. arXiv:2309.13007. https://arxiv.org/abs/2309.13007
- **Method summary:** Round-table of *diverse/heterogeneous* LLMs; multi-round discussion with grouped answers + confidences; confidence-weighted voting.
- **Criteria:** A no ("diversity originating from different models is critical to its superior performance" — abstract) | B yes | C no | D no | E no | F no | G no | H no | I no (all see all grouped answers) | J yes (weighted voting) | K yes
- **Verdict:** RELATED ONLY — diversity via heterogeneous models, not measured/selected via embeddings.

### 9. Corex
- **Citation:** Qiushi Sun, Zhangyue Yin, Xiang Li, Zhiyong Wu, Xipeng Qiu, Lingpeng Kong. COLM 2024. arXiv:2310.00280. https://arxiv.org/abs/2310.00280
- **Method summary:** Discuss/Review/Retrieve modes. Review mode: one randomly selected primary agent's solution is iteratively reviewed by other agents in a sequential chain. Discuss mode: agents randomly divided into two groups + judge.
- **Criteria:** A partial (homogeneous GPT-3.5 main, roles/personas assigned) | B partial (only primary agent solves initially in Review mode) | C no | D no | E no (selection explicitly random: "a single agent Ap is randomly selected", §3.2) | F no | G no (sequential chain, not one-to-one) | H no (reviewers review the primary; not reciprocal) | I no (reviewers see predecessors' output) | J partial | K yes
- **Verdict:** RELATED ONLY — review chains exist but assignment is random and one-directional.

### 10. PiCO — Ning et al., "Peer Review in LLMs Based on Consistency Optimization"
- **Citation:** ICLR 2025; arXiv:2402.01830. https://arxiv.org/abs/2402.01830
- **Method summary:** Unsupervised LLM ranking: models answer questions; answer pairs ("battle pairs") are *randomly* constructed and *randomly* assigned reviewers; learnable confidence weights optimized for consistency; reviewer elimination.
- **Criteria:** A no (heterogeneous model pool) | B partial | C no | D no | E no | F no | G partial (pair-based, but pairing + reviewer assignment explicitly random: "we randomly construct a battle pair … Each battle pair will randomly assign 'reviewers'", §2.2) | H partial | I no | J yes (weighted score ranking) | K yes
- **Verdict:** RELATED ONLY — the only verified pair-based mutual-assessment system found, and pairing is random — useful contrast for FPRR's distance-based pairing claim.

### 11. DyLAN — Dynamic LLM-Powered Agent Network
- **Citation:** Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, Diyi Yang. arXiv:2310.02170 (2023). https://arxiv.org/abs/2310.02170
- **Method summary:** Team optimization via Agent Importance Score from *peer ratings* (LLM verbalized scoring), top-contributory agents selected; LLM Ranker deactivates low performers between rounds; layered feed-forward topology.
- **Criteria:** A partial (same backbone, distinct role prompts) | B partial | C no | D no | E partial (dynamic topology, but driven by LLM-ranker contribution ranking, not embedding distance: "edges will only be added for these top-ranked agents") | F no (selects top-contributory — most similar-to-correct, not farthest) | G no (many-to-many layered edges) | H partial (unidirectional feed-forward rating) | I no (all previous-layer responses visible) | J yes | K yes
- **Verdict:** RELATED ONLY — dynamic topology, wrong selection signal.

### 12. Sparse MAD — Li et al., "Improving Multi-Agent Debate with Sparse Communication Topology"
- **Citation:** Yunxuan Li, Yibing Du, Jiageng Zhang, Le Hou, Peter Grabowski, Yeqing Li, Eugene Ie. Findings of EMNLP 2024. arXiv:2406.11776. https://arxiv.org/abs/2406.11776
- **Method summary:** Systematic comparison of static sparse graph families (ring, tree, random, regular) vs. fully-connected MAD. Topologies content-independent.
- **Criteria:** A yes | B yes | C no | D no | E no ("We focus on static graphs in this work", §3.1 — topology chosen from graph families, not data-derived) | F no | G no | H no | I partial (neighbor-limited, fixed) | J partial (vote) | K yes
- **Verdict:** RELATED ONLY — topology sparsity without any semantic selection.

### 13. GroupDebate
- **Citation:** Tongxuan Liu et al. arXiv:2409.14051 (2024). https://arxiv.org/abs/2409.14051
- **Method summary:** M agents randomly divided into N groups; intra-group debate + inter-group summary exchange; majority vote.
- **Criteria:** A yes | B yes | C no | D no | E no ("Initialize and shuffle the agents randomly", Algorithm 1) | F no | G no (groups, not pairs) | H no | I partial (group-limited) | J partial | K yes
- **Verdict:** RELATED ONLY — grouping is random.

### 14. DAR — Nguyen et al., "Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention"
- **Citation:** arXiv:2603.20640 (Mar 2026, within cutoff). https://arxiv.org/abs/2603.20640 (abstract + full HTML method §3 read)
- **Method summary:** Standard homogeneous MAD + a filter agent (same LLM) that each round retains a subset of responses that "differ the most from each other and from the majority vote"; retained responses are broadcast to all; final answer majority vote. Sentence embeddings (all-MiniLM-L6-v2) appear only as an *evaluation metric*.
- **Criteria:** A yes ("The module is implemented using the same backbone LLM as the generation agents", §3.4) | B yes | C partial (embeddings post-hoc only: "We measure diversity as the average pairwise embedding distance (1 − cosine similarity) among retained responses", §4.3.1) | D partial (same) | E no (selection is LLM-judged subset filtering; topology is broadcast) | F partial (dissimilarity-driven *retention*, not pairing; criterion is LLM judgment, not computed distance) | G no | H no | I no (broadcast) | J yes (vote; Algorithm 1 line 18) | K yes ("training-free", §3.4)
- **Verdict:** STRONG OVERLAP on F-flavor — closest to dissimilarity-based selection, but it is subset retention with broadcast, not one-to-one pairing with reciprocal review.

### 15. Multiagent Finetuning — Subramaniam et al.
- **Citation:** Vighnesh Subramaniam et al. ICLR 2025. arXiv:2501.05707. https://arxiv.org/abs/2501.05707
- **Method summary:** Same base model splits into generator/critic roles via independent finetuning on debate traces. Notable: appendix analyzes diversity via embedding dissimilarity between agents' responses — strictly an analysis metric.
- **Criteria:** C yes (analysis only: "we analyze diversity by measuring the embedding dissimilarity between responses of different agents", §C.4) | D partial (analysis only) | E no | F no | G no | H partial (all-to-all debate critique) | K **no** (finetuning is the point)
- **Verdict:** RELATED ONLY — excludes itself on K; embedding dissimilarity never operationalized.

### 16. PiFlow — "Principle-Aware Scientific Discovery with Multi-Agent Collaboration"
- **Citation:** arXiv:2505.15047 (2025). https://arxiv.org/abs/2505.15047
- **Method summary:** Scientific-discovery MAS; diversifies *principles/hypotheses* via max–min embedding distance ("the strategy of maximizing the minimum embedding distance between principles is not an arbitrary heuristic"). Farthest-point-style selection in embedding space, applied to hypothesis diversification — not solver pairing, no review stage.
- **Criteria:** C partial (principle embeddings) | F partial (farthest-selection flavor, different substrate) | E/G/H no | J yes | K yes
- **Verdict:** RELATED ONLY — the closest "farthest-in-embedding-space" mechanism; different object (hypotheses), no pairing, no reciprocal review.

### 17. MARS — Wang et al., "Toward More Efficient Multi-Agent Collaboration for LLM Reasoning"
- **Citation:** arXiv:2509.20502 (within cutoff). https://arxiv.org/abs/2509.20502
- **Method summary:** Author → m independent reviewers → meta-reviewer → revision loop. "Reviewers in MARS operate independently … avoiding costly reviewer-to-reviewer interactions." Reviewers review the author's solution, not each other.
- **Criteria:** A yes | B no (only author solves) | C/D/E/F/G no | H no | I no | J yes (meta-reviewer aggregation) | K yes
- **Verdict:** RELATED ONLY — star topology, non-reciprocal review.

### 18. LM vs LM — Cohen et al., "Detecting Factual Errors via Cross Examination"
- **Citation:** Roi Cohen, May Hamri, Mor Geva, Amir Globerson. EMNLP 2023. arXiv:2305.13281. https://arxiv.org/abs/2305.13281
- **Method summary:** One LM (examinee) generates a claim; a second LM (examiner) generates questions to expose inconsistencies; a judge decides factuality. Directional, asymmetric.
- **Criteria:** A no | H no ("a multi-turn interaction between the LM that generated the claim and another LM (acting as an examiner)" — one-directional) | I yes | G yes (fixed pair) | C/D/E/F no | J partial | K yes
- **Verdict:** RELATED ONLY — shares the "cross-examination" term; no reciprocal review, no pairing selection.

### 19. PRD — Li, Patel, Du, "Peer Rank and Discussion Improve Large Language Model based Evaluations"
- **Citation:** TMLR 2024. arXiv:2307.02762. https://arxiv.org/abs/2307.02762
- **Method summary:** LLMs as peer evaluators ranking each other's answers; peer discussion where two LLMs agree on preference between two answers. Evaluation framework, not collaborative solving; answer pairing exhaustive/tournament.
- **Criteria:** A no | C/D/E/F/G no | H partial (mutual evaluation among all peers) | J partial (ranking aggregation) | K yes
- **Verdict:** RELATED ONLY.

### 20. AutoBench — Loi et al.
- **Citation:** arXiv:2510.22593 (within cutoff). https://arxiv.org/abs/2510.22593
- **Method summary:** Heterogeneous open models rotate as task generator/contestant/judge; "reciprocal peer assessment" via complete n×n pairwise judging; iterative weight aggregation. Leaderboard/evaluation system.
- **Criteria:** A no | B yes | C/D/E/F no | G no | H yes (all judge all — exhaustive, no selection) | I no | J yes | K yes
- **Verdict:** RELATED ONLY — reciprocal but exhaustive; evaluation setting, not answer improvement.

### 21. LivingArena
- **Citation:** arXiv:2607.24780 (Jul 2026, within cutoff; verified via abstract page).
- **Method summary:** Peer-probing tournament: models alternate as questioner/answerer in exhaustive round-robin ("a full round-robin tournament of (10 choose 2)=45 unique pairs"); judge panel grades; Elo aggregation.
- **Criteria:** A no | G partial (pairwise matches, exhaustive/adversarial, not distance-selected) | H no | J yes (Elo) | K yes | others no
- **Verdict:** RELATED ONLY.

### 22. GPTSwarm — Zhuge et al., "Language Agents as Optimizable Graphs"
- **Citation:** ICML 2024. arXiv:2402.16823. https://arxiv.org/abs/2402.16823
- **Method summary:** Agents as computational graphs; node (prompt) and edge (connectivity) optimization via REINFORCE-style policy gradient.
- **Criteria:** C/D no | E partial (topology optimized by utility-driven RL, not semantic distance) | F/G/H/I no | J partial | K **no** (training/optimization loop over graph parameters)
- **Verdict:** RELATED ONLY — topology is learned by RL, and K fails.

### 23. AgentPrune ("Cut the Crap")
- **Citation:** Guibin Zhang et al. ICLR 2025. arXiv:2410.02506. https://arxiv.org/abs/2410.02506
- **Method summary:** Formalizes communication redundancy; trains differentiable low-rank graph masks (policy gradient), one-shot prunes edges, fixes topology.
- **Criteria:** C/D no | E partial (topology pruned by learned importance masks, not embedding distance) | F/G/H/I no | J yes | K **no** (trainable masks)
- **Verdict:** RELATED ONLY.

### 24. G-Designer
- **Citation:** Guibin Zhang et al. ICML 2025. arXiv:2410.11782. https://arxiv.org/abs/2410.11782
- **Method summary:** Variational graph auto-encoder encodes agent *profiles* and decodes task-specific communication topologies.
- **Criteria:** C partial (profile encodings, not response embeddings) | E partial (learned encoder) | D/F/G/H/I no | K no (trained VGAE)
- **Verdict:** RELATED ONLY.

### 25. MacNet — Qian et al., "Scaling Large Language Model-based Multi-Agent Collaboration"
- **Citation:** NeurIPS 2024. arXiv:2406.07155. https://arxiv.org/abs/2406.07155
- **Method summary:** Agents on fixed DAG topologies (chain, star, tree, mesh, random, layered); scaling study to 1000+ agents.
- **Criteria:** A yes | E no (topology chosen a priori) | F/G no | H partial (edge agents deliver critiques; not reciprocal pairs) | others no/partial | K yes
- **Verdict:** RELATED ONLY.

### 26. AgentVerse — Chen et al.
- **Citation:** ICLR 2024. arXiv:2308.10848. https://arxiv.org/abs/2308.10848
- **Method summary:** Expert recruitment → collaborative decision-making → execution → evaluation; dynamic composition by LLM judgment.
- **Criteria:** A no (heterogeneous recruited experts) | E no (composition by LLM judgment) | C/D/F/G/H/I no | J yes | K yes
- **Verdict:** NOT RELEVANT to the pairing mechanism.

### 27. More Agents Is All You Need
- **Citation:** Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, Deheng Ye. TMLR 2024. arXiv:2402.05120. https://arxiv.org/abs/2402.05120
- **Method summary:** Pure sampling-and-voting with same-model instances; no communication at all.
- **Criteria:** A yes | B yes | C–I no | J yes (vote) | K yes
- **Verdict:** RELATED ONLY — closest on A+B+K, but the negation of E–I.

### 28. DynaDebate — Li et al., "Breaking Homogeneity in Multi-Agent Debate with Dynamic Path Generation"
- **Citation:** arXiv:2601.05746 (Jan 2026, within cutoff). https://arxiv.org/abs/2601.05746 (abstract/intro verified)
- **Method summary:** Homogeneous agents assigned distinct reasoning paths to inject diversity, then debate. Diversity is prompt-assigned, not embedding-measured, not used for pairing.
- **Criteria:** A yes | B partial | C/D/E/F/G/H no | J partial | K yes
- **Verdict:** RELATED ONLY.

### 29. AgentReview — Jin et al., "Exploring Peer Review Dynamics with LLM Agents"
- **Citation:** Findings of EMNLP 2024. arXiv:2406.12708. https://arxiv.org/abs/2406.12708
- **Method summary:** LLM-agent simulation of the conference peer-review pipeline (reviewers → rebuttal → AC discussion → meta-review) to study reviewer biases. Reviewers review *manuscripts*; no reciprocal solver-solver review; assignment not distance-based.
- **Criteria:** H no (author–reviewer asymmetry) | C/D/E/F/G no | K yes
- **Verdict:** NOT RELEVANT as prior art (peer-review simulation, not answer critique pairing).

### 30. Ha & Mitchell — "Beyond Correctness: Distance-Based Social Dynamics of Multi-Agent Debate"
- **Citation:** Seungwoong Ha, Melanie Mitchell. ICML 2026, OpenReview o4v0sr8Mpy (Apr 2026, within cutoff). **Partially verified** — OpenReview bot-blocked the fetcher; assessment from abstract + figure captions only.
- **Method summary:** Analysis paper measuring embedding-space distances in debate ("agents are more likely to revise when their answers are farther from the correct solution"; "distance contraction after revision"). Distance is an analytical lens, not a partner-selection mechanism; no pairing, no review topology.
- **Criteria:** C yes | D yes | E no (analysis only) | F/G/H no
- **Verdict:** RELATED ONLY — worth citing as related use of response-embedding distance in MAD analysis. Low novelty risk; human spot-check recommended only for completeness.

### 31. ThoughtComm — Zhao et al., "Thought Communication in Multiagent Collaboration"
- **Citation:** arXiv:2510.20733 (within cutoff). https://arxiv.org/abs/2510.20733
- **Method summary:** Extracts latent thoughts from hidden states via a sparsity-regularized autoencoder; routes latent thoughts; prefix injection. Trains modules.
- **Criteria:** C partial (hidden-state representations, not sentence embeddings; authors note text embeddings as future work) | E/F/G/H no | K **no**
- **Verdict:** NOT RELEVANT (fails K; wrong substrate).

### 32. PER study — Yang et al., "Precise but Uncoupled: Reviewer Precision Does Not Guarantee Critique Uptake in Multi-Agent Math Reasoning"
- **Citation:** arXiv:2607.15388 (Jul 2026, within cutoff). https://arxiv.org/abs/2607.15388
- **Method summary:** Compares planner-executor-reviewer hierarchy vs. broadcast peer discussion on Omni-MATH with matched gpt-oss-120b actors. Asymmetric dedicated reviewer role.
- **Criteria:** A yes | B yes | C/D/E/F/G no | H no (asymmetric, not reciprocal) | I no | J yes | K yes
- **Verdict:** NOT RELEVANT to pairing mechanism.

### 33. DySCo — "Dynamic Trust-Aware Sparse Communication Topology for LLM-Based Multi-Agent Consensus"
- **Citation:** arXiv:2606.01828 (Jun 2026, within cutoff). Verified to exist; trust/reputation-based edge selection, not semantic distance.
- **Verdict:** NOT RELEVANT (trust-based, not distance-based).

### 34. "From Replication to Redesign: Exploring Pairwise Comparisons for LLM-Based Peer Review" — Zhang et al.
- **Citation:** arXiv:2506.11343 (2025). https://arxiv.org/abs/2506.11343
- **Method summary:** LLMs do pairwise comparisons of *manuscripts* for conference review. No reciprocal solver-solver review, no distance pairing.
- **Verdict:** NOT RELEVANT (manuscript evaluation).

### 35. Reviewer-assignment lineage (TPMS → LLM era) — verified as a *different task*
- Charlin & Zemel, "The Toronto Paper Matching System" (ICML 2013 workshop). TF-IDF similarity of reviewer publications to submissions + assignment optimization. https://www.cs.utoronto.ca/~lcharlin/papers/tpms.pdf
- Mimno & McCallum (2007), expertise modeling for reviewer assignment (TPMS lineage; not adapted to LLM answer critique anywhere verifiable).
- PeerReview4All — Stelmakh et al., arXiv:1806.06237, JMLR 2021 — similarity-based paper→reviewer matching *maximizing* competence similarity under constraints.
- Zhang et al., "Chain-of-Factors Paper-Reviewer Matching", WWW 2025 — LLM-generated review topics as shared matching space for paper↔reviewer matching.
- Bagheri et al., "Leveraging knowledge graphs and LLMs for content-based reviewer assignment", J. Intell. Inf. Syst. 2025/2026 (Springer, DOI 10.1007/s10844-025-01004-9).
- Leyton-Brown et al., "Vulnerability of Text-Matching in ML/AI Conference Reviewer Assignments to Collusions", arXiv:2412.06606 — robustness of similarity-score reviewer assignment.
- Also checked: GAR (arXiv:2412.10415), Agent Reviewers (Lu et al., ICML 2025, PMLR 267:40803), RATE (arXiv:2601.19637 — within-cutoff ID reported by one search; not independently re-verified), ReviewAgents (arXiv:2503.08506).
- **Key finding:** this entire lineage assigns reviewers to *manuscripts* by *high* similarity (expertise matching — the opposite of farthest pairing), is one-directional (reviewer → paper), and involves no reciprocal critique or answer synthesis. **No adaptation of reviewer assignment to pair LLM agents for mutual critique of generated answers was found or could be verified to exist** (9+ query formulations tried, including synonyms for pairing/matching/reciprocal critique/farthest selection).
- **Verdict:** NOT RELEVANT as prior art; important as a contrast — FPRR inverts the similarity direction (farthest vs. most-similar) and changes the substrate (answer↔answer reciprocal, not reviewer→manuscript).

---

## Consolidated criteria table (critical criteria C–I + K)

| # | Paper | C | D | E | F | G | H | I | K |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Xu et al. 2311.08152 | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (all-to-all)** | ✗ | ✓ |
| 2 | S²-MAD 2502.04790 | ✓ | ✓ | ~ | ✗ (similarity gate) | ✗ | ✗ | ✗ | ✓ |
| 3 | CortexDebate 2507.03928 | ✓* | ✓ | ~ | ~ | ✗ | ✗ | ~ | ✓ |
| 4 | FORD 2305.11595 | ✗ | ✗ | ✗ | ✗ | ✓ (fixed pair) | ✓ | ✓ | ✓ |
| 5 | Du et al. 2305.14325 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 6 | EoT 2312.01823 | ✗ | ✗ | ✗ (fixed topo) | ✗ | ~ (tree leaves) | ~ | ~ | ✓ |
| 7 | ChatEval 2308.07201 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 8 | ReConcile 2309.13007 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 9 | Corex 2310.00280 | ✗ | ✗ | ✗ (random) | ✗ | ✗ | ✗ | ✗ | ✓ |
| 10 | PiCO 2402.01830 | ✗ | ✗ | ✗ | ✗ | ~ (random pairs) | ~ | ✗ | ✓ |
| 11 | DyLAN 2310.02170 | ✗ | ✗ | ~ (LLM ratings) | ✗ | ✗ | ~ | ✗ | ✓ |
| 12 | Sparse MAD 2406.11776 | ✗ | ✗ | ✗ (static) | ✗ | ✗ | ✗ | ~ | ✓ |
| 13 | GroupDebate 2409.14051 | ✗ | ✗ | ✗ (random) | ✗ | ✗ | ✗ | ~ | ✓ |
| 14 | DAR 2603.20640 | ~ (analysis) | ~ (analysis) | ✗ | ~ (retention, not pairing) | ✗ | ✗ | ✗ | ✓ |
| 15 | Multiagent Finetuning 2501.05707 | ~ (analysis) | ~ (analysis) | ✗ | ✗ | ✗ | ~ | ✗ | ✗ |
| 16 | PiFlow 2505.15047 | ~ (principles) | ~ | ✗ | ~ (max-min, different object) | ✗ | ✗ | ✗ | ✓ |
| 17 | MARS 2509.20502 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 18 | LM vs LM 2305.13281 | ✗ | ✗ | ✗ | ✗ | ✓ (fixed) | ✗ (one-way) | ✓ | ✓ |
| 19 | PRD 2307.02762 | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✗ | ✓ |
| 20 | AutoBench 2510.22593 | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (exhaustive) | ✗ | ✓ |
| 21 | LivingArena 2607.24780 | ✗ | ✗ | ✗ | ✗ | ~ (round-robin) | ✗ | ✗ | ✓ |
| 22 | GPTSwarm 2402.16823 | ✗ | ✗ | ~ (RL) | ✗ | ✗ | ✗ | ✗ | ✗ |
| 23 | AgentPrune 2410.02506 | ✗ | ✗ | ~ (learned masks) | ✗ | ✗ | ✗ | ✗ | ✗ |
| 24 | G-Designer 2410.11782 | ~ (profiles) | ✗ | ~ (learned) | ✗ | ✗ | ✗ | ✗ | ✗ |
| 25 | MacNet 2406.07155 | ✗ | ✗ | ✗ (a priori) | ✗ | ✗ | ~ | ✗ | ✓ |
| 26 | AgentVerse 2308.10848 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 27 | More Agents 2402.05120 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 28 | DynaDebate 2601.05746 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 29 | AgentReview 2406.12708 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 30 | Ha & Mitchell ICML'26 | ✓ | ✓ | ✗ (analysis) | ✗ | ✗ | ✗ | ✗ | ✓ |
| 31 | ThoughtComm 2510.20733 | ~ (hidden states) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 32 | PER study 2607.15388 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 33 | DySCo 2606.01828 | ✗ | ✗ | ✗ (trust-based) | ✗ | ✗ | ✗ | ✗ | ✓ |
| 34 | Pairwise LLM review 2506.11343 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 35 | TPMS lineage | n/a — different task (reviewer→manuscript, similarity-maximizing, one-directional) | | | | | | | |

*CortexDebate C: output-level cosine similarity, but the encoder is unspecified in the sections read (possibly lexical, not sentence-embedding).

## E+F+G+H conjunction and stop-condition check

- **E+F+G+H together: ZERO papers.**
- **Full stop-condition chain (same-model independent generation → embedding-distance farthest pairing → reciprocal pairwise review → synthesizer aggregation): ZERO papers.**
- Closest single papers by axis:
  - Reciprocal critique (H): Xu et al. 2311.08152 — all-to-all, no embeddings, no pairing.
  - Distance → topology (C+D+E): CortexDebate 2507.03928 — heterogeneous models, multi-factor thresholded graph (not matching), debate not review.
  - Dissimilarity selection (F): DAR 2603.20640 — LLM-judged subset retention + broadcast; PiFlow 2505.15047 — max-min distance over hypotheses, no pairing/review.
  - One-to-one pairing (G): FORD 2305.11595 — fixed heterogeneous pair; PiCO — random pairs.

## Caveats

1. CortexDebate's similarity encoder is unspecified in the sections read (lexical vs. sentence-embedding). Immaterial to the verdict: it fails G and H regardless.
2. Ha & Mitchell (ICML 2026) verified via abstract/figure captions only (OpenReview bot wall). It is an analysis paper, not a pairing method; novelty risk negligible.
3. A few 2026 arXiv IDs (2603.20640, 2607.24780, 2606.01828, 2607.15388, 2601.05746) were verified by subagent fetches of abstract/HTML pages; none are close to the stop condition even as described.
4. Coverage limitation: searches concentrated on English arXiv/venue papers via web search (~40 query formulations across four searchers, including synonyms for reciprocal/mutual critique, farthest/dissimilar pairing, perfect matching, embedding routing, reviewer assignment). A very recent workshop paper or non-indexed preprint could in principle exist, but nothing surfaced.
