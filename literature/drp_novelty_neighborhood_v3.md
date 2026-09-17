# Novelty Neighborhood Report v3 — DRP (Disagreement Decomposition and Blind Repair)

**Date:** 2026-09-17
**Literature cutoff:** 2026-09-17（仅计入此前公开的工作；检索中出现的 2026-09-17 之后日期条目一律排除）
**Base:** 本报告在 v2（`literature/drp_novelty_gate.md`，GO-WITH-REPOSITIONING）之上扩展，不重复 v2 已验证内容。
**Method note:** 所有引用均实际 fetch 验证（arXiv abstract/API、Semantic Scholar Graph API、OpenAlex API）。无任何凭记忆条目。

---

## 1. 检索协议（本轮新增）

- **Forward（cited-by）**：Semantic Scholar `/paper/arXiv:{id}/citations`；S2 持续 429 时以 OpenAlex `filter=doi:10.48550/arxiv.{id}` 交叉核对 cited_by_count。
- **Backward（references）**：Semantic Scholar `/paper/arXiv:{id}/references`（OpenAlex 对这六篇 arXiv 记录均无 referenced_works 数据，弃用）。
- **Semantic neighborhood**：S2 推荐接口 `/recommendations/v1/papers/forpaper/arXiv:{id}`（DR、GuardedRepair）+ OpenAlex 全文检索六组关键词（"gated self-correction"、"blind verification disagreement"、"answer selection candidate pool multi-agent"、"epistemic credit disagreement verification"、"crux disagreement resolution LLM"、"asymmetric gate answer revision keep original"）。
- **验证层级**：所有进入清单的论文均经 arXiv API/abstract 页核对标题、作者、日期、abstract；仅 snippet 级命中的单独标注。

## 2. 任务 2 结果：Ji et al. (arXiv:2608.25937) — P0 精读

**验证后的准确引用**（arXiv API + abstract 页双重确认）：

> Jia-Hao Ji, Sijie Li, Jiabei Cheng, Zixi She, Jin-Tai Yu, Zhiyuan Yuan. "Candidate supply and answer selection shape the value of LLM judging in multi-agent systems." arXiv:2608.25937 [cs.AI], v1 2026-08-26, v2 2026-08-30. 预印本，无 venue；S2 cited-by = 0（2026-09-17 查询）。

**方法要点**（abstract 精读）：将多智能体推理概念化为「candidate generation → peer communication → terminal selection」的演化流水线；指出无质量控制的共识可呈现 "memetic drift"。两个研究问题：(1) LLM judge 何时提供有效选择压力（为候选提供正确性信号）；(2) 何时用该信号改善最终报告答案。规模：15,336 题（MMLU-Pro、GPQA、MedXpertQA、MuSR；HLE 单独分析）绘制 judge 可靠性地图；81,390 个固定候选池（来自 16,278 题、五个 benchmark）做 replay 实验。三个发现：

1. 正确答案常常**已经存在于候选中**，系统仍收敛并报告错误答案；
2. Judge 可靠性不是模型固定属性，随任务、生成器、正确答案稀有度变化；
3. 仅改动最终答案选择规则（answer frequency + judge evaluation 结合）将准确率从 63.82% 提到 70.82–70.95%，主要靠**拯救被多数错误压倒的正确答案**。

结论：生成更多候选的价值取决于额外样本能否让正确答案「在场、频繁或可识别」；通过隔离 generation/recognition/selection，为「保护已生成的正确答案不丢失」的架构设计建立诊断基础。

**与我们的关系评估**：

- **直接预占我们第一篇论文的核心叙事之一**：「正确答案已在候选集中但被系统丢失」「候选供给（supply）封顶 judge 收益」「judge 可靠性非常数」这三条均已由 Ji et al. 以大规模 replay 实验发表（2026-08-26，早于截止日三周）。第一篇论文若以此为主打发现，**必须引用 Ji et al. 并重新定位增量**（我们的增量只能落在：同模型同源 N 采样设定、crux 级机制归因、双向 transition 漏斗，而非「candidate supply caps judging」这一现象本身）。
- **对 DRP 反而构成强化**：Ji et al. 明确定位为**诊断性**工作（"establish a diagnostic basis for designing multi-agent architectures that protect generated correct answers from being lost"）——它画出了问题地图（generation/recognition/selection 三环节隔离），但**没有给出机制**：没有 disagreement 的 crux 级表示、没有 blind verification、没有 repair、没有 incumbent 保护门，唯一的干预是选择规则（frequency+judge）。DRP 可以把自己定位为「回答 Ji et al. 呼吁的候选架构」——把威胁转化为引用杠杆。
- **注意**：Ji et al. 的五维覆盖在 selection 端最深，preservation 端仅有动机性表述（protect...from being lost），无机制实现；disagreement 仅答案级频率分布。
- Ji et al. 的 62 条参考文献中**不含** DR / GuardedRepair / DISC / PVD / CoVe（对 S2 references 全量 grep 确认），即两条文献线尚未交汇——我们先交汇它们，本身就是 related work 的增量。

## 3. 任务 1 结果：四向雪球

### 3.1 Forward（cited-by）—— 2026 年种子几乎无人引用

| 种子 | S2 cited-by | OpenAlex cited_by | 截止内相关引用 |
|---|---|---|---|
| GuardedRepair (2605.24613) | 0 | 0 | 无 |
| DISC (2606.21724) | 1 | 0 | 仅一篇综述：Mingguang Chen, Licheng Wang, Bo Qu, "Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops," arXiv:2607.07663, 2026-07-08（自改进综述，DISC 被归入 bounded self-refinement；相关度低） |
| DR (2607.01251) | 0 | 0 | 无 |
| PVD (2605.25133) | 0 | 0 | 无 |
| CoVe (2309.11495) | ≥200（S2 分页，采样前 200 条） | 43（索引滞后） | 见下 |
| Ji et al. (2608.25937) | 0 | 0 | 无 |

**含义**：v2 判定的两个最危险先验（GuardedRepair、DISC）在截止日前**均无后续工作承接**——它们预占了组件但没有形成生态；同时意味着审稿人通过 cited-by 发现它们的概率中等偏低（但我们仍必须在 related work 主动处理）。

CoVe 引用中（前 200 条采样，关键词过滤 + 逐条 arXiv 验证）相关的新发现：

| 论文 | 验证方式 | 相关度 |
|---|---|---|
| "Verify-Gated Completion as Admission Control in a Governed Multi-Agent Runtime," arXiv:2605.17998, 2026-05-18 | arXiv API abstract | **高**（gate 语义）：agent 可 propose completion，只读 verifier 决定 admission，证据不足时 **fail-closed**，事件 trace 保留审计路径 —— 是「默认拒绝 + 审计留痕」门控在 agent runtime 层的先验，但不涉及答案修复/分歧/crux |
| "Route, Don't Fix: Regime-Dependent Decoding Correction and a Trajectory-Gated Router for Reliable Clinical LLM Answer Selection," arXiv:2609.14825, 2026-09-13 | arXiv API abstract | 中：trajectory-gated router 决定**是否施加**解码纠正（routing 优于 fixing）——「条件性修改权」思想在临床 MCQA 的实例 |
| "ConsistencyGate: Preventing Memory Contamination in LLM Agents via Self-Consistency Admission Control," arXiv:2607.22962, 2026-07-25 | arXiv API abstract | 中：写入时 admission gate（self-consistency 判定）防止幻觉事实进入持久记忆 —— gate 思想在 memory-write 域 |
| "LEDGERMIND: Provenance-Constrained Multimodal Agentic Reasoning with a Structured Evidence Ledger," arXiv:2607.28374, 2026-07-30 | arXiv API abstract | 中（C4 相关）：Structured Evidence Ledger 作为轨迹状态，决策 claim 须引用 ledger 证据 —— provenance/certificate 相邻概念，但是多模态 VQA 域、非修改证书 |
| "Cost-Pragmatic Quality Gating and Selection-Fusion Multi-Model Combiners," arXiv:2607.13551, 2026-07-15 | S2 元数据（snippet 级） | 低-中：gating + selection-fusion，BioASQ 域 |

### 3.2 Backward（references）—— 无新的机制级威胁

五篇种子的参考文献（S2，全量拉取后关键词过滤）落在 v2 已覆盖的谱系内：debate 线（1805.00899, 2311.08702, 2409.16636, 2509.23055, 2510.20963）、self-correction 线（2303.17651 Self-Refine, 2305.11738 CRITIC, 2310.01738 "LLMs Cannot Self-Correct Yet", 2412.19513 Confidence v.s. Critique）、verification 线（2110.14168, 2305.20050 Let's Verify Step by Step, 2306.03872 Deductive Verification of CoT, 2305.03268 Verify-and-Edit, 2307.03987 "A Stitch in Time"）、selective classification 线（1705.08500, 1901.09192 SelectiveNet, "Noise-free Selective Classification"）、SelfCheckGPT/SelfCheckGPT 类（2303.08896）。值得点名的新增谱系节点（均经 S2 元数据确认）：

- **GuardedRepair 引 selective classification 谱系**（Geifman & El-Yaniv 1705.08500 等）：其 gate 的智识血统是 selective prediction，DRP 写 related work 时可沿此线定位「门控 = 选择性预测在修改权上的迁移」。
- **DISC 引 2412.19513（Confidence v.s. Critique: A Decomposition of Self-Correction Capability）**：recognition（critique）与 correction 能力分解 —— 与我们的五维框架中 recognition/generation 分离相呼应，可引。
- **DR 引 2509.23055（Peacemaker or Troublemaker: sycophancy in MAD）与 2510.20963（ColMAD）**：MAD 失败模式线，v2 已覆盖。

### 3.3 Semantic neighborhood —— 三个新的最近邻

S2 推荐接口（基于 DR 与 GuardedRepair）+ OpenAlex 关键词检索，经 arXiv 逐条验证后的新发现：

| 论文 | 来源 | 验证方式 | 相关度 |
|---|---|---|---|
| **AgentAuditor**：Wei Yang, Shixuan Li, Heng Ping, Peiyu Zhang, Paul Bogdan, Jesse Thomason. "Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and LLM-as-Judge." arXiv:2602.09341, v1 2026-02-10 | Ji refs backward | arXiv API abstract 精读 | **极高（新最近邻）**：将 agent traces 组织为显式表示 agreements/**divergences** 的 Reasoning Tree，在**关键分歧点**比较 branch 级证据，把全局裁决变成**局部化验证**；另提 ACPO 训练 adjudicator 抗误导性多数线索。这是「crux 来自分歧结构」的最接近先验——分歧点定位 + 局部证据核验。差异：trace 树分支级而非答案级 crux 抽取；adjudicator 看全部 trace（无 blindness）；无 repair、无 gate、无 incumbent 保护；目标是聚合准确率 |
| **Minority Sentinel**："When to Overturn Majority Voting in Multi-Agent LLM Debates." arXiv:2606.29270, v1 2026-06-28 | Ji refs backward | arXiv API abstract 精读 | **高**：指出 LLM 错误强相关导致多数票系统性压制正确少数派（"Minority Truth"），约 1/4 分歧案例中少数派正确（10pp 理论挽回空间）；轻量 meta-classifier 决定何时推翻多数票。与 Ji 发现 3 同属「拯救被压倒的正确答案」线，但是 selection 端条件 overturn，非 incumbent 保护门、非 repair |
| **VRR-Stop**："Verify, Repair, Repeat, or Stop? Robust Stopping for Noisy Verify-Repair Loops in LLM Agents." arXiv:2607.17641, v1 2026-07-20 | GuardedRepair 推荐邻域 | arXiv API abstract 精读 | **高（preservation 维新先验）**：显式建模 verifier 与 repairer 双方噪声；指出 repair 可**损坏已正确的 plan**、报告接受率升而真实有效性降；四参数噪声模型分离 verifier 误收/误拒与 repairer 修复/损坏行为，belief filtering 决定停止。是「无 deterministic checker 时保护 incumbent 不被 repair 腐蚀」的最接近先验——但以停止规则形式出现，非 τ 门 + KEEP 默认 + 净修复目标 |
| CASE："A decodability criterion predicts when hidden-state selection beats majority voting in large language models." arXiv:2608.17124, v1 2026-08-17 | Ji refs backward | arXiv API abstract | 中-高：**同模型 N 采样**设定（与 C7 同），hidden-state 正确性信号的线性 gate 做 dynamic selection，并给出该信号何时可信的 decodability 判据。是「同模型采样 + 选择端 gate」的先验，但无分歧表示、无 verification、无 repair |
| "A Layered Analysis of Disagreement And Answer Quality in Multi-Agent LLM Debate." arXiv:2609.08016, v1 2026-09-07 | DR 推荐邻域 | arXiv API abstract | 中：四层测量（报告一致/文本反驳/立场持续/logprob 立场）检验 MAD 中「分歧是否真实」——分歧质量诊断，非分歧利用机制 |
| "A Theory of Post-hoc Debate Judgement." arXiv:2608.19002, v1 2026-08-19 | DR 推荐邻域 | arXiv API abstract | 中：debate 裁决的形式化性质理论 —— recognition 端理论化 |
| "Evolution without an Oracle: Driving Effective Evolution with LLM Judges (MADE)." arXiv:2511.19489, v1 2025-11-23 | Ji refs backward | arXiv API abstract | 中：无 oracle 条件下把模糊指令分解为**可验证子需求**以稳定 judge 选择压力 —— 「无 deterministic checker 时提取选择信号」的 EC 域实例 |
| "Diagnosis Before Recovery: Turning Agent Failures into Selective Self-Correction." arXiv:2608.11772, v1 2026-08-12 | GuardedRepair 推荐邻域 | arXiv API abstract | 中：类型化失败信号触发**选择性**自纠（仅当失败约束下一次 repair）；代码 agent 域，触发器是编译器/测试等确定性信号 |
| "When Is Collective Intelligence a Lottery? Multi-Agent Scaling Laws for Memetic Drift in LLMs (QSG)." arXiv:2603.24676, v1 2026-03-25 | Ji refs backward | arXiv API abstract | 低-中：memetic drift 的机制模型（mutual in-context learning），Ji et al. 的概念前驱 |
| "Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?" arXiv:2508.17536, 2025-08-24 | Ji refs backward | S2 元数据 | 低：selection 规则比较 |

OpenAlex 六组关键词检索（限 2024-01-01 至 2026-09-17）：**"epistemic credit" 组合在 LLM/多智能体空间零命中**（仅教育/社会科学哲学文献）；其余组合无超出上表的新机制级命中。「epistemic credit from disagreement」这一表述与问题框架目前无人占据。

## 4. 任务 3：五维问题定义对照表

generation（候选如何产生）/ disagreement（分歧如何发现与表示）/ recognition（如何识别谁更可能对）/ selection（如何选或合成最终答案）/ preservation（是否保护 incumbent / 防 C→W）。

| Prior | generation | disagreement | recognition | selection | preservation |
|---|---|---|---|---|---|
| **Ji et al. 2608.25937** ⚠️新 | MAS 运行产生的候选池，固定后 replay（81,390 池/16,278 题/5 benchmark；异质生成源） | **答案级**：候选频率分布；无质量控制共识 = memetic drift | LLM judge 正确性信号；可靠性随任务/生成器/正确率稀有度变化（绘制 reliability map） | 终端选择规则消融：vote vs judge vs **frequency+judge 组合**（63.82%→70.82–70.95%） | **无机制**；仅在动机层提出「保护已生成正确答案不丢失」，干预只落在选择规则 |
| **AgentAuditor 2602.09341** ⚠️新 | 四种 MAS 框架的多 agent traces（异质） | **分歧点级（最接近 crux）**：Reasoning Tree 显式表示 agreement/divergence，定位 critical divergence points | 分歧点处 branch 级证据比较（局部化验证）+ ACPO 训练 adjudicator 抗多数误导 | adjudicator 聚合裁决（超 majority vote 与 LLM-as-judge，+5%） | 无 |
| **Minority Sentinel 2606.29270** ⚠️新 | 3 个异质 LLM 辩论 | 答案级：多数 vs 少数 divergent cases（约 1/4 中少数派正确） | meta-classifier 提取特征识别 Minority Truth | **条件性 overturn 多数票** | ~（overturn 本身由分类器门控——防止误推翻，但方向是推翻多数而非保护 incumbent；无 C→W 记账） |
| **CASE 2608.17124** ⚠️新 | **同模型 N 采样（与 C7 相同设定）** | 答案级投票分布（相关错误使多数票失效） | 隐藏态正确性信号（linear gate），附 decodability 可信判据 | 动态 selection combiner vs majority voting | 无 |
| **VRR-Stop 2607.17641** ⚠️新 | 单条 plan/trace（代码、数学、工具使用域） | 无分歧概念；verifier vs repairer 噪声建模 | **带噪 verifier**：四参数噪声模型分离误收/误拒，belief filtering | verify-repair 循环中的接受/继续/停止决策 | **~/✓（最接近）**：停止规则显式防止 repair 损坏已正确 plan；但是循环停止而非 KEEP 默认 + τ 门 + 净修复目标 |
| **GuardedRepair 2605.24613**（v2） | 单条 cached trace（异质 repair 模型） | 无（确定性符号诊断触发） | 确定性符号 guard（算术/约束/语义风险图） | 通过全部 gate 才替换 | ✓（keep-original 默认 + bounded-harm 约束，fixed/broken 双边记账） |
| **DISC 2606.21724**（v2） | 单解答（推荐跨模型角色分配） | 无（CoVe 式自我提问） | 独立验证问题 + binary judgment | verify-judge-correct 多轮 | ✓（judgment gate 阻止损害正确答案的 rewrite；I/D 双边诊断） |
| **DR 2607.01251**（v2） | 异质双 agent 立场（GPT-4o×Claude 等） | **crux 级**（Def 3.1，持立场 consultant 互抽） | weak judge 看全部 transcript | retain/adopt 二选一，无合成无修复 | ~（retain 选项存在但对称、无 τ） |
| **CoVe 2309.11495**（v2） | 单模型自我问答 | 无 | **blind 独立回答验证问题**（精确先验） | 无条件重写 | 无 |
| **PVD 2605.25133**（v2） | 单 prover 候选 + verifier 挑战 | 主张级（sub-claim 分解 + targeted challenges，单候选非分歧触发） | prover-verifier 对话（verifier 可见 claims） | ANC 选择性报告 | ~（selective prediction 弃权，非修改权门） |
| **Verify-Gated Completion 2605.17998** ⚠️新 | governed runtime 中专门化 agent 的完成声明 | 无 | 只读 verifier 核验 admission | admit/reject 完成声明 | ✓（fail-closed 默认 + 审计 trace；但对象是 runtime 完成声明，非答案修复） |
| **ConsistencyGate 2607.22962** ⚠️新 | 单 agent 多轮轨迹 | 无 | self-consistency 判定写入事实 | admit/block 记忆写入 | ✓（write-time admission 保护记忆库完整性；域为 memory，非答案 incumbent） |
| **Route, Don't Fix 2609.14825** ⚠️新 | 单模型解码（logit 信号） | 无 | regime 诊断（何时纠正有效） | **gated router 决定是否施加纠正** | ~（不纠正即保留原解码——「不改」为默认分支的雏形；临床 MCQA 域） |
| **ColMAD / Michael et al. / SAFE / ReConcile**（v2） | 各异（见 v2 表） | 无~答案级 | 互见辩论/cross-exam/检索验证 | vote/讨论收敛 | 无 |
| **MADE 2511.19489**（新） | EC 变异产生候选 | 无 | **无 oracle**：分解为可验证子需求稳定 judge 信号 | 演化选择压力 | 无 |
| **DRP（本方法）** | **同 checkpoint/prompt/decoding N 采样** | **crux 级，由同源群体分歧触发，中立 agent 抽取** | **blind 独立验证（不可见候选）** | 仅失败者 blind repair + certificate | **τ 门 KEEP 默认 + NetRepair 目标** |

## 5. 更新后的 Verdict

**维持 GO-WITH-REPOSITIONING，且 v3 雪球未触发 stop 条件**：仍无单篇实现「同模型 N 采样分歧 → 中立 crux → blind 验证 → 仅失败者 blind repair + 证书 → τ 门 → NetRepair」全链；五个 2026 年种子 cited-by 全为 0–1，无后续工作缝合这条链。

**但空间被进一步压缩，三处必须处理**：

1. **Ji et al.（最大新威胁，P0 确认）**：预占了「正确答案已在候选中却被丢失」「候选供给封顶 judge 收益」「judge 可靠性非常数」「generation/recognition/selection 三环节隔离」的诊断框架与大规模证据。我们第一篇论文的相关发现**必须改述为确认并扩展 Ji et al.**，增量限定在同源设定与机制归因。反过来说，Ji et al. 明确呼吁「设计保护已生成正确答案的架构」却未给出机制——DRP 正是候选答案，这反而**强化**了 DRP 的动机。
2. **AgentAuditor 预占「分歧点级局部验证」**：crux-from-disagreement 的最接近先验从 DR（对话 crux）变为 AgentAuditor（trace 树分歧点 + 局部证据核验）。差异仍可防守（答案级中立 crux 抽取、blindness、repair、gate 均无），但论文必须把它列为 C1 的第一最近邻，并说明 DRP 的分歧来自**同源采样群体**而非异质框架 trace 树。
3. **preservation 维的新先验群**：VRR-Stop（带噪 verify-repair 下防止 repair 腐蚀正确 plan）+ Minority Sentinel（条件性 overturn）+ Route-Don't-Fix（gated correction）+ Verify-Gated Completion（fail-closed admission）从四个方向包围 C5。它们无一使用「τ 证据阈值 + KEEP 默认 + 净修复目标」组合，但「修改权需要门控」这一思想在 2026 年 5–9 月已多点开花——**C5 单独作为新颖性声明的空间进一步收窄**。

**对「epistemic credit from disagreement without deterministic checker」定位的评估**：

- **该定位仍然成立且是目前最可防守的核心**。理由：(a) 占据该问题邻近空间的几篇工作（Ji et al.、AgentAuditor、Minority Sentinel、CASE、MADE）全部绕开了「验证者信息隔离」这一变量——它们要么给 judge/adjudicator 看全部信息，要么用隐藏态/频率等旁路信号；blindness 作为抗锚定机制（除 CoVe 单组件先验外）没有任何论文与 disagreement-driven crux 耦合；(b) 「epistemic credit」表述在 LLM/MAS 文献空间经 OpenAlex 关键词检索确认零占据；(c) VRR-Stop 证明了「带噪验证下的保护」有独立需求，但停留在停止规则——「从分歧结构中提取 credit 并据此分配修改权」的因果链无人实现。
- **可防守核心修订为**：「同源 N 采样分歧 → 中立 crux（区别于 AgentAuditor 的 trace 树分歧点与 DR 的对话 crux）× 端到端 blindness（区别于所有 judge/adjudicator 先验）× τ 门修改权 + NetRepair（区别于 GuardedRepair 的确定性 gate 与 VRR-Stop 的停止规则）」的三元耦合，包装在 Ji et al. 的 generation/recognition/selection 地图 + 我们新增的 disagreement/preservation 两维之下。
- **新增必引清单**：Ji et al. 2608.25937（动机与诊断框架，正面引用）、AgentAuditor 2602.09341（C1 最近邻）、Minority Sentinel 2606.29270（少数派拯救，selection 端替代路线）、VRR-Stop 2607.17641（preservation 最近邻）、CASE 2608.17124（同源采样 + 选择端 gate）、Verify-Gated Completion 2605.17998（fail-closed 门控先例）、MADE 2511.19489（无 oracle 选择压力）。

## 6. Caveats

- S2 Graph API 全程处于严格限流（429 频发）；CoVe cited-by 仅采样前 200 条（S2 分页 next=200 未继续；OpenAlex cited_by=43 显示索引远未收全），存在漏检尾部相关引用的风险。
- OpenAlex 对六篇 arXiv 种子无 referenced_works 数据，backward 全部依赖 S2 references 端点（DR 46 条、GuardedRepair 16 条、DISC 16 条、PVD 37 条、CoVe 54 条、Ji 62 条，均已全量拉取并关键词过滤）。
- 新增论文中 2607.13551、2508.17536、2603.24676、2609.08016、2608.19002 按 abstract/元数据评估，未做全文精读；相关度评级可能随精读调整。
- 「GuardedRepair 代码已开源（GitHub: guarded-repair）」为 v2 记录，本轮未复核。
