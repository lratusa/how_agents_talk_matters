# Novelty Gate Report v2 — DRP (Disagreement Decomposition and Blind Repair)

**Date:** 2026-09-17
**Literature cutoff:** 2026-09-17 (nothing after this date considered)
**Decision:** **GO-WITH-REPOSITIONING** — 无单篇完整实现全链；但「asymmetric revision gate + 双向 transition 目标」已被 GuardedRepair (arXiv:2605.24613) 与 DISC (arXiv:2606.21724) 实质预占，不能作为核心新颖性声明。可防守的核心在别处（见 §6）。

---

## 1. Method under evaluation (working name DRP)

设定：同一 LLM（同 checkpoint、同 prompt、同 decoding）独立采样 N 个答案。对存在答案分歧的题：

- **C1. 分歧触发的 crux 抽取**：中立 agent（不看对错立场）抽取「最小决定性命题」（decisive claim / crux）；
- **C2. Blind independent verification**：全新 agent 看不到任何候选答案，独立验证该命题（重查模型权重知识）；
- **C3. 只对失败答案修复（blind repair）**：修复 agent 看原题 + 命题验证结果，不看候选答案；
- **C4. Correction certificate**：每次修改附结构化证书（原答案、新答案、crux、验证结果、失败理由、通过理由）；
- **C5. Asymmetric revision gate / monotonic gate**：原答案有保留权，证据增量超过阈值 τ 才允许修改，否则 KEEP；
- **C6. NetRepair 目标**：P(W→C) − λP(C→W)、Beneficial Transition Ratio、双向漏斗（rescue 链 vs corruption 链），而非 accuracy；
- **C7. 同模型同源设定**：同一 checkpoint/prompt/decoding，采样是唯一多样性来源。

## 2. Search protocol

七条检索流，关键先验全部实际 fetch 验证（arXiv abstract 页 + 可得时 HTML 全文方法部分），禁止凭记忆：

1. DR 全文精读（arXiv:2607.01251 HTML 全文，含 Definition 1–3、Table 1 伪代码、实验设置）；
2. CoVe / PVD / ColMAD / ReConcile / SAFE / Michael et al. abstract 页验证；
3. "gated self-correction" / "revision gate" / keep-original threshold 检索 → 命中 DISC；
4. "correction certificate" / provenance 检索 → 无同义命中（仅有无关的 provenance 基础设施与专利文书概念）；
5. W→C / C→W transition metric 谱系检索 → 命中 GuardedRepair + 4 篇 transition-rate 论文；
6. 同模型多采样分歧 → crux → blind verify → gated repair 整链检索 → 无命中；
7. GuardedRepair HTML 全文精读（方法、触发策略、acceptance policy、fixed/broken 评估）。

## 3. Verified citations（全部实际 fetch 确认存在）

| 简称 | 准确引用 | 验证方式 |
|---|---|---|
| **DR** | Chacha Chen*, Teng Wu*, Liwen Sun, Han Liu, Shi Feng, Chenhao Tan. "Collaborative Disagreement Resolution for Scalable Oversight." ICML 2026. arXiv:2607.01251 [cs.CY], v1 2026-06-02. （注：任务书称 "Jiang et al."，实际作者为 Chen/Wu 等；Yuyang Jiang 仅为 arXiv 提交人。） | abstract + HTML 全文 |
| **CoVe** | Shehzaad Dhuliawala et al. (Meta AI). "Chain-of-Verification Reduces Hallucination in Large Language Models." arXiv:2309.11495, v1 2023-09-20. | abstract 页 |
| **PVD** | João Sedoc et al. "Prover-Verifier Deliberation for Selective LLM Prediction." arXiv:2605.25133, v1 2026-05-24. | abstract 页 |
| **ColMAD** | Yongqiang Chen, Gian Niu, Jiaheng Cheng, Bo Han, Masashi Sugiyama. "Towards Scalable Oversight with Collaborative Multi-Agent Debate in Error Detection." arXiv:2510.20963, v1 2025-10-23, v2 2026-07-14. | abstract 页 |
| **Michael et al.** | Julian Michael, Salim Mahdi, David Rein, Jackson Petty, Julien Dirani, Vishakh Padmakumar, Samuel R. Bowman. "Debate Helps Supervise Unreliable Experts." arXiv:2311.08702, v1 2023-11-15. | abstract 页 |
| **SAFE** | Jerry Wei et al. (Google DeepMind). "Long-form factuality in large language models" (LongFact + SAFE). arXiv:2403.18802, v1 2024-03-27; NeurIPS 2024. | abstract 页 |
| **ReConcile** | Justin Chih-Yao Chen, Swarnadeep Saha, Mohit Bansal. "ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs." arXiv:2309.13007, v1 2023-09-22; ACL 2024. | abstract 页 |
| **GuardedRepair**（新发现，任务书未列） | "Guarded Repair for Harm-Aware Post-hoc Replacement of LLM Mathematical Reasoning." arXiv:2605.24613, v1 2026-05-23. 代码已开源（GitHub: guarded-repair）。 | HTML 全文精读 |
| **DISC**（新发现，任务书未列） | Shen Yin et al. "Denoising Iterative Self-Correction: Structured Verification Loops for Reliable LLM Reasoning." arXiv:2606.21724, v1 2026-06-19. | abstract 页 |
| Transition-metric 谱系 | "LLMs Can't Handle Peer Pressure" (arXiv:2508.18321, 2025-08)；"The Cost of Consensus" (arXiv:2605.00914, 2026-04)；"From Process Loss to Assembly Bonus" (arXiv:2609.13261, 2026-09-06，截止前)；S²R (OpenReview, ICR/CIR 指标) | snippet 级 |

## 4. Component comparison matrix

列 = DRP 组件 C1–C7（见 §1）。✓ = 如规格实现；~ = 部分/不同实现；✗ = 缺失。

| Prior | C1 分歧触发crux | C2 blind verify | C3 仅修复失败+blind repair | C4 certificate | C5 asymmetric gate | C6 NetRepair目标 | C7 同模型同源 |
|---|---|---|---|---|---|---|---|
| **DR** (2607.01251) | ~ （crux 是核心机制 Def 3.1，但由持立场的两名 consultant 在多轮对话中从对方论证抽取；异质双agent，非中立agent对N个同模型样本） | ✗ （consultant 互看论证；judge 看全部 transcript） | ✗ （action 仅 retain/adopt 两个既有候选，无 repair） | ✗ （transcript 留痕但非结构化修改证书） | ~ （retain/adopt 中 retain 是保留选项，但 adopt 对称、无证据阈值 τ 保护 incumbent） | ~ （定义了 C→C/W→C/C→W/W→W 四分类 Persistence/Recovery/Overthinking/Stubbornness 作诊断+校准分析，但目标是 judge accuracy，无 λ 净修复目标） | ✗ （GPT-4o×Claude Sonnet 4；GLM-4.6×Kimi K2，均为异质对；judge 更弱） |
| **CoVe** (2309.11495) | ✗ （verification questions 由自身 draft 规划，非跨样本分歧触发） | **✓** （独立回答验证问题，"so the answers are not biased by other responses" —— blind verification 的精确先验） | ✗ （无条件修正自己的回答，非选择性只修失败者） | ✗ | ✗ （无条件重新生成 final response，无 KEEP 门） | ✗ | ✓ （单模型自我 deliberation） |
| **PVD** (2605.25133) | ~ （checkable sub-claims 分解 + targeted challenges 定位争点，但针对单一候选答案、非分歧触发） | ✗ （verifier 必须看到 prover 的 claims 才能 challenge） | ~ （Accept/Challenge/Reject 对话内修订；非 blind repair） | ~ （structured confidence verdict 类似裁决凭证，但非逐次修改的 provenance 证书） | ~ （Accept+NoChange (ANC) 子集有"默认不变"色彩，但那是 selective prediction 的选择机制，非保护 incumbent 的 τ 门） | ✗ （coverage-precision，非净修复） | ✗ （Sonnet 4.6 prover × Haiku 4.5 verifier；跨族 GPT/Gemini） |
| **ColMAD** (2510.20963) | ✗ （协作式辩论纠错检测；无 crux 隔离） | ✗ （agent 互见消息） | ✗ | ✗ | ✗ | ✗ （error detection accuracy） | ~ （MAD 通常为同模型多实例，但机制不依赖同源性） |
| **Michael et al.** (2311.08702) | ~ （cross-examination 让 debater 选择性揭示关键 quote 给 judge —— 对抗式、面向 judge 的"关键证据抽取"，非中立 crux） | ✗ | ✗ | ✗ | ✗ | ✗ （judge accuracy 84% vs consultancy 74%） | ✗ （human/AI debaters + judge，信息不对称设定） |
| **SAFE** (2403.18802) | ~ （atomic fact decomposition 是"分解为可验证命题"的先验，但针对单条长回答、非分歧触发的最小 crux） | ✗ （search-augmented 外部检索验证，非 blind 重查权重知识） | ✗ | ✗ | ✗ | ✗ （F1@K） | ✗ |
| **ReConcile** (2309.13007) | ✗ （多轮圆桌讨论 + confidence-weighted voting，无 crux） | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ （多模型多样性是关键贡献） |
| **GuardedRepair** (2605.24613) ⚠️ | ✗ （trigger 是单条 cached trace 上的确定性符号诊断：算术检查/约束覆盖/语义风险图；无多样本分歧、无 crux） | ✗ （guard 是确定性符号检查而非 LLM blind verification；但 repair attempt 2 "把初始 trace 只当 warning signal" 有 blind repair 雏形） | **✓** （仅 triggered 的可疑 trace 进入修复；默认动作就是保留原 trace） | ✗ （JSON 候选 + 日志，无"为何失败/为何通过"的结构化修改证书） | **✓** （形式化为 max F(π) s.t. B(π)≤ε；仅当候选通过全部 deterministic gates 才替换，否则 KEEP 原 trace —— 这就是 asymmetric revision gate 的精确先验） | **✓/~** （fixed/broken 双边记账、HarmRate、accepted-repair precision、fix–harm frontier —— 功能上等价于 NetRepair 评估精神，但无 λ 参数化目标、无 BTR 命名） | ✗ （deepseek-v4-pro 修 deepseek-v4-flash；单条 trace 非 N 样本） |
| **DISC** (2606.21724) ⚠️ | ✗ （CoVe 血统：对自身解答出 verification questions，当作"噪声测量"） | ~ （CoVe 式独立验证问题） | ✗ （每条解答都过 verify-judge-correct 多轮） | ✗ | **✓** （binary judgment gate 显式阻止会损害已正确答案的 rewrite） | ~ （improvement-to-degradation ratio (precision) + repair rate (recall) 配对诊断 —— 双边 tradeoff，但 precision/recall 框架而非净收益目标） | ✗ （明确推荐跨模型角色分配以缓解自我确认偏差 —— 与 C7 相反） |
| Transition-metric 谱系 (2508.18321; 2605.00914; 2609.13261; S²R) | ✗ | ✗ | ✗ | ✗ | ✗ | ~ （W→C/C→W flip rate、ICR/CIR 已成 MAD/self-correction 文献标准诊断量；但均非设计目标函数） | ~ |

## 5. Closest prior art（论文中必须显式区分）

按危险度排序：

1. **GuardedRepair (arXiv:2605.24613)** —— 对 C3+C5+C6 构成**近乎精确的组件级先验**：harm-aware selective replacement、keep-original 默认、guarded acceptance、fixed/broken 双边记账、形式化 max-fix-s.t.-bounded-harm。差异：确定性符号 guard 而非 LLM blind crux verification；单条 cached trace 而非同模型 N 样本分歧触发；无 crux、无 certificate；异质 repair 模型。**DRP 不可把 gate 或净修复评估作为核心新颖性声明。**
2. **DR (arXiv:2607.01251, ICML 2026)** —— 对 C1 的 crux 概念与"以分歧为触发条件"构成精确概念先验，且其四分类 action taxonomy（含 C→W "Overthinking"、W→C "Recovery"）是 DRP transition 漏斗的直接先验。差异：异质双 agent 多轮对话、互相可见、retain/adopt 无修复、目标是 weak-judge oversight 而非同源采样群体的净修复。
3. **DISC (arXiv:2606.21724)** —— C5 的 judgment gate（保护正确答案不被改写）+ 双边 precision/recall 诊断。差异：单解答 CoVe 血统、无分歧触发、无 certificate，且明确走向跨模型（反 C7）。
4. **CoVe (arXiv:2309.11495)** —— C2 blind independent answering 的精确先验。差异：自我核查、无分歧触发、无门控。
5. **PVD (arXiv:2605.25133)** —— sub-claim 可检验分解 + ANC 选择性机制。差异：verifier 可见 claims、coverage-precision 目标、跨模型。
6. **Michael et al. (arXiv:2311.08702)** —— cross-examination 的"关键证据选择性揭示"。对抗式、无 blind、无 gate。
7. Transition-metric 谱系（2508.18321 / 2605.00914 / 2609.13261 / S²R）—— W→C/C→W flip rate 已是标准诊断；NetRepair 作为**指标**增量小，作为**设计目标 + λ 权衡 + 双向因果漏斗**仍有框架性贡献空间，但需引用此谱系并明确定位。

**区分「精确先验」与「仅相关」**：精确先验 = CoVe（C2）、GuardedRepair（C3/C5/C6 的确定性单 trace 版）、DISC（C5 门）、DR（C1 crux + 分歧触发 + transition taxonomy）。仅相关 = PVD、ColMAD、Michael et al.、SAFE、ReConcile、metric 谱系。

## 6. Stop-condition assessment 与决策问题

**Stop 条件**（单篇完整实现：同模型 N 采样分歧 → 中立 crux 抽取 → blind 独立验证 → 仅失败者 blind repair + 证书 → τ 门控 KEEP 默认 → NetRepair 目标）：**未触发。** 没有任何单篇同时满足 C1+C2+C5；GuardedRepair 缺 C1/C2/C4/C7；DR 缺 C2/C3/C4/C5/C7；DISC 缺 C1/C4/C7。

**决策问题 1：「asymmetric revision gate + NetRepair 目标」能否承担「不可被 DR+CoVe+PVD 直接组合覆盖」的角色？**

- 对 DR+CoVe+PVD 的直接组合而言：是，不可直接覆盖 —— 三者均无 KEEP 默认 + 证据阈值 τ 的 incumbent 保护机制（PVD 的 ANC 是选择性报告的筛选，不是修改权控制），也无净修复目标函数。
- **但此角色在论文层面守不住**：GuardedRepair（2026-05）与 DISC（2026-06）已在 DRP 之前公开发表了该组合的本质（harm-aware gate + 双边 transition 评估）。把核心新颖性押在 C5+C6 上会撞上直接先验，审稿人一票即可击穿。

**决策问题 2：「crux 来自同模型多样本分歧 + blind repair + 门控修改权」整条链是否被任何单篇完整实现？**

- 否。没有任何论文把 crux 定位建立在**同源 N 样本群体的分歧结构**上（DR 是异质双 agent 对话、CoVe 是自我提问、SAFE 是单回答分解），也没有论文把 blindness 作为贯穿 verify 与 repair 两阶段的**信息隔离机制**（CoVe 只在 answering 阶段 blind；GuardedRepair 的 attempt 2 只是 prompt 技巧）。

## 7. Verdict

**GO-WITH-REPOSITIONING。**

组合可覆盖大部分，但存在可区分点。建议的定位改写：

1. **核心机制声明改为**：「在同源（同 checkpoint/prompt/decoding）N 采样群体中，以答案分歧为触发、由中立 agent 抽取最小决定性命题（crux），并以全程信息隔离（verifier 与 repairer 均不可见候选答案）的 blind verification + blind repair 执行修复，修改权受不对称门控约束」。不可被 DR+CoVe+PVD 直接组合覆盖的部分是：**分歧驱动的 crux 定位（C1 的同源群体版本）× 端到端信息隔离（C2+C3 的 blindness 作为抗锚定/抗相关性机制）× 门控修改权（C5）三者的耦合** —— 单看每个零件都有先验，三者耦合的因果结构（分歧告诉你在哪查、blindness 保证证据独立、gate 保证 incumbent 优先）无先验实现。
2. **明确不作核心声明**：asymmetric gate（cite GuardedRepair、DISC 为最近邻并区分确定性 guard vs LLM blind 验证）、blind verification 单独组件（cite CoVe）、crux 概念本身（cite DR 及 Double Crux 血统）、W→C/C→W 记账（cite DR 的 action taxonomy 与 transition-metric 谱系）。
3. **C4 correction certificate 是安全的次级贡献**：检索未发现 LLM 纠错文献中的同义机制（仅有无关的 provenance 基础设施与跨领域的 "Research Question Certificate" 等用法），可作为 auditability 贡献声明，但不宜单独承担核心新颖性。
4. **NetRepair 的 λ 参数化与双向因果漏斗**（rescue 链 vs corruption 链作为机制分析而非仅终点指标）可作为评估框架贡献，但必须引用 GuardedRepair 的 fixed/broken、DISC 的 I/D-ratio、DR 的四分类并说明增量（从记账/诊断提升为带权重的设计目标 + 机制归因）。

**风险提示**：GuardedRepair 与 DISC 均为 2026 年 5–6 月的预印本，索引较新；建议正式开工前对二者做引用雪球（cited-by）检查，并在论文 related work 中给予显眼位置。若审稿语境是 MAD 社区，DR（ICML 2026）是必须正面对话的工作。

## 8. Caveats

- ColMAD 与 transition-metric 谱系中的部分论文仅按 abstract/snippet 评估（方法细节未全文精读）；不影响结论，因为它们的缺失组件在 abstract 层面即可判定。
- DR 论文任务书误记为 "Jiang et al."，实际作者为 Chen*, Wu* 等（Yuyang Jiang 仅为 arXiv 提交人）；引用时须用正确作者。
- 2026 年 9 月索引可能存在滞后；但七条独立检索流 + 最近邻全文精读使漏掉精确重复的概率较低。
