# 执行总结（Executive Conclusion）

项目：Farthest-Pair Reciprocal Review（FPRR/RGFM）——同模型多实例按回答语义距离最远配对互审能否提升推理准确率。
日期：2026-09-18。主模型 deepseek-flash（thinking disabled，T=0.8）；嵌入 bge-m3。全部结果来自真实 API 运行（约 14 万次调用），无任何捏造数据。

## 最终分类：Result D（无意义效应）+ 一条探索性线索

**主结论**：在受控条件下（所有配对条件共享同一组缓存初始回答、同一合成器、同一预算），FPRR 不优于随机配对、最近配对、最大权匹配、多数投票或算力匹配自反思。MMLU-Pro（500 题 × 3 seeds）C5−C3=−0.5pp，SuperGPQA（500 题 × 4 seeds）C5−C3=−0.7pp；四个主比较的 Holm 校正 p 值全部 ≥0.92，配对 bootstrap CI 全跨 0。H1/H2/H3 均不成立。更尖锐的是：在较难的 SuperGPQA 上，**不做互审、直接把候选答案交给合成器（C2）反而显著优于任何互审条件**（C2−C5=+1.5pp，CI [−2.6,−0.4]，p=0.010）——互审在该基准上是净成本。

**机制发现（论文真正的贡献）**：

1. **Exposure ≠ Resolution**。语义距离确实单调提升错误检出率（最低十分位 0.00–0.11 → 最高 0.30–0.44，logistic β=+12.2/+7.5，p<0.0001），但在 MMLU-Pro 上该效应完全不传导到修复与采纳级（p=0.20/0.84）；在 SuperGPQA 上能传导（三级全显著），但反向漏斗同步放大。
2. **反向漏斗**：false opposition 与正确答案 destabilization 随距离上升；互审是「active but net-zero」（MMLU 27 救 23 毁，BTR=0.54；SuperGPQA 49 救 79 毁，BTR=0.38）。
3. **新知识确实存在但送不达**：互审在 0–4% 的全员错误题上生成了初始集中不存在的正确答案（证伪「信息全在 R_x 中」的强论断），但端到端送达率 ≤1.5%，多数提案死在采纳前。
4. **失败模式归因被数据修正**：「最远=最差」的 outlier amplification 假说被否决（max-distance 边端点错误率≈基准率）；真正的机制是**语义差异让审阅者把「不同的正确」误判为「错误」**。
5. **瓶颈定位**：不是拓扑，而是 epistemic credit assignment——系统无法区分 corrective disagreement 与 false opposition。稀释论反驳被正面排除（recoverable 子集上两基准方向一致为负）。

**探索性线索**：距离消融（200 题单 seed，共享初始回答）显示，在 SuperGPQA 上用 NLI 矛盾概率替代语义余弦距离做最远配对，C5 比多数投票高 5.0pp（CI 不跨 0），配对比较 D4−D2=+3.0pp（CI [+0.5,+6.0]，7:1 discordant，p=0.077）。即「认识对立」比「文本差异」更接近有用信号。该结果不在预注册主族内，脆弱且需复制，不改变 Result D 主分类。

## 何时有效、何时失败

有效场景（本研究外推，需谨慎）：高多样性 regime、存在廉价验证的任务、分歧感知（而非语义）路由。失败场景：低多样性（多数题全员同答案）、同源权重导致的共享误解（8–30% 的题不可恢复）、以及任何把「发现错误」与「修正错误」混为一谈的设计。

## 是否值得发表

诚实判断：作为主会论文偏弱（单模型、null 结果、部分消融未跑）；作为 workshop 或 negative-result + 机制分析的论文**有真实价值**——受控设计（同缓存初始回答）、双向漏斗、三级回归、delivery 核验都是可复用的分析框架，且直接催生了后续方向。后续线（DRP / Epistemic Credit Assignment under Disagreement）已通过三轮新颖性审查（GO-WITH-REPOSITIONING），核心定位为「无 deterministic checker 条件下从分歧中提取可靠信用」，最小因果实验设计已冻结（docs/credit_assignment_design.md），尚未投入付费实验。

## 完整性声明

未跑项（预算耗尽，余额 ¥0.43）：N∈{2,6,8} 消融、配对种子方差、MMLU-Pro D4——设计已冻结于 preregistration Amendment 8，论文中如实标注 not run。SuperGPQA 为 4 seeds，MMLU-Pro 为 3 seeds；跨 seed 共享题目使 CI 轻微反保守（已在论文中声明）。环境无 LaTeX 工具链，论文交付为 main.tex + pandoc 渲染的 main.html，未生成 PDF。
