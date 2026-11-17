

## 决策确认记录（2026-11-16）

用户已逐项确认会话中的第 1–7 项均采用 A，并在第 8 项提出后授权：“后续都按照你推荐的方案”。因此，本任务剩余的未明确实现细节直接采用下列推荐方案，不再逐项请求确认。这里的会话编号与下文按主题合并的设计编号不同。

| 会话编号 | 已确定的方案 | 确认方式 |
|---|---|---|
| 1 | 一条记录关联的全部原始消息作为完整 fine 单元；替换超预算则保留 coarse | 用户逐项选择 A |
| 2 | 无效动作不重试；第二次无效时保留第一次已接受证据 | 用户逐项选择 A |
| 3 | 提示词引导 JSON，执行前严格校验；不做解码期间的语法 mask | 用户逐项选择 A |
| 4 | GRPO 采样及 old/current/reference 概率计算统一使用温度 0.7；SFT 使用温度 1 | 用户逐项选择 A |
| 5 | 组内总体标准差（分母 G）；零方差 advantage 为 0，保留 KL | 用户逐项选择 A |
| 6 | AdamW，weight_decay=0 | 用户逐项选择 A |
| 7 | 全局梯度 L2 范数裁剪阈值 1.0 | 用户逐项选择 A |
| 8 | 前 100 次更新线性预热，随后保持学习率 1e-6 | 按后续统一授权采用推荐 A |

其余细节采用下文的现有推荐实现：SFT 按 dev action-token CE 选模、patience=1；GRPO 每 50 update 验证，以准确率选模、相同时取较低记忆成本；使用固定提示序列化、完整元数据和稳定 ID 排序；构建采用保守合并；仓库标准化先使用确定性的子摘要去重拼接；缺少原始 split 时使用固定且可追溯的划分；adaptive gate 独立训练并冻结。严格粒度对照需锁定 hybrid/top-5，完整系统保留动作控制能力。依据是减少未报告的自由度、保持概率计算一致，并使数据和实验条件可追溯；具体代价与边界保留在下文。

本次核对确认这些选择已由现有代码实现；`configs/paper.json` 进一步显式记录 weight_decay=0 和 max_grad_norm=1.0，便于检查。用户对工程决策的认可不等于已验证论文结果，原始数据、示范和权重的缺失仍按文末说明处理。

| 论文协议 | 实现 | 对应文件 |
|---|---|---|
| M/D/L/V＋P1/P2，两层父/叶结构 | 数据类、唯一 ID、源链接和父子关系验证 | `mrmem/data.py` |
| >24h / cosine<.75 分段，episodic .80/top10 | 无 query/answer 的固定构建流程 | `mrmem/preparation.py` |
| 六字段动作、union/dedup、top-k 前置 | 严格 schema 与执行器 | `mrmem/environment.py` |
| 两阶段，细化替换粗摘要，保留其他证据 | 条件第二次调用、事务式预算检查 | `mrmem/environment.py` |
| Qwen3 embedding 前768维＋L2，BM25(1.2,.75)，RRF60 | 本地 embedding、稳定排序、融合 | `mrmem/models.py`, `ranking.py` |
| SFT lr1e-5/batch16/最多3 epoch，scope2 epoch | 只对 completion loss，owner 隔离 | `mrmem/training.py` |
| GRPO G8、64轨迹、4 epoch、minibatch8 | 逐轨迹计算并累积真实 minibatch 梯度 | `mrmem/training.py` |
| trajectory ratio、clip .2、KL(ref||policy) .01 | 精确全词表 KL；action logp 求和 | `models.py`, `training.py` |
| correctness-gated length bonus，B2048 | 空记忆无长度奖励，错误答案零奖励 | `mrmem/rewards.py` |
| LoCoMo train含category5、eval不含 | 数据属性及评测过滤 | `data.py`, `evaluation.py` |
| 500 记录保护/随机条件，增长500/1000/1500 | 受保护集合、显式 donor 白名单、操作 manifest | `mrmem/preparation.py` |
| 2×2、SFT/RL decomposition、粒度与组件消融 | 条件矩阵、独立配置与运行命令 | `configs/experiments.json` |
| 2027–2029，10k owner bootstrap，QA权重 | 严格配对、seed先平均、cluster重采样 | `mrmem/evaluation.py` |

## 未明确细节的具体设计

1. **完整单元预算。** 一条 coarse record 或一条 record 对应的全部 fine source messages 是一个完整单元。超预算就跳过；fine 替换超预算则保留该 coarse record。按排名顺序尝试，后面较小单元仍可进入。不同 leaf 共享 source 时保留各自记录中的源消息，因此可能重复计费；paper 只明确 record 去重，未明确跨记录消息去重。
2. **调用控制。** expand/stop 必须恰好一真；expand 仅允许第一次 coarse 调用。第二次只能选择第一次所选 scopes 的 descendant leaves，必须 fine 且 stop。无效第一次调用消耗一次并终止；无效第二次保留已接受 coarse。论文未完整指定无效动作后的恢复流程，本实现不自动修复、重采样或无限重试。
3. **JSON 生成。** 使用 JSON 提示＋严格 schema/可见 ID/预算验证，未加入 vocabulary-level grammar mask。论文的非零 invalid-action rate 不能证明其是否使用语法约束；这里采用用户已确认的提示＋校验方案，将实际错误动作保留在采样分布和成本中，不通过后处理隐藏。若后续加入受限解码，必须同步对 rollout/current/reference log-prob 与 KL 使用相同约束，不能只在 generate 端 mask。
4. **概率分布一致性。** RL 的 .7 温度应用于采样、old/current/reference 三者。top-p=1、top-k=0、无 repetition penalty。SFT CE 使用温度1。禁用 dropout；checkpointing 开启时仅保留 HF 所需 train mode。KL 是全词表 forward KL，包含动作所有生成 token（含 EOS），不混入 prompt/工具返回 token。
5. **组统计与优化。** 总体 std（correction=0），FP32 reward/advantage/log-softmax，零方差组保留 KL；AdamW weight_decay=0、grad_norm=1、100 update 线性 warmup 后恒定 LR。采样问题有放回，同一问题的 G 条轨迹共享状态。概率比数值溢出时显式报错，不用隐性 log-ratio clamp 改写算法。PyTorch Adam 参数状态采用模型实际 dtype；本实现没有单独 FP32 master-weight 分布式优化器。
6. **SFT 选模。** 按 dev action-token 平均 CE 选 best，patience1；训练按样本 shuffle，在 batch 内按有效 action token 数归一化。多步示范拆成多个动作训练样本；训练输入由重放构建，不加入 teacher 的 gold answer。multi-format 数据由调用者提供合法的各格式动作，不自动把长 parent set 展开后偷偷截断成 flat。
7. **提示序列化。** 策略使用稳定的 plain completion 格式，适用于 base checkpoint；答案模型在有模板时使用官方 chat template，关闭 thinking。策略 prompt 中字段、排序和 source 引文格式为本实现设计；论文未提供逐字模板。上下文超长显式报错，不自动删元数据。
8. **embedding/ranking。** Qwen3 最后有效 token pooling；不增加论文未指定的 query instruction prefix。BM25 文档是 leaf M/D/L/V JSON 和 parent summary；IDF/平均长度取该仓库对应层级的全语料，scope 仅过滤候选；相同分数按 ID 升序。内存缓存 embedding 以文本为键，避免重复前向。
9. **固定构建。** 语义边界取相邻消息向量；profile/domain 的候选要求抽取出的 M/D/L 精确一致，再由温度0模型判定等价。这是保守设计，避免错误合并实体；规范化标签质量影响召回。profile 当前值放在 value，历史状态保留在 summary 和不可变 source links。
10. **标准化后父摘要。** 重新按 M/D/L 分组，使用去重的 descendant summaries 拼接，保证未凭空创造事实。它不等同于论文未公布的摘要压缩提示，可能显著影响 metadata token 成本；正式实验可先在所有 resize 条件上用同一个固定摘要模型重建摘要并冻结文件。不得为某个 cell 单独压缩。
11. **划分与种子。** 十折 inner dev 采用循环 assignment，PersonaMem 使用固定随机 user partition，均需与获得的原始 manifest 对齐。一次训练使用同 seed 的 shuffling、采样和验证；验证保存/恢复 torch RNG，不扰动下一轮训练。不同条件共享 seed 但动作长度不同仍会消耗不同随机数，这只是 paired seeds，不保证逐 token common random numbers。
12. **自适应 gate。** 从训练标注 SFT 一个单独文字分类器，greedy 解码 coarse/fine；错误标签直接报错。不拿支持证据标签/答案在 test 时当 oracle，不固定 fine 比率。gate 模型路径和标注需提供，输入/输出计入 online 成本。
13. **粒度对照与全系统。** fixed_coarse/fixed_fine/parent_fine 固定 hybrid＋top5。全系统 two_stage 允许六字段控制中的 rank mode/top_k；如果做补充材料严格固定 hybrid/top5 的粒度对照，需要设置 `lock_executor=true`（配置说明以代码 Interface 为准）。

## 必须区分的结果来源

代码输出的任何 accuracy 都来自实际模型预测；不填充论文数值来替代实验。示例只使用两条虚构记录。论文报告的低 invalid-action rate、22.4% fine ratio、约8个 resolved leaves、65.8% token reduction 是观测结果，不能反过来作为硬约束写进训练器。

未提供数据时不能核验跨 benchmark 分布、原始示范数、original split user IDs 和逐折最优超参数。当前实现不承诺论文准确率、吞吐或 latency 数值的一致性。
