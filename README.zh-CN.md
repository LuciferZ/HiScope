
## 1. 先运行离线验证

在 `code/` 下执行：

```bash
python3 scripts/demo.py
python3 -m unittest discover -s tests -v
python3 -m mrmem --help
```

示例复现补充材料 S2 的两步轨迹：先取偏好和短途步行的摘要，再仅细化步行限制，保留偏好摘要。示例使用空白分词计数，仅验证执行语义。真实实验使用**答案模型 tokenizer**计算终端记忆长度，使用**策略 tokenizer**计算动作长度。

测试包含真实 PyTorch 参数更新，以及随机初始化的微型 GPT-2 的动作概率、KL 方向和梯度检查；不下载模型、不访问评判 API。它们不是论文数据集上的性能实验。

## 2. 环境与模型

复用项目已有依赖，无新增依赖或自动安装步骤。核心需要 Python ≥3.10、PyTorch、Transformers；统计使用 NumPy。示例执行器可只用标准库运行。

`configs/paper.json` 给出论文超参数和模型角色。将其中 `models.*.path` 替换为本地权重路径；默认 `local_only=true`，不会自动下载。策略默认以 **Qwen3-8B-Base** 开始，答案模型为 Qwen3-8B，embedding 为 Qwen3-Embedding-8B。

当前训练器是单进程、全参数训练参考实现，通过逐轨迹累积形成有效 minibatch；策略、参考、答案、embedding 可以放在不同 GPU。8B 全参数训练需要为权重、梯度、Adam 状态及激活分配足够显存；配置里的四个 device 是角色示例，并不保证每张 48GB 卡能容纳。先用本地 0.6B 或 4B 模型验证数据与流程，再按硬件容量运行。不要将此实现的吞吐/延迟作为论文服务器的复现。

评判模型通过 OpenAI-compatible `/chat/completions` 接口调用，无需新增 SDK。配置 `judge.base_url` 为自己的评判服务，模型使用论文的 GPT-4 Turbo；凭证从 `MRMEM_JUDGE_API_KEY` 环境变量读取，不写入代码、配置或日志。

## 3. 数据契约

推荐布局：

```text
data/
  questions.jsonl
  repositories/<owner_id>.json
  splits/locomo_fixed.json
  teacher_traces.jsonl
  demonstrations.jsonl
models/
checkpoints/
runs/
```

`owner_id` 对 LoCoMo 表示 conversation，对 PersonaMem 表示 user。每个仓库使用 [examples/repository.json](examples/repository.json) 的结构：

- `sources`：不可变原始消息，含 ID、ISO 8601 时间戳及正文。
- `records`：`id, category, topic, attribute, value, summary, source_ids`；category 为 `episodic/profile/domain`。
- `parents`：唯一 ID、摘要、子记录 ID；每个 `(category,topic,attribute)` 对应一个父节点，每个叶子恰好有一个父节点。
- `owner_id`：必须与问题及仓库文件名一致。

每行问题采用 [examples/questions.jsonl](examples/questions.jsonl) 格式，包含 `id, owner_id, dataset, query, answer, category, objective`，可选 `list_answer, support_ids`。`objective` 是**预先固定的训练正确性判定掩码**，必须显式给出，不能根据模型是否答对动态决定。`support_ids` 为抽取后叶记录 ID，不是原始消息 ID。PersonaMem 将选项写入 query，answer 保存标准选项标签。原始 benchmark 字段和抽取后的支持证据映射需要在数据准备时转换到此契约。

训练时：objective=true 使用规范化 EM，否则调用二元 judge。评测时：LoCoMo 全部使用 judge，PersonaMem 使用选项 EM。LoCoMo category 5 保留在训练数据中，仅在评测时排除。

生成划分：

```bash
python3 -m mrmem splits --output data/splits/locomo_fixed.json
python3 -m mrmem splits --fold 0 --output data/splits/locomo_fold0.json
python3 -m mrmem splits --dataset personamem --owners data/user_ids.json --output data/splits/personamem.json
```

LoCoMo 固定划分与补充材料一致。十折测试 owner 依次为文中的十个 conversation，内层验证 owner 采用循环下一个，剩余八个训练。论文没有给出逐折验证 owner，该选择属于实现决策。PersonaMem 原始 user ID 划分未提供，默认按 seed 2027 随机分成 126/18/36；获得原始划分后应替换 manifest。

## 4. 构建与标准化仓库

已有经过核验的抽取仓库时可直接加载。由原始消息构建时，创建配置：

```json
{
  "sources": "data/raw/conv-26.json",
  "extractor": {"path": "models/Qwen3-8B", "device": "cuda:0", "dtype": "bfloat16"},
  "embedding": {"path": "models/Qwen3-Embedding-8B", "device": "cuda:1", "dtype": "bfloat16"},
  "max_tokens": 4096,
  "output": "data/repositories/conv-26.json"
}
```

原始消息文件为 `{"owner_id":"conv-26","sources":[...]}`。执行 `python3 -m mrmem construct --config configs/construction.json`。构建仅访问历史消息，不访问问题和答案。时间/语义切分、episodic 候选阈值和候选数遵循论文；模型输出不合法时直接报错，避免静默写入错误源链接。

500 记录标准化、1,000/1,500 记录增长共用 `resize`：

```json
{
  "repository": "data/natural/conv-26.json",
  "size": 500,
  "seed": 2027,
  "protected_ids": ["r000001", "r000008"],
  "donors": [],
  "allowed_donor_owners": [],
  "random_subsample": false,
  "output": "data/repositories/conv-26.json"
}
```

执行 `python3 -m mrmem resize --config configs/resize.json`。`protected_ids` 应包含该 owner 所有标注问题的支持叶记录，**包括其测试标注**，这是论文证据保持的 factorial 构建协议，不能据此声称这是无标注部署流程。对自然仓库的系统实验无需这一标准化步骤。

补齐时必须显式列出合法 donor owner：十折 test 从该折 train+dev 补齐；固定 test 两个对话仅互相补齐；PersonaMem 不跨用户补齐。增长实验从 PersonaMem **训练用户**抽样。工具验证 donor 是否在给定白名单，白名单与 split 的一致性由实验配置负责。导入时命名空间化 source/record ID，合并相同 `(M,D,L)` 的父节点。操作输出伴随 `.manifest.json`，记录增删 ID、种子和仓库哈希。

证据敏感性实验设置 `random_subsample=true`，种子为 `training_seed + fold_index`；在训练/评测配置中显式设置 `allow_missing_support=true`。必须保留被删除支持证据对应的问题，不能筛掉这些问题。复用保护条件 SFT，但重新训练 FL/HL。

## 5. SFT → GRPO

人审后的 teacher trace 格式见 [examples/teacher_traces.jsonl](examples/teacher_traces.jsonl)。它只保存 owner、question ID 和动作列表，不保存答案到策略输入。`compile-demos` 用实际执行器重放轨迹，重新生成每一步只含 query/metadata/history/evidence 的策略输入：

```bash
python3 -m mrmem compile-demos --config configs/paper.json --traces data/teacher_traces.jsonl --output data/demonstrations.jsonl
python3 -m mrmem sft --config configs/sft_run.json
python3 -m mrmem grpo --config configs/grpo_run.json
```

两个运行配置均可从 `paper.json` 复制并修改：

1. SFT：policy 指向 base 权重，`output` 取新的目录，填好 `split/demonstrations`。全系统最多 3 epoch；`interface.protocol=scope` 最多 2 epoch。每 epoch 验证 loss，保存 `best/last`，一次未改善提前停止。训练只取 train owners，验证只取 dev owners。
2. GRPO：policy 路径改为对应折和 seed 的 `SFT输出/best`，`output` 换为新的目录。冻结参考自动从**同一个起始 checkpoint**加载。1000 update、8 组×8 轨迹、4 epoch/batch、minibatch 8、100 warmup、学习率 1e-6、clip .2、KL .01、长度权重 .7。
3. 从 base 训练：跳过 SFT，policy 指向 base；冻结参考也自动变为 base。
4. no-length：设置 `training.length_weight=0` 后独立训练。

Scope-only 示范 completion 为 `{"subspace_ids":[...]}`。共享 multi-format SFT 的 teacher traces 需要覆盖 flat、hierarchical、leaf_only、opaque 等所有使用的格式；每条 trace 可通过 `interface` 覆盖默认设置。先将各格式重放成一个 demonstrations.jsonl，然后为同一折、seed 训练一个共享 scope SFT，再分别启动 FL/HL 的 GRPO。**不要分别训练 FL-SFT/HL-SFT 后把它们称为共享初始化。**

`compile-demos` 验证格式和可执行性；语义支持对齐、人审标记和示范覆盖率需由真实数据准备流程核验。本仓库不将合成轨迹冒充论文的 GPT-4 Turbo＋人审示范。

SFT 使用动作 token 的交叉熵，prompt token 不计 loss。GRPO 在两次动作的所有生成 token（含实际生成的 EOS）上求 log-prob 总和，先求**轨迹概率比**再裁剪；旧策略 log-prob 在更新前保存。KL 是全词表 `KL(ref || policy)` 的动作 token 求和，非抽样近似。奖励标准差使用总体标准差，零方差组 advantage 为 0，但 KL 仍有效。

## 6. 批量实验与评测

```bash
python3 scripts/make_experiments.py --folds tenfold --conditions FC HC FL HL FL_SFT HL_SFT --output experiments/coarse
python3 scripts/make_experiments.py --folds tenfold --granularity fine --conditions FC HC FL HL --output experiments/fine
python3 scripts/make_experiments.py --folds tenfold --conditions hierarchical_leaf_only opaque_masked --output experiments/components
python3 scripts/make_experiments.py --conditions MRMem flat_two_stage fixed_coarse fixed_fine parent_fine adaptive no_length_reward from_base --output experiments/system
```

输出 `tasks.json` 包含真实命令数组及各运行配置。生成器**只写配置**，不自动启动 GPU 训练或评判请求；运行其中命令即可执行。默认寻找 `checkpoints/fold_N/seed_N/scope_sft/best` 或 `full_sft/best`，固定划分用 `fixed/`。各折/seed 的 SFT 需要先按第 5 节训练好。

矩阵生成器专用于 LoCoMo；PersonaMem 使用生成的 user split，在独立配置中指定即可。factorial 使用标准化仓库；自然仓库系统结果、标准化 factorial 结果和 fine granularity 条件应写入不同运行目录，不能混合分析。

`--grid` 生成论文给出的验证网格。FL/HL 每折独立选择，不能使用 test 分数选择。HC 的 `k_topic` 从 1..5 的 dev accuracy 选择；主 RRF comparator 取最高 accuracy，可用 `select --tolerance 0`；长度权重候选允许距最高 accuracy 0.01 内优先最低记忆成本（默认 tolerance .01）。生成器不会替你依据测试结果做选择。若单独跑 `FC/HC` 而不做网格，`k_topic=3` 是默认值，并非声称所有折最优。

GRPO 每 50 update 验证一次（实现选择，可改）；`best` 按 validation accuracy 选，完全相等时按记忆成本。`last` 为最终 update。跨 hyperparameter 的容忍度选模和单次训练内部选 checkpoint 是两个步骤。

评测配置的 policy 改为所选 checkpoint，给定独立 output，然后：

```bash
python3 -m mrmem evaluate --config configs/eval_run.json --partition test
python3 -m mrmem evaluate --config configs/dev_run.json --partition dev
python3 -m mrmem select --candidates data/validation_candidates.json --output runs/selected.json
```

`method=rrf` 是主 conventional comparator；FC 全叶 RRF，HC 父 RRF→descendant RRF，固定 top-5。`zero_shot=true` 配合 base checkpoint 使用温度 0；learned 和 SFT-only 评测保持 .7。固定细粒度显示父摘要和所有叶元数据；parent_fine 只显示父摘要；两阶段初始只显示父摘要，第二步显示选定父范围的所有子叶摘要/元数据。

自适应粒度需要额外训练好的 `models.granularity_gate`。该模型输入 query＋parent metadata，输出严格的 `coarse` 或 `fine`，可通过通用 SFT 入口训练二分类文字标签。标注只能来自 train/dev owners。实际推理先调用冻结 gate，再按决定显示父元数据或父＋全叶元数据，gate 的输入/输出计入 online tokens。测试集 fine 比例由模型实际输出决定，不写死为论文的 22.4%。

输出包括 `predictions.jsonl`、`metrics.json`、`provenance.json`。每个问题记录 owner、seed、condition、prediction、correctness、证据 ID、终端记忆 token、累计 policy/answer token、调用数、无效动作和延迟。延迟包含本地实现的完整逐题处理，不能直接对照论文的专门延迟测量。

## 7. 配对统计

```bash
python3 -m mrmem analyze --predictions runs/FC/predictions.jsonl runs/HC/predictions.jsonl runs/FL/predictions.jsonl runs/HL/predictions.jsonl --output runs/factorial.json
python3 -m mrmem analyze --predictions runs/system/predictions.jsonl runs/fine/predictions.jsonl --weights '{"MRMem":1,"fixed_fine":-1}' --output runs/disclosure_gap.json
python3 -m mrmem analyze --predictions runs/system/predictions.jsonl runs/fine/predictions.jsonl --weights '{"MRMem":1,"fixed_fine":-1}' --metric memory_tokens --output runs/token_gap.json
```

按实际路径传入全部折、三个 seed 的 per-question 文件。分析器要求各 cell 的 owner/question/seed 覆盖完全一致，拒绝静默取交集。确定性 comparator 也使用三个对应 seed 标签运行/记录，以便严格配对。

先对每个问题求配对三 seed 平均，再按 owner 聚类，以 QA 数加权做 10,000 次配对 bootstrap。两 owner 固定 split 只给点估计，不输出貌似可靠的 CI。`FL_SFT/HL_SFT` 同时提供时自动计算 SFT interaction、RL increment 和 RL share 点估计；不对该比值杜撰置信区间。sign test 标记为 nominal，不能消除十折共享训练历史的依赖。

