# 代码目录

```text
code/
├── README.md                  中文运行说明与数据契约
├── IMPLEMENTATION_NOTES.md    论文映射及未明确细节的实现决策
├── VALIDATION.md              本地测试结果与环境记录
├── CODEMULU.md                本目录索引
├── configs/
│   ├── paper.json             模型角色、核心训练超参数
│   └── experiments.json       实验条件、种子和验证网格
├── mrmem/
│   ├── __main__.py            python -m mrmem 入口
│   ├── cli.py                 训练、评测、构建和分析命令
│   ├── data.py                记忆/问题结构、校验与数据划分
│   ├── preparation.py         构建、整合、标准化和增长
│   ├── ranking.py             dense、BM25、RRF
│   ├── environment.py         动作协议、元数据视图、两阶段执行器
│   ├── models.py              HF模型、精确动作概率/KL、评判适配器
│   ├── rollout.py             在线检索→回答→奖励轨迹
│   ├── rewards.py             EM、judge分流、正确性门控奖励
│   ├── training.py            SFT、GRPO、验证与checkpoint
│   └── evaluation.py          逐题评测、配对统计和验证选模
├── scripts/
│   ├── demo.py                无模型下载的两阶段示例
│   └── make_experiments.py    生成按条件/折/seed隔离的配置
├── examples/
│   ├── repository.json        两条记录及源消息示例
│   ├── questions.jsonl        问题输入示例
│   └── teacher_traces.jsonl   示范轨迹示例
└── tests/
    ├── test_protocol.py       执行器、预算、隔离及统计测试
    └── test_training.py       SFT/GRPO更新、KL及Hugging Face梯度测试
```
