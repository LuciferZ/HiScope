# 本地验证记录

日期：2026-11-15。验证在 CPU 上完成；当前会话无法初始化 CUDA。

实际环境：Python 3.12.13、PyTorch 2.10.0+cu129、Transformers 4.57.6、NumPy 2.2.6、tokenizers 0.22.2。没有安装或修改项目依赖。该环境不同于论文报告的固定版本，因此不声称已验证其所有版本组合。

验证结果：

- 执行器/数据/奖励/统计测试 24 项通过。
- 训练和模型适配测试 8 项通过，包括实际 SFT 参数更新、在线 GRPO 参数更新、冻结 reference 不变、轨迹概率比、零方差 advantage、精确 forward KL、微型 GPT-2 梯度，以及本地 HF 权重加载/采样/保存/重新加载。
- `scripts/demo.py` 复现补充材料 S2 的两阶段证据替换逻辑。
- `scripts/make_experiments.py` 成功生成十折 × 三 seed × 六条件的 180 个任务及相应配置；未启动真实训练。
- Python 模块编译及 CLI help 检查通过。

最后补充了 PersonaMem 选项 A 的回归测试：标准英文 EM 会删除冠词 a，不能直接用于选项标签，否则空答案可能错误匹配 A。PersonaMem 使用非空、大小写规范化后的选项标签匹配。

运行方式：

```bash
python3 -m unittest discover -s tests -v
python3 scripts/demo.py
python3 -m compileall -q mrmem scripts tests
```

微型模型和合成数据仅用于检验代码行为。
