# TinyWM

极简 RSSM 世界模型（~100M 参数）：Dreamer 风格，但剥到最简，核心模型只有一个文件。

在 LIBERO 仿真演示数据上训练，从单帧出发做开环想象（imagine） rollout。

## 架构

```
帧 (3×128×128)
   │
Encoder: stride-2 CNN ×5 → embedding (768-D)
   │
RSSM:  GRU 确定性状态 h (3072-D) + 高斯随机潜变量 z (32-D)
       posterior q(z|h, embed) / prior p(z|h)，KL balancing
   ├─→ Decoder: deconv 从 (h, z) 重建帧
   └─→ Reward head: MLP 从 (h, z) 预测奖励
```

- 损失 = 重建 (MSE) + β·KL (β=0.15) + 奖励 MSE
- KL balancing：prior 慢速（0.8）、posterior 快速（0.2）
- 总计约 100M 参数，单卡可训

## 文件

| 文件 | 说明 |
|---|---|
| `world_model.py` | 全部模型代码：Encoder / RSSM / Decoder / Reward（约 160 行） |
| `data.py` | LIBERO hdf5 数据集：随机取 (帧序列, 动作) 窗口 |
| `train.py` | 训练循环 + 训练后开环想象自检 |

## 训练

```bash
pip install torch h5py numpy
python train.py --data ../datasets_libero/libero_goal --steps 20000 --batch 32
# 冒烟测试：python train.py --smoke
```

训练结束自动保存 `wm.pt`，并打印一次 imagine 输出的形状 `(B, T+1, 3, H, W)`。

## 想象（开环 rollout）

```python
from world_model import WorldModel
model = WorldModel().cuda()
model.load_state_dict(torch.load("wm.pt"))
frames = model.imagine(obs0, actions)  # obs0: (B,3,H,W)，返回 (B,T+1,3,H,W)
```

## 备注

- 模型权重 `wm.pt`（约 380M）不入库，按上面命令自行训练生成
- 有意保持最小实现：没有 actor-critic、没有数据增强、没有 EMA，留作后续扩展

## 参考

- [DreamerV3](https://github.com/danijar/dreamerv3) — 学习在想象中学策略的通用框架
- DreamerV2 / RSSM（Hafner et al.）
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — 基准与数据来源
