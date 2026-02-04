# AERO: Adversarial Example Research for Optimization

> 🚧 **Work in Progress** - 对抗样本攻击与迁移性研究

## 项目概述

本项目研究 **FreezeOut + FGSM (FF)** 攻击方法，重点探索：
1. **Negative 样本策略** - 不同目标选择策略对攻击效果的影响
2. **迁移性能** - 对抗样本从源模型迁移到目标模型的能力

## 当前进度

### ✅ 已完成
- [x] 实验框架搭建
- [x] 6 种 Negative 策略实现
- [x] 19 个预训练模型下载 (CIFAR-10)
- [x] 初步实验完成
- [x] 结果可视化

### 🔄 进行中
- [ ] 迁移性能瓶颈分析
- [ ] 添加迁移增强技术 (MI, DI, TI)
- [ ] FreezeOut 各阶段迁移性分析

### 📋 计划中
- [ ] 完整实验报告
- [ ] 论文相关分析

## 实验结果

### Negative 策略对比 (2024-02-04)

**实验配置:**
- 源模型: ResNet-56 (94.22% acc)
- 目标模型: VGG16-BN, MobileNetV2, ShuffleNetV2
- 攻击: FF, ε=8/255, 10 steps, 500 samples

**Target Success Rate:**

| Strategy | Whitebox | vgg16_bn | mobilenetv2_x1_0 | shufflenetv2_x1_0 |
|---|---|---|---|---|
| **most_confusing** | **60.2%** | **13.8%** | **20.0%** | **16.4%** |
| **multi_target** | **60.2%** | **13.8%** | **20.0%** | **16.4%** |
| semantic | 53.8% | 9.4% | 13.8% | 11.8% |
| random | 36.8% | 6.2% | 9.0% | 6.8% |
| least_likely | 27.4% | 3.8% | 3.4% | 3.2% |
| dynamic_topk | 27.4% | 3.8% | 3.4% | 3.2% |

### 可视化

<p align="center">
  <img src="experiments/results/strategy_comparison.png" width="80%" />
</p>

<p align="center">
  <img src="experiments/results/transfer_heatmap.png" width="60%" />
</p>

### 关键发现

1. **most_confusing 策略最优** - 选择模型最容易混淆的类别（非真实类别中概率最高的）效果最好
2. **least_likely 策略最差** - 选择最不可能的类别反而最难攻击成功
3. **迁移率普遍较低** (~3-20%) - 需要进一步分析和优化

### 迁移瓶颈分析 (2024-02-04)

**梯度相似性分析：**
| Target Model | Cosine Similarity | Sign Match Rate |
|--------------|-------------------|-----------------|
| vgg16_bn | 0.086 | 52.1% |
| mobilenetv2 | 0.111 | 52.7% |
| shufflenetv2 | 0.108 | 52.8% |

**关键发现：**
- ⚠️ **梯度相似度极低** (~0.09-0.11) - 这是迁移率低的主要原因
- ⚠️ **Sign Match ~52%** - 接近随机，说明梯度方向几乎不相关
- ✅ **扰动 95.8% 是低频** - 低频扰动通常更容易迁移

**改进方向：**
1. 使用输入变换 (DI, TI, SI) 增加梯度多样性
2. 使用多模型集成攻击
3. 使用 Momentum 累积梯度 (MI-FGSM)

## 项目结构

```
aero/
├── README.md
├── experiments/
│   ├── negative_strategies.py    # 6 种 Negative 策略实现
│   ├── hub_models.py             # torch.hub 模型加载器
│   ├── run_ff_experiment.py      # FF 实验脚本
│   ├── visualize_results.py      # 可视化脚本
│   ├── checkpoints/              # 预训练模型 (19 个)
│   └── results/                  # 实验结果和图表
├── research/
│   └── ff_research_plan.md       # 研究计划
└── data/                         # CIFAR-10 数据集
```

## 快速开始

```bash
# 1. 下载预训练模型
cd experiments
python download_models.py --hub

# 2. 运行实验
python run_ff_experiment.py

# 3. 可视化结果
python visualize_results.py
```

## Negative 策略说明

| 策略 | 描述 |
|------|------|
| `random` | 随机选择一个非真实类别 |
| `least_likely` | 选择预测概率最低的类别 |
| `most_confusing` | 选择非真实类别中概率最高的（最易混淆） |
| `semantic` | 基于语义相似性选择（如 cat→dog） |
| `multi_target` | 同时向多个高概率类别优化 |
| `dynamic_topk` | 动态惩罚 top-k 高概率非目标类别 |

## 参考文献

- [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) - 迁移攻击框架
- [TAA-Bench](https://github.com/KxPlaug/TAA-Bench) - 迁移攻击 benchmark
- [pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models) - 预训练模型

## License

MIT
