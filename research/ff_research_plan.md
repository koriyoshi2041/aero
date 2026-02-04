# FF (FreezeOut + FGSM) 研究计划

**目标**: 深入研究 FF 方法，找到迁移性能突破口，对比 Negative 样本策略

---

## 一、前人工作调研

### 核心代码库

| 仓库 | 描述 | 用途 |
|------|------|------|
| [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) | PyTorch 迁移攻击框架 | 主要参考，方法最全 |
| [TAA-Bench](https://github.com/KxPlaug/TAA-Bench) | 10 种迁移攻击的 benchmark | 对比基线 |
| [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) | MI-FGSM, DI-FGSM, TI-FGSM 等 | 基础攻击实现 |
| [Translation-Invariant-Attacks](https://github.com/dongyp13/Translation-Invariant-Attacks) | TI-FGSM 官方实现 | 迁移增强参考 |

### 关键论文

**迁移攻击增强:**
1. **MI-FGSM** (Dong et al., 2018) - Momentum 增强迁移
2. **DI-FGSM** (Xie et al., 2019) - Input Diversity 增强迁移
3. **TI-FGSM** (Dong et al., 2019) - Translation Invariance 增强迁移
4. **SI-FGSM** (Lin et al., 2020) - Scale Invariance 增强迁移

**Target 选择策略:**
1. **Dynamic Loss** (ScienceDirect 2024) - 动态选择 top-k 高概率标签惩罚
2. **Non-robust Feature Alignment** (ScienceDirect 2023) - 动态 top-k 标签选择
3. **CVPR 2023: Towards Transferable Targeted Adversarial Examples** - 目标迁移基准

**综合性:**
1. **TAA-Bench** (arXiv 2024) - 迁移攻击综合 benchmark
2. **TransferAttackEval** (TPAMI 2025) - 最新评估框架

---

## 二、Negative 样本策略实验

### 2.1 策略定义

| 策略 | 实现 | 预期效果 |
|------|------|----------|
| **Random** | `target = random.choice(other_classes)` | 基线 |
| **Least-Likely** | `target = argmin(logits)` | 最大决策边界 |
| **Most-Confusing** | `target = argmax(logits[not_true])` | 最易混淆 |
| **Semantic-Similar** | 预定义语义映射 (cat→dog) | 语义相似 |
| **Multi-Target** | 同时向 top-k 错误类别优化 | 分散攻击 |
| **Dynamic-Top-K** | 每步选当前 top-k 惩罚 | 动态调整 |

### 2.2 实验设计

```
数据集: CIFAR-10
模型: ResNet-18 (源) → VGG-16, DenseNet-121 (目标)
攻击: FF (FreezeOut + FGSM)
评估: 
  - 白盒成功率
  - 迁移成功率
  - 扰动大小 (L2, L∞)
```

### 2.3 代码结构

```python
# negative_strategies.py

def random_target(logits, true_label, num_classes):
    """随机选择非真实类别"""
    candidates = [i for i in range(num_classes) if i != true_label]
    return random.choice(candidates)

def least_likely_target(logits, true_label):
    """选择预测概率最低的类别"""
    probs = F.softmax(logits, dim=-1)
    probs[true_label] = 1.0  # 排除真实类别
    return probs.argmin()

def most_confusing_target(logits, true_label):
    """选择模型最容易混淆的类别（除真实类别外最高概率）"""
    probs = F.softmax(logits, dim=-1)
    probs[true_label] = -1.0  # 排除真实类别
    return probs.argmax()

def multi_target_loss(logits, targets, weights=None):
    """同时向多个目标类别优化"""
    loss = 0
    for t, w in zip(targets, weights or [1]*len(targets)):
        loss += w * F.cross_entropy(logits, t)
    return loss

def dynamic_topk_loss(logits, true_label, k=3):
    """动态惩罚 top-k 非目标高概率类别"""
    probs = F.softmax(logits, dim=-1)
    probs[true_label] = -1.0
    topk_classes = probs.topk(k).indices
    # 最大化这些类别的 loss（减小它们的概率）
    return -F.cross_entropy(logits, topk_classes)
```

---

## 三、迁移性能突破口分析

### 3.1 分析维度

1. **Freeze 阶段 vs 迁移性能**
   - 记录每个 freeze 阶段生成的 AE 的迁移成功率
   - 分析哪个阶段对迁移最有帮助

2. **梯度多样性分析**
   - 计算不同阶段梯度的余弦相似度
   - FreezeOut 是否导致梯度过于"专一"？

3. **特征层分析**
   - 对比 AE 在源模型和目标模型的特征表示
   - 使用 CKA (Centered Kernel Alignment) 分析

4. **频率域分析**
   - 分析扰动的高频/低频成分
   - 高频扰动是否影响迁移？

### 3.2 改进方向

基于分析可能的改进：
- **Multi-model gradient**: 使用多个模型的梯度
- **Feature-level attack**: 在特征层而非输出层攻击
- **Gradient diversity**: 增加梯度扰动或随机性
- **Input transformation**: DI, TI, SI 等变换

---

## 四、实验时间表

| 阶段 | 任务 | 时间 |
|------|------|------|
| Phase 1 | 搭建实验框架，实现 6 种 negative 策略 | 2h |
| Phase 2 | 运行 negative 策略对比实验 | 3h |
| Phase 3 | 迁移性能深度分析 | 3h |
| Phase 4 | 基于分析尝试改进 | 2h |

---

## 五、参考实现

从 TransferAttack 借鉴的核心代码模式：

```python
# 标准迁移攻击框架
class TransferAttack:
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
    
    def forward(self, images, labels):
        images = images.clone().detach()
        adv_images = images.clone().detach()
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            
            # 计算 loss（这里可以插入不同的 negative 策略）
            loss = self.get_loss(outputs, labels)
            
            # 计算梯度
            grad = torch.autograd.grad(loss, adv_images)[0]
            
            # 更新（可以加 momentum, input diversity 等）
            adv_images = self.update(adv_images, grad)
            
            # 投影到 epsilon ball
            adv_images = torch.clamp(adv_images, images - self.eps, images + self.eps)
            adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images
```

---

## 六、下一步

1. **Clone TransferAttack** 作为代码基础
2. **实现 FF + 6 种 negative 策略**
3. **运行对比实验**
4. **分析结果，找突破口**
