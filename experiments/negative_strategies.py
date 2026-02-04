"""
Negative Sample Strategies for Targeted Adversarial Attacks

实现 6 种不同的 negative 样本策略，用于对比实验
"""

import torch
import torch.nn.functional as F
import random
from typing import List, Optional, Tuple


class NegativeStrategy:
    """Negative 样本策略基类"""
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
    
    def get_target(self, logits: torch.Tensor, true_label: int) -> int:
        """获取目标类别"""
        raise NotImplementedError
    
    def get_loss(self, logits: torch.Tensor, true_label: int, 
                 target: Optional[int] = None) -> torch.Tensor:
        """计算 loss"""
        if target is None:
            target = self.get_target(logits, true_label)
        # 默认：最小化到目标类别的距离（targeted attack）
        target_tensor = torch.tensor([target], device=logits.device)
        return F.cross_entropy(logits.unsqueeze(0), target_tensor)


class RandomNegative(NegativeStrategy):
    """随机选择一个错误类别"""
    
    def get_target(self, logits: torch.Tensor, true_label: int) -> int:
        candidates = [i for i in range(self.num_classes) if i != true_label]
        return random.choice(candidates)


class LeastLikelyNegative(NegativeStrategy):
    """选择预测概率最低的类别（最大化决策边界距离）"""
    
    def get_target(self, logits: torch.Tensor, true_label: int) -> int:
        probs = F.softmax(logits, dim=-1).clone()
        probs[true_label] = float('inf')  # 排除真实类别
        return probs.argmin().item()


class MostConfusingNegative(NegativeStrategy):
    """选择最容易混淆的类别（除真实类别外概率最高）"""
    
    def get_target(self, logits: torch.Tensor, true_label: int) -> int:
        probs = F.softmax(logits, dim=-1).clone()
        probs[true_label] = float('-inf')  # 排除真实类别
        return probs.argmax().item()


class SemanticNegative(NegativeStrategy):
    """基于语义相似性选择目标类别
    
    CIFAR-10 类别: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    语义相似组:
    - 交通工具: airplane(0), automobile(1), ship(8), truck(9)
    - 动物: bird(2), cat(3), deer(4), dog(5), frog(6), horse(7)
    """
    
    CIFAR10_SEMANTIC_MAP = {
        0: [8, 1, 9],       # airplane → ship, automobile, truck
        1: [9, 0, 8],       # automobile → truck, airplane, ship
        2: [6, 3, 4],       # bird → frog, cat, deer
        3: [5, 4, 2],       # cat → dog, deer, bird
        4: [7, 5, 3],       # deer → horse, dog, cat
        5: [3, 7, 4],       # dog → cat, horse, deer
        6: [2, 4, 3],       # frog → bird, deer, cat
        7: [4, 5, 3],       # horse → deer, dog, cat
        8: [0, 1, 9],       # ship → airplane, automobile, truck
        9: [1, 8, 0],       # truck → automobile, ship, airplane
    }
    
    def get_target(self, logits: torch.Tensor, true_label: int) -> int:
        similar_classes = self.CIFAR10_SEMANTIC_MAP.get(true_label, [])
        if similar_classes:
            return similar_classes[0]  # 返回最相似的
        # fallback to random
        candidates = [i for i in range(self.num_classes) if i != true_label]
        return random.choice(candidates)


class MultiTargetNegative(NegativeStrategy):
    """同时向多个目标类别优化
    
    使用 top-k 最高概率的非真实类别作为目标
    """
    
    def __init__(self, num_classes: int = 10, k: int = 3):
        super().__init__(num_classes)
        self.k = k
    
    def get_target(self, logits: torch.Tensor, true_label: int) -> List[int]:
        probs = F.softmax(logits, dim=-1).clone()
        probs[true_label] = float('-inf')
        topk = probs.topk(self.k).indices.tolist()
        return topk
    
    def get_loss(self, logits: torch.Tensor, true_label: int,
                 target: Optional[List[int]] = None) -> torch.Tensor:
        if target is None:
            target = self.get_target(logits, true_label)
        
        # 多目标 loss: 最小化到所有目标类别的平均距离
        total_loss = 0
        for t in target:
            target_tensor = torch.tensor([t], device=logits.device)
            total_loss += F.cross_entropy(logits.unsqueeze(0), target_tensor)
        return total_loss / len(target)


class DynamicTopKNegative(NegativeStrategy):
    """动态 Top-K 惩罚策略
    
    参考: "Dynamic loss yielding more transferable targeted adversarial examples"
    
    在优化过程中动态选择当前 top-k 高概率的非目标类别进行惩罚，
    确保对抗样本不仅接近目标类别，还远离其他可能的类别
    """
    
    def __init__(self, num_classes: int = 10, k: int = 3, 
                 target_weight: float = 1.0, penalty_weight: float = 0.5):
        super().__init__(num_classes)
        self.k = k
        self.target_weight = target_weight
        self.penalty_weight = penalty_weight
        self._fixed_target = None  # 固定的目标类别
    
    def set_fixed_target(self, target: int):
        """设置固定的目标类别"""
        self._fixed_target = target
    
    def get_target(self, logits: torch.Tensor, true_label: int) -> int:
        if self._fixed_target is not None:
            return self._fixed_target
        # 默认使用 least-likely
        probs = F.softmax(logits, dim=-1).clone()
        probs[true_label] = float('inf')
        return probs.argmin().item()
    
    def get_loss(self, logits: torch.Tensor, true_label: int,
                 target: Optional[int] = None) -> torch.Tensor:
        if target is None:
            target = self.get_target(logits, true_label)
        
        # 1. 目标 loss: 最小化到目标类别的距离
        target_tensor = torch.tensor([target], device=logits.device)
        target_loss = F.cross_entropy(logits.unsqueeze(0), target_tensor)
        
        # 2. 惩罚 loss: 最大化到 top-k 非目标高概率类别的距离
        probs = F.softmax(logits, dim=-1).clone()
        probs[true_label] = float('-inf')
        probs[target] = float('-inf')
        
        if self.k > 0:
            topk_classes = probs.topk(min(self.k, self.num_classes - 2)).indices
            penalty_loss = 0
            for cls in topk_classes:
                cls_tensor = torch.tensor([cls.item()], device=logits.device)
                # 最大化这些类别的 loss（负号）
                penalty_loss -= F.cross_entropy(logits.unsqueeze(0), cls_tensor)
            penalty_loss /= len(topk_classes)
        else:
            penalty_loss = 0
        
        return self.target_weight * target_loss + self.penalty_weight * penalty_loss


# 策略注册
STRATEGIES = {
    'random': RandomNegative,
    'least_likely': LeastLikelyNegative,
    'most_confusing': MostConfusingNegative,
    'semantic': SemanticNegative,
    'multi_target': MultiTargetNegative,
    'dynamic_topk': DynamicTopKNegative,
}


def get_strategy(name: str, **kwargs) -> NegativeStrategy:
    """获取策略实例"""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)


# 测试
if __name__ == '__main__':
    # 模拟 logits
    logits = torch.randn(10)
    true_label = 3
    
    print("Testing negative strategies:")
    print(f"Logits: {F.softmax(logits, dim=-1).numpy().round(3)}")
    print(f"True label: {true_label}")
    print()
    
    for name, cls in STRATEGIES.items():
        strategy = cls()
        target = strategy.get_target(logits, true_label)
        loss = strategy.get_loss(logits, true_label)
        print(f"{name:20s} → target: {target}, loss: {loss.item():.4f}")
