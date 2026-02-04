"""
快速验证实验框架

使用随机初始化的模型验证代码流程
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_model
from negative_strategies import get_strategy, STRATEGIES


def quick_test():
    """快速测试所有组件"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. 测试模型
    print("\n1. Testing models...")
    x = torch.randn(4, 3, 32, 32).to(device)
    for name in ['resnet18', 'vgg16', 'densenet121']:
        model = get_model(name).to(device)
        out = model(x)
        print(f"  {name}: {x.shape} -> {out.shape}")
    
    # 2. 测试 negative 策略
    print("\n2. Testing negative strategies...")
    logits = torch.randn(10).to(device)
    true_label = 3
    
    for name in STRATEGIES:
        strategy = get_strategy(name)
        target = strategy.get_target(logits, true_label)
        loss = strategy.get_loss(logits, true_label)
        print(f"  {name:20s} -> target: {target}, loss: {loss.item():.4f}")
    
    # 3. 测试完整攻击流程
    print("\n3. Testing attack pipeline...")
    
    # 加载少量数据
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    subset = Subset(testset, range(32))
    loader = DataLoader(subset, batch_size=8)
    
    # 模型
    model = get_model('resnet18').to(device)
    model.eval()
    
    # 攻击参数
    eps = 8/255
    alpha = 2/255
    steps = 5
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # 简化版 FF 攻击
        adv_images = images.clone().detach()
        
        # 获取目标
        with torch.no_grad():
            logits = model(images)
        
        strategy = get_strategy('least_likely')
        targets = []
        for i in range(len(labels)):
            t = strategy.get_target(logits[i], labels[i].item())
            targets.append(t)
        targets = torch.tensor(targets, device=device)
        
        # 攻击
        for step in range(steps):
            adv_images.requires_grad = True
            outputs = model(adv_images)
            
            # 计算 loss
            loss = F.cross_entropy(outputs, targets)
            
            model.zero_grad()
            loss.backward()
            
            grad = adv_images.grad.data
            adv_images = adv_images - alpha * grad.sign()  # targeted
            
            perturbation = torch.clamp(adv_images - images, -eps, eps)
            adv_images = torch.clamp(images + perturbation, 0, 1).detach()
        
        # 评估
        with torch.no_grad():
            clean_pred = model(images).argmax(dim=1)
            adv_pred = model(adv_images).argmax(dim=1)
        
        clean_acc = (clean_pred == labels).float().mean().item()
        target_success = (adv_pred == targets).float().mean().item()
        
        print(f"  Clean accuracy: {clean_acc*100:.1f}%")
        print(f"  Target success: {target_success*100:.1f}%")
        
        # 扰动统计
        pert = adv_images - images
        print(f"  L2 norm: {pert.norm(p=2, dim=[1,2,3]).mean().item():.4f}")
        print(f"  Linf: {pert.abs().max().item():.4f}")
        
        break  # 只测一个 batch
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    quick_test()
