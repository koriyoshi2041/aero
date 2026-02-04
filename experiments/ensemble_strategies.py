"""
集成策略深度研究

探索不同的集成方式：
1. 梯度平均 (baseline)
2. 梯度加权 (按模型置信度)
3. 损失加权
4. 随机选择模型
5. 不同模型数量的影响
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hub_models import get_hub_model
from negative_strategies import get_strategy


class EnsembleAttackVariants:
    """不同集成策略的实现"""
    
    def __init__(self, models, eps=8/255, alpha=2/255, steps=10,
                 strategy='average', momentum=1.0, device=None):
        self.models = models
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.strategy = strategy
        self.momentum = momentum
        self.device = device
    
    def _get_ensemble_gradient(self, adv, targets):
        """根据策略计算集成梯度"""
        grads = []
        losses = []
        confidences = []
        
        for model in self.models:
            model.eval()
            outputs = model(adv)
            loss = F.cross_entropy(outputs, targets)
            grad = torch.autograd.grad(loss, adv, retain_graph=True)[0]
            grads.append(grad)
            losses.append(loss.item())
            
            # 置信度 = 目标类的概率
            probs = F.softmax(outputs, dim=1)
            conf = probs.gather(1, targets.unsqueeze(1)).mean().item()
            confidences.append(conf)
        
        if self.strategy == 'average':
            # 简单平均
            return sum(grads) / len(grads)
        
        elif self.strategy == 'loss_weighted':
            # 损失加权（损失大的权重大）
            weights = torch.softmax(torch.tensor(losses), dim=0)
            weighted_grad = sum(w * g for w, g in zip(weights, grads))
            return weighted_grad
        
        elif self.strategy == 'confidence_weighted':
            # 置信度加权（置信度低的权重大，因为更难攻击）
            inv_conf = [1.0 / (c + 0.1) for c in confidences]
            weights = [c / sum(inv_conf) for c in inv_conf]
            weighted_grad = sum(w * g for w, g in zip(weights, grads))
            return weighted_grad
        
        elif self.strategy == 'random':
            # 随机选择一个模型
            idx = np.random.randint(len(grads))
            return grads[idx]
        
        elif self.strategy == 'max_grad':
            # 选择梯度最大的
            grad_norms = [g.abs().mean().item() for g in grads]
            idx = np.argmax(grad_norms)
            return grads[idx]
        
        else:
            return sum(grads) / len(grads)
    
    def __call__(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        adv = images.clone().detach()
        momentum_g = torch.zeros_like(images)
        
        for _ in range(self.steps):
            adv.requires_grad = True
            
            grad = self._get_ensemble_gradient(adv, targets)
            
            # Momentum
            momentum_g = self.momentum * momentum_g + grad / (grad.abs().mean() + 1e-8)
            
            adv = adv.detach() - self.alpha * momentum_g.sign()
            delta = torch.clamp(adv - images, -self.eps, self.eps)
            adv = torch.clamp(images + delta, 0, 1)
        
        return adv.detach()


def run_ensemble_strategy_comparison():
    """比较不同集成策略"""
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    print("\nLoading models...")
    
    # 源模型集合
    ensemble_models = [
        get_hub_model('resnet56', pretrained=True, device=device),
        get_hub_model('vgg16_bn', pretrained=True, device=device),
        get_hub_model('mobilenetv2_x1_0', pretrained=True, device=device),
    ]
    for m in ensemble_models:
        m.eval()
    
    single_model = ensemble_models[0]
    
    # 目标模型
    target_models = {
        'resnet20': get_hub_model('resnet20', pretrained=True, device=device),
        'vgg13_bn': get_hub_model('vgg13_bn', pretrained=True, device=device),
        'shufflenetv2_x1_0': get_hub_model('shufflenetv2_x1_0', pretrained=True, device=device),
        'repvgg_a0': get_hub_model('repvgg_a0', pretrained=True, device=device),
    }
    for m in target_models.values():
        m.eval()
    
    # 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset = Subset(testset, range(300))
    loader = DataLoader(subset, batch_size=50, shuffle=False)
    
    # 集成策略
    strategies = ['average', 'loss_weighted', 'confidence_weighted', 'random', 'max_grad']
    
    results = {}
    
    print("\n" + "="*70)
    print("ENSEMBLE STRATEGY COMPARISON")
    print("="*70)
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        
        attacker = EnsembleAttackVariants(
            ensemble_models, eps=8/255, alpha=2/255, steps=10,
            strategy=strategy, device=device
        )
        
        all_adv = []
        all_labels = []
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = attacker(images, labels)
            all_adv.append(adv_images)
            all_labels.append(labels)
        
        all_adv = torch.cat(all_adv)
        all_labels = torch.cat(all_labels)
        
        results[strategy] = {}
        
        # 迁移测试
        transfer_results = []
        for target_name, target_model in target_models.items():
            with torch.no_grad():
                out = target_model(all_adv)
                pred = out.argmax(dim=1)
                misclass = (pred != all_labels).float().mean().item()
            results[strategy][target_name] = misclass
            transfer_results.append(misclass)
            print(f"  → {target_name}: {misclass*100:.1f}%")
        
        results[strategy]['avg_transfer'] = np.mean(transfer_results)
        print(f"  Avg Transfer: {np.mean(transfer_results)*100:.1f}%")
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/ensemble_strategies_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 排名
    print("\n" + "="*70)
    print("STRATEGY RANKING")
    print("="*70)
    sorted_strategies = sorted(results.items(), key=lambda x: x[1]['avg_transfer'], reverse=True)
    for i, (name, res) in enumerate(sorted_strategies, 1):
        print(f"{i}. {name}: {res['avg_transfer']*100:.1f}%")
    
    return results


def run_model_count_experiment():
    """测试不同模型数量对集成效果的影响"""
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"\nDevice: {device}")
    
    print("\nLoading all available models...")
    
    all_models = {
        'resnet56': get_hub_model('resnet56', pretrained=True, device=device),
        'resnet44': get_hub_model('resnet44', pretrained=True, device=device),
        'resnet20': get_hub_model('resnet20', pretrained=True, device=device),
        'vgg16_bn': get_hub_model('vgg16_bn', pretrained=True, device=device),
        'vgg13_bn': get_hub_model('vgg13_bn', pretrained=True, device=device),
        'mobilenetv2_x1_0': get_hub_model('mobilenetv2_x1_0', pretrained=True, device=device),
        'shufflenetv2_x1_0': get_hub_model('shufflenetv2_x1_0', pretrained=True, device=device),
        'repvgg_a0': get_hub_model('repvgg_a0', pretrained=True, device=device),
    }
    for m in all_models.values():
        m.eval()
    
    # 目标模型（不在源模型中）
    target_model = get_hub_model('shufflenetv2_x0_5', pretrained=True, device=device)
    target_model.eval()
    
    # 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset = Subset(testset, range(200))
    loader = DataLoader(subset, batch_size=50, shuffle=False)
    
    # 不同数量的集成
    model_names = list(all_models.keys())
    
    results = {}
    
    print("\n" + "="*70)
    print("MODEL COUNT VS TRANSFER RATE")
    print("="*70)
    
    for n_models in [1, 2, 3, 4, 5, 6, 7, 8]:
        if n_models > len(model_names):
            break
        
        selected_names = model_names[:n_models]
        selected_models = [all_models[name] for name in selected_names]
        
        print(f"\n--- {n_models} model(s): {', '.join(selected_names)} ---")
        
        attacker = EnsembleAttackVariants(
            selected_models, eps=8/255, alpha=2/255, steps=10,
            strategy='average', device=device
        )
        
        all_adv = []
        all_labels = []
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = attacker(images, labels)
            all_adv.append(adv_images)
            all_labels.append(labels)
        
        all_adv = torch.cat(all_adv)
        all_labels = torch.cat(all_labels)
        
        with torch.no_grad():
            out = target_model(all_adv)
            pred = out.argmax(dim=1)
            misclass = (pred != all_labels).float().mean().item()
        
        results[n_models] = {
            'models': selected_names,
            'transfer': misclass
        }
        print(f"  Transfer: {misclass*100:.1f}%")
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/model_count_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 总结
    print("\n" + "="*70)
    print("SUMMARY: Model Count vs Transfer")
    print("="*70)
    print(f"{'# Models':<12} {'Transfer':<12}")
    print("-"*24)
    for n, res in results.items():
        print(f"{n:<12} {res['transfer']*100:>8.1f}%")
    
    return results


def main():
    results = {}
    
    # 1. 集成策略比较
    results['strategies'] = run_ensemble_strategy_comparison()
    
    # 2. 模型数量影响
    results['model_count'] = run_model_count_experiment()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)


if __name__ == '__main__':
    main()
