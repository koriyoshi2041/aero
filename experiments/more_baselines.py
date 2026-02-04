"""
更多 Baseline 对比

对比 TransferAttack 仓库中的更多方法
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


class SIFGSM:
    """Scale-Invariant FGSM (SI-FGSM)"""
    
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, 
                 scale_copies=5, device=None):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.scale_copies = scale_copies
        self.device = device
    
    def __call__(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        adv = images.clone().detach()
        
        for _ in range(self.steps):
            adv.requires_grad = True
            
            # 多尺度梯度
            total_grad = torch.zeros_like(images)
            
            for i in range(self.scale_copies):
                scale = 1.0 / (2 ** i)
                scaled = adv * scale
                outputs = self.model(scaled)
                loss = F.cross_entropy(outputs, targets)
                grad = torch.autograd.grad(loss, adv, retain_graph=True)[0]
                total_grad += grad
            
            avg_grad = total_grad / self.scale_copies
            
            adv = adv.detach() - self.alpha * avg_grad.sign()
            delta = torch.clamp(adv - images, -self.eps, self.eps)
            adv = torch.clamp(images + delta, 0, 1)
        
        return adv.detach()


class VMIFGSM:
    """Variance-tuning MI-FGSM (VMI-FGSM)"""
    
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10,
                 momentum=1.0, beta=1.5, sample_num=20, device=None):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.momentum = momentum
        self.beta = beta
        self.sample_num = sample_num
        self.device = device
    
    def __call__(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        adv = images.clone().detach()
        momentum_g = torch.zeros_like(images)
        variance = torch.zeros_like(images)
        
        for _ in range(self.steps):
            adv.requires_grad = True
            
            # 计算当前梯度
            outputs = self.model(adv)
            loss = F.cross_entropy(outputs, targets)
            grad = torch.autograd.grad(loss, adv)[0]
            
            # 计算方差（使用邻域采样）
            grad_samples = []
            for _ in range(min(self.sample_num, 5)):  # 减少采样数以加速
                noise = torch.randn_like(adv) * (self.beta * self.eps)
                adv_sample = adv.detach() + noise
                adv_sample.requires_grad = True
                out_sample = self.model(adv_sample)
                loss_sample = F.cross_entropy(out_sample, targets)
                grad_sample = torch.autograd.grad(loss_sample, adv_sample)[0]
                grad_samples.append(grad_sample)
            
            # 计算方差
            grad_stack = torch.stack(grad_samples)
            current_variance = grad_stack.var(dim=0)
            variance = variance + current_variance
            
            # 方差调整的梯度
            adjusted_grad = grad / (torch.sqrt(variance) + 1e-8)
            
            # Momentum
            momentum_g = self.momentum * momentum_g + adjusted_grad / (adjusted_grad.abs().mean() + 1e-8)
            
            adv = adv.detach() - self.alpha * momentum_g.sign()
            delta = torch.clamp(adv - images, -self.eps, self.eps)
            adv = torch.clamp(images + delta, 0, 1)
        
        return adv.detach()


class ADMIX:
    """Admix Attack - 混合输入"""
    
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10,
                 admix_portion=0.2, num_admix=3, device=None):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.admix_portion = admix_portion
        self.num_admix = num_admix
        self.device = device
    
    def __call__(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        batch_size = images.size(0)
        
        adv = images.clone().detach()
        
        for _ in range(self.steps):
            adv.requires_grad = True
            
            total_grad = torch.zeros_like(images)
            
            # 原始梯度
            outputs = self.model(adv)
            loss = F.cross_entropy(outputs, targets)
            grad = torch.autograd.grad(loss, adv, retain_graph=True)[0]
            total_grad += grad
            
            # Admix 梯度
            for _ in range(self.num_admix):
                # 随机选择混合样本
                indices = torch.randperm(batch_size)
                mixed = adv * (1 - self.admix_portion) + adv[indices] * self.admix_portion
                mixed.requires_grad = True
                
                out_mixed = self.model(mixed)
                loss_mixed = F.cross_entropy(out_mixed, targets)
                grad_mixed = torch.autograd.grad(loss_mixed, mixed)[0]
                total_grad += grad_mixed
            
            avg_grad = total_grad / (1 + self.num_admix)
            
            adv = adv.detach() - self.alpha * avg_grad.sign()
            delta = torch.clamp(adv - images, -self.eps, self.eps)
            adv = torch.clamp(images + delta, 0, 1)
        
        return adv.detach()


def run_baseline_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    print("\nLoading models...")
    source_model = get_hub_model('resnet56', pretrained=True, device=device)
    source_model.eval()
    
    target_models = {
        'vgg16_bn': get_hub_model('vgg16_bn', pretrained=True, device=device),
        'mobilenetv2_x1_0': get_hub_model('mobilenetv2_x1_0', pretrained=True, device=device),
        'shufflenetv2_x1_0': get_hub_model('shufflenetv2_x1_0', pretrained=True, device=device),
        'resnet20': get_hub_model('resnet20', pretrained=True, device=device),
    }
    for m in target_models.values():
        m.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset = Subset(testset, range(300))
    loader = DataLoader(subset, batch_size=50, shuffle=False)
    
    # 导入已有的攻击
    from ff_attack import FFAttack, FFDIAttack
    from transfer_attacks import MIFGSM, DIFGSM, TIFGSM
    
    attacks = {
        'I-FGSM': FFAttack(source_model, epsilon=8/255, alpha=2/255, epoch=10, 
                          freeze_epochs=1, device=device),  # 1 epoch = no freeze
        'MI-FGSM': MIFGSM(source_model, eps=8/255, alpha=2/255, steps=10),
        'DI-FGSM': DIFGSM(source_model, eps=8/255, alpha=2/255, steps=10),
        'TI-FGSM': TIFGSM(source_model, eps=8/255, alpha=2/255, steps=10),
        'SI-FGSM': SIFGSM(source_model, eps=8/255, alpha=2/255, steps=10, device=device),
        'VMI-FGSM': VMIFGSM(source_model, eps=8/255, alpha=2/255, steps=10, device=device),
        'Admix': ADMIX(source_model, eps=8/255, alpha=2/255, steps=10, device=device),
        'FF': FFAttack(source_model, epsilon=8/255, alpha=2/255, epoch=10, device=device),
        'FF-DI': FFDIAttack(source_model, epsilon=8/255, alpha=2/255, epoch=10, device=device),
    }
    
    results = {}
    
    print("\n" + "="*70)
    print("COMPREHENSIVE BASELINE COMPARISON")
    print("="*70)
    
    for attack_name, attacker in attacks.items():
        print(f"\n--- {attack_name} ---")
        
        all_adv = []
        all_labels = []
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # 非目标攻击
            if hasattr(attacker, 'attack'):
                adv_images = attacker.attack(images, labels)
            else:
                adv_images = attacker(images, labels)
            all_adv.append(adv_images)
            all_labels.append(labels)
        
        all_adv = torch.cat(all_adv)
        all_labels = torch.cat(all_labels)
        
        results[attack_name] = {}
        
        # 白盒
        with torch.no_grad():
            out = source_model(all_adv)
            pred = out.argmax(dim=1)
            misclass = (pred != all_labels).float().mean().item()
        results[attack_name]['whitebox'] = misclass
        print(f"  Whitebox: {misclass*100:.1f}%")
        
        # 迁移
        transfer_results = []
        for target_name, target_model in target_models.items():
            with torch.no_grad():
                out = target_model(all_adv)
                pred = out.argmax(dim=1)
                misclass = (pred != all_labels).float().mean().item()
            results[attack_name][target_name] = misclass
            transfer_results.append(misclass)
            print(f"  → {target_name}: {misclass*100:.1f}%")
        
        results[attack_name]['avg_transfer'] = np.mean(transfer_results)
        print(f"  Avg Transfer: {np.mean(transfer_results)*100:.1f}%")
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/baseline_comparison_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 排名
    print("\n" + "="*70)
    print("RANKING BY TRANSFER RATE")
    print("="*70)
    
    sorted_attacks = sorted(results.items(), key=lambda x: x[1]['avg_transfer'], reverse=True)
    
    print(f"{'Rank':<6} {'Attack':<15} {'Whitebox':<12} {'Avg Transfer':<12}")
    print("-"*45)
    for i, (name, res) in enumerate(sorted_attacks, 1):
        wb = res['whitebox'] * 100
        tr = res['avg_transfer'] * 100
        print(f"{i:<6} {name:<15} {wb:>8.1f}%    {tr:>8.1f}%")
    
    return results


if __name__ == '__main__':
    results = run_baseline_comparison()
