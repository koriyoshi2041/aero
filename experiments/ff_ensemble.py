"""
FF + 集成攻击

基于发现的改进：既然单模型梯度正交，那就用多模型的 FF 攻击。
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


class FFEnsembleAttack:
    """
    FreezeOut + Ensemble Attack
    
    在多个模型上同时应用 FreezeOut 策略，取平均梯度。
    """
    
    def __init__(self, models, eps=8/255, alpha=2/255, steps=10, 
                 freeze_epochs=3, momentum=1.0, device=None):
        self.models = models
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.freeze_epochs = freeze_epochs
        self.momentum = momentum
        self.device = device
        
        # 为每个模型获取层组
        self.model_layer_groups = []
        for model in models:
            groups = self._get_layer_groups(model)
            self.model_layer_groups.append(groups)
    
    def _get_layer_groups(self, model):
        """获取模型的层组"""
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                layers.append(module)
        
        n = len(layers)
        if n == 0:
            return []
        
        group_size = max(1, n // self.freeze_epochs)
        groups = []
        for i in range(0, n, group_size):
            groups.append(layers[i:i+group_size])
        
        return groups
    
    def _freeze_groups(self, model_idx, num_groups):
        """冻结指定模型的前 num_groups 组"""
        for group in self.model_layer_groups[model_idx][:num_groups]:
            for module in group:
                for param in module.parameters():
                    param.requires_grad = False
    
    def _unfreeze_all(self):
        """解冻所有模型的所有层"""
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = True
    
    def __call__(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        adv = images.clone().detach()
        momentum_g = torch.zeros_like(images)
        
        steps_per_stage = max(1, self.steps // self.freeze_epochs)
        
        for stage in range(self.freeze_epochs):
            # 在所有模型上冻结对应阶段的层
            for model_idx in range(len(self.models)):
                self._freeze_groups(model_idx, stage)
            
            for _ in range(steps_per_stage):
                adv.requires_grad = True
                
                # 计算所有模型的平均梯度
                total_grad = torch.zeros_like(images)
                
                for model in self.models:
                    model.eval()
                    outputs = model(adv)
                    loss = F.cross_entropy(outputs, targets)
                    grad = torch.autograd.grad(loss, adv, retain_graph=False)[0]
                    total_grad += grad
                
                avg_grad = total_grad / len(self.models)
                
                # Momentum
                momentum_g = self.momentum * momentum_g + avg_grad / (avg_grad.abs().mean() + 1e-8)
                
                # Update
                adv = adv.detach() - self.alpha * momentum_g.sign()
                
                # Project
                delta = torch.clamp(adv - images, -self.eps, self.eps)
                adv = torch.clamp(images + delta, 0, 1)
        
        # 解冻
        self._unfreeze_all()
        
        return adv.detach()


def run_ff_ensemble_experiment():
    """运行 FF + 集成实验"""
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    print("\nLoading models...")
    
    # 集成源模型（不同架构）
    ensemble_models = [
        get_hub_model('resnet56', pretrained=True, device=device),
        get_hub_model('vgg16_bn', pretrained=True, device=device),
        get_hub_model('mobilenetv2_x1_0', pretrained=True, device=device),
    ]
    for m in ensemble_models:
        m.eval()
    
    # 单一源模型
    single_model = get_hub_model('resnet56', pretrained=True, device=device)
    single_model.eval()
    
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
    
    # 攻击方法对比
    from ff_attack import FFAttack, FFDIAttack
    
    attacks = {
        'FF (single)': FFAttack(single_model, epsilon=8/255, alpha=2/255, epoch=10, device=device),
        'FF-DI (single)': FFDIAttack(single_model, epsilon=8/255, alpha=2/255, epoch=10, device=device),
        'FF-Ensemble (3 models)': FFEnsembleAttack(ensemble_models, eps=8/255, alpha=2/255, steps=10, device=device),
    }
    
    strategy = get_strategy('most_confusing')
    results = {}
    
    print("\n" + "="*70)
    print("FF + ENSEMBLE ATTACK COMPARISON")
    print("="*70)
    
    for attack_name, attacker in attacks.items():
        print(f"\n--- {attack_name} ---")
        
        all_adv = []
        all_labels = []
        all_targets = []
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                logits = single_model(images)
            targets = torch.tensor([strategy.get_target(logits[i], labels[i].item()) 
                                   for i in range(len(labels))], device=device)
            
            if 'Ensemble' in attack_name:
                adv_images = attacker(images, targets)
            else:
                adv_images = attacker(images, targets)
            
            all_adv.append(adv_images)
            all_labels.append(labels)
            all_targets.append(targets)
        
        all_adv = torch.cat(all_adv)
        all_labels = torch.cat(all_labels)
        all_targets = torch.cat(all_targets)
        
        results[attack_name] = {}
        
        # 白盒（用 ResNet56）
        with torch.no_grad():
            out = single_model(all_adv)
            pred = out.argmax(dim=1)
            target_success = (pred == all_targets).float().mean().item()
            misclass = (pred != all_labels).float().mean().item()
        results[attack_name]['whitebox'] = {'target': target_success, 'misclass': misclass}
        print(f"  Whitebox: Target={target_success*100:.1f}%, Misclass={misclass*100:.1f}%")
        
        # 迁移
        transfer_results = []
        for target_name, target_model in target_models.items():
            with torch.no_grad():
                out = target_model(all_adv)
                pred = out.argmax(dim=1)
                target_success = (pred == all_targets).float().mean().item()
                misclass = (pred != all_labels).float().mean().item()
            results[attack_name][target_name] = {'target': target_success, 'misclass': misclass}
            transfer_results.append(misclass)
            print(f"  → {target_name}: Target={target_success*100:.1f}%, Misclass={misclass*100:.1f}%")
        
        results[attack_name]['avg_transfer'] = np.mean(transfer_results)
        print(f"  Avg Transfer (misclass): {np.mean(transfer_results)*100:.1f}%")
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/ff_ensemble_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 总结
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Attack':<25} {'Avg Transfer (misclass)':<20}")
    print("-"*45)
    for name in attacks.keys():
        avg = results[name]['avg_transfer'] * 100
        print(f"{name:<25} {avg:>15.1f}%")
    
    return results


if __name__ == '__main__':
    results = run_ff_ensemble_experiment()
