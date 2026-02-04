"""
FF 消融实验

分析 FreezeOut 的每个组件贡献多少：
1. 渐进冻结 vs 不冻结
2. 冻结阶段数量
3. 冻结顺序（从前到后 vs 从后到前）
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


class AblationFFAttack:
    """可配置的 FF 攻击，用于消融实验"""
    
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10,
                 freeze_epochs=3, freeze_direction='forward', 
                 use_freezeout=True, device=None):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.freeze_epochs = freeze_epochs
        self.freeze_direction = freeze_direction  # 'forward' or 'backward'
        self.use_freezeout = use_freezeout
        self.device = device
        
        self.layer_groups = self._get_layer_groups()
    
    def _get_layer_groups(self):
        layers = []
        for name, module in self.model.named_modules():
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
    
    def _freeze_groups(self, num_groups):
        if self.freeze_direction == 'forward':
            # 从前往后冻结（标准 FreezeOut）
            for group in self.layer_groups[:num_groups]:
                for module in group:
                    for param in module.parameters():
                        param.requires_grad = False
        else:
            # 从后往前冻结（逆向）
            for group in self.layer_groups[-num_groups:]:
                for module in group:
                    for param in module.parameters():
                        param.requires_grad = False
    
    def _unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def __call__(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        adv = images.clone().detach()
        
        if not self.use_freezeout:
            # 标准 PGD，不使用 FreezeOut
            for _ in range(self.steps):
                adv.requires_grad = True
                outputs = self.model(adv)
                loss = F.cross_entropy(outputs, targets)
                grad = torch.autograd.grad(loss, adv)[0]
                adv = adv.detach() - self.alpha * grad.sign()
                delta = torch.clamp(adv - images, -self.eps, self.eps)
                adv = torch.clamp(images + delta, 0, 1)
        else:
            # FreezeOut
            steps_per_stage = max(1, self.steps // self.freeze_epochs)
            
            for stage in range(self.freeze_epochs):
                self._freeze_groups(stage)
                
                for _ in range(steps_per_stage):
                    adv.requires_grad = True
                    outputs = self.model(adv)
                    loss = F.cross_entropy(outputs, targets)
                    grad = torch.autograd.grad(loss, adv)[0]
                    adv = adv.detach() - self.alpha * grad.sign()
                    delta = torch.clamp(adv - images, -self.eps, self.eps)
                    adv = torch.clamp(images + delta, 0, 1)
            
            self._unfreeze_all()
        
        return adv.detach()


def run_ablation():
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
    
    # 消融配置
    ablation_configs = {
        'No FreezeOut (baseline)': {'use_freezeout': False},
        'FreezeOut (2 stages)': {'use_freezeout': True, 'freeze_epochs': 2},
        'FreezeOut (3 stages)': {'use_freezeout': True, 'freeze_epochs': 3},
        'FreezeOut (5 stages)': {'use_freezeout': True, 'freeze_epochs': 5},
        'FreezeOut (backward)': {'use_freezeout': True, 'freeze_epochs': 3, 'freeze_direction': 'backward'},
    }
    
    strategy = get_strategy('most_confusing')
    results = {}
    
    print("\n" + "="*70)
    print("FF ABLATION STUDY")
    print("="*70)
    
    for config_name, config in ablation_configs.items():
        print(f"\n--- {config_name} ---")
        
        attacker = AblationFFAttack(
            source_model, 
            eps=8/255, 
            alpha=2/255, 
            steps=10,
            device=device,
            **config
        )
        
        all_adv = []
        all_labels = []
        all_targets = []
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                logits = source_model(images)
            targets = torch.tensor([strategy.get_target(logits[i], labels[i].item()) 
                                   for i in range(len(labels))], device=device)
            
            adv_images = attacker(images, targets)
            all_adv.append(adv_images)
            all_labels.append(labels)
            all_targets.append(targets)
        
        all_adv = torch.cat(all_adv)
        all_labels = torch.cat(all_labels)
        all_targets = torch.cat(all_targets)
        
        results[config_name] = {}
        
        # 白盒
        with torch.no_grad():
            out = source_model(all_adv)
            pred = out.argmax(dim=1)
            target_success = (pred == all_targets).float().mean().item()
            misclass = (pred != all_labels).float().mean().item()
        results[config_name]['whitebox'] = {'target': target_success, 'misclass': misclass}
        print(f"  Whitebox: Target={target_success*100:.1f}%, Misclass={misclass*100:.1f}%")
        
        # 迁移
        transfer_results = []
        for target_name, target_model in target_models.items():
            with torch.no_grad():
                out = target_model(all_adv)
                pred = out.argmax(dim=1)
                target_success = (pred == all_targets).float().mean().item()
                misclass = (pred != all_labels).float().mean().item()
            results[config_name][target_name] = {'target': target_success, 'misclass': misclass}
            transfer_results.append(target_success)
            print(f"  → {target_name}: Target={target_success*100:.1f}%")
        
        results[config_name]['avg_transfer'] = np.mean(transfer_results)
        print(f"  Avg Transfer (target): {np.mean(transfer_results)*100:.1f}%")
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/ff_ablation_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 总结
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    print(f"{'Config':<30} {'Whitebox':<15} {'Avg Transfer':<15}")
    print("-"*60)
    for name in ablation_configs.keys():
        wb = results[name]['whitebox']['target'] * 100
        tr = results[name]['avg_transfer'] * 100
        print(f"{name:<30} {wb:>10.1f}%     {tr:>10.1f}%")
    
    return results


if __name__ == '__main__':
    results = run_ablation()
