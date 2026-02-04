"""
Ensemble Attack - 集成攻击

使用多个源模型的梯度来生成对抗样本，理论上可以大幅提高迁移率。

参考:
- Liu et al., "Delving into Transferable Adversarial Examples and Black-box Attacks"
- Dong et al., "Boosting Adversarial Attacks with Momentum"
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


class EnsembleAttack:
    """
    Ensemble I-FGSM Attack
    
    使用多个模型的平均梯度进行攻击
    """
    
    def __init__(self, models, eps=8/255, alpha=2/255, steps=10, 
                 momentum=1.0, input_diversity=False, device=None):
        self.models = models
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.momentum = momentum
        self.input_diversity = input_diversity
        self.device = device or next(models[0].parameters()).device
        
    def _input_transform(self, x):
        """Input diversity transformation"""
        if not self.input_diversity:
            return x
        
        # Random resize and pad
        rnd = torch.randint(28, 33, (1,)).item()
        rescaled = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
        
        h_rem = 32 - rnd
        w_rem = 32 - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem + 1, (1,)).item()
        pad_right = w_rem - pad_left
        
        padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom])
        return padded
        
    def __call__(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        adv = images.clone().detach()
        momentum_g = torch.zeros_like(images)
        
        for _ in range(self.steps):
            adv.requires_grad = True
            
            # 计算所有模型的平均梯度
            total_grad = torch.zeros_like(images)
            
            for model in self.models:
                model.eval()
                x = self._input_transform(adv)
                outputs = model(x)
                loss = F.cross_entropy(outputs, targets)
                grad = torch.autograd.grad(loss, adv, retain_graph=False)[0]
                total_grad += grad
            
            # 平均
            avg_grad = total_grad / len(self.models)
            
            # Momentum
            momentum_g = self.momentum * momentum_g + avg_grad / (avg_grad.abs().mean() + 1e-8)
            
            # Update
            adv = adv.detach() - self.alpha * momentum_g.sign()
            
            # Project
            delta = torch.clamp(adv - images, -self.eps, self.eps)
            adv = torch.clamp(images + delta, 0, 1)
        
        return adv.detach()


def run_ensemble_experiment():
    """运行集成攻击实验"""
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    # 加载模型
    print("\nLoading models...")
    
    # 集成源模型（不同架构）
    ensemble_models = [
        get_hub_model('resnet56', pretrained=True, device=device),
        get_hub_model('vgg16_bn', pretrained=True, device=device),
        get_hub_model('mobilenetv2_x1_0', pretrained=True, device=device),
    ]
    for m in ensemble_models:
        m.eval()
    
    # 单一源模型（对照）
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
    
    # 攻击方法
    attacks = {
        'Single (ResNet56)': EnsembleAttack([single_model], eps=8/255, alpha=2/255, steps=10, device=device),
        'Ensemble (3 models)': EnsembleAttack(ensemble_models, eps=8/255, alpha=2/255, steps=10, device=device),
        'Ensemble + MI': EnsembleAttack(ensemble_models, eps=8/255, alpha=2/255, steps=10, momentum=1.0, device=device),
        'Ensemble + MI + DI': EnsembleAttack(ensemble_models, eps=8/255, alpha=2/255, steps=10, momentum=1.0, input_diversity=True, device=device),
    }
    
    strategy = get_strategy('most_confusing')
    results = {}
    
    print("\n" + "="*70)
    print("ENSEMBLE ATTACK COMPARISON")
    print("="*70)
    
    for attack_name, attacker in attacks.items():
        print(f"\n--- {attack_name} ---")
        
        all_adv = []
        all_labels = []
        all_targets = []
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # 使用 ResNet56 计算 targets（保持一致性）
            with torch.no_grad():
                logits = single_model(images)
            targets = torch.tensor([strategy.get_target(logits[i], labels[i].item()) 
                                   for i in range(len(labels))], device=device)
            
            adv_images = attacker(images, targets)
            all_adv.append(adv_images)
            all_labels.append(labels)
            all_targets.append(targets)
        
        all_adv = torch.cat(all_adv)
        all_labels = torch.cat(all_labels)
        all_targets = torch.cat(all_targets)
        
        results[attack_name] = {}
        
        # 测试迁移
        for target_name, target_model in target_models.items():
            with torch.no_grad():
                target_out = target_model(all_adv)
                target_pred = target_out.argmax(dim=1)
                
                target_success = (target_pred == all_targets).float().mean().item()
                misclass = (target_pred != all_labels).float().mean().item()
            
            results[attack_name][target_name] = {
                'target_success': target_success,
                'misclass': misclass,
            }
            
            print(f"  → {target_name}: Target={target_success*100:.1f}%, Misclass={misclass*100:.1f}%")
        
        # 平均
        avg_target = np.mean([v['target_success'] for v in results[attack_name].values()])
        avg_misclass = np.mean([v['misclass'] for v in results[attack_name].values()])
        results[attack_name]['average'] = {
            'target_success': avg_target,
            'misclass': avg_misclass,
        }
        print(f"  Average: Target={avg_target*100:.1f}%, Misclass={avg_misclass*100:.1f}%")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'results/ensemble_attack_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # 总结
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Attack':<25} {'Avg Target Success':<20} {'Avg Misclass':<15}")
    print("-"*60)
    for attack_name in attacks.keys():
        avg_t = results[attack_name]['average']['target_success'] * 100
        avg_m = results[attack_name]['average']['misclass'] * 100
        print(f"{attack_name:<25} {avg_t:>15.1f}%     {avg_m:>10.1f}%")
    
    return results


if __name__ == '__main__':
    results = run_ensemble_experiment()
