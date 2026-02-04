"""
最佳组合实验

测试所有增强技术的组合：
- MI (Momentum)
- DI (Input Diversity)
- TI (Translation Invariance)
- Ensemble
- FF (FreezeOut) - 虽然消融显示没用，但测试组合效果
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
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hub_models import get_hub_model


class CombinedAttack:
    """组合多种增强技术的攻击"""
    
    def __init__(self, models, eps=8/255, alpha=2/255, steps=10,
                 use_momentum=True, use_di=True, use_ti=False,
                 momentum=1.0, di_prob=0.5, device=None):
        self.models = models if isinstance(models, list) else [models]
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.use_momentum = use_momentum
        self.use_di = use_di
        self.use_ti = use_ti
        self.momentum = momentum
        self.di_prob = di_prob
        self.device = device
        
        # TI kernel
        if use_ti:
            self.ti_kernel = self._get_ti_kernel()
    
    def _get_ti_kernel(self, kernel_size=5):
        """高斯核用于 TI"""
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        return kernel.to(self.device)
    
    def _input_diversity(self, x):
        """DI 变换"""
        if not self.use_di or np.random.random() > self.di_prob:
            return x
        
        rnd = np.random.randint(28, 33)
        rescaled = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
        
        h_rem = 32 - rnd
        w_rem = 32 - rnd
        pad_top = np.random.randint(0, h_rem + 1)
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem + 1)
        pad_right = w_rem - pad_left
        
        padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom])
        return padded
    
    def _apply_ti(self, grad):
        """TI 平滑"""
        if not self.use_ti:
            return grad
        
        # 对每个通道应用卷积
        grad_smooth = F.conv2d(
            grad, self.ti_kernel.expand(3, 1, -1, -1),
            padding=2, groups=3
        )
        return grad_smooth
    
    def __call__(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        adv = images.clone().detach()
        momentum_g = torch.zeros_like(images)
        
        for _ in range(self.steps):
            adv.requires_grad = True
            
            # 计算集成梯度
            total_grad = torch.zeros_like(images)
            
            for model in self.models:
                model.eval()
                x = self._input_diversity(adv)
                outputs = model(x)
                loss = F.cross_entropy(outputs, labels)
                grad = torch.autograd.grad(loss, adv, retain_graph=False)[0]
                
                # TI
                grad = self._apply_ti(grad)
                
                total_grad += grad
            
            avg_grad = total_grad / len(self.models)
            
            # Momentum
            if self.use_momentum:
                momentum_g = self.momentum * momentum_g + avg_grad / (avg_grad.abs().mean() + 1e-8)
                update = momentum_g.sign()
            else:
                update = avg_grad.sign()
            
            adv = adv.detach() + self.alpha * update  # + for untargeted
            delta = torch.clamp(adv - images, -self.eps, self.eps)
            adv = torch.clamp(images + delta, 0, 1)
        
        return adv.detach()


def run_combination_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    print("\nLoading models...")
    
    # 单模型
    single_model = get_hub_model('resnet56', pretrained=True, device=device)
    single_model.eval()
    
    # 集成模型
    ensemble_models = [
        get_hub_model('resnet56', pretrained=True, device=device),
        get_hub_model('vgg16_bn', pretrained=True, device=device),
        get_hub_model('mobilenetv2_x1_0', pretrained=True, device=device),
    ]
    for m in ensemble_models:
        m.eval()
    
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
    
    # 测试的组合
    configs = {
        'I-FGSM (baseline)': {
            'models': [single_model],
            'use_momentum': False,
            'use_di': False,
            'use_ti': False,
        },
        'MI-FGSM': {
            'models': [single_model],
            'use_momentum': True,
            'use_di': False,
            'use_ti': False,
        },
        'DI-FGSM': {
            'models': [single_model],
            'use_momentum': False,
            'use_di': True,
            'use_ti': False,
        },
        'MI-DI-FGSM': {
            'models': [single_model],
            'use_momentum': True,
            'use_di': True,
            'use_ti': False,
        },
        'MI-TI-FGSM': {
            'models': [single_model],
            'use_momentum': True,
            'use_di': False,
            'use_ti': True,
        },
        'MI-DI-TI-FGSM': {
            'models': [single_model],
            'use_momentum': True,
            'use_di': True,
            'use_ti': True,
        },
        'Ensemble': {
            'models': ensemble_models,
            'use_momentum': False,
            'use_di': False,
            'use_ti': False,
        },
        'Ensemble + MI': {
            'models': ensemble_models,
            'use_momentum': True,
            'use_di': False,
            'use_ti': False,
        },
        'Ensemble + MI + DI': {
            'models': ensemble_models,
            'use_momentum': True,
            'use_di': True,
            'use_ti': False,
        },
        'Ensemble + MI + DI + TI': {
            'models': ensemble_models,
            'use_momentum': True,
            'use_di': True,
            'use_ti': True,
        },
    }
    
    results = {}
    
    print("\n" + "="*70)
    print("COMBINATION EXPERIMENT")
    print("="*70)
    
    for config_name, config in configs.items():
        print(f"\n--- {config_name} ---")
        
        attacker = CombinedAttack(
            models=config['models'],
            eps=8/255, alpha=2/255, steps=10,
            use_momentum=config['use_momentum'],
            use_di=config['use_di'],
            use_ti=config['use_ti'],
            device=device
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
        
        results[config_name] = {}
        
        # 白盒（用 ResNet56）
        with torch.no_grad():
            out = single_model(all_adv)
            pred = out.argmax(dim=1)
            misclass = (pred != all_labels).float().mean().item()
        results[config_name]['whitebox'] = misclass
        print(f"  Whitebox: {misclass*100:.1f}%")
        
        # 迁移
        transfer_results = []
        for target_name, target_model in target_models.items():
            with torch.no_grad():
                out = target_model(all_adv)
                pred = out.argmax(dim=1)
                misclass = (pred != all_labels).float().mean().item()
            results[config_name][target_name] = misclass
            transfer_results.append(misclass)
            print(f"  → {target_name}: {misclass*100:.1f}%")
        
        results[config_name]['avg_transfer'] = np.mean(transfer_results)
        print(f"  Avg Transfer: {np.mean(transfer_results)*100:.1f}%")
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/combination_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 排名
    print("\n" + "="*70)
    print("RANKING BY TRANSFER RATE")
    print("="*70)
    sorted_configs = sorted(results.items(), key=lambda x: x[1]['avg_transfer'], reverse=True)
    print(f"{'Rank':<6} {'Method':<25} {'Whitebox':<12} {'Transfer':<12}")
    print("-"*55)
    for i, (name, res) in enumerate(sorted_configs, 1):
        wb = res['whitebox'] * 100
        tr = res['avg_transfer'] * 100
        print(f"{i:<6} {name:<25} {wb:>8.1f}%    {tr:>8.1f}%")
    
    return results


if __name__ == '__main__':
    results = run_combination_experiment()
