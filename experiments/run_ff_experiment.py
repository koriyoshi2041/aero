"""
FF (FreezeOut + FGSM) + Negative Strategy 实验

使用 torch.hub 预训练模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from negative_strategies import get_strategy, STRATEGIES
from hub_models import get_hub_model, list_hub_models, get_model_accuracy


class FFAttacker:
    """FreezeOut + FGSM 攻击器"""
    
    def __init__(self, model: nn.Module, strategy_name: str = 'random',
                 eps: float = 8/255, alpha: float = 2/255, steps: int = 10,
                 freeze_epochs: int = 3, num_classes: int = 10):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.freeze_epochs = freeze_epochs
        self.num_classes = num_classes
        self.strategy = get_strategy(strategy_name, num_classes=num_classes)
        self.strategy_name = strategy_name
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行 targeted FF 攻击
        
        Returns:
            adv_images: 对抗样本
            targets: 目标标签
        """
        self.model.eval()
        device = images.device
        batch_size = images.size(0)
        
        # 获取初始 logits 用于确定目标
        with torch.no_grad():
            init_logits = self.model(images)
        
        # 为每个样本确定目标类别
        targets = []
        for i in range(batch_size):
            target = self.strategy.get_target(init_logits[i], labels[i].item())
            targets.append(target if isinstance(target, int) else target[0])
        targets = torch.tensor(targets, device=device)
        
        # 初始化对抗样本
        adv_images = images.clone().detach()
        
        # FreezeOut 简化版：分阶段攻击
        steps_per_epoch = max(1, self.steps // self.freeze_epochs)
        
        for epoch in range(self.freeze_epochs):
            for step in range(steps_per_epoch):
                adv_images.requires_grad = True
                outputs = self.model(adv_images)
                
                # Targeted attack: 最小化到目标类别的 CE loss
                loss = F.cross_entropy(outputs, targets)
                
                self.model.zero_grad()
                loss.backward()
                
                grad = adv_images.grad.data
                # Targeted: 梯度下降
                adv_images = adv_images - self.alpha * grad.sign()
                
                # 投影到 epsilon ball
                perturbation = torch.clamp(adv_images - images, -self.eps, self.eps)
                adv_images = torch.clamp(images + perturbation, 0, 1).detach()
        
        return adv_images, targets


def evaluate(model: nn.Module, adv_images: torch.Tensor, 
             true_labels: torch.Tensor, target_labels: torch.Tensor) -> Dict:
    """评估攻击效果"""
    model.eval()
    with torch.no_grad():
        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)
    
    # Targeted 成功率
    target_success = (preds == target_labels).float().mean().item()
    # Untargeted 成功率（误分类率）
    misclass_rate = (preds != true_labels).float().mean().item()
    # 目标类别置信度
    probs = F.softmax(outputs, dim=1)
    target_conf = probs.gather(1, target_labels.unsqueeze(1)).mean().item()
    
    return {
        'target_success': target_success,
        'misclass_rate': misclass_rate,
        'target_confidence': target_conf,
    }


def run_experiment():
    """运行完整实验"""
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    # 实验配置
    config = {
        'source_model': 'resnet56',  # 白盒模型
        'target_models': ['vgg16_bn', 'mobilenetv2_x1_0', 'shufflenetv2_x1_0'],  # 迁移目标
        'strategies': ['random', 'least_likely', 'most_confusing', 
                      'semantic', 'multi_target', 'dynamic_topk'],
        'eps': 8/255,
        'alpha': 2/255,
        'steps': 10,
        'freeze_epochs': 3,
        'num_samples': 500,
        'batch_size': 50,
    }
    
    print(f"Config: {config}")
    print()
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 使用子集
    indices = np.random.choice(len(testset), config['num_samples'], replace=False)
    subset = Subset(testset, indices)
    loader = DataLoader(subset, batch_size=config['batch_size'], shuffle=False)
    
    # 加载模型
    print("Loading models...")
    source_model = get_hub_model(config['source_model'], pretrained=True, device=device)
    source_model.eval()
    print(f"  Source: {config['source_model']} (acc={get_model_accuracy(config['source_model']):.2f}%)")
    
    target_models = {}
    for name in config['target_models']:
        model = get_hub_model(name, pretrained=True, device=device)
        model.eval()
        target_models[name] = model
        print(f"  Target: {name} (acc={get_model_accuracy(name):.2f}%)")
    print()
    
    # 结果存储
    results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'strategies': {},
    }
    
    # 对每种策略进行实验
    for strategy_name in config['strategies']:
        print(f"{'='*60}")
        print(f"Strategy: {strategy_name}")
        print('='*60)
        
        attacker = FFAttacker(
            model=source_model,
            strategy_name=strategy_name,
            eps=config['eps'],
            alpha=config['alpha'],
            steps=config['steps'],
            freeze_epochs=config['freeze_epochs'],
        )
        
        # 收集结果
        whitebox_results = []
        transfer_results = {name: [] for name in config['target_models']}
        perturbations = []
        
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            # 执行攻击
            adv_images, targets = attacker.attack(images, labels)
            
            # 白盒评估
            wb_eval = evaluate(source_model, adv_images, labels, targets)
            whitebox_results.append(wb_eval)
            
            # 迁移评估
            for name, model in target_models.items():
                tr_eval = evaluate(model, adv_images, labels, targets)
                transfer_results[name].append(tr_eval)
            
            # 扰动统计
            pert = adv_images - images
            perturbations.append({
                'l2': pert.norm(p=2, dim=[1,2,3]).mean().item(),
                'linf': pert.abs().max().item(),
            })
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)}")
        
        # 计算平均值
        avg_whitebox = {
            'target_success': np.mean([r['target_success'] for r in whitebox_results]),
            'misclass_rate': np.mean([r['misclass_rate'] for r in whitebox_results]),
            'target_confidence': np.mean([r['target_confidence'] for r in whitebox_results]),
        }
        
        avg_transfer = {}
        for name in config['target_models']:
            avg_transfer[name] = {
                'target_success': np.mean([r['target_success'] for r in transfer_results[name]]),
                'misclass_rate': np.mean([r['misclass_rate'] for r in transfer_results[name]]),
                'target_confidence': np.mean([r['target_confidence'] for r in transfer_results[name]]),
            }
        
        avg_pert = {
            'l2': np.mean([p['l2'] for p in perturbations]),
            'linf': np.mean([p['linf'] for p in perturbations]),
        }
        
        results['strategies'][strategy_name] = {
            'whitebox': avg_whitebox,
            'transfer': avg_transfer,
            'perturbation': avg_pert,
        }
        
        # 打印结果
        print(f"\n  Whitebox: target_success={avg_whitebox['target_success']*100:.1f}%, misclass={avg_whitebox['misclass_rate']*100:.1f}%")
        for name in config['target_models']:
            t = avg_transfer[name]
            print(f"  Transfer→{name}: target_success={t['target_success']*100:.1f}%, misclass={t['misclass_rate']*100:.1f}%")
        print(f"  Perturbation: L2={avg_pert['l2']:.4f}, Linf={avg_pert['linf']:.4f}")
        print()
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    result_path = f"results/ff_negative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {result_path}")
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    headers = ['Strategy', 'Whitebox'] + config['target_models']
    print(f"{'Strategy':<18} {'Whitebox':>10} " + " ".join([f"{n:>15}" for n in config['target_models']]))
    print("-"*80)
    
    for strategy_name in config['strategies']:
        r = results['strategies'][strategy_name]
        wb = f"{r['whitebox']['target_success']*100:.1f}%"
        transfers = [f"{r['transfer'][n]['target_success']*100:.1f}%" for n in config['target_models']]
        print(f"{strategy_name:<18} {wb:>10} " + " ".join([f"{t:>15}" for t in transfers]))
    
    return results


if __name__ == '__main__':
    results = run_experiment()
