"""
FF (FreezeOut + FGSM) + Negative Strategy 对比实验

实验设计:
- 6 种 negative 策略
- 白盒 + 迁移评估
- CIFAR-10 数据集
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
from typing import Dict, List, Tuple, Optional
import argparse

# 添加父目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from negative_strategies import get_strategy, STRATEGIES
from models import get_model  # 假设有这个模块


class FFAttack:
    """FreezeOut + FGSM 攻击"""
    
    def __init__(self, model: nn.Module, negative_strategy: str = 'random',
                 eps: float = 8/255, alpha: float = 2/255, steps: int = 10,
                 freeze_epochs: int = 3, num_classes: int = 10,
                 strategy_kwargs: dict = None):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.freeze_epochs = freeze_epochs
        self.num_classes = num_classes
        
        # 初始化 negative 策略
        strategy_kwargs = strategy_kwargs or {}
        self.strategy = get_strategy(negative_strategy, 
                                     num_classes=num_classes,
                                     **strategy_kwargs)
        self.strategy_name = negative_strategy
    
    def _get_layer_groups(self) -> List[List[nn.Module]]:
        """获取模型的层组（用于 FreezeOut）"""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                layers.append(module)
        
        # 简单分组：平均分成 freeze_epochs 组
        n = len(layers)
        group_size = max(1, n // self.freeze_epochs)
        groups = []
        for i in range(0, n, group_size):
            groups.append(layers[i:i+group_size])
        return groups
    
    def _freeze_layers(self, layers: List[nn.Module]):
        """冻结指定层"""
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
    
    def _unfreeze_all(self):
        """解冻所有层"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor,
               targeted: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        执行 FF 攻击
        
        Args:
            images: 原始图像 [B, C, H, W]
            labels: 真实标签 [B]
            targeted: 是否为 targeted 攻击
        
        Returns:
            adv_images: 对抗样本
            info: 攻击信息
        """
        self.model.eval()
        device = images.device
        batch_size = images.size(0)
        
        # 获取层组
        layer_groups = self._get_layer_groups()
        
        # 初始化对抗样本
        adv_images = images.clone().detach()
        
        # 获取初始 logits 用于确定目标
        with torch.no_grad():
            init_logits = self.model(images)
        
        # 为每个样本确定目标类别
        targets = []
        for i in range(batch_size):
            target = self.strategy.get_target(init_logits[i], labels[i].item())
            targets.append(target if isinstance(target, int) else target[0])
        targets = torch.tensor(targets, device=device)
        
        # 记录攻击信息
        info = {
            'strategy': self.strategy_name,
            'targets': targets.cpu().tolist(),
            'freeze_stages': [],
            'losses': [],
        }
        
        # FreezeOut 攻击过程
        steps_per_epoch = max(1, self.steps // self.freeze_epochs)
        
        for epoch in range(self.freeze_epochs):
            # 冻结前面的层
            self._unfreeze_all()
            for g in range(epoch):
                if g < len(layer_groups):
                    self._freeze_layers(layer_groups[g])
            
            info['freeze_stages'].append({
                'epoch': epoch,
                'frozen_groups': epoch,
                'active_groups': len(layer_groups) - epoch
            })
            
            # 在当前 freeze 状态下执行 FGSM 步骤
            for step in range(steps_per_epoch):
                adv_images.requires_grad = True
                
                # 前向传播
                outputs = self.model(adv_images)
                
                # 计算 loss（使用策略的 loss 函数）
                total_loss = 0
                for i in range(batch_size):
                    loss = self.strategy.get_loss(
                        outputs[i], 
                        labels[i].item(),
                        targets[i].item() if isinstance(targets[i], torch.Tensor) else targets[i]
                    )
                    total_loss += loss
                total_loss /= batch_size
                
                # 反向传播
                self.model.zero_grad()
                total_loss.backward()
                
                info['losses'].append(total_loss.item())
                
                # FGSM 更新
                grad = adv_images.grad.data
                if targeted:
                    # targeted: 最小化到目标的距离 → 梯度下降
                    adv_images = adv_images - self.alpha * grad.sign()
                else:
                    # untargeted: 最大化原始类别的 loss → 梯度上升
                    adv_images = adv_images + self.alpha * grad.sign()
                
                # 投影到 epsilon ball
                perturbation = torch.clamp(adv_images - images, -self.eps, self.eps)
                adv_images = torch.clamp(images + perturbation, 0, 1).detach()
        
        # 解冻所有层
        self._unfreeze_all()
        
        # 计算最终扰动信息
        perturbation = adv_images - images
        info['perturbation'] = {
            'l2': perturbation.norm(p=2, dim=[1,2,3]).mean().item(),
            'linf': perturbation.abs().max().item(),
        }
        
        return adv_images, info


def evaluate_attack(model: nn.Module, adv_images: torch.Tensor, 
                    true_labels: torch.Tensor, target_labels: torch.Tensor,
                    targeted: bool = True) -> Dict:
    """
    评估攻击效果
    
    Returns:
        results: 包含成功率等指标
    """
    model.eval()
    with torch.no_grad():
        outputs = model(adv_images)
        predictions = outputs.argmax(dim=1)
    
    if targeted:
        # targeted: 预测为目标类别即成功
        success = (predictions == target_labels).float().mean().item()
    else:
        # untargeted: 预测不是真实类别即成功
        success = (predictions != true_labels).float().mean().item()
    
    # 置信度
    probs = F.softmax(outputs, dim=1)
    if targeted:
        target_confidence = probs.gather(1, target_labels.unsqueeze(1)).mean().item()
    else:
        target_confidence = 0
    
    return {
        'success_rate': success,
        'target_confidence': target_confidence,
        'predictions': predictions.cpu().tolist(),
    }


def run_experiment(args):
    """运行实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 使用子集加速实验
    if args.num_samples < len(testset):
        indices = np.random.choice(len(testset), args.num_samples, replace=False)
        testset = Subset(testset, indices)
    
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    # 加载模型
    print("Loading models...")
    
    # 源模型（白盒）
    source_model = get_model(args.source_model).to(device)
    source_model.load_state_dict(
        torch.load(f'checkpoints/{args.source_model}_cifar10.pth', 
                   map_location=device, weights_only=True)
    )
    source_model.eval()
    
    # 目标模型（迁移）
    target_models = {}
    for name in args.target_models:
        model = get_model(name).to(device)
        model.load_state_dict(
            torch.load(f'checkpoints/{name}_cifar10.pth',
                       map_location=device, weights_only=True)
        )
        model.eval()
        target_models[name] = model
    
    # 实验结果
    results = {
        'config': vars(args),
        'timestamp': datetime.now().isoformat(),
        'strategies': {},
    }
    
    # 对每种策略进行实验
    for strategy_name in args.strategies:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy_name}")
        print('='*60)
        
        strategy_results = {
            'whitebox': {'success_rates': [], 'confidences': []},
            'transfer': {name: {'success_rates': [], 'confidences': []} 
                        for name in args.target_models},
            'perturbations': {'l2': [], 'linf': []},
        }
        
        # 创建攻击器
        attacker = FFAttack(
            model=source_model,
            negative_strategy=strategy_name,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            freeze_epochs=args.freeze_epochs,
            num_classes=10,
        )
        
        # 对每个 batch 进行攻击
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            
            # 执行攻击
            adv_images, attack_info = attacker.attack(images, labels, targeted=True)
            targets = torch.tensor(attack_info['targets'], device=device)
            
            # 白盒评估
            whitebox_eval = evaluate_attack(
                source_model, adv_images, labels, targets, targeted=True
            )
            strategy_results['whitebox']['success_rates'].append(whitebox_eval['success_rate'])
            strategy_results['whitebox']['confidences'].append(whitebox_eval['target_confidence'])
            
            # 迁移评估
            for name, model in target_models.items():
                transfer_eval = evaluate_attack(
                    model, adv_images, labels, targets, targeted=True
                )
                strategy_results['transfer'][name]['success_rates'].append(transfer_eval['success_rate'])
                strategy_results['transfer'][name]['confidences'].append(transfer_eval['target_confidence'])
            
            # 扰动统计
            strategy_results['perturbations']['l2'].append(attack_info['perturbation']['l2'])
            strategy_results['perturbations']['linf'].append(attack_info['perturbation']['linf'])
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(testloader)}")
        
        # 计算平均值
        strategy_results['whitebox']['avg_success'] = np.mean(strategy_results['whitebox']['success_rates'])
        strategy_results['whitebox']['avg_confidence'] = np.mean(strategy_results['whitebox']['confidences'])
        
        for name in args.target_models:
            strategy_results['transfer'][name]['avg_success'] = np.mean(
                strategy_results['transfer'][name]['success_rates']
            )
            strategy_results['transfer'][name]['avg_confidence'] = np.mean(
                strategy_results['transfer'][name]['confidences']
            )
        
        strategy_results['perturbations']['avg_l2'] = np.mean(strategy_results['perturbations']['l2'])
        strategy_results['perturbations']['avg_linf'] = np.mean(strategy_results['perturbations']['linf'])
        
        results['strategies'][strategy_name] = strategy_results
        
        # 打印结果
        print(f"\n  Results for {strategy_name}:")
        print(f"    Whitebox Success: {strategy_results['whitebox']['avg_success']*100:.2f}%")
        for name in args.target_models:
            print(f"    Transfer to {name}: {strategy_results['transfer'][name]['avg_success']*100:.2f}%")
        print(f"    Avg L2: {strategy_results['perturbations']['avg_l2']:.4f}")
        print(f"    Avg Linf: {strategy_results['perturbations']['avg_linf']:.4f}")
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    result_path = f"results/ff_negative_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_path}")
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Strategy':<20} {'Whitebox':>10} " + " ".join([f"{n:>12}" for n in args.target_models]))
    print("-"*80)
    for strategy_name in args.strategies:
        r = results['strategies'][strategy_name]
        whitebox = f"{r['whitebox']['avg_success']*100:.1f}%"
        transfers = [f"{r['transfer'][n]['avg_success']*100:.1f}%" for n in args.target_models]
        print(f"{strategy_name:<20} {whitebox:>10} " + " ".join([f"{t:>12}" for t in transfers]))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='FF + Negative Strategy Experiment')
    
    # 攻击参数
    parser.add_argument('--eps', type=float, default=8/255, help='Perturbation bound')
    parser.add_argument('--alpha', type=float, default=2/255, help='Step size')
    parser.add_argument('--steps', type=int, default=10, help='Number of attack steps')
    parser.add_argument('--freeze-epochs', type=int, default=3, help='Number of freeze epochs')
    
    # 模型参数
    parser.add_argument('--source-model', type=str, default='resnet18', help='Source model')
    parser.add_argument('--target-models', nargs='+', default=['vgg16', 'densenet121'],
                       help='Target models for transfer evaluation')
    
    # 策略参数
    parser.add_argument('--strategies', nargs='+', 
                       default=['random', 'least_likely', 'most_confusing', 
                               'semantic', 'multi_target', 'dynamic_topk'],
                       help='Negative strategies to test')
    
    # 实验参数
    parser.add_argument('--num-samples', type=int, default=500, help='Number of test samples')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    results = run_experiment(args)
    return results


if __name__ == '__main__':
    main()
