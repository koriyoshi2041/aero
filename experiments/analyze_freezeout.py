"""
FreezeOut Stage Analysis

Analyze how different freeze stages affect transferability
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
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hub_models import get_hub_model
from negative_strategies import get_strategy


class FreezeOutAttacker:
    """FreezeOut attack with stage-wise tracking"""
    
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, 
                 freeze_epochs=5, decay=1.0):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.freeze_epochs = freeze_epochs
        self.decay = decay
        
        # Get layer groups
        self.layer_groups = self._get_layer_groups()
        print(f"  Found {len(self.layer_groups)} layer groups for FreezeOut")
    
    def _get_layer_groups(self):
        """Group layers for progressive freezing"""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append((name, module))
        
        # Divide into groups
        n = len(layers)
        group_size = max(1, n // self.freeze_epochs)
        groups = []
        for i in range(0, n, group_size):
            groups.append(layers[i:i+group_size])
        
        return groups
    
    def _freeze_groups(self, num_groups):
        """Freeze first num_groups groups"""
        for i, group in enumerate(self.layer_groups[:num_groups]):
            for name, module in group:
                for param in module.parameters():
                    param.requires_grad = False
    
    def _unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def attack_with_tracking(self, images, labels, targets):
        """
        Attack and track intermediate results at each freeze stage
        
        Returns:
            final_adv: Final adversarial images
            stage_results: Results at each freeze stage
        """
        self.model.eval()
        device = images.device
        
        adv_images = images.clone().detach()
        momentum = torch.zeros_like(images)
        
        stage_results = []
        steps_per_stage = max(1, self.steps // self.freeze_epochs)
        
        for stage in range(self.freeze_epochs):
            # Configure freezing for this stage
            self._unfreeze_all()
            self._freeze_groups(stage)
            
            # Attack steps for this stage
            for step in range(steps_per_stage):
                adv_images.requires_grad = True
                outputs = self.model(adv_images)
                loss = F.cross_entropy(outputs, targets)
                
                self.model.zero_grad()
                loss.backward()
                
                grad = adv_images.grad.data
                grad = grad / (grad.abs().mean(dim=[1,2,3], keepdim=True) + 1e-8)
                momentum = self.decay * momentum + grad
                
                adv_images = adv_images - self.alpha * momentum.sign()
                perturbation = torch.clamp(adv_images - images, -self.eps, self.eps)
                adv_images = torch.clamp(images + perturbation, 0, 1).detach()
            
            # Record intermediate result
            stage_results.append({
                'stage': stage,
                'frozen_groups': stage,
                'active_groups': len(self.layer_groups) - stage,
                'adv_images': adv_images.clone(),
            })
        
        self._unfreeze_all()
        return adv_images, stage_results


def evaluate(model, adv_images, true_labels, target_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)
    
    return {
        'target_success': (preds == target_labels).float().mean().item(),
        'misclass_rate': (preds != true_labels).float().mean().item(),
    }


def run_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    # Config
    config = {
        'source_model': 'resnet56',
        'target_models': ['vgg16_bn', 'mobilenetv2_x1_0', 'shufflenetv2_x1_0'],
        'freeze_epochs': 5,
        'eps': 8/255,
        'alpha': 2/255,
        'steps': 15,
        'num_samples': 200,
        'batch_size': 50,
    }
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    indices = np.random.choice(len(testset), config['num_samples'], replace=False)
    subset = Subset(testset, indices)
    loader = DataLoader(subset, batch_size=config['batch_size'], shuffle=False)
    
    # Models
    print("\nLoading models...")
    source_model = get_hub_model(config['source_model'], pretrained=True, device=device)
    source_model.eval()
    
    target_models = {}
    for name in config['target_models']:
        model = get_hub_model(name, pretrained=True, device=device)
        model.eval()
        target_models[name] = model
    
    # Strategy
    strategy = get_strategy('most_confusing')
    
    # Attacker
    attacker = FreezeOutAttacker(
        source_model,
        eps=config['eps'],
        alpha=config['alpha'],
        steps=config['steps'],
        freeze_epochs=config['freeze_epochs'],
    )
    
    # Collect results per stage
    stage_whitebox = {i: [] for i in range(config['freeze_epochs'])}
    stage_transfer = {
        name: {i: [] for i in range(config['freeze_epochs'])}
        for name in config['target_models']
    }
    
    print("\n" + "="*60)
    print("Running FreezeOut Stage Analysis...")
    print("="*60)
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        # Get targets
        with torch.no_grad():
            logits = source_model(images)
        targets = torch.tensor([
            strategy.get_target(logits[i], labels[i].item())
            for i in range(len(labels))
        ], device=device)
        
        # Attack with tracking
        _, stage_results = attacker.attack_with_tracking(images, labels, targets)
        
        # Evaluate each stage
        for result in stage_results:
            stage = result['stage']
            adv = result['adv_images']
            
            # Whitebox
            wb = evaluate(source_model, adv, labels, targets)
            stage_whitebox[stage].append(wb['target_success'])
            
            # Transfer
            for name, model in target_models.items():
                tr = evaluate(model, adv, labels, targets)
                stage_transfer[name][stage].append(tr['target_success'])
        
        print(f"  Batch {batch_idx + 1}/{len(loader)}")
    
    # Aggregate results
    results = {
        'config': {k: str(v) if isinstance(v, float) else v for k, v in config.items()},
        'stages': {},
    }
    
    print("\n" + "="*60)
    print("STAGE-WISE RESULTS")
    print("="*60)
    print(f"\n{'Stage':<8} {'Frozen':<8} {'Whitebox':>10} " + 
          " ".join([f"{n:>12}" for n in config['target_models']]))
    print("-" * 70)
    
    for stage in range(config['freeze_epochs']):
        wb_avg = np.mean(stage_whitebox[stage]) * 100
        tr_avgs = {name: np.mean(stage_transfer[name][stage]) * 100 
                   for name in config['target_models']}
        
        results['stages'][stage] = {
            'whitebox': wb_avg,
            'transfer': tr_avgs,
            'avg_transfer': np.mean(list(tr_avgs.values())),
        }
        
        print(f"{stage:<8} {stage:<8} {wb_avg:>9.1f}% " + 
              " ".join([f"{tr_avgs[n]:>11.1f}%" for n in config['target_models']]))
    
    # Save
    os.makedirs('results', exist_ok=True)
    result_path = f"results/freezeout_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_path}")
    
    # Plot
    plot_stage_analysis(results, config['target_models'])
    
    return results


def plot_stage_analysis(results, target_models, save_path='results/freezeout_stage_analysis.png'):
    """Plot stage-wise results"""
    stages = list(results['stages'].keys())
    
    whitebox = [results['stages'][s]['whitebox'] for s in stages]
    avg_transfer = [results['stages'][s]['avg_transfer'] for s in stages]
    transfers = {
        name: [results['stages'][s]['transfer'][name] for s in stages]
        for name in target_models
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: All metrics
    ax1.plot(stages, whitebox, 'o-', label='Whitebox', linewidth=2, markersize=8)
    ax1.plot(stages, avg_transfer, 's-', label='Avg Transfer', linewidth=2, markersize=8)
    
    for name in target_models:
        ax1.plot(stages, transfers[name], '--', label=f'{name}', alpha=0.7)
    
    ax1.set_xlabel('Freeze Stage (# frozen groups)')
    ax1.set_ylabel('Target Success Rate (%)')
    ax1.set_title('FreezeOut: Performance vs Freeze Stage')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Whitebox vs Transfer tradeoff
    ax2.scatter(whitebox, avg_transfer, c=stages, cmap='viridis', s=200, edgecolors='black')
    for i, stage in enumerate(stages):
        ax2.annotate(f'Stage {stage}', (whitebox[i], avg_transfer[i]),
                    textcoords="offset points", xytext=(5, 5))
    
    ax2.set_xlabel('Whitebox Success Rate (%)')
    ax2.set_ylabel('Avg Transfer Success Rate (%)')
    ax2.set_title('Whitebox vs Transfer at Each Stage')
    ax2.grid(alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(min(stages), max(stages)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label='Freeze Stage')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    results = run_analysis()
