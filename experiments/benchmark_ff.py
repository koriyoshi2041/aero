"""
Benchmark FF Attack vs Baselines

Compare FF-FGSM with I-FGSM, MI-FGSM, DI-FGSM on transferability.
Based on TransferAttack framework methodology.
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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hub_models import get_hub_model, get_model_accuracy
from transfer_attacks import IFGSM, MIFGSM, DIFGSM, MIDIFGSM
from ff_attack import FFAttack, FFDIAttack


class BenchmarkRunner:
    """Run comprehensive attack benchmarks"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                   else 'mps' if torch.backends.mps.is_available() 
                                   else 'cpu')
        print(f"Device: {self.device}")
        
        # Load models
        self.source_model = self._load_model(config['source_model'])
        self.target_models = {
            name: self._load_model(name) 
            for name in config['target_models']
        }
        
        # Load data
        self.loader = self._load_data()
        
    def _load_model(self, name: str) -> nn.Module:
        """Load pretrained model"""
        model = get_hub_model(name, pretrained=True, device=self.device)
        model.eval()
        print(f"  Loaded {name} (acc={get_model_accuracy(name):.1f}%)")
        return model
    
    def _load_data(self) -> DataLoader:
        """Load test data"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010)),
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        indices = np.random.choice(len(testset), self.config['num_samples'], replace=False)
        subset = Subset(testset, indices)
        return DataLoader(subset, batch_size=self.config['batch_size'], shuffle=False)
    
    def _evaluate(self, model: nn.Module, adv_images: torch.Tensor, 
                  labels: torch.Tensor) -> dict:
        """Evaluate attack on a model"""
        model.eval()
        with torch.no_grad():
            outputs = model(adv_images)
            preds = outputs.argmax(dim=1)
        
        # Untargeted success = misclassification rate
        success = (preds != labels).float().mean().item()
        return {'success_rate': success}
    
    def _create_attack(self, attack_name: str) -> object:
        """Create attack instance"""
        params = {
            'eps': self.config['eps'],
            'alpha': self.config['alpha'],
            'steps': self.config['steps'],
        }
        
        if attack_name == 'ifgsm':
            return IFGSM(self.source_model, **params)
        elif attack_name == 'mifgsm':
            return MIFGSM(self.source_model, **params, decay=1.0)
        elif attack_name == 'difgsm':
            return DIFGSM(self.source_model, **params, diversity_prob=0.5)
        elif attack_name == 'midifgsm':
            return MIDIFGSM(self.source_model, **params, decay=1.0, diversity_prob=0.5)
        elif attack_name == 'ff':
            return FFAttack(self.source_model, 
                           epsilon=params['eps'], alpha=params['alpha'],
                           epoch=params['steps'], freeze_epochs=3)
        elif attack_name == 'ff_di':
            return FFDIAttack(self.source_model,
                             epsilon=params['eps'], alpha=params['alpha'],
                             epoch=params['steps'], freeze_epochs=3,
                             diversity_prob=0.5)
        else:
            raise ValueError(f"Unknown attack: {attack_name}")
    
    def run(self) -> dict:
        """Run all benchmarks"""
        results = {
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'attacks': {},
        }
        
        for attack_name in self.config['attacks']:
            print(f"\n{'='*60}")
            print(f"Attack: {attack_name.upper()}")
            print('='*60)
            
            attack = self._create_attack(attack_name)
            
            # Collect results
            whitebox_results = []
            transfer_results = defaultdict(list)
            
            for batch_idx, (images, labels) in enumerate(self.loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Generate adversarial examples
                if attack_name.startswith('ff'):
                    # FF attacks return delta
                    delta = attack(images, labels)
                    adv_images = torch.clamp(images + delta, 0, 1)
                else:
                    # Transfer attacks return adv_images
                    adv_images = attack.attack(images, labels)
                
                # Evaluate whitebox
                wb = self._evaluate(self.source_model, adv_images, labels)
                whitebox_results.append(wb['success_rate'])
                
                # Evaluate transfer
                for name, model in self.target_models.items():
                    tr = self._evaluate(model, adv_images, labels)
                    transfer_results[name].append(tr['success_rate'])
                
                if (batch_idx + 1) % 2 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(self.loader)}")
            
            # Aggregate
            avg_whitebox = np.mean(whitebox_results) * 100
            avg_transfer = {name: np.mean(vals) * 100 
                          for name, vals in transfer_results.items()}
            overall_transfer = np.mean(list(avg_transfer.values()))
            
            results['attacks'][attack_name] = {
                'whitebox': avg_whitebox,
                'transfer': avg_transfer,
                'avg_transfer': overall_transfer,
            }
            
            print(f"\n  Whitebox: {avg_whitebox:.1f}%")
            for name in self.config['target_models']:
                print(f"  Transfer→{name}: {avg_transfer[name]:.1f}%")
            print(f"  Avg Transfer: {overall_transfer:.1f}%")
        
        return results
    
    def save_results(self, results: dict, prefix: str = 'benchmark'):
        """Save results to file"""
        os.makedirs('results', exist_ok=True)
        path = f"results/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {path}")
        return path
    
    def print_summary(self, results: dict):
        """Print summary table"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        attacks = list(results['attacks'].keys())
        targets = self.config['target_models']
        
        # Header
        print(f"\n{'Attack':<12} {'Whitebox':>10} {'Avg Trans':>10} " + 
              " ".join([f"{t[:10]:>12}" for t in targets]))
        print("-"*80)
        
        # Data rows
        for attack in attacks:
            r = results['attacks'][attack]
            wb = f"{r['whitebox']:.1f}%"
            avg = f"{r['avg_transfer']:.1f}%"
            trans = [f"{r['transfer'][t]:.1f}%" for t in targets]
            print(f"{attack:<12} {wb:>10} {avg:>10} " + 
                  " ".join([f"{t:>12}" for t in trans]))
        
        # Find best
        best_transfer = max(attacks, key=lambda a: results['attacks'][a]['avg_transfer'])
        print(f"\n🏆 Best Transfer: {best_transfer.upper()} ({results['attacks'][best_transfer]['avg_transfer']:.1f}%)")


def main():
    config = {
        'source_model': 'resnet56',
        'target_models': ['vgg16_bn', 'mobilenetv2_x1_0', 'shufflenetv2_x1_0', 'resnet20'],
        'attacks': ['ifgsm', 'mifgsm', 'difgsm', 'midifgsm', 'ff', 'ff_di'],
        'eps': 8/255,
        'alpha': 2/255,
        'steps': 10,
        'num_samples': 300,
        'batch_size': 50,
    }
    
    print("="*60)
    print("FF Attack Benchmark")
    print("="*60)
    print(f"\nConfig:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    runner = BenchmarkRunner(config)
    results = runner.run()
    runner.save_results(results)
    runner.print_summary(results)
    
    return results


if __name__ == '__main__':
    results = main()
