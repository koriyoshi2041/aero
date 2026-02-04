"""
Transfer Enhancement Experiment

Compare FGSM, I-FGSM, MI-FGSM, DI-FGSM, TI-FGSM, MI-DI-FGSM
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transfer_attacks import get_attack, ATTACKS
from hub_models import get_hub_model, get_model_accuracy
from negative_strategies import get_strategy


def evaluate(model, adv_images, true_labels, target_labels):
    """Evaluate attack success"""
    model.eval()
    with torch.no_grad():
        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)
    
    target_success = (preds == target_labels).float().mean().item()
    misclass_rate = (preds != true_labels).float().mean().item()
    
    return {
        'target_success': target_success,
        'misclass_rate': misclass_rate,
    }


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    # Config
    config = {
        'source_model': 'resnet56',
        'target_models': ['vgg16_bn', 'mobilenetv2_x1_0', 'shufflenetv2_x1_0'],
        'attacks': ['ifgsm', 'mifgsm', 'difgsm', 'tifgsm', 'midifgsm'],
        'eps': 8/255,
        'alpha': 2/255,
        'steps': 10,
        'num_samples': 200,
        'batch_size': 25,
        'negative_strategy': 'most_confusing',
    }
    
    print(f"Config: {json.dumps({k: str(v) if isinstance(v, float) else v for k, v in config.items()}, indent=2)}")
    
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
    print(f"  Source: {config['source_model']} (acc={get_model_accuracy(config['source_model']):.2f}%)")
    
    target_models = {}
    for name in config['target_models']:
        model = get_hub_model(name, pretrained=True, device=device)
        model.eval()
        target_models[name] = model
        print(f"  Target: {name} (acc={get_model_accuracy(name):.2f}%)")
    
    # Strategy
    strategy = get_strategy(config['negative_strategy'])
    
    # Results
    results = {
        'config': {k: str(v) if isinstance(v, float) else v for k, v in config.items()},
        'timestamp': datetime.now().isoformat(),
        'attacks': {},
    }
    
    # Run experiments
    for attack_name in config['attacks']:
        print(f"\n{'='*60}")
        print(f"Attack: {attack_name.upper()}")
        print('='*60)
        
        attacker = get_attack(
            attack_name, source_model,
            eps=config['eps'], alpha=config['alpha'], steps=config['steps']
        )
        
        whitebox_results = []
        transfer_results = {name: [] for name in config['target_models']}
        perturbations = []
        
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            # Get targets
            with torch.no_grad():
                logits = source_model(images)
            targets = torch.tensor([
                strategy.get_target(logits[i], labels[i].item())
                for i in range(len(labels))
            ], device=device)
            
            # Attack
            adv_images = attacker.attack(images, labels, targets)
            
            # Evaluate
            wb_eval = evaluate(source_model, adv_images, labels, targets)
            whitebox_results.append(wb_eval)
            
            for name, model in target_models.items():
                tr_eval = evaluate(model, adv_images, labels, targets)
                transfer_results[name].append(tr_eval)
            
            # Perturbation stats
            pert = adv_images - images
            perturbations.append({
                'l2': pert.norm(p=2, dim=[1,2,3]).mean().item(),
                'linf': pert.abs().max().item(),
            })
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)}")
        
        # Aggregate
        avg_whitebox = {
            'target_success': np.mean([r['target_success'] for r in whitebox_results]),
            'misclass_rate': np.mean([r['misclass_rate'] for r in whitebox_results]),
        }
        
        avg_transfer = {}
        for name in config['target_models']:
            avg_transfer[name] = {
                'target_success': np.mean([r['target_success'] for r in transfer_results[name]]),
                'misclass_rate': np.mean([r['misclass_rate'] for r in transfer_results[name]]),
            }
        
        # Average transfer rate
        avg_transfer_rate = np.mean([avg_transfer[n]['target_success'] for n in config['target_models']])
        
        results['attacks'][attack_name] = {
            'whitebox': avg_whitebox,
            'transfer': avg_transfer,
            'avg_transfer_rate': avg_transfer_rate,
            'perturbation': {
                'l2': np.mean([p['l2'] for p in perturbations]),
                'linf': np.mean([p['linf'] for p in perturbations]),
            },
        }
        
        print(f"\n  Whitebox: {avg_whitebox['target_success']*100:.1f}%")
        for name in config['target_models']:
            t = avg_transfer[name]
            print(f"  Transfer→{name}: {t['target_success']*100:.1f}%")
        print(f"  Avg Transfer: {avg_transfer_rate*100:.1f}%")
    
    # Save
    os.makedirs('results', exist_ok=True)
    result_path = f"results/transfer_attacks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_path}")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Attack':<12} {'Whitebox':>10} {'Avg Transfer':>12} " + 
          " ".join([f"{n:>15}" for n in config['target_models']]))
    print("-"*80)
    
    for attack_name in config['attacks']:
        r = results['attacks'][attack_name]
        wb = f"{r['whitebox']['target_success']*100:.1f}%"
        avg_tr = f"{r['avg_transfer_rate']*100:.1f}%"
        transfers = [f"{r['transfer'][n]['target_success']*100:.1f}%" for n in config['target_models']]
        print(f"{attack_name:<12} {wb:>10} {avg_tr:>12} " + " ".join([f"{t:>15}" for t in transfers]))
    
    return results


if __name__ == '__main__':
    results = run_experiment()
