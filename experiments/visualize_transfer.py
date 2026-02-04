"""
Visualize transfer attack results
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150


def load_latest_results():
    results_dir = 'results'
    files = [f for f in os.listdir(results_dir) if f.startswith('transfer_attacks_')]
    if not files:
        raise FileNotFoundError("No transfer attack result files found")
    latest = sorted(files)[-1]
    with open(os.path.join(results_dir, latest)) as f:
        return json.load(f), latest


def plot_transfer_comparison(results, save_path='results/transfer_attack_comparison.png'):
    """Compare transfer attack methods"""
    attacks = list(results['attacks'].keys())
    target_models = results['config']['target_models']
    
    # Data
    whitebox = [results['attacks'][a]['whitebox']['target_success'] * 100 for a in attacks]
    avg_transfer = [results['attacks'][a]['avg_transfer_rate'] * 100 for a in attacks]
    
    transfers = {
        model: [results['attacks'][a]['transfer'][model]['target_success'] * 100 for a in attacks]
        for model in target_models
    }
    
    # Plot
    x = np.arange(len(attacks))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Bars
    ax.bar(x - 2*width, whitebox, width, label='Whitebox', color='#2ecc71')
    ax.bar(x - 1*width, avg_transfer, width, label='Avg Transfer', color='#f39c12')
    
    colors = ['#3498db', '#e74c3c', '#9b59b6']
    for i, model in enumerate(target_models):
        ax.bar(x + i*width, transfers[model], width, label=f'{model}', color=colors[i])
    
    ax.set_xlabel('Attack Method')
    ax.set_ylabel('Target Success Rate (%)')
    ax.set_title('Transfer Attack Comparison\n(Using most_confusing target strategy)')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in attacks])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 70)
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, (wb, tr) in enumerate(zip(whitebox, avg_transfer)):
        ax.annotate(f'{wb:.0f}%', xy=(i - 2*width, wb), ha='center', va='bottom', fontsize=8)
        ax.annotate(f'{tr:.0f}%', xy=(i - 1*width, tr), ha='center', va='bottom', fontsize=8, color='#f39c12')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_whitebox_transfer_tradeoff(results, save_path='results/whitebox_transfer_tradeoff.png'):
    """Whitebox vs Transfer tradeoff"""
    attacks = list(results['attacks'].keys())
    
    whitebox = [results['attacks'][a]['whitebox']['target_success'] * 100 for a in attacks]
    avg_transfer = [results['attacks'][a]['avg_transfer_rate'] * 100 for a in attacks]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(attacks)))
    
    for i, attack in enumerate(attacks):
        ax.scatter(whitebox[i], avg_transfer[i], s=200, c=[colors[i]], 
                  label=attack.upper(), alpha=0.8, edgecolors='black')
        ax.annotate(attack.upper(), (whitebox[i], avg_transfer[i]),
                   textcoords="offset points", xytext=(8, 8), fontsize=10)
    
    ax.set_xlabel('Whitebox Success Rate (%)')
    ax.set_ylabel('Average Transfer Success Rate (%)')
    ax.set_title('Whitebox vs Transfer Tradeoff\n(Higher right = better transfer without losing whitebox)')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    
    # Highlight best transfer
    best_idx = np.argmax(avg_transfer)
    ax.annotate('Best Transfer!', (whitebox[best_idx], avg_transfer[best_idx]),
               textcoords="offset points", xytext=(20, -20), fontsize=12,
               arrowprops=dict(arrowstyle='->', color='red'),
               color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    results, filename = load_latest_results()
    print(f"Loaded: {filename}")
    
    plot_transfer_comparison(results)
    plot_whitebox_transfer_tradeoff(results)
    
    # Print summary
    print("\n" + "="*60)
    print("TRANSFER ATTACK RESULTS SUMMARY")
    print("="*60)
    
    attacks = list(results['attacks'].keys())
    for attack in attacks:
        r = results['attacks'][attack]
        print(f"\n{attack.upper()}:")
        print(f"  Whitebox: {r['whitebox']['target_success']*100:.1f}%")
        print(f"  Avg Transfer: {r['avg_transfer_rate']*100:.1f}%")


if __name__ == '__main__':
    main()
