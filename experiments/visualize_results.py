"""
实验结果可视化
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150


def load_latest_results():
    """加载最新的实验结果"""
    results_dir = 'results'
    files = [f for f in os.listdir(results_dir) if f.startswith('ff_negative_')]
    if not files:
        raise FileNotFoundError("No result files found")
    
    latest = sorted(files)[-1]
    with open(os.path.join(results_dir, latest)) as f:
        return json.load(f), latest


def plot_strategy_comparison(results, save_path='results/strategy_comparison.png'):
    """策略对比柱状图"""
    strategies = list(results['strategies'].keys())
    target_models = results['config']['target_models']
    
    # 准备数据
    whitebox = [results['strategies'][s]['whitebox']['target_success'] * 100 for s in strategies]
    transfers = {
        model: [results['strategies'][s]['transfer'][model]['target_success'] * 100 for s in strategies]
        for model in target_models
    }
    
    # 绘图
    x = np.arange(len(strategies))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 柱子
    bars1 = ax.bar(x - 1.5*width, whitebox, width, label='Whitebox (ResNet-56)', color='#2ecc71')
    for i, model in enumerate(target_models):
        colors = ['#3498db', '#e74c3c', '#9b59b6']
        ax.bar(x + (i-0.5)*width, transfers[model], width, label=f'Transfer→{model}', color=colors[i])
    
    # 标签
    ax.set_xlabel('Negative Sample Strategy')
    ax.set_ylabel('Target Success Rate (%)')
    ax.set_title('FF Attack: Negative Strategy Comparison\n(Targeted Attack Success Rate)')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_transfer_heatmap(results, save_path='results/transfer_heatmap.png'):
    """迁移成功率热力图"""
    strategies = list(results['strategies'].keys())
    models = ['whitebox'] + results['config']['target_models']
    
    # 准备数据矩阵
    data = []
    for s in strategies:
        row = [results['strategies'][s]['whitebox']['target_success'] * 100]
        for m in results['config']['target_models']:
            row.append(results['strategies'][s]['transfer'][m]['target_success'] * 100)
        data.append(row)
    
    data = np.array(data)
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_xticklabels(['Whitebox\n(ResNet-56)'] + results['config']['target_models'])
    ax.set_yticklabels(strategies)
    
    # 添加数值
    for i in range(len(strategies)):
        for j in range(len(models)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%',
                          ha='center', va='center', color='black', fontsize=10)
    
    ax.set_title('FF Attack: Target Success Rate Heatmap\n(Higher is Better for Attacker)')
    fig.colorbar(im, ax=ax, label='Success Rate (%)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_whitebox_vs_transfer(results, save_path='results/whitebox_vs_transfer.png'):
    """白盒 vs 迁移散点图"""
    strategies = list(results['strategies'].keys())
    target_models = results['config']['target_models']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c', '#9b59b6']
    markers = ['o', 's', '^']
    
    for idx, model in enumerate(target_models):
        whitebox = [results['strategies'][s]['whitebox']['target_success'] * 100 for s in strategies]
        transfer = [results['strategies'][s]['transfer'][model]['target_success'] * 100 for s in strategies]
        
        ax.scatter(whitebox, transfer, c=colors[idx], marker=markers[idx], 
                  s=150, label=f'Transfer→{model}', alpha=0.8)
        
        # 添加策略名标签
        for i, s in enumerate(strategies):
            ax.annotate(s, (whitebox[i], transfer[i]), 
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8, alpha=0.7)
    
    # 对角线
    ax.plot([0, 70], [0, 70], 'k--', alpha=0.3, label='y=x')
    
    ax.set_xlabel('Whitebox Success Rate (%)')
    ax.set_ylabel('Transfer Success Rate (%)')
    ax.set_title('FF Attack: Whitebox vs Transfer Performance\n(Higher Transfer Rate = Better Transferability)')
    ax.legend()
    ax.set_xlim(20, 70)
    ax.set_ylim(0, 25)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_markdown_table(results):
    """生成 Markdown 表格"""
    strategies = list(results['strategies'].keys())
    target_models = results['config']['target_models']
    
    # 表头
    header = "| Strategy | Whitebox | " + " | ".join(target_models) + " |"
    separator = "|" + "---|" * (2 + len(target_models))
    
    rows = [header, separator]
    for s in strategies:
        wb = results['strategies'][s]['whitebox']['target_success'] * 100
        transfers = [results['strategies'][s]['transfer'][m]['target_success'] * 100 for m in target_models]
        row = f"| {s} | {wb:.1f}% | " + " | ".join([f"{t:.1f}%" for t in transfers]) + " |"
        rows.append(row)
    
    return "\n".join(rows)


def main():
    results, filename = load_latest_results()
    print(f"Loaded results from: {filename}")
    
    # 生成可视化
    plot_strategy_comparison(results)
    plot_transfer_heatmap(results)
    plot_whitebox_vs_transfer(results)
    
    # 生成表格
    table = generate_markdown_table(results)
    print("\nMarkdown Table:")
    print(table)
    
    # 保存表格
    with open('results/results_table.md', 'w') as f:
        f.write("# FF Attack Results\n\n")
        f.write("## Target Success Rate\n\n")
        f.write(table)
    print("\nSaved: results/results_table.md")


if __name__ == '__main__':
    main()
