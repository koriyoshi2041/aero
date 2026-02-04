"""
可视化梯度正交性 - 展示迁移差的根本原因
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 10))

# ============================================================
# 1. 梯度相似度对比 (左上)
# ============================================================
ax1 = fig.add_subplot(2, 2, 1)

models = ['VGG-16', 'MobileNetV2', 'ShuffleNetV2']
cosine_sim = [0.086, 0.111, 0.108]
sign_match = [0.521, 0.527, 0.528]

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, cosine_sim, width, label='Cosine Similarity', color='#e74c3c')
bars2 = ax1.bar(x + width/2, sign_match, width, label='Sign Match Rate', color='#3498db')

# 添加随机基线
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Baseline (0.5)')
ax1.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)

ax1.set_ylabel('Similarity Score', fontsize=12)
ax1.set_title('Gradient Similarity: ResNet-56 vs Target Models\n(Source → Target)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend(loc='upper right')
ax1.set_ylim(0, 0.7)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# ============================================================
# 2. 输入层梯度相似度 - 核心发现 (右上)
# ============================================================
ax2 = fig.add_subplot(2, 2, 2)

channels = ['Red (R)', 'Green (G)', 'Blue (B)', 'Overall']
input_grad_sim = [0.0005, 0.0005, 0.0005, 0.0005]

colors = ['#e74c3c', '#27ae60', '#3498db', '#9b59b6']
bars = ax2.bar(channels, input_grad_sim, color=colors, edgecolor='black', linewidth=1.5)

ax2.set_ylabel('Cosine Similarity', fontsize=12)
ax2.set_title('🚨 Input-Level Gradient Similarity\n(ResNet-56 vs VGG-16)', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 0.01)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加注释
ax2.text(0.5, 0.7, '≈ 0 means ORTHOGONAL gradients!\nPerturbations optimized for ResNet\nare random directions for VGG', 
         transform=ax2.transAxes, fontsize=11, va='center', ha='center',
         bbox=dict(boxstyle='round', facecolor='#fff3cd', edgecolor='#ffc107', alpha=0.9))

# ============================================================
# 3. Epsilon vs Transfer - 增大扰动没用 (左下)
# ============================================================
ax3 = fig.add_subplot(2, 2, 3)

epsilons = [2, 4, 8, 16, 32]
whitebox = [9.7, 9.7, 7.7, 8.3, 8.0]
transfer = [10.0, 9.3, 9.0, 8.7, 8.7]

ax3.plot(epsilons, whitebox, 'o-', label='Whitebox', color='#2ecc71', linewidth=2, markersize=8)
ax3.plot(epsilons, transfer, 's-', label='Avg Transfer', color='#e74c3c', linewidth=2, markersize=8)

ax3.set_xlabel('Perturbation Budget ε (/255)', fontsize=12)
ax3.set_ylabel('Success Rate (%)', fontsize=12)
ax3.set_title('Increasing ε Does NOT Improve Transfer\n(Targeted Attack)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.set_ylim(0, 15)
ax3.grid(True, alpha=0.3)

# 添加注释
ax3.annotate('16x larger ε\nbut same transfer rate!', 
            xy=(32, 8.7), xytext=(25, 12),
            arrowprops=dict(arrowstyle='->', color='#e74c3c'),
            fontsize=10, color='#e74c3c')

# ============================================================
# 4. 集成攻击效果 (右下)
# ============================================================
ax4 = fig.add_subplot(2, 2, 4)

methods = ['Single\n(ResNet56)', 'Ensemble\n(3 models)', 'Ensemble\n+ MI', 'Ensemble\n+ MI + DI']
target_success = [17.6, 20.4, 20.4, 20.7]

colors = ['#95a5a6', '#3498db', '#9b59b6', '#27ae60']
bars = ax4.bar(methods, target_success, color=colors, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('Avg Target Success Rate (%)', fontsize=12)
ax4.set_title('Ensemble Attack Improves Transfer\n(Multi-model gradient averaging)', fontsize=13, fontweight='bold')
ax4.set_ylim(0, 25)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax4.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加提升标注
ax4.annotate('+3.1%', xy=(3, 20.7), xytext=(3, 23),
            fontsize=12, ha='center', color='#27ae60', fontweight='bold')
ax4.annotate('', xy=(3, 21.5), xytext=(0, 18.5),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

plt.tight_layout()
plt.savefig('results/gradient_orthogonality_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✓ Saved to results/gradient_orthogonality_analysis.png")

# ============================================================
# 第二张图：核心概念图
# ============================================================
fig2, ax = plt.subplots(figsize=(12, 8))

# 画决策边界示意图
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# 两个模型的决策边界
theta1 = np.pi / 6  # ResNet 方向
theta2 = np.pi / 2.2  # VGG 方向 (几乎正交)

# 决策边界线
x = np.linspace(-3, 3, 100)

# ResNet 决策边界
y1 = np.tan(theta1) * x
ax.plot(x, y1, 'b-', linewidth=3, label='ResNet Decision Boundary')

# VGG 决策边界
y2 = np.tan(theta2) * x
ax.plot(x, y2, 'r-', linewidth=3, label='VGG Decision Boundary')

# 原始样本
ax.scatter([1], [0.5], s=200, c='green', marker='*', zorder=5, label='Original Sample')

# ResNet 的攻击方向 (垂直于 ResNet 边界)
attack_dir = np.array([np.cos(theta1 + np.pi/2), np.sin(theta1 + np.pi/2)]) * 0.8
ax.annotate('', xy=(1 + attack_dir[0], 0.5 + attack_dir[1]), xytext=(1, 0.5),
            arrowprops=dict(arrowstyle='->', color='blue', lw=3))
ax.text(1 + attack_dir[0] + 0.2, 0.5 + attack_dir[1], 'ResNet\nGradient', fontsize=11, color='blue')

# VGG 的最优攻击方向 (垂直于 VGG 边界)
vgg_dir = np.array([np.cos(theta2 + np.pi/2), np.sin(theta2 + np.pi/2)]) * 0.8
ax.annotate('', xy=(1 + vgg_dir[0], 0.5 + vgg_dir[1]), xytext=(1, 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=3))
ax.text(1 + vgg_dir[0] - 0.3, 0.5 + vgg_dir[1] + 0.3, 'VGG\nGradient', fontsize=11, color='red')

# 标注角度
arc = mpatches.Arc((1, 0.5), 0.8, 0.8, angle=0, theta1=np.degrees(theta1 + np.pi/2), 
                   theta2=np.degrees(theta2 + np.pi/2), color='purple', linewidth=2)
ax.add_patch(arc)
ax.text(0.6, 1.2, '≈ 90°', fontsize=14, color='purple', fontweight='bold')

ax.set_xlabel('Feature Dimension 1', fontsize=12)
ax.set_ylabel('Feature Dimension 2', fontsize=12)
ax.set_title('Why Transfer Fails: Gradient Directions are Nearly Orthogonal\n(Conceptual Illustration)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# 添加文字说明
textstr = '''Key Insight:
• ResNet gradient ⊥ VGG gradient (cosine sim ≈ 0)
• Perturbations that cross ResNet's boundary
  may be parallel to VGG's boundary
• This is why increasing ε doesn't help!'''
ax.text(-2.8, -2.5, textstr, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig('results/gradient_orthogonality_concept.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✓ Saved to results/gradient_orthogonality_concept.png")

plt.show()
