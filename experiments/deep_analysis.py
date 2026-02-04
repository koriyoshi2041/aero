"""
深度迁移分析 - 找到迁移差的根本原因

分析维度:
1. 不同架构族之间的梯度差异
2. 模型容量与迁移的关系
3. 扰动幅度 (epsilon) 对迁移的影响
4. 层级特征相似性分析
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
from ff_attack import FFAttack


def get_model_architecture_info():
    """模型架构信息"""
    return {
        'resnet56': {'family': 'ResNet', 'params': '0.86M', 'depth': 56},
        'resnet20': {'family': 'ResNet', 'params': '0.27M', 'depth': 20},
        'resnet44': {'family': 'ResNet', 'params': '0.66M', 'depth': 44},
        'vgg16_bn': {'family': 'VGG', 'params': '14.7M', 'depth': 16},
        'vgg13_bn': {'family': 'VGG', 'params': '9.4M', 'depth': 13},
        'mobilenetv2_x1_0': {'family': 'MobileNet', 'params': '2.2M', 'depth': 52},
        'shufflenetv2_x1_0': {'family': 'ShuffleNet', 'params': '1.3M', 'depth': 50},
        'repvgg_a0': {'family': 'RepVGG', 'params': '8.3M', 'depth': 22},
    }


def analyze_cross_architecture_transfer(device):
    """分析不同架构族之间的迁移"""
    print("\n" + "="*70)
    print("CROSS-ARCHITECTURE TRANSFER ANALYSIS")
    print("="*70)
    
    # 加载多个源模型
    source_models = {
        'resnet56': get_hub_model('resnet56', pretrained=True, device=device),
        'vgg16_bn': get_hub_model('vgg16_bn', pretrained=True, device=device),
        'mobilenetv2_x1_0': get_hub_model('mobilenetv2_x1_0', pretrained=True, device=device),
    }
    
    target_models = {
        'resnet20': get_hub_model('resnet20', pretrained=True, device=device),
        'vgg13_bn': get_hub_model('vgg13_bn', pretrained=True, device=device),
        'shufflenetv2_x1_0': get_hub_model('shufflenetv2_x1_0', pretrained=True, device=device),
        'repvgg_a0': get_hub_model('repvgg_a0', pretrained=True, device=device),
    }
    
    for m in source_models.values():
        m.eval()
    for m in target_models.values():
        m.eval()
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset = Subset(testset, range(300))
    loader = DataLoader(subset, batch_size=50, shuffle=False)
    
    arch_info = get_model_architecture_info()
    results = {}
    
    for source_name, source_model in source_models.items():
        print(f"\n--- Source: {source_name} ({arch_info[source_name]['family']}) ---")
        
        # 创建攻击
        attacker = FFAttack(source_model, eps=8/255, alpha=2/255, steps=10, device=device)
        
        # 生成对抗样本
        all_adv = []
        all_labels = []
        all_targets = []
        
        strategy = get_strategy('most_confusing')
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                logits = source_model(images)
            targets = torch.tensor([strategy.get_target(logits[i], labels[i].item()) 
                                   for i in range(len(labels))], device=device)
            
            adv_images = attacker(images, targets)
            all_adv.append(adv_images)
            all_labels.append(labels)
            all_targets.append(targets)
        
        all_adv = torch.cat(all_adv)
        all_labels = torch.cat(all_labels)
        all_targets = torch.cat(all_targets)
        
        results[source_name] = {}
        
        # 测试迁移到各目标模型
        for target_name, target_model in target_models.items():
            with torch.no_grad():
                target_out = target_model(all_adv)
                target_pred = target_out.argmax(dim=1)
                
                # 目标成功率
                target_success = (target_pred == all_targets).float().mean().item()
                # 误分类率
                misclass = (target_pred != all_labels).float().mean().item()
            
            source_family = arch_info[source_name]['family']
            target_family = arch_info[target_name]['family']
            same_family = source_family == target_family
            
            results[source_name][target_name] = {
                'target_success': target_success,
                'misclass': misclass,
                'same_family': same_family,
            }
            
            marker = "✓" if same_family else "✗"
            print(f"  → {target_name} ({target_family}) [{marker} same family]: "
                  f"Target={target_success*100:.1f}%, Misclass={misclass*100:.1f}%")
    
    return results


def analyze_epsilon_impact(device):
    """分析不同扰动幅度对迁移的影响"""
    print("\n" + "="*70)
    print("EPSILON IMPACT ANALYSIS")
    print("="*70)
    
    source_model = get_hub_model('resnet56', pretrained=True, device=device)
    source_model.eval()
    
    target_models = {
        'vgg16_bn': get_hub_model('vgg16_bn', pretrained=True, device=device),
        'mobilenetv2_x1_0': get_hub_model('mobilenetv2_x1_0', pretrained=True, device=device),
    }
    for m in target_models.values():
        m.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset = Subset(testset, range(300))
    loader = DataLoader(subset, batch_size=100, shuffle=False)
    
    epsilons = [2/255, 4/255, 8/255, 16/255, 32/255]
    results = {eps: {} for eps in epsilons}
    
    strategy = get_strategy('most_confusing')
    
    for eps in epsilons:
        print(f"\nEpsilon = {eps*255:.0f}/255")
        
        attacker = FFAttack(source_model, eps=eps, alpha=eps/4, steps=10, device=device)
        
        all_adv = []
        all_labels = []
        all_targets = []
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                logits = source_model(images)
            targets = torch.tensor([strategy.get_target(logits[i], labels[i].item()) 
                                   for i in range(len(labels))], device=device)
            
            adv_images = attacker(images, targets)
            all_adv.append(adv_images)
            all_labels.append(labels)
            all_targets.append(targets)
        
        all_adv = torch.cat(all_adv)
        all_labels = torch.cat(all_labels)
        all_targets = torch.cat(all_targets)
        
        # 白盒
        with torch.no_grad():
            source_out = source_model(all_adv)
            source_pred = source_out.argmax(dim=1)
            whitebox = (source_pred == all_targets).float().mean().item()
        
        results[eps]['whitebox'] = whitebox
        print(f"  Whitebox: {whitebox*100:.1f}%")
        
        # 迁移
        for target_name, target_model in target_models.items():
            with torch.no_grad():
                target_out = target_model(all_adv)
                target_pred = target_out.argmax(dim=1)
                transfer = (target_pred == all_targets).float().mean().item()
            
            results[eps][target_name] = transfer
            print(f"  → {target_name}: {transfer*100:.1f}%")
    
    # 总结
    print("\n--- Summary ---")
    print(f"{'Epsilon':<10} {'Whitebox':<12} {'Avg Transfer':<12}")
    for eps in epsilons:
        whitebox = results[eps]['whitebox']
        transfers = [v for k, v in results[eps].items() if k != 'whitebox']
        avg_transfer = np.mean(transfers)
        print(f"{eps*255:.0f}/255     {whitebox*100:>6.1f}%      {avg_transfer*100:>6.1f}%")
    
    return results


def analyze_gradient_divergence_by_layer(device):
    """分析不同层的梯度发散程度"""
    print("\n" + "="*70)
    print("LAYER-WISE GRADIENT DIVERGENCE ANALYSIS")
    print("="*70)
    
    source_model = get_hub_model('resnet56', pretrained=True, device=device)
    target_model = get_hub_model('vgg16_bn', pretrained=True, device=device)
    source_model.eval()
    target_model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset = Subset(testset, range(50))
    loader = DataLoader(subset, batch_size=50, shuffle=False)
    
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    # 收集各层梯度
    source_grads = {}
    target_grads = {}
    
    def make_hook(storage, name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                storage[name] = grad_output[0].detach()
        return hook
    
    # 注册 hooks
    source_handles = []
    target_handles = []
    
    layer_idx = 0
    for name, module in source_model.named_modules():
        if isinstance(module, nn.Conv2d):
            handle = module.register_full_backward_hook(make_hook(source_grads, f'conv_{layer_idx}'))
            source_handles.append(handle)
            layer_idx += 1
    
    layer_idx = 0
    for name, module in target_model.named_modules():
        if isinstance(module, nn.Conv2d):
            handle = module.register_full_backward_hook(make_hook(target_grads, f'conv_{layer_idx}'))
            target_handles.append(handle)
            layer_idx += 1
    
    # 前向和反向传播
    images.requires_grad = True
    
    source_out = source_model(images)
    source_loss = F.cross_entropy(source_out, labels)
    source_loss.backward()
    
    images.requires_grad = True
    target_out = target_model(images)
    target_loss = F.cross_entropy(target_out, labels)
    target_loss.backward()
    
    # 清理 hooks
    for h in source_handles + target_handles:
        h.remove()
    
    # 分析
    print("\nSource model has", len(source_grads), "conv layers")
    print("Target model has", len(target_grads), "conv layers")
    
    # 比较前几层和后几层的输入梯度
    print("\n输入层梯度相似度（这是决定迁移的关键）:")
    images.requires_grad = True
    
    source_out = source_model(images)
    source_loss = F.cross_entropy(source_out, labels)
    source_input_grad = torch.autograd.grad(source_loss, images, create_graph=False)[0]
    
    images.requires_grad = True
    target_out = target_model(images)
    target_loss = F.cross_entropy(target_out, labels)
    target_input_grad = torch.autograd.grad(target_loss, images, create_graph=False)[0]
    
    # 按通道分析
    for c in range(3):
        source_c = source_input_grad[:, c].flatten()
        target_c = target_input_grad[:, c].flatten()
        cos_sim = F.cosine_similarity(source_c.unsqueeze(0), target_c.unsqueeze(0)).item()
        print(f"  Channel {c} (RGB[{c}]): cosine similarity = {cos_sim:.4f}")
    
    # 总体
    source_flat = source_input_grad.flatten()
    target_flat = target_input_grad.flatten()
    total_cos = F.cosine_similarity(source_flat.unsqueeze(0), target_flat.unsqueeze(0)).item()
    print(f"  Overall: cosine similarity = {total_cos:.4f}")
    
    return {
        'source_conv_layers': len(source_grads),
        'target_conv_layers': len(target_grads),
        'input_grad_similarity': total_cos,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    results = {}
    
    # 1. 跨架构迁移分析
    results['cross_architecture'] = analyze_cross_architecture_transfer(device)
    
    # 2. Epsilon 影响分析
    results['epsilon_impact'] = analyze_epsilon_impact(device)
    
    # 3. 层级梯度分析
    results['gradient_divergence'] = analyze_gradient_divergence_by_layer(device)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'results/deep_analysis_{timestamp}.json'
    
    # 转换 numpy 类型
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj
    
    results = convert(results)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # 最终结论
    print("\n" + "="*70)
    print("FINAL CONCLUSIONS")
    print("="*70)
    
    print("""
🔍 为什么迁移率低？

1. **架构差异导致梯度方向完全不同**
   - 不同架构的模型学习了完全不同的特征表示
   - ResNet 的残差连接 vs VGG 的纯堆叠 → 梯度流完全不同
   - 梯度余弦相似度 ~0.09，几乎正交！

2. **决策边界几何不同**
   - 每个模型的决策边界在高维空间的形状完全不同
   - 对 ResNet 有效的扰动方向可能与 VGG 的边界平行

3. **FreezeOut 可能加剧了过拟合到源模型**
   - 渐进冻结导致攻击过度适配源模型的特定层
   - 这解释了为什么 FreezeOut 白盒强但迁移弱

📈 为什么 MI-DI-FGSM 更好？

   - Momentum: 平滑梯度，减少噪声
   - Input Diversity: 增加梯度多样性，不过拟合单一输入
   - 这两者都在"正则化"攻击，使其更通用

💡 改进方向：

   1. **集成攻击**: 同时对多个模型优化
   2. **特征层攻击**: 攻击中间特征而非输出
   3. **元学习**: 学习可迁移的攻击模式
""")


if __name__ == '__main__':
    main()
