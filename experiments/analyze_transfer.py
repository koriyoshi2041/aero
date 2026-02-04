"""
迁移性能深度分析

分析 FF 攻击迁移性能低的原因
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hub_models import get_hub_model
from negative_strategies import get_strategy


def analyze_gradient_similarity(source_model, target_models, images, labels, device):
    """
    分析源模型和目标模型梯度的相似性
    
    如果梯度相似，说明攻击方向可以迁移
    """
    results = {}
    
    # 计算源模型梯度
    images.requires_grad = True
    source_out = source_model(images)
    source_loss = F.cross_entropy(source_out, labels)
    source_grad = torch.autograd.grad(source_loss, images)[0]
    
    for name, model in target_models.items():
        # 计算目标模型梯度
        images.requires_grad = True
        target_out = model(images)
        target_loss = F.cross_entropy(target_out, labels)
        target_grad = torch.autograd.grad(target_loss, images)[0]
        
        # 计算梯度余弦相似度
        source_flat = source_grad.view(source_grad.size(0), -1)
        target_flat = target_grad.view(target_grad.size(0), -1)
        
        cos_sim = F.cosine_similarity(source_flat, target_flat, dim=1).mean().item()
        
        # 计算梯度方向一致性
        sign_match = (source_grad.sign() == target_grad.sign()).float().mean().item()
        
        results[name] = {
            'cosine_similarity': cos_sim,
            'sign_match_rate': sign_match,
        }
    
    return results


def analyze_feature_alignment(source_model, target_models, images, device):
    """
    分析源模型和目标模型特征表示的相似性
    
    使用最后一层卷积特征
    """
    results = {}
    
    # 提取源模型特征
    def get_features(model, x):
        features = []
        def hook(module, input, output):
            features.append(output.detach())
        
        # 注册 hook 到最后一个卷积层
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        handle = last_conv.register_forward_hook(hook)
        with torch.no_grad():
            model(x)
        handle.remove()
        
        return features[0] if features else None
    
    source_feat = get_features(source_model, images)
    
    if source_feat is None:
        return {'error': 'Could not extract features'}
    
    for name, model in target_models.items():
        target_feat = get_features(model, images)
        
        if target_feat is not None:
            # CKA (Centered Kernel Alignment) 简化版
            # 使用特征均值的相关性
            source_mean = source_feat.mean(dim=[2, 3]).view(source_feat.size(0), -1)
            target_mean = target_feat.mean(dim=[2, 3]).view(target_feat.size(0), -1)
            
            # 归一化
            source_norm = F.normalize(source_mean, dim=1)
            target_norm = F.normalize(target_mean, dim=1)
            
            # 相关性
            correlation = (source_norm * target_norm).sum(dim=1).mean().item()
            
            results[name] = {
                'feature_correlation': correlation,
                'source_feat_dim': source_feat.shape[1],
                'target_feat_dim': target_feat.shape[1],
            }
        else:
            results[name] = {'error': 'Could not extract features'}
    
    return results


def analyze_decision_boundary_distance(model, images, labels, adv_images, targets, device):
    """
    分析对抗样本到决策边界的距离
    """
    with torch.no_grad():
        clean_logits = model(images)
        adv_logits = model(adv_images)
    
    # 计算 logit margin
    clean_margin = clean_logits.gather(1, labels.unsqueeze(1)) - \
                   clean_logits.gather(1, targets.unsqueeze(1))
    adv_margin = adv_logits.gather(1, labels.unsqueeze(1)) - \
                 adv_logits.gather(1, targets.unsqueeze(1))
    
    return {
        'clean_margin_mean': clean_margin.mean().item(),
        'adv_margin_mean': adv_margin.mean().item(),
        'margin_change': (clean_margin - adv_margin).mean().item(),
    }


def analyze_perturbation_frequency(images, adv_images):
    """
    分析扰动的频率特性
    
    高频扰动可能不容易迁移
    """
    perturbation = adv_images - images
    
    # 2D FFT
    fft = torch.fft.fft2(perturbation)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shift)
    
    # 计算能量分布
    h, w = magnitude.shape[-2:]
    center_h, center_w = h // 2, w // 2
    
    # 低频区域（中心 1/4）
    low_freq_region = magnitude[..., center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
    low_freq_energy = low_freq_region.pow(2).sum().item()
    
    # 总能量
    total_energy = magnitude.pow(2).sum().item()
    
    low_freq_ratio = low_freq_energy / (total_energy + 1e-8)
    
    return {
        'low_freq_ratio': low_freq_ratio,
        'high_freq_ratio': 1 - low_freq_ratio,
    }


def run_transfer_analysis():
    """运行完整的迁移分析"""
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    # 加载模型
    print("\nLoading models...")
    source_model = get_hub_model('resnet56', pretrained=True, device=device)
    source_model.eval()
    
    target_models = {
        'vgg16_bn': get_hub_model('vgg16_bn', pretrained=True, device=device),
        'mobilenetv2_x1_0': get_hub_model('mobilenetv2_x1_0', pretrained=True, device=device),
        'shufflenetv2_x1_0': get_hub_model('shufflenetv2_x1_0', pretrained=True, device=device),
    }
    for model in target_models.values():
        model.eval()
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    subset = Subset(testset, range(100))
    loader = DataLoader(subset, batch_size=50, shuffle=False)
    
    # 分析
    print("\n" + "="*60)
    print("TRANSFER ANALYSIS")
    print("="*60)
    
    all_results = defaultdict(list)
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # 1. 梯度相似性分析
        grad_results = analyze_gradient_similarity(
            source_model, target_models, images.clone(), labels, device
        )
        for name, res in grad_results.items():
            all_results[f'gradient_{name}'].append(res)
        
        # 2. 生成对抗样本
        strategy = get_strategy('most_confusing')
        with torch.no_grad():
            logits = source_model(images)
        targets = torch.tensor([strategy.get_target(logits[i], labels[i].item()) 
                               for i in range(len(labels))], device=device)
        
        # 简单 FGSM 攻击
        eps = 8/255
        images.requires_grad = True
        outputs = source_model(images)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        adv_images = images - eps * images.grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        # 3. 决策边界分析
        for name, model in target_models.items():
            boundary_results = analyze_decision_boundary_distance(
                model, images.detach(), labels, adv_images, targets, device
            )
            all_results[f'boundary_{name}'].append(boundary_results)
        
        # 4. 频率分析
        freq_results = analyze_perturbation_frequency(images.detach(), adv_images)
        all_results['frequency'].append(freq_results)
    
    # 汇总结果
    print("\n1. Gradient Similarity (Source vs Target)")
    print("-" * 50)
    for name in target_models.keys():
        cos_sims = [r['cosine_similarity'] for r in all_results[f'gradient_{name}']]
        sign_matches = [r['sign_match_rate'] for r in all_results[f'gradient_{name}']]
        print(f"  {name}:")
        print(f"    Cosine Similarity: {np.mean(cos_sims):.4f}")
        print(f"    Sign Match Rate: {np.mean(sign_matches)*100:.1f}%")
    
    print("\n2. Decision Boundary Analysis")
    print("-" * 50)
    for name in target_models.keys():
        margins = [r['margin_change'] for r in all_results[f'boundary_{name}']]
        print(f"  {name}: Margin Change = {np.mean(margins):.4f}")
    
    print("\n3. Perturbation Frequency Analysis")
    print("-" * 50)
    low_ratios = [r['low_freq_ratio'] for r in all_results['frequency']]
    print(f"  Low Frequency Ratio: {np.mean(low_ratios)*100:.1f}%")
    print(f"  High Frequency Ratio: {(1-np.mean(low_ratios))*100:.1f}%")
    
    # 分析结论
    print("\n" + "="*60)
    print("ANALYSIS CONCLUSIONS")
    print("="*60)
    
    avg_cos_sim = np.mean([np.mean([r['cosine_similarity'] for r in all_results[f'gradient_{name}']])
                          for name in target_models.keys()])
    
    if avg_cos_sim < 0.3:
        print("⚠️  Low gradient similarity detected!")
        print("   → Models have different gradient directions")
        print("   → Consider using input transformations (DI, TI) to improve")
    
    if np.mean(low_ratios) > 0.5:
        print("✓  Perturbations are mostly low-frequency")
        print("   → Should transfer better")
    else:
        print("⚠️  High-frequency perturbations detected")
        print("   → May not transfer well")
        print("   → Consider low-pass filtering or frequency regularization")
    
    return all_results


if __name__ == '__main__':
    results = run_transfer_analysis()
