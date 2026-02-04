"""
特征层梯度相似度分析

核心问题：输出层梯度正交，那中间层呢？
如果中间层梯度更相似，特征层攻击可能是突破口。
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


class FeatureExtractor:
    """提取中间层特征和梯度"""
    
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.gradients = {}
        self.hooks = []
        
    def _get_activation(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook
    
    def _get_gradient(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0]
        return hook
    
    def register_hooks(self, layer_names=None):
        """注册 hooks 到指定层"""
        layer_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_name = f'conv_{layer_idx}'
                self.hooks.append(module.register_forward_hook(self._get_activation(layer_name)))
                self.hooks.append(module.register_full_backward_hook(self._get_gradient(layer_name)))
                layer_idx += 1
        return layer_idx
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}
        self.gradients = {}


def analyze_layer_gradient_similarity(device):
    """分析各层梯度的相似度"""
    print("\n" + "="*70)
    print("FEATURE LAYER GRADIENT SIMILARITY ANALYSIS")
    print("="*70)
    
    # 加载模型
    source_model = get_hub_model('resnet56', pretrained=True, device=device)
    target_model = get_hub_model('vgg16_bn', pretrained=True, device=device)
    source_model.eval()
    target_model.eval()
    
    # 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset = Subset(testset, range(100))
    loader = DataLoader(subset, batch_size=50, shuffle=False)
    
    # 特征提取器
    source_extractor = FeatureExtractor(source_model)
    target_extractor = FeatureExtractor(target_model)
    
    source_layers = source_extractor.register_hooks()
    target_layers = target_extractor.register_hooks()
    
    print(f"Source model (ResNet-56): {source_layers} conv layers")
    print(f"Target model (VGG-16): {target_layers} conv layers")
    
    # 分析梯度
    results = {'early': [], 'middle': [], 'late': [], 'output': []}
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        
        # Source forward + backward
        source_out = source_model(images)
        source_loss = F.cross_entropy(source_out, labels)
        source_model.zero_grad()
        source_loss.backward(retain_graph=True)
        
        source_input_grad = images.grad.clone()
        source_conv_grads = {k: v.clone() for k, v in source_extractor.gradients.items()}
        
        # Target forward + backward
        images.grad = None
        images.requires_grad = True
        target_out = target_model(images)
        target_loss = F.cross_entropy(target_out, labels)
        target_model.zero_grad()
        target_loss.backward()
        
        target_input_grad = images.grad.clone()
        target_conv_grads = {k: v.clone() for k, v in target_extractor.gradients.items()}
        
        # 计算输入层梯度相似度
        source_flat = source_input_grad.view(source_input_grad.size(0), -1)
        target_flat = target_input_grad.view(target_input_grad.size(0), -1)
        input_sim = F.cosine_similarity(source_flat, target_flat, dim=1).mean().item()
        results['output'].append(input_sim)
        
        # 比较对应位置的层（按比例映射）
        source_keys = sorted(source_conv_grads.keys(), key=lambda x: int(x.split('_')[1]))
        target_keys = sorted(target_conv_grads.keys(), key=lambda x: int(x.split('_')[1]))
        
        # 早期层（前 1/4）
        s_early = source_keys[:len(source_keys)//4]
        t_early = target_keys[:len(target_keys)//4]
        
        # 中间层（1/4 - 3/4）
        s_mid = source_keys[len(source_keys)//4:3*len(source_keys)//4]
        t_mid = target_keys[len(target_keys)//4:3*len(target_keys)//4]
        
        # 后期层（后 1/4）
        s_late = source_keys[3*len(source_keys)//4:]
        t_late = target_keys[3*len(target_keys)//4:]
        
        # 计算各区域的梯度相似度（使用全局平均池化后比较）
        def compute_region_similarity(s_keys, t_keys, s_grads, t_grads):
            sims = []
            # 取每个区域的第一层比较
            if s_keys and t_keys:
                s_grad = s_grads[s_keys[0]]
                t_grad = t_grads[t_keys[0]]
                # 全局平均池化到相同维度
                s_pooled = F.adaptive_avg_pool2d(s_grad, 1).view(s_grad.size(0), -1)
                t_pooled = F.adaptive_avg_pool2d(t_grad, 1).view(t_grad.size(0), -1)
                # 截断到相同通道数
                min_c = min(s_pooled.size(1), t_pooled.size(1))
                sim = F.cosine_similarity(s_pooled[:, :min_c], t_pooled[:, :min_c], dim=1).mean().item()
                return sim
            return 0
        
        early_sim = compute_region_similarity(s_early, t_early, source_conv_grads, target_conv_grads)
        mid_sim = compute_region_similarity(s_mid, t_mid, source_conv_grads, target_conv_grads)
        late_sim = compute_region_similarity(s_late, t_late, source_conv_grads, target_conv_grads)
        
        results['early'].append(early_sim)
        results['middle'].append(mid_sim)
        results['late'].append(late_sim)
    
    # 清理
    source_extractor.remove_hooks()
    target_extractor.remove_hooks()
    
    # 汇总
    print("\n" + "-"*50)
    print("GRADIENT SIMILARITY BY LAYER DEPTH")
    print("-"*50)
    print(f"  Early layers (first 1/4):  {np.mean(results['early']):.4f}")
    print(f"  Middle layers (1/4-3/4):   {np.mean(results['middle']):.4f}")
    print(f"  Late layers (last 1/4):    {np.mean(results['late']):.4f}")
    print(f"  Output (input grad):       {np.mean(results['output']):.4f}")
    
    # 结论
    print("\n" + "="*50)
    print("IMPLICATIONS")
    print("="*50)
    
    if np.mean(results['early']) > np.mean(results['late']):
        print("✅ Early layers have HIGHER gradient similarity!")
        print("   → Feature-level attacks on early layers may transfer better")
    else:
        print("⚠️ Early layers don't have higher similarity")
        print("   → Feature-level attacks may not help")
    
    return {
        'early_sim': float(np.mean(results['early'])),
        'middle_sim': float(np.mean(results['middle'])),
        'late_sim': float(np.mean(results['late'])),
        'output_sim': float(np.mean(results['output'])),
    }


def run_feature_attack_experiment(device):
    """特征层攻击实验"""
    print("\n" + "="*70)
    print("FEATURE-LEVEL ATTACK EXPERIMENT")
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
    loader = DataLoader(subset, batch_size=50, shuffle=False)
    
    # 获取中间层
    feature_layers = []
    for name, module in source_model.named_modules():
        if isinstance(module, nn.Conv2d):
            feature_layers.append((name, module))
    
    # 选择早期、中期、后期的代表层
    early_idx = len(feature_layers) // 4
    mid_idx = len(feature_layers) // 2
    late_idx = 3 * len(feature_layers) // 4
    
    attack_layers = {
        'output': None,  # 标准输出层攻击
        'early': feature_layers[early_idx],
        'middle': feature_layers[mid_idx],
        'late': feature_layers[late_idx],
    }
    
    results = {}
    
    for layer_name, layer_info in attack_layers.items():
        print(f"\n--- Attacking at: {layer_name} ---")
        
        all_adv = []
        all_labels = []
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # PGD 攻击
            adv = images.clone().detach()
            eps = 8/255
            alpha = 2/255
            
            for _ in range(10):
                adv.requires_grad = True
                
                if layer_info is None:
                    # 标准输出层攻击
                    out = source_model(adv)
                    loss = F.cross_entropy(out, labels)
                else:
                    # 特征层攻击
                    layer_module = layer_info[1]
                    features = []
                    
                    def hook(module, input, output):
                        features.append(output)
                    
                    handle = layer_module.register_forward_hook(hook)
                    _ = source_model(adv)
                    handle.remove()
                    
                    # 最大化特征扰动
                    feat = features[0]
                    loss = -feat.pow(2).mean()  # 负的 L2 范数，最大化扰动
                
                grad = torch.autograd.grad(loss, adv)[0]
                adv = adv.detach() + alpha * grad.sign()
                delta = torch.clamp(adv - images, -eps, eps)
                adv = torch.clamp(images + delta, 0, 1)
            
            all_adv.append(adv.detach())
            all_labels.append(labels)
        
        all_adv = torch.cat(all_adv)
        all_labels = torch.cat(all_labels)
        
        # 测试
        results[layer_name] = {}
        
        # 白盒
        with torch.no_grad():
            out = source_model(all_adv)
            pred = out.argmax(dim=1)
            misclass = (pred != all_labels).float().mean().item()
        results[layer_name]['whitebox'] = misclass
        print(f"  Whitebox: {misclass*100:.1f}%")
        
        # 迁移
        for target_name, target_model in target_models.items():
            with torch.no_grad():
                out = target_model(all_adv)
                pred = out.argmax(dim=1)
                misclass = (pred != all_labels).float().mean().item()
            results[layer_name][target_name] = misclass
            print(f"  → {target_name}: {misclass*100:.1f}%")
    
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    results = {}
    
    # 1. 层级梯度相似度分析
    results['gradient_similarity'] = analyze_layer_gradient_similarity(device)
    
    # 2. 特征层攻击实验
    results['feature_attack'] = run_feature_attack_experiment(device)
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/feature_layer_analysis_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved")


if __name__ == '__main__':
    main()
