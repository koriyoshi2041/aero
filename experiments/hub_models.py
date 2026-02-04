"""
加载 torch.hub 预训练模型的包装器

支持 chenyaofo/pytorch-cifar-models 的模型
"""

import torch
import torch.nn as nn
import os

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')

# 模型映射
HUB_MODELS = {
    'resnet20': 'cifar10_resnet20',
    'resnet32': 'cifar10_resnet32',
    'resnet44': 'cifar10_resnet44',
    'resnet56': 'cifar10_resnet56',
    'vgg11_bn': 'cifar10_vgg11_bn',
    'vgg13_bn': 'cifar10_vgg13_bn',
    'vgg16_bn': 'cifar10_vgg16_bn',
    'vgg19_bn': 'cifar10_vgg19_bn',
    'mobilenetv2_x0_5': 'cifar10_mobilenetv2_x0_5',
    'mobilenetv2_x0_75': 'cifar10_mobilenetv2_x0_75',
    'mobilenetv2_x1_0': 'cifar10_mobilenetv2_x1_0',
    'mobilenetv2_x1_4': 'cifar10_mobilenetv2_x1_4',
    'shufflenetv2_x0_5': 'cifar10_shufflenetv2_x0_5',
    'shufflenetv2_x1_0': 'cifar10_shufflenetv2_x1_0',
    'shufflenetv2_x1_5': 'cifar10_shufflenetv2_x1_5',
    'shufflenetv2_x2_0': 'cifar10_shufflenetv2_x2_0',
    'repvgg_a0': 'cifar10_repvgg_a0',
    'repvgg_a1': 'cifar10_repvgg_a1',
    'repvgg_a2': 'cifar10_repvgg_a2',
}

# 模型精度参考（来自 chenyaofo/pytorch-cifar-models）
MODEL_ACCURACY = {
    'resnet20': 92.60,
    'resnet32': 93.53,
    'resnet44': 93.63,
    'resnet56': 94.22,
    'vgg11_bn': 92.79,
    'vgg13_bn': 94.00,
    'vgg16_bn': 94.00,
    'vgg19_bn': 93.91,
    'mobilenetv2_x0_5': 91.85,
    'mobilenetv2_x0_75': 93.03,
    'mobilenetv2_x1_0': 93.79,
    'mobilenetv2_x1_4': 94.23,
    'shufflenetv2_x0_5': 90.05,
    'shufflenetv2_x1_0': 93.40,
    'shufflenetv2_x1_5': 93.47,
    'shufflenetv2_x2_0': 93.71,
    'repvgg_a0': 94.48,
    'repvgg_a1': 94.96,
    'repvgg_a2': 95.21,
}


def get_hub_model(name: str, pretrained: bool = True, device: str = 'cpu') -> nn.Module:
    """
    获取 torch.hub 预训练模型
    
    Args:
        name: 模型名称
        pretrained: 是否加载预训练权重
        device: 设备
    
    Returns:
        model: 模型实例
    """
    if name not in HUB_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(HUB_MODELS.keys())}")
    
    hub_name = HUB_MODELS[name]
    
    # 从 torch.hub 加载模型
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        hub_name,
        pretrained=pretrained,
        verbose=False,
        trust_repo=True
    )
    
    return model.to(device)


def get_local_hub_model(name: str, device: str = 'cpu') -> nn.Module:
    """
    从本地 checkpoint 加载模型
    
    需要先用 download_models.py --hub 下载模型
    """
    if name not in HUB_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(HUB_MODELS.keys())}")
    
    # 先从 hub 获取模型结构
    model = get_hub_model(name, pretrained=False, device='cpu')
    
    # 加载本地权重
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{name}_cifar10.pth')
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}, using random weights")
    
    return model.to(device)


def list_hub_models():
    """列出所有可用的 hub 模型"""
    return list(HUB_MODELS.keys())


def get_model_accuracy(name: str) -> float:
    """获取模型在 CIFAR-10 上的精度"""
    return MODEL_ACCURACY.get(name, 0.0)


# 测试
if __name__ == '__main__':
    import torch
    
    print("Testing hub models...")
    print(f"Available models: {list_hub_models()}")
    print()
    
    x = torch.randn(2, 3, 32, 32)
    
    # 测试几个代表性模型
    for name in ['resnet56', 'vgg16_bn', 'mobilenetv2_x1_0']:
        try:
            model = get_hub_model(name, pretrained=True, device='cpu')
            model.eval()
            with torch.no_grad():
                out = model(x)
            print(f"  {name}: {x.shape} -> {out.shape}, acc={get_model_accuracy(name):.2f}%")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
