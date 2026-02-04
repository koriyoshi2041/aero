"""
下载 CIFAR-10 预训练模型

使用 chenyaofo/pytorch-cifar-models 和 kuangliu/pytorch-cifar
"""

import torch
import os
import sys

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def download_from_torch_hub():
    """从 torch.hub 下载预训练模型"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 可用模型列表
    # https://github.com/chenyaofo/pytorch-cifar-models
    models = {
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
    
    print("Downloading models from torch.hub (chenyaofo/pytorch-cifar-models)...")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print()
    
    downloaded = []
    failed = []
    
    for name, hub_name in models.items():
        save_path = os.path.join(CHECKPOINT_DIR, f'{name}_cifar10.pth')
        if os.path.exists(save_path):
            print(f"  [SKIP] {name} - already exists")
            downloaded.append(name)
            continue
        
        try:
            print(f"  [DOWNLOADING] {name}...")
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models", 
                hub_name, 
                pretrained=True,
                verbose=False
            )
            torch.save(model.state_dict(), save_path)
            print(f"  [OK] {name} -> {save_path}")
            downloaded.append(name)
        except Exception as e:
            print(f"  [FAILED] {name}: {e}")
            failed.append(name)
    
    print()
    print(f"Downloaded: {len(downloaded)}, Failed: {len(failed)}")
    if failed:
        print(f"Failed models: {failed}")
    
    return downloaded, failed


def train_custom_models():
    """
    训练自定义模型（ResNet-18, VGG-16, DenseNet-121）
    
    这些模型结构与 torch hub 的不同，需要自己训练
    """
    from models import get_model
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Training device: {device}")
    
    # 数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    models_to_train = ['resnet18', 'vgg16', 'densenet121']
    
    for model_name in models_to_train:
        save_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_cifar10.pth')
        if os.path.exists(save_path):
            print(f"\n[SKIP] {model_name} - already exists")
            continue
        
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print('='*60)
        
        model = get_model(model_name).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
        best_acc = 0
        epochs = 100  # 减少 epochs 加快训练
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            
            # Testing
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            test_acc = 100. * correct / total
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
            
            # Save best
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), save_path)
            
            scheduler.step()
        
        print(f"Best accuracy: {best_acc:.2f}%")
        print(f"Saved to: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hub', action='store_true', help='Download from torch.hub')
    parser.add_argument('--train', action='store_true', help='Train custom models')
    parser.add_argument('--all', action='store_true', help='Download and train all')
    args = parser.parse_args()
    
    if args.all or args.hub:
        download_from_torch_hub()
    
    if args.all or args.train:
        train_custom_models()
    
    if not (args.hub or args.train or args.all):
        print("Usage:")
        print("  python download_models.py --hub    # Download from torch.hub")
        print("  python download_models.py --train  # Train custom models")
        print("  python download_models.py --all    # Both")


if __name__ == '__main__':
    main()
