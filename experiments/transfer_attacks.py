"""
Transfer Enhancement Attacks

Implements MI-FGSM, DI-FGSM, TI-FGSM and combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import scipy.stats as st


class BaseAttack:
    """Base class for attacks"""
    
    def __init__(self, model: nn.Module, eps: float = 8/255, 
                 alpha: float = 2/255, steps: int = 10):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor,
               targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class FGSM(BaseAttack):
    """Fast Gradient Sign Method (single step)"""
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor,
               targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        images = images.clone().detach()
        images.requires_grad = True
        
        outputs = self.model(images)
        
        if targets is not None:
            # Targeted attack
            loss = F.cross_entropy(outputs, targets)
            self.model.zero_grad()
            loss.backward()
            adv_images = images - self.eps * images.grad.sign()
        else:
            # Untargeted attack
            loss = F.cross_entropy(outputs, labels)
            self.model.zero_grad()
            loss.backward()
            adv_images = images + self.eps * images.grad.sign()
        
        return torch.clamp(adv_images, 0, 1).detach()


class IFGSM(BaseAttack):
    """Iterative FGSM (PGD without random start)"""
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor,
               targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        adv_images = images.clone().detach()
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            
            if targets is not None:
                loss = F.cross_entropy(outputs, targets)
                self.model.zero_grad()
                loss.backward()
                adv_images = adv_images - self.alpha * adv_images.grad.sign()
            else:
                loss = F.cross_entropy(outputs, labels)
                self.model.zero_grad()
                loss.backward()
                adv_images = adv_images + self.alpha * adv_images.grad.sign()
            
            # Project to epsilon ball
            perturbation = torch.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = torch.clamp(images + perturbation, 0, 1).detach()
        
        return adv_images


class MIFGSM(BaseAttack):
    """
    Momentum Iterative FGSM
    
    Paper: Boosting Adversarial Attacks with Momentum (CVPR 2018)
    Key idea: Accumulate gradients with momentum to stabilize update direction
    """
    
    def __init__(self, model: nn.Module, eps: float = 8/255,
                 alpha: float = 2/255, steps: int = 10, decay: float = 1.0):
        super().__init__(model, eps, alpha, steps)
        self.decay = decay
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor,
               targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        adv_images = images.clone().detach()
        momentum = torch.zeros_like(images)
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            
            if targets is not None:
                loss = F.cross_entropy(outputs, targets)
            else:
                loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            grad = adv_images.grad.data
            # Normalize gradient
            grad = grad / (grad.abs().mean(dim=[1,2,3], keepdim=True) + 1e-8)
            # Accumulate momentum
            momentum = self.decay * momentum + grad
            
            if targets is not None:
                adv_images = adv_images - self.alpha * momentum.sign()
            else:
                adv_images = adv_images + self.alpha * momentum.sign()
            
            perturbation = torch.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = torch.clamp(images + perturbation, 0, 1).detach()
        
        return adv_images


class DIFGSM(BaseAttack):
    """
    Diverse Input FGSM
    
    Paper: Improving Transferability of Adversarial Examples with Input Diversity (CVPR 2019)
    Key idea: Apply random transformations to increase gradient diversity
    """
    
    def __init__(self, model: nn.Module, eps: float = 8/255,
                 alpha: float = 2/255, steps: int = 10,
                 diversity_prob: float = 0.5, resize_rate: float = 0.9):
        super().__init__(model, eps, alpha, steps)
        self.diversity_prob = diversity_prob
        self.resize_rate = resize_rate
    
    def input_diversity(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random resizing and padding"""
        if torch.rand(1).item() > self.diversity_prob:
            return x
        
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        
        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]
        
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32).item()
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem + 1, size=(1,), dtype=torch.int32).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem + 1, size=(1,), dtype=torch.int32).item()
        pad_right = w_rem - pad_left
        
        padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0)
        
        return padded
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor,
               targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        adv_images = images.clone().detach()
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            
            # Apply input diversity
            diverse_inputs = self.input_diversity(adv_images)
            outputs = self.model(diverse_inputs)
            
            if targets is not None:
                loss = F.cross_entropy(outputs, targets)
            else:
                loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            grad = adv_images.grad.data
            
            if targets is not None:
                adv_images = adv_images - self.alpha * grad.sign()
            else:
                adv_images = adv_images + self.alpha * grad.sign()
            
            perturbation = torch.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = torch.clamp(images + perturbation, 0, 1).detach()
        
        return adv_images


class TIFGSM(BaseAttack):
    """
    Translation-Invariant FGSM
    
    Paper: Evading Defenses to Transferable Adversarial Examples (CVPR 2019)
    Key idea: Use kernel convolution to smooth gradients
    """
    
    def __init__(self, model: nn.Module, eps: float = 8/255,
                 alpha: float = 2/255, steps: int = 10,
                 kernel_size: int = 5):
        super().__init__(model, eps, alpha, steps)
        self.kernel_size = kernel_size
        self.kernel = self._gkern(kernel_size, 3).to(torch.float32)
    
    def _gkern(self, kernlen: int, nsig: int) -> torch.Tensor:
        """Generate Gaussian kernel"""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
        # Expand for 3 channels
        kernel = kernel.expand(3, 1, -1, -1)
        return kernel
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor,
               targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        adv_images = images.clone().detach()
        kernel = self.kernel.to(images.device)
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            
            if targets is not None:
                loss = F.cross_entropy(outputs, targets)
            else:
                loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            grad = adv_images.grad.data
            
            # Apply Gaussian kernel convolution
            grad = F.conv2d(grad, kernel, padding=self.kernel_size//2, groups=3)
            
            if targets is not None:
                adv_images = adv_images - self.alpha * grad.sign()
            else:
                adv_images = adv_images + self.alpha * grad.sign()
            
            perturbation = torch.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = torch.clamp(images + perturbation, 0, 1).detach()
        
        return adv_images


class MIDIFGSM(BaseAttack):
    """
    MI-DI-FGSM: Combination of Momentum and Diversity
    """
    
    def __init__(self, model: nn.Module, eps: float = 8/255,
                 alpha: float = 2/255, steps: int = 10,
                 decay: float = 1.0, diversity_prob: float = 0.5):
        super().__init__(model, eps, alpha, steps)
        self.decay = decay
        self.diversity_prob = diversity_prob
    
    def input_diversity(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.diversity_prob:
            return x
        
        img_size = x.shape[-1]
        rnd = torch.randint(low=int(img_size * 0.9), high=img_size + 1, size=(1,)).item()
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        
        h_rem = img_size - rnd
        w_rem = img_size - rnd
        pad_top = torch.randint(low=0, high=h_rem + 1, size=(1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem + 1, size=(1,)).item()
        pad_right = w_rem - pad_left
        
        padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0)
        return padded
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor,
               targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        adv_images = images.clone().detach()
        momentum = torch.zeros_like(images)
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            diverse_inputs = self.input_diversity(adv_images)
            outputs = self.model(diverse_inputs)
            
            if targets is not None:
                loss = F.cross_entropy(outputs, targets)
            else:
                loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            grad = adv_images.grad.data
            grad = grad / (grad.abs().mean(dim=[1,2,3], keepdim=True) + 1e-8)
            momentum = self.decay * momentum + grad
            
            if targets is not None:
                adv_images = adv_images - self.alpha * momentum.sign()
            else:
                adv_images = adv_images + self.alpha * momentum.sign()
            
            perturbation = torch.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = torch.clamp(images + perturbation, 0, 1).detach()
        
        return adv_images


# Attack registry
ATTACKS = {
    'fgsm': FGSM,
    'ifgsm': IFGSM,
    'mifgsm': MIFGSM,
    'difgsm': DIFGSM,
    'tifgsm': TIFGSM,
    'midifgsm': MIDIFGSM,
}


def get_attack(name: str, model: nn.Module, **kwargs) -> BaseAttack:
    """Get attack instance by name"""
    if name not in ATTACKS:
        raise ValueError(f"Unknown attack: {name}. Available: {list(ATTACKS.keys())}")
    return ATTACKS[name](model, **kwargs)


# Test
if __name__ == '__main__':
    print("Testing transfer attacks...")
    print(f"Available attacks: {list(ATTACKS.keys())}")
