"""
FF-FGSM: FreezeOut FGSM Attack

Based on TransferAttack framework (https://github.com/Trustworthy-AI-Group/TransferAttack)

Key idea: Progressively freeze early layers during attack iterations.
As layers are frozen, gradients only flow through unfrozen (later) layers,
encouraging perturbations that are more model-agnostic.

Arguments:
    model_name (str): the name of surrogate model for attack.
    epsilon (float): the perturbation budget.
    alpha (float): the step size.
    epoch (int): the number of iterations.
    decay (float): the decay factor for momentum calculation.
    freeze_epochs (int): number of progressive freeze stages.
    targeted (bool): targeted/untargeted attack.
    random_start (bool): whether using random initialization for delta.
    norm (str): the norm of perturbation, l2/linfty.
    loss (str): the loss function.
    device (torch.device): the device for data.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class FFAttack:
    """
    FF-FGSM: FreezeOut FGSM Attack
    
    Progressively freezes layers during attack to improve transferability.
    Based on the hypothesis that perturbations from unfrozen later layers
    are more transferable across different architectures.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        epsilon: float = 16/255,
        alpha: float = 1.6/255,
        epoch: int = 10,
        decay: float = 1.0,
        freeze_epochs: int = 3,
        targeted: bool = False,
        random_start: bool = False,
        norm: str = 'linfty',
        device: Optional[torch.device] = None,
        **kwargs
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.freeze_epochs = freeze_epochs
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        self.device = device or next(model.parameters()).device
        
        # Get layer groups for progressive freezing
        self.layer_groups = self._get_layer_groups()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def _get_layer_groups(self) -> List[List[Tuple[str, nn.Module]]]:
        """
        Group model layers for progressive freezing.
        
        Returns:
            List of layer groups, where each group is a list of (name, module) tuples
        """
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                layers.append((name, module))
        
        # Divide into groups
        n = len(layers)
        if n == 0:
            return []
        
        group_size = max(1, n // self.freeze_epochs)
        groups = []
        for i in range(0, n, group_size):
            groups.append(layers[i:i+group_size])
        
        return groups
    
    def _freeze_groups(self, num_groups: int):
        """Freeze the first num_groups layer groups"""
        for group in self.layer_groups[:num_groups]:
            for name, module in group:
                for param in module.parameters():
                    param.requires_grad = False
    
    def _unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _init_delta(self, data: torch.Tensor) -> torch.Tensor:
        """Initialize perturbation"""
        delta = torch.zeros_like(data, device=self.device)
        if self.random_start:
            if self.norm == 'linfty':
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(0, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0, 1)
                delta *= r / (n + 1e-8) * self.epsilon
            delta = torch.clamp(delta, -data, 1-data)
        delta.requires_grad = True
        return delta
    
    def _update_delta(self, delta: torch.Tensor, data: torch.Tensor, 
                      grad: torch.Tensor) -> torch.Tensor:
        """Update perturbation with projection"""
        if self.norm == 'linfty':
            delta = torch.clamp(delta + self.alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-8)
            delta = delta + scaled_grad * self.alpha
            delta_flat = delta.view(delta.size(0), -1)
            delta = delta_flat.renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        
        delta = torch.clamp(delta, -data, 1-data)
        return delta.detach().requires_grad_(True)
    
    def forward(self, data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Execute FF-FGSM attack
        
        Arguments:
            data (N, C, H, W): input images
            label (N,): ground-truth labels (untargeted) or target labels (targeted)
        
        Returns:
            delta (N, C, H, W): adversarial perturbation
        """
        self.model.eval()
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        # Initialize
        delta = self._init_delta(data)
        momentum = torch.zeros_like(data)
        
        # Calculate iterations per freeze stage
        steps_per_stage = max(1, self.epoch // self.freeze_epochs)
        current_step = 0
        
        for stage in range(self.freeze_epochs):
            # Configure freezing for this stage
            self._unfreeze_all()
            self._freeze_groups(stage)
            
            # Attack steps for this stage
            stage_steps = steps_per_stage
            if stage == self.freeze_epochs - 1:
                # Last stage gets remaining steps
                stage_steps = self.epoch - current_step
            
            for _ in range(stage_steps):
                # Forward pass
                logits = self.model(data + delta)
                
                # Loss
                loss = self.loss_fn(logits, label)
                if not self.targeted:
                    loss = loss  # maximize loss for untargeted
                else:
                    loss = -loss  # minimize loss for targeted
                
                # Backward
                self.model.zero_grad()
                loss.backward()
                
                grad = delta.grad.data
                
                # Momentum
                grad_norm = grad.abs().mean(dim=(1, 2, 3), keepdim=True)
                grad = grad / (grad_norm + 1e-8)
                momentum = self.decay * momentum + grad
                
                # Update
                delta = self._update_delta(delta, data, momentum)
                
                current_step += 1
        
        # Cleanup
        self._unfreeze_all()
        
        return delta.detach()
    
    def __call__(self, data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.forward(data, label)


class FFMIAttack(FFAttack):
    """
    FF-MI-FGSM: FreezeOut + Momentum Attack
    
    Combines FreezeOut progressive freezing with momentum-based gradient accumulation.
    """
    pass  # Same as FFAttack, momentum is built-in


class FFDIAttack(FFAttack):
    """
    FF-DI-FGSM: FreezeOut + Diverse Input Attack
    
    Combines FreezeOut with input diversity transformation.
    """
    
    def __init__(self, *args, diversity_prob: float = 0.5, 
                 resize_rate: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.diversity_prob = diversity_prob
        self.resize_rate = resize_rate
    
    def _input_diversity(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random resizing and padding for input diversity"""
        if torch.rand(1).item() > self.diversity_prob:
            return x
        
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        
        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]
        
        rnd = torch.randint(low=img_size, high=img_resize + 1, size=(1,)).item()
        rescaled = torch.nn.functional.interpolate(
            x, size=[rnd, rnd], mode='bilinear', align_corners=False
        )
        
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem + 1, size=(1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem + 1, size=(1,)).item()
        pad_right = w_rem - pad_left
        
        padded = torch.nn.functional.pad(
            rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0
        )
        
        return padded
    
    def forward(self, data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Execute FF-DI-FGSM attack with input diversity"""
        self.model.eval()
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        delta = self._init_delta(data)
        momentum = torch.zeros_like(data)
        
        steps_per_stage = max(1, self.epoch // self.freeze_epochs)
        current_step = 0
        
        for stage in range(self.freeze_epochs):
            self._unfreeze_all()
            self._freeze_groups(stage)
            
            stage_steps = steps_per_stage
            if stage == self.freeze_epochs - 1:
                stage_steps = self.epoch - current_step
            
            for _ in range(stage_steps):
                # Apply input diversity
                adv_input = self._input_diversity(data + delta)
                logits = self.model(adv_input)
                
                loss = self.loss_fn(logits, label)
                if not self.targeted:
                    loss = loss
                else:
                    loss = -loss
                
                self.model.zero_grad()
                loss.backward()
                
                grad = delta.grad.data
                grad_norm = grad.abs().mean(dim=(1, 2, 3), keepdim=True)
                grad = grad / (grad_norm + 1e-8)
                momentum = self.decay * momentum + grad
                
                delta = self._update_delta(delta, data, momentum)
                current_step += 1
        
        self._unfreeze_all()
        return delta.detach()


# Registry for easy access
FF_ATTACKS = {
    'ff': FFAttack,
    'ff_mi': FFMIAttack,
    'ff_di': FFDIAttack,
}


def get_ff_attack(name: str, model: nn.Module, **kwargs) -> FFAttack:
    """Get FF attack by name"""
    if name not in FF_ATTACKS:
        raise ValueError(f"Unknown attack: {name}. Available: {list(FF_ATTACKS.keys())}")
    return FF_ATTACKS[name](model, **kwargs)
