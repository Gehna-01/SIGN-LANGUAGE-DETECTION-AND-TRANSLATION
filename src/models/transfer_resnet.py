"""
Advanced Vision Model: Transfer Learning with Pretrained ResNet-18
==================================================================
Transfer Learning vision architecture for 29-class Sign Language Recognition.

Key Architectural Characteristics:
- Backbone: ResNet-18 pretrained on ImageNet-1K (11.2M parameters).
- Residual Skip Connections: F(x) + x identity mappings solve vanishing gradients and feature degradation across deep layers.
- Custom Sign Classification Head: Replaces 1000-class ImageNet head with a regularized two-stage projection:
    Linear(512 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(128 -> 29).
- Flexible Fine-Tuning Regimes: Supports full fine-tuning, linear probing (frozen backbone), and gradual unfreezing (freeze layers 1-3, tune layer 4 + head).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class SignResNet18Transfer(nn.Module):
    """
    Pretrained ResNet-18 Sign Classifier for 29-class ASL recognition.
    
    Args:
        num_classes (int): Number of target categories (default: 29).
        pretrained (bool): Load ImageNet-1K pretrained weights (default: True).
        fine_tune_mode (str): One of:
            - 'full': Fine-tune all layers.
            - 'freeze_all': Freeze entire backbone, train classifier head only.
            - 'freeze_shallow': Freeze conv1, bn1, layer1, layer2, layer3; fine-tune layer4 + head.
        dropout_rate (float): Bottleneck dropout probability (default: 0.4).
    """
    def __init__(
        self,
        num_classes: int = 29,
        pretrained: bool = True,
        fine_tune_mode: str = "freeze_shallow",
        dropout_rate: float = 0.4
    ):
        super(SignResNet18Transfer, self).__init__()
        self.num_classes = num_classes
        self.fine_tune_mode = fine_tune_mode
        self.dropout_rate = dropout_rate
        
        # 1. Load Pretrained ResNet-18 Backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # 2. Extract Feature Dimensionality before original FC
        in_features = self.backbone.fc.in_features  # 512
        
        # 3. Replace ImageNet FC with Custom Regularized Sign Classification Head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.75),
            nn.Linear(128, num_classes)
        )
        
        # Initialize the newly added custom head weights
        self._initialize_head()
        
        # 4. Apply Layer Freezing Strategy
        self.set_fine_tune_mode(fine_tune_mode)

    def _initialize_head(self):
        """
        Initializes the new classifier head layers using Xavier Normal.
        """
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def set_fine_tune_mode(self, mode: str):
        """
        Configures layer gradient requirements based on fine-tuning regime.
        """
        self.fine_tune_mode = mode
        
        if mode == "full":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif mode == "freeze_all":
            # Freeze all parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze only the custom head
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif mode == "freeze_shallow":
            # Freeze initial stem and shallow residual blocks
            shallow_modules = [
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3
            ]
            for module in shallow_modules:
                for param in module.parameters():
                    param.requires_grad = False
                    
            # Train layer4 (high-level semantic patterns) and the classification head
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown fine_tune_mode: {mode}. Choose 'full', 'freeze_all', or 'freeze_shallow'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized class logits [B, 29].
        """
        return self.backbone(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference pass returning normalized class probabilities [B, 29].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def summary(self, input_size=(1, 3, 64, 64)):
        """
        Prints a detailed architecture and parameter status summary.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        head_params = sum(p.numel() for p in self.backbone.fc.parameters())
        backbone_params = total_params - head_params
        
        print("=" * 85)
        print("ADVANCED VISION MODEL: RESNET-18 TRANSFER LEARNING ARCHITECTURE")
        print("=" * 85)
        print(f"Base Backbone:        Pretrained ResNet-18 (ImageNet-1K)")
        print(f"Fine-Tuning Regime:   {self.fine_tune_mode.upper()}")
        print(f"Target Categories:    {self.num_classes} ASL classes")
        print(f"Input Shape:          {input_size}")
        print("-" * 85)
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"Frozen Parameters:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"Backbone Parameters:  {backbone_params:,}")
        print(f"Custom Head Params:   {head_params:,}")
        print(f"Model Memory Footprint: {(total_params * 4) / (1024**2):.2f} MB (FP32)")
        print("=" * 85)


if __name__ == "__main__":
    # Unit test and verification
    model = SignResNet18Transfer(num_classes=29, pretrained=True, fine_tune_mode="freeze_shallow")
    model.summary(input_size=(2, 3, 64, 64))
    
    dummy_input = torch.randn(4, 3, 64, 64)
    logits = model(dummy_input)
    probs = model.predict_probabilities(dummy_input)
    
    print("\nFORWARD PASS VALIDATION:")
    print(f"Input batch shape:       {list(dummy_input.shape)}")
    print(f"Output logits shape:     {list(logits.shape)} (Expected: [4, 29])")
    print(f"Softmax probabilities:   {list(probs.shape)}")
    print(f"Probabilities sum check: {probs.sum(dim=1).detach().numpy()} (All sum to 1.0)")
    
    assert logits.shape == (4, 29), f"Output shape mismatch: {logits.shape}"
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0, 1.0, 1.0, 1.0])), "Probabilities failed sum check"
    print("\n[SUCCESS] Pretrained ResNet-18 transfer model verified cleanly!")
