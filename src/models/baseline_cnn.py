"""
Baseline CNN Sign Classifier Architecture
=========================================
Modular PyTorch Convolutional Neural Network designed for 29-class isolated ASL
fingerspelling and operational token recognition (A-Z, del, nothing, space).

Architectural Highlights:
- 4-Stage Conv-BatchNorm-ReLU-MaxPool feature extractor with progressive channel expansion (3 -> 32 -> 64 -> 128 -> 256).
- Global Adaptive Average Pooling: Decouples spatial input dimensions from classifier head.
- Regularized classification head with Dropout (0.4 and 0.3) and BatchNorm1d to mitigate overfitting.
- Kaiming (He) Normal weight initialization optimized for non-saturating ReLU activations.
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    A single Convolutional Block consisting of:
    Conv2d (3x3, pad 1) -> BatchNorm2d -> ReLU -> MaxPool2d (2x2, stride 2)
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False  # Bias omitted due to subsequent BatchNorm2d
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SignBaselineCNN(nn.Module):
    """
    Baseline CNN Sign Classifier for 29 isolated sign classes.
    
    Args:
        num_classes (int): Number of target gesture categories (default 29).
        in_channels (int): Number of input image channels (default 3 for RGB).
        dropout_rate (float): Classifier dropout probability (default 0.4).
    """
    def __init__(self, num_classes: int = 29, in_channels: int = 3, dropout_rate: float = 0.4):
        super(SignBaselineCNN, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # -------------------------------------------------------------
        # Stage 1: Spatial Feature Extraction Backbone (4 Conv Blocks)
        # -------------------------------------------------------------
        # Input: [B, 3, 64, 64]
        self.block1 = ConvBlock(in_channels, 32, dropout_prob=0.0)      # -> [B, 32, 32, 32]
        self.block2 = ConvBlock(32, 64, dropout_prob=0.0)              # -> [B, 64, 16, 16]
        self.block3 = ConvBlock(64, 128, dropout_prob=0.1)             # -> [B, 128, 8, 8]
        self.block4 = ConvBlock(128, 256, dropout_prob=0.1)            # -> [B, 256, 4, 4]
        
        # -------------------------------------------------------------
        # Stage 2: Spatial Aggregation & Latent Bottleneck
        # -------------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))                # -> [B, 256, 1, 1]
        
        # -------------------------------------------------------------
        # Stage 3: Regularized Dense Classifier Head
        # -------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),                                              # -> [B, 256]
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.75),                        # 0.3 dropout
            nn.Linear(128, num_classes)                               # -> [B, num_classes] (logits)
        )
        
        # Initialize weights with Kaiming He Normal for Conv and Xavier for Linear
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Applies Kaiming (He) normal initialization to all Conv2d layers
        and Xavier normal initialization to all Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing unnormalized class logits.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference forward pass applying softmax to return normalized probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    @staticmethod
    def calculate_receptive_field():
        r"""
        Computes layer-by-layer theoretical receptive field (RF) and feature map strides.
        Formula:
          RF_out = RF_in + (k - 1) * Stride_in
          Stride_out = Stride_in * s
        """
        layers = [
            ("Input", "-", 1, 1, 1),
            ("Block 1 Conv (3x3)", 3, 1, 3, 1),
            ("Block 1 MaxPool (2x2)", 2, 2, 4, 2),
            ("Block 2 Conv (3x3)", 3, 1, 8, 2),
            ("Block 2 MaxPool (2x2)", 2, 2, 10, 4),
            ("Block 3 Conv (3x3)", 3, 1, 18, 4),
            ("Block 3 MaxPool (2x2)", 2, 2, 22, 8),
            ("Block 4 Conv (3x3)", 3, 1, 38, 8),
            ("Block 4 MaxPool (2x2)", 2, 2, 46, 16),
            ("AdaptiveAvgPool (4x4)", 4, 4, 64, 64),
        ]
        return layers

    def summary(self, input_size=(1, 3, 64, 64)):
        """
        Prints a detailed architecture summary with tensor shapes and parameter counts.
        """
        print("=" * 85)
        print("BASELINE CNN SIGN CLASSIFIER ARCHITECTURE SUMMARY")
        print("=" * 85)
        print(f"Target Classes: {self.num_classes} | Input Channels: {self.in_channels}")
        print(f"Input Shape: {input_size}")
        print("-" * 85)
        print(f"{'Layer / Block':<28} | {'Output Shape':<22} | {'Param Count':<12} | {'Receptive Field':<16}")
        print("-" * 85)

        x = torch.zeros(input_size)
        total_params = 0
        trainable_params = 0

        # Layer-by-layer forward tracing
        shapes = []
        
        # Block 1
        x = self.block1(x)
        p1 = sum(p.numel() for p in self.block1.parameters())
        print(f"{'Stage 1: ConvBlock 1':<28} | {str(list(x.shape)):<22} | {p1:<12,d} | {'4 x 4 (s=2)':<16}")
        
        # Block 2
        x = self.block2(x)
        p2 = sum(p.numel() for p in self.block2.parameters())
        print(f"{'Stage 1: ConvBlock 2':<28} | {str(list(x.shape)):<22} | {p2:<12,d} | {'10 x 10 (s=4)':<16}")
        
        # Block 3
        x = self.block3(x)
        p3 = sum(p.numel() for p in self.block3.parameters())
        print(f"{'Stage 1: ConvBlock 3':<28} | {str(list(x.shape)):<22} | {p3:<12,d} | {'22 x 22 (s=8)':<16}")
        
        # Block 4
        x = self.block4(x)
        p4 = sum(p.numel() for p in self.block4.parameters())
        print(f"{'Stage 1: ConvBlock 4':<28} | {str(list(x.shape)):<22} | {p4:<12,d} | {'46 x 46 (s=16)':<16}")
        
        # Global AvgPool
        x = self.global_pool(x)
        print(f"{'Stage 2: AdaptiveAvgPool':<28} | {str(list(x.shape)):<22} | {'0':<12} | {'64 x 64 (s=64)':<16}")
        
        # Classifier
        x = self.classifier(x)
        pc = sum(p.numel() for p in self.classifier.parameters())
        print(f"{'Stage 3: Classifier Head':<28} | {str(list(x.shape)):<22} | {pc:<12,d} | {'Global':<16}")
        
        print("-" * 85)
        for p in self.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
                
        conv_params = p1 + p2 + p3 + p4
        print(f"Total Parameters:      {total_params:,}")
        print(f"Trainable Parameters:  {trainable_params:,}")
        print(f"Conv Backbone Params:  {conv_params:,} ({conv_params/total_params*100:.1f}%)")
        print(f"Classifier Head Params:{pc:,} ({pc/total_params*100:.1f}%)")
        
        # Model memory footprint estimation (FP32 = 4 bytes per param)
        param_memory_mb = (total_params * 4) / (1024 ** 2)
        print(f"Estimated Parameter Size: {param_memory_mb:.2f} MB (Ultra-lightweight for Edge Kiosks)")
        print("=" * 85)


if __name__ == "__main__":
    # Standalone execution and validation
    model = SignBaselineCNN(num_classes=29, in_channels=3)
    model.summary(input_size=(2, 3, 64, 64))
    
    # Forward pass test with synthetic batch
    dummy_input = torch.randn(4, 3, 64, 64)
    logits = model(dummy_input)
    probs = model.predict_probabilities(dummy_input)
    
    print("\nFORWARD PASS VALIDATION:")
    print(f"Input batch shape:       {list(dummy_input.shape)}")
    print(f"Logits output shape:     {list(logits.shape)} (Expected: [4, 29])")
    print(f"Probabilities shape:     {list(probs.shape)}")
    print(f"Probabilities sum check: {probs.sum(dim=1).detach().numpy()} (Expected: [1.0, 1.0, 1.0, 1.0])")
    
    assert logits.shape == (4, 29), f"Shape mismatch: {logits.shape}"
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0, 1.0, 1.0, 1.0])), "Probabilities do not sum to 1.0"
    print("\n[SUCCESS] Baseline CNN architecture verified cleanly!")
