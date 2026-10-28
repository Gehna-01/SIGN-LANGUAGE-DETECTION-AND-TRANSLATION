r"""
Activation Functions Comparison and Analysis
=============================================
Implementation, analytical derivatives, saturation analysis, and vanishing
gradient diagnostics for fundamental neural network activation functions:
1. Heaviside Step Function
2. Sigmoid (Logistic) Function
3. Hyperbolic Tangent (Tanh) Function
4. Rectified Linear Unit (ReLU)
5. Leaky Rectified Linear Unit (LeakyReLU)

This module provides from-scratch NumPy implementations, gradient propagation
simulations across deep layers, and automated visualization generator.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Core Activation Functions and Analytical Derivatives
# -------------------------------------------------------------------------

def step_function(z: np.ndarray) -> np.ndarray:
    """
    Heaviside Step Function:
    f(z) = 1 if z >= 0 else 0
    Output range: {0, 1} (discrete binary)
    """
    return np.where(z >= 0.0, 1.0, 0.0)


def step_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of Heaviside Step:
    f'(z) = 0 for z != 0, undefined (Dirac delta) at z = 0.
    In practice, gradient is 0 almost everywhere, making gradient descent impossible.
    """
    return np.zeros_like(z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Logistic Sigmoid Function:
    sigma(z) = 1 / (1 + exp(-z))
    Output range: (0, 1)
    Zero-centered: No (strictly positive)
    """
    # Numerically stable sigmoid formulation
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Analytical derivative of Sigmoid:
    sigma'(z) = sigma(z) * (1 - sigma(z))
    Maximum derivative: 0.25 at z = 0
    Saturation zones: |z| > 3 (sigma'(z) -> 0)
    """
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    """
    Hyperbolic Tangent Function:
    tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z)) = 2 * sigma(2z) - 1
    Output range: (-1, 1)
    Zero-centered: Yes
    """
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Analytical derivative of Tanh:
    tanh'(z) = 1 - tanh^2(z)
    Maximum derivative: 1.0 at z = 0
    Saturation zones: |z| > 3 (tanh'(z) -> 0)
    """
    t = np.tanh(z)
    return 1.0 - t ** 2


def relu(z: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU):
    ReLU(z) = max(0, z)
    Output range: [0, +inf)
    Zero-centered: No
    Sparsity: Exactly 0 for z <= 0
    """
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Subgradient derivative of ReLU:
    ReLU'(z) = 1 if z > 0 else 0 (subgradient at z = 0 treated as 0 or 1)
    Constant gradient: 1.0 for z > 0 (completely non-saturating)
    Vulnerability: 'Dying ReLU' if z <= 0 across all training data.
    """
    return np.where(z > 0.0, 1.0, 0.0)


def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky Rectified Linear Unit (LeakyReLU):
    LeakyReLU(z) = z if z > 0 else alpha * z
    Output range: (-inf, +inf)
    Avoids Dying ReLU by retaining small non-zero slope for negative inputs.
    """
    return np.where(z > 0.0, z, alpha * z)


def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Analytical derivative of LeakyReLU:
    LeakyReLU'(z) = 1.0 if z > 0 else alpha
    Non-zero gradient everywhere, eliminating complete dead-neuron traps.
    """
    return np.where(z > 0.0, 1.0, alpha)


# -------------------------------------------------------------------------
# Vanishing Gradient Deep Propagation Simulation
# -------------------------------------------------------------------------

def simulate_gradient_propagation(layers: int = 10, preactivations: np.ndarray = None):
    r"""
    Simulates the backpropagated gradient magnitude across 'layers' consecutive layers
    for different activation functions.
    
    Gradient update by Chain Rule:
      dLoss / dz_1 = (dLoss / dz_L) * \prod_{l=1}^{L-1} (W_{l+1}^T * f'(z_l))
    Assuming normalized weight scale ||W|| = 1.0 for isolated activation study:
      Effective gradient scale \approx \prod_{l=1}^{L} f'(z_l)
    """
    if preactivations is None:
        # Operating points with modest preactivations (mild saturation):
        preactivations = np.array([0.5, 0.8, 1.0, 0.4, 0.7, 1.2, 0.3, 0.6, 0.9, 1.1][:layers])
    
    grad_history = {
        'Step': [],
        'Sigmoid': [],
        'Tanh': [],
        'ReLU': [],
        'LeakyReLU': []
    }
    
    cum_step = 1.0
    cum_sig = 1.0
    cum_tanh = 1.0
    cum_relu = 1.0
    cum_lrelu = 1.0
    
    for l in range(layers):
        z = preactivations[l % len(preactivations)]
        
        cum_step *= step_derivative(np.array([z]))[0]
        cum_sig *= sigmoid_derivative(np.array([z]))[0]
        cum_tanh *= tanh_derivative(np.array([z]))[0]
        cum_relu *= relu_derivative(np.array([z]))[0]
        cum_lrelu *= leaky_relu_derivative(np.array([z]), alpha=0.01)[0]
        
        grad_history['Step'].append(cum_step)
        grad_history['Sigmoid'].append(cum_sig)
        grad_history['Tanh'].append(cum_tanh)
        grad_history['ReLU'].append(cum_relu)
        grad_history['LeakyReLU'].append(cum_lrelu)
        
    return grad_history


# -------------------------------------------------------------------------
# Visualization Suite
# -------------------------------------------------------------------------

def generate_comparison_plots(save_path: str):
    """
    Generates a 3-panel comprehensive comparison figure:
    1. Function Profiles f(z)
    2. Analytical Derivatives f'(z)
    3. Vanishing Gradient Propagation Across 10 Layers
    """
    z = np.linspace(-5.0, 5.0, 1000)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.22)
    
    colors = {
        'Step': '#7f7f7f',
        'Sigmoid': '#1f77b4',
        'Tanh': '#2ca02c',
        'ReLU': '#d62728',
        'LeakyReLU': '#9467bd'
    }
    
    # ---------------------------------------------------------
    # Panel 1: Activation Function Shapes f(z)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(z, step_function(z), label='Step (Heaviside)', color=colors['Step'], linestyle='--', linewidth=2.0)
    ax1.plot(z, sigmoid(z), label='Sigmoid (Logistic)', color=colors['Sigmoid'], linewidth=2.5)
    ax1.plot(z, tanh(z), label='Tanh', color=colors['Tanh'], linewidth=2.5)
    ax1.plot(z, relu(z), label='ReLU', color=colors['ReLU'], linewidth=2.5)
    ax1.plot(z, leaky_relu(z, alpha=0.1), label='LeakyReLU (alpha=0.1)', color=colors['LeakyReLU'], linestyle='-.', linewidth=2.2)
    
    ax1.axhline(0.0, color='black', linestyle=':', alpha=0.6)
    ax1.axvline(0.0, color='black', linestyle=':', alpha=0.6)
    ax1.set_xlim([-5.0, 5.0])
    ax1.set_ylim([-2.0, 5.0])
    ax1.set_title('(A) Activation Functions Profile: f(z)', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Preactivation Input (z)', fontsize=12)
    ax1.set_ylabel('Activation Output f(z)', fontsize=12)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Annotation on saturation
    ax1.annotate('Sigmoid / Tanh\nSaturation Regions', xy=(3.8, 1.0), xytext=(2.2, 2.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='#333333'),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', alpha=0.8))

    # ---------------------------------------------------------
    # Panel 2: Analytical Derivatives f'(z)
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(z, step_derivative(z), label="Step' (0 everywhere)", color=colors['Step'], linestyle='--', linewidth=2.0)
    ax2.plot(z, sigmoid_derivative(z), label="Sigmoid' (max 0.25)", color=colors['Sigmoid'], linewidth=2.5)
    ax2.plot(z, tanh_derivative(z), label="Tanh' (max 1.0)", color=colors['Tanh'], linewidth=2.5)
    ax2.plot(z, relu_derivative(z), label="ReLU' (1 for z>0, 0 else)", color=colors['ReLU'], linewidth=2.5)
    ax2.plot(z, leaky_relu_derivative(z, alpha=0.1), label="LeakyReLU' (1 or 0.1)", color=colors['LeakyReLU'], linestyle='-.', linewidth=2.2)
    
    ax2.axhline(0.0, color='black', linestyle=':', alpha=0.6)
    ax2.axvline(0.0, color='black', linestyle=':', alpha=0.6)
    ax2.set_xlim([-5.0, 5.0])
    ax2.set_ylim([-0.05, 1.15])
    ax2.set_title("(B) Analytical Derivatives: f'(z)", fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Preactivation Input (z)', fontsize=12)
    ax2.set_ylabel("Derivative Value f'(z)", fontsize=12)
    ax2.legend(loc='upper right', frameon=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Annotations on derivatives
    ax2.annotate('Sigmoid Max = 0.25\n(Severe Vanishing)', xy=(0.0, 0.25), xytext=(-3.8, 0.45),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color=colors['Sigmoid']),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f2ff', alpha=0.9))
    ax2.annotate('ReLU Constant Gradient = 1.0\n(Zero Saturation for z > 0)', xy=(2.0, 1.0), xytext=(-0.5, 0.8),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color=colors['ReLU']),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6e6', alpha=0.9))

    # ---------------------------------------------------------
    # Panel 3: Vanishing Gradient Propagation Across 10 Layers
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    
    layers = 10
    grad_data = simulate_gradient_propagation(layers=layers)
    layer_indices = np.arange(1, layers + 1)
    
    # Plot log-scale gradient decay
    ax3.plot(layer_indices, grad_data['ReLU'], marker='^', linewidth=2.5, color=colors['ReLU'], label='ReLU: Active Pathways Preserve Gradient exactly (1.0^L = 1.0)')
    ax3.plot(layer_indices, grad_data['LeakyReLU'], marker='d', linewidth=2.5, color=colors['LeakyReLU'], linestyle='-.', label='LeakyReLU: Preserves Non-zero Negative Gradient (prevents dead neurons)')
    ax3.plot(layer_indices, grad_data['Tanh'], marker='s', linewidth=2.5, color=colors['Tanh'], label='Tanh: Moderate Decay (max 1.0, decays under non-zero preactivations)')
    ax3.plot(layer_indices, grad_data['Sigmoid'], marker='o', linewidth=2.5, color=colors['Sigmoid'], label='Sigmoid: Exponential Decay (f\'(z) <= 0.25 -> 0.25^10 ~ 1e-7)')
    ax3.plot(layer_indices, grad_data['Step'], marker='x', linewidth=2.0, color=colors['Step'], linestyle='--', label='Step: Vanishes Instantly at Layer 1 (f\'(z) = 0)')
    
    ax3.set_yscale('log')
    ax3.set_xticks(layer_indices)
    ax3.set_xlim([0.8, layers + 0.2])
    ax3.set_ylim([1e-10, 5.0])
    ax3.set_title('(C) Backpropagated Gradient Magnitude Across 10 Consecutive Layers (Log Scale)', fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlabel('Layer Depth (Backpropagation Flow: Layer L -> Layer 1)', fontsize=12)
    ax3.set_ylabel('Effective Gradient Scale (Log10)', fontsize=12)
    ax3.legend(loc='lower left', frameon=True, fontsize=11)
    ax3.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Highlight vanishing problem on sigmoid
    ax3.annotate(f"Sigmoid at Layer 10: {grad_data['Sigmoid'][-1]:.2e}\n(Severe Vanishing Gradient)",
                 xy=(10, grad_data['Sigmoid'][-1]), xytext=(7.2, 1e-7),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color=colors['Sigmoid']),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f2ff', alpha=0.9))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved comprehensive activation comparison plot to: {save_path}")


# -------------------------------------------------------------------------
# CLI Verification & Numerical Diagnostics
# -------------------------------------------------------------------------

def run_numerical_diagnostics():
    test_points = np.array([-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
    
    print("=" * 95)
    print("STEP 15: ACTIVATION FUNCTIONS NUMERICAL EVALUATION TABLE")
    print("=" * 95)
    print(f"{'Input (z)':<10} | {'Step':<8} | {'Sigmoid':<10} | {'Tanh':<10} | {'ReLU':<10} | {'LeakyReLU (0.01)':<15}")
    print("-" * 95)
    
    for z_val in test_points:
        z_arr = np.array([z_val])
        st = step_function(z_arr)[0]
        sg = sigmoid(z_arr)[0]
        th = tanh(z_arr)[0]
        rl = relu(z_arr)[0]
        lrl = leaky_relu(z_arr, alpha=0.01)[0]
        print(f"{z_val:<10.1f} | {st:<8.4f} | {sg:<10.4f} | {th:<10.4f} | {rl:<10.4f} | {lrl:<15.4f}")
        
    print("=" * 95)
    print("\nANALYTICAL DERIVATIVES EVALUATION TABLE f'(z):")
    print("=" * 95)
    print(f"{'Input (z)':<10} | {'Step d/dz':<10} | {'Sigmoid d/dz':<14} | {'Tanh d/dz':<12} | {'ReLU d/dz':<12} | {'LeakyReLU d/dz':<16}")
    print("-" * 95)
    
    for z_val in test_points:
        z_arr = np.array([z_val])
        d_st = step_derivative(z_arr)[0]
        d_sg = sigmoid_derivative(z_arr)[0]
        d_th = tanh_derivative(z_arr)[0]
        d_rl = relu_derivative(z_arr)[0]
        d_lrl = leaky_relu_derivative(z_arr, alpha=0.01)[0]
        print(f"{z_val:<10.1f} | {d_st:<10.4f} | {d_sg:<14.4f} | {d_th:<12.4f} | {d_rl:<12.4f} | {d_lrl:<16.4f}")
    print("=" * 95)
    
    print("\n10-LAYER VANISHING GRADIENT SIMULATION (Mild Preactivation Regime):")
    grad_sim = simulate_gradient_propagation(layers=10)
    for l_idx in range(10):
        print(f"Layer {l_idx+1:2d} | Sigmoid Grad: {grad_sim['Sigmoid'][l_idx]:.3e} | Tanh Grad: {grad_sim['Tanh'][l_idx]:.3e} | ReLU Grad: {grad_sim['ReLU'][l_idx]:.3e} | LeakyReLU Grad: {grad_sim['LeakyReLU'][l_idx]:.3e}")
    print("=" * 95)


if __name__ == "__main__":
    run_numerical_diagnostics()
    asset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "activation_functions_comparison.png"))
    generate_comparison_plots(asset_path)
