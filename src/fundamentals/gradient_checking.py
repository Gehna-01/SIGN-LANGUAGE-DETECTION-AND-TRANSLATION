"""
Comprehensive Numerical Gradient Checking on Multilayer Perceptron
University Hackathon Project 13: Sign Language Recognition and Translation Assistant
Author: Gehna
Date: 2026-10-28

Verifies the mathematical correctness of analytical backpropagation gradients:
- Tests all 17 parameters (W1, b1, W2, b2)
- Compares analytical gradients against two-sided central finite differences
- Uses relative error: ||g_anal - g_num|| / (||g_anal|| + ||g_num||)
- Demonstrates intentional bug detection (proving gradient checking works)
"""

import math
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.fundamentals.mlp_from_scratch import MultilayerPerceptron

def flatten_parameters(mlp: MultilayerPerceptron) -> np.ndarray:
    """Flattens all network parameters into a single 1D vector."""
    return np.concatenate([
        mlp.W1.flatten(),
        mlp.b1.flatten(),
        mlp.W2.flatten(),
        mlp.b2.flatten()
    ])

def unflatten_parameters(mlp: MultilayerPerceptron, theta: np.ndarray):
    """Restores parameters from a 1D vector into the MLP's matrices."""
    idx = 0
    
    # W1: 2 x 4 (8 elements)
    w1_size = mlp.W1.size
    mlp.W1 = theta[idx:idx + w1_size].reshape(mlp.W1.shape)
    idx += w1_size
    
    # b1: 1 x 4 (4 elements)
    b1_size = mlp.b1.size
    mlp.b1 = theta[idx:idx + b1_size].reshape(mlp.b1.shape)
    idx += b1_size
    
    # W2: 4 x 1 (4 elements)
    w2_size = mlp.W2.size
    mlp.W2 = theta[idx:idx + w2_size].reshape(mlp.W2.shape)
    idx += w2_size
    
    # b2: 1 x 1 (1 element)
    b2_size = mlp.b2.size
    mlp.b2 = theta[idx:idx + b2_size].reshape(mlp.b2.shape)

def flatten_gradients(grads: dict) -> np.ndarray:
    """Flattens all computed analytical gradients into a single 1D vector."""
    return np.concatenate([
        grads['dW1'].flatten(),
        grads['db1'].flatten(),
        grads['dW2'].flatten(),
        grads['db2'].flatten()
    ])

def check_mlp_gradients(mlp: MultilayerPerceptron, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-7):
    """
    Computes two-sided central difference numerical gradient for every parameter
    and compares against analytical backpropagation gradient.
    """
    # 1. Compute Analytical Gradients via Backpropagation
    y_pred = mlp.forward(X)
    grads_dict = mlp.backward(y)
    analytical_grads = flatten_gradients(grads_dict)
    
    # 2. Compute Numerical Gradients via Finite Differences
    theta = flatten_parameters(mlp)
    num_params = len(theta)
    numerical_grads = np.zeros(num_params)
    
    for i in range(num_params):
        # Perturb theta[i] + epsilon
        theta_plus = np.copy(theta)
        theta_plus[i] += epsilon
        unflatten_parameters(mlp, theta_plus)
        loss_plus = mlp.compute_loss(y, mlp.forward(X))
        
        # Perturb theta[i] - epsilon
        theta_minus = np.copy(theta)
        theta_minus[i] -= epsilon
        unflatten_parameters(mlp, theta_minus)
        loss_minus = mlp.compute_loss(y, mlp.forward(X))
        
        # Two-sided symmetric central difference
        numerical_grads[i] = (loss_plus - loss_minus) / (2.0 * epsilon)
        
    # Restore original parameters
    unflatten_parameters(mlp, theta)
    
    # 3. Relative Error Metric
    diff_norm = np.linalg.norm(analytical_grads - numerical_grads)
    anal_norm = np.linalg.norm(analytical_grads)
    num_norm  = np.linalg.norm(numerical_grads)
    
    relative_error = diff_norm / (anal_norm + num_norm + 1e-15)
    return analytical_grads, numerical_grads, relative_error


def run_gradient_checking_demo():
    print("=" * 65)
    print("STEP 12: FINITE-DIFFERENCE GRADIENT CHECKING ON MLP")
    print("=" * 65)
    
    os.makedirs('docs/assets', exist_ok=True)
    
    X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
    y = np.array([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])
                  
    mlp = MultilayerPerceptron(n_in=2, n_hidden=4, n_out=1, seed=42)
    
    print("\n[1] EVALUATING ANALYTICAL BACKPROP VS. FINITE DIFFERENCES")
    print(f"Total Parameters to Verify: 17 scalar weights and biases")
    print(f"Perturbation Step Epsilon:  1e-7 (Two-sided Central Difference)")
    
    anal_grads, num_grads, rel_error = check_mlp_gradients(mlp, X, y, epsilon=1e-7)
    
    param_names = [
        "W1[0,0]", "W1[0,1]", "W1[0,2]", "W1[0,3]",
        "W1[1,0]", "W1[1,1]", "W1[1,2]", "W1[1,3]",
        "b1[0,0]", "b1[0,1]", "b1[0,2]", "b1[0,3]",
        "W2[0,0]", "W2[1,0]", "W2[2,0]", "W2[3,0]",
        "b2[0,0]"
    ]
    
    print("\nParam Name   |  Analytical Grad  |  Numerical Grad   | Absolute Diff")
    print("--------------------------------------------------------------------")
    for name, a_g, n_g in zip(param_names, anal_grads, num_grads):
        diff = abs(a_g - n_g)
        print(f"{name:12s} |   {a_g:14.8f}  |   {n_g:14.8f}  |  {diff:.2e}")
        
    print(f"\nFinal Calculated Relative Error: {rel_error:.2e}")
    assert rel_error < 1e-7, f"Gradient checking failed! Relative error {rel_error} >= 1e-7"
    print("  -> Assertion PASSED: Relative Error < 1e-7 (Backpropagation is mathematically verified!).")

    # 2. Bug Detection Sensitivity Test
    print("\n[2] INTENTIONAL BUG DETECTION TEST (VERIFYING GRADIENT CHECK RIGOR)")
    # Corrupt analytical gradient intentionally: simulate developer forgetting sigma'(z)
    corrupted_grads = np.copy(anal_grads)
    corrupted_grads[0] *= -1.0  # flipped sign
    corrupted_grads[12] += 0.5  # corrupted magnitude
    
    diff_bug = np.linalg.norm(corrupted_grads - num_grads)
    rel_error_bug = diff_bug / (np.linalg.norm(corrupted_grads) + np.linalg.norm(num_grads) + 1e-15)
    print(f"Relative Error with Corrupted Analytical Gradients: {rel_error_bug:.2e}")
    assert rel_error_bug > 0.01, "Error: Gradient check failed to catch the intentional bug!"
    print(f"  -> Assertion PASSED: Gradient check successfully flagged the bug (Rel Error = {rel_error_bug:.2e} >> 1e-7).")

    # 3. Visual Comparison Chart
    indices = np.arange(len(param_names))
    plt.figure(figsize=(14, 5))
    width = 0.35
    plt.bar(indices - width/2, anal_grads, width=width, label='Analytical Grad (Backprop)', color='#2563EB', alpha=0.85)
    plt.bar(indices + width/2, num_grads, width=width, label='Numerical Grad (Finite Diff)', color='#DC2626', alpha=0.85)
    plt.xticks(indices, param_names, rotation=45, ha='right', fontsize=9)
    plt.ylabel("Gradient Value", fontsize=11)
    plt.title("Step 12: Analytical vs. Numerical Finite-Difference Gradients (All 17 Parameters)", fontsize=13, fontweight='bold', pad=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot_path = 'docs/assets/gradient_check_comparison.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nSaved gradient comparison plot to: {plot_path}")

    print("\n" + "=" * 65)
    print("ALL GRADIENT CHECKING VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == '__main__':
    run_gradient_checking_demo()
