"""
Sigmoid Neuron Implementation From Scratch
University Hackathon Project 13: Sign Language Recognition and Translation Assistant
Author: Gehna
Date: 2026-10-28

Implements the fundamental Sigmoid Neuron from first principles:
- Continuous real-valued inputs x in R^n
- Learnable real-valued weights w and bias b
- Linear pre-activation: z = sum(w_i * x_i) + b
- Continuous non-linear activation: a = sigma(z) = 1 / (1 + exp(-z))
- Analytical derivative: sigma'(z) = sigma(z) * (1 - sigma(z))
- Binary Cross-Entropy (BCE) loss
- Gradient computation and Gradient Descent update rule
- Visualizes the sigmoid curve and its derivative
"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z: float) -> float:
    """
    Numerically stable sigmoid function:
    sigma(z) = 1 / (1 + exp(-z))
    Clamped to prevent overflow for large negative z.
    """
    z_clamped = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z_clamped))

def sigmoid_derivative(a: float) -> float:
    """
    Derivative of sigmoid with respect to z:
    d(sigma)/dz = sigma(z) * (1 - sigma(z)) = a * (1 - a)
    """
    return a * (1.0 - a)

class SigmoidNeuron:
    """
    A single artificial neuron with continuous sigmoid activation
    and gradient descent learning.
    """
    def __init__(self, n_features: int, lr: float = 0.5):
        # Initialize small random weights and zero bias
        np.random.seed(42)
        self.weights = [np.random.randn() * 0.1 for _ in range(n_features)]
        self.bias = 0.0
        self.lr = lr

    def forward(self, x: list) -> float:
        """
        Computes pre-activation z and activated output a = sigma(z).
        """
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def compute_bce_loss(self, X: list, y: list) -> float:
        """
        Binary Cross-Entropy Loss:
        L = - (1/N) * sum( y*ln(a) + (1-y)*ln(1-a) )
        """
        N = len(X)
        eps = 1e-15
        total_loss = 0.0
        for xi, yi in zip(X, y):
            a = self.forward(xi)
            a_clamped = min(max(a, eps), 1.0 - eps)
            total_loss += -(yi * math.log(a_clamped) + (1.0 - yi) * math.log(1.0 - a_clamped))
        return total_loss / N

    def train_step(self, X: list, y: list):
        """
        Performs one epoch of batch gradient descent using BCE loss:
        dL/dz = a - y
        dL/dw_i = (1/N) * sum( (a - y) * x_i )
        dL/db   = (1/N) * sum( a - y )
        """
        N = len(X)
        n_features = len(self.weights)
        
        grad_w = [0.0] * n_features
        grad_b = 0.0
        
        for xi, yi in zip(X, y):
            a = self.forward(xi)
            error = a - yi  # dL/dz for sigmoid + BCE
            
            for j in range(n_features):
                grad_w[j] += error * xi[j]
            grad_b += error
            
        # Parameter updates: theta = theta - lr * grad
        for j in range(n_features):
            self.weights[j] -= self.lr * (grad_w[j] / N)
        self.bias -= self.lr * (grad_b / N)


def run_sigmoid_demo():
    print("=" * 60)
    print("STEP 10: SIGMOID NEURON FROM FIRST PRINCIPLES")
    print("=" * 60)
    
    os.makedirs('docs/assets', exist_ok=True)
    
    # 1. Mathematical Behaviour & Derivative Proof
    print("\n[1] SIGMOID FUNCTION & DERIVATIVE PROPERTIES")
    test_z_values = [-5.0, -2.0, 0.0, 2.0, 5.0]
    print("   z   |  sigma(z)  |  sigma'(z) = a*(1-a)")
    print("------------------------------------------")
    for z in test_z_values:
        a = sigmoid(z)
        da = sigmoid_derivative(a)
        print(f" {z:5.1f} |   {a:6.4f}   |   {da:6.4f}")
    
    # Assert standard known values
    assert abs(sigmoid(0.0) - 0.5) < 1e-9
    assert abs(sigmoid_derivative(0.5) - 0.25) < 1e-9
    print("  -> Assertion PASSED: sigma(0) = 0.5 and peak derivative sigma'(0) = 0.25 verified.")

    # 2. Generate Sigmoid Curve & Derivative Visualization
    z_range = np.linspace(-7, 7, 300)
    sigma_vals = [sigmoid(z) for z in z_range]
    deriv_vals = [sigmoid_derivative(a) for a in sigma_vals]
    
    plt.figure(figsize=(10, 4.5))
    plt.plot(z_range, sigma_vals, label=r'Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$', color='#2563EB', linewidth=2.5)
    plt.plot(z_range, deriv_vals, label=r"Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$", color='#DC2626', linewidth=2, linestyle='--')
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    plt.title("Sigmoid Activation Function and its Derivative", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Linear Pre-activation (z)", fontsize=11)
    plt.ylabel("Activation Value", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plot_path = 'docs/assets/sigmoid_derivatives.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved sigmoid visualization to: {plot_path}")

    # 3. Demonstration of Smooth Learning via Gradient Descent
    print("\n[2] DEMONSTRATING GRADIENT DESCENT LEARNING")
    print("Training a Sigmoid Neuron to solve the continuous OR logic problem:")
    X_train = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y_train = [0.0, 1.0, 1.0, 1.0]
    
    neuron = SigmoidNeuron(n_features=2, lr=2.0)
    
    initial_loss = neuron.compute_bce_loss(X_train, y_train)
    print(f"Initial BCE Loss (Epoch 0): {initial_loss:.4f}")
    
    losses = []
    for epoch in range(1, 201):
        neuron.train_step(X_train, y_train)
        loss = neuron.compute_bce_loss(X_train, y_train)
        losses.append(loss)
        if epoch in [1, 20, 50, 100, 200]:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Weights: [{neuron.weights[0]:.2f}, {neuron.weights[1]:.2f}] | Bias: {neuron.bias:.2f}")
            
    final_loss = losses[-1]
    assert final_loss < initial_loss, "Learning failed: Loss did not decrease!"
    assert final_loss < 0.15, f"Expected loss < 0.15, got {final_loss:.4f}"
    print(f"  -> Assertion PASSED: Loss smoothly decreased from {initial_loss:.4f} to {final_loss:.4f}.")

    # 4. Predictions Verification
    print("\n[3] FINAL TRAINED SIGMOID PREDICTIONS")
    print(" Input (x1, x2) | Target | Sigmoid Output a | Predicted Class (threshold=0.5)")
    print("-------------------------------------------------------------------------")
    for xi, yi in zip(X_train, y_train):
        a = neuron.forward(xi)
        pred = 1 if a >= 0.5 else 0
        print(f"   ({xi[0]:.0f}, {xi[1]:.0f})     |   {yi:.0f}    |     {a:.4f}      |                  {pred}")
        assert pred == int(yi), f"Prediction error on input {xi}!"
    print("  -> Assertion PASSED: All 4 logic outputs match targets 100%.")

    print("\n" + "=" * 60)
    print("ALL SIGMOID NEURON DEMONSTRATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    run_sigmoid_demo()
