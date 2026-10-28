"""
Optimizer Comparison Sandbox (Implemented Completely From Scratch)
University Hackathon Project 13: Sign Language Recognition and Translation Assistant
Author: Gehna
Date: 2026-10-28

Compares 4 fundamental optimization algorithms on an identical controlled problem:
1. Batch Gradient Descent (BGD)
2. Stochastic Gradient Descent (SGD, mini-batch=16)
3. Classical Momentum (Polyak Momentum)
4. Nesterov Accelerated Gradient (NAG)

Maintains identical architecture, weights initialization, dataset, and evaluation.
Generates Loss vs. Epoch convergence curves.
"""

import copy
import math
import os
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(Z):
    Z_clamped = np.clip(Z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-Z_clamped))

def sigmoid_derivative(A):
    return A * (1.0 - A)

class ControlledMLP:
    """
    A 2 -> 8 -> 1 MLP for fair optimizer benchmarking.
    """
    def __init__(self, W1, b1, W2, b2):
        self.W1 = np.copy(W1)
        self.b1 = np.copy(b1)
        self.W2 = np.copy(W2)
        self.b2 = np.copy(b2)
        self.cache = {}

    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = sigmoid(Z2)
        self.cache = {'X': X, 'A1': A1, 'A2': A2}
        return A2

    def compute_loss(self, y_true, y_pred):
        N = y_true.shape[0]
        eps = 1e-15
        p = np.clip(y_pred, eps, 1.0 - eps)
        return float(- (1.0 / N) * np.sum(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def backward(self, y_true):
        N = y_true.shape[0]
        X = self.cache['X']
        A1 = self.cache['A1']
        A2 = self.cache['A2']

        delta2 = A2 - y_true
        dW2 = (1.0 / N) * np.dot(A1.T, delta2)
        db2 = (1.0 / N) * np.sum(delta2, axis=0, keepdims=True)

        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(A1)
        dW1 = (1.0 / N) * np.dot(X.T, delta1)
        db1 = (1.0 / N) * np.sum(delta1, axis=0, keepdims=True)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

# =====================================================================
# OPTIMIZER ALGORITHMS (FROM SCRATCH)
# =====================================================================

def train_batch_gd(mlp: ControlledMLP, X, y, epochs=150, lr=0.5):
    """
    1. Batch Gradient Descent (BGD)
    Uses entire dataset for each gradient step:
    theta_{t+1} = theta_t - lr * grad
    """
    loss_history = []
    for _ in range(epochs):
        y_pred = mlp.forward(X)
        loss = mlp.compute_loss(y, y_pred)
        loss_history.append(loss)

        grads = mlp.backward(y)
        mlp.W1 -= lr * grads['dW1']
        mlp.b1 -= lr * grads['db1']
        mlp.W2 -= lr * grads['dW2']
        mlp.b2 -= lr * grads['db2']
    return loss_history

def train_sgd(mlp: ControlledMLP, X, y, epochs=150, lr=0.5, batch_size=16, seed=42):
    """
    2. Mini-batch Stochastic Gradient Descent (SGD)
    Updates parameters on small random batches:
    theta_{t+1} = theta_t - lr * grad_batch
    """
    np.random.seed(seed)
    N = X.shape[0]
    loss_history = []

    for _ in range(epochs):
        # Measure epoch loss on full dataset for fair comparison
        epoch_loss = mlp.compute_loss(y, mlp.forward(X))
        loss_history.append(epoch_loss)

        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start_idx in range(0, N, batch_size):
            X_batch = X_shuffled[start_idx:start_idx + batch_size]
            y_batch = y_shuffled[start_idx:start_idx + batch_size]

            mlp.forward(X_batch)
            grads = mlp.backward(y_batch)

            mlp.W1 -= lr * grads['dW1']
            mlp.b1 -= lr * grads['db1']
            mlp.W2 -= lr * grads['dW2']
            mlp.b2 -= lr * grads['db2']
    return loss_history

def train_momentum(mlp: ControlledMLP, X, y, epochs=150, lr=0.5, gamma=0.9):
    """
    3. Classical Momentum (Polyak Momentum)
    v_{t+1} = gamma * v_t + lr * grad
    theta_{t+1} = theta_t - v_{t+1}
    """
    loss_history = []
    # Velocity buffers
    v_W1 = np.zeros_like(mlp.W1)
    v_b1 = np.zeros_like(mlp.b1)
    v_W2 = np.zeros_like(mlp.W2)
    v_b2 = np.zeros_like(mlp.b2)

    for _ in range(epochs):
        y_pred = mlp.forward(X)
        loss = mlp.compute_loss(y, y_pred)
        loss_history.append(loss)

        grads = mlp.backward(y)

        v_W1 = gamma * v_W1 + lr * grads['dW1']
        v_b1 = gamma * v_b1 + lr * grads['db1']
        v_W2 = gamma * v_W2 + lr * grads['dW2']
        v_b2 = gamma * v_b2 + lr * grads['db2']

        mlp.W1 -= v_W1
        mlp.b1 -= v_b1
        mlp.W2 -= v_W2
        mlp.b2 -= v_b2
    return loss_history

def train_nesterov(mlp: ControlledMLP, X, y, epochs=150, lr=0.5, gamma=0.9):
    """
    4. Nesterov Accelerated Gradient (NAG)
    Applies lookahead momentum step: theta_lookahead = theta - gamma * v
    Evaluates gradient at theta_lookahead:
    v_{t+1} = gamma * v_t + lr * grad(theta_lookahead)
    theta_{t+1} = theta_t - v_{t+1}
    """
    loss_history = []
    v_W1 = np.zeros_like(mlp.W1)
    v_b1 = np.zeros_like(mlp.b1)
    v_W2 = np.zeros_like(mlp.W2)
    v_b2 = np.zeros_like(mlp.b2)

    for _ in range(epochs):
        y_pred = mlp.forward(X)
        loss = mlp.compute_loss(y, y_pred)
        loss_history.append(loss)

        # 1. Step to lookahead position
        orig_W1, orig_b1 = np.copy(mlp.W1), np.copy(mlp.b1)
        orig_W2, orig_b2 = np.copy(mlp.W2), np.copy(mlp.b2)

        mlp.W1 -= gamma * v_W1
        mlp.b1 -= gamma * v_b1
        mlp.W2 -= gamma * v_W2
        mlp.b2 -= gamma * v_b2

        # 2. Evaluate gradient at lookahead position
        mlp.forward(X)
        grads_lookahead = mlp.backward(y)

        # 3. Restore parameters and apply update with lookahead gradient
        mlp.W1 = orig_W1
        mlp.b1 = orig_b1
        mlp.W2 = orig_W2
        mlp.b2 = orig_b2

        v_W1 = gamma * v_W1 + lr * grads_lookahead['dW1']
        v_b1 = gamma * v_b1 + lr * grads_lookahead['db1']
        v_W2 = gamma * v_W2 + lr * grads_lookahead['dW2']
        v_b2 = gamma * v_b2 + lr * grads_lookahead['db2']

        mlp.W1 -= v_W1
        mlp.b1 -= v_b1
        mlp.W2 -= v_W2
        mlp.b2 -= v_b2
    return loss_history


def run_optimizer_benchmark():
    print("=" * 65)
    print("STEP 13: OPTIMIZER COMPARISON (BGD, SGD, MOMENTUM, NAG)")
    print("=" * 65)
    
    os.makedirs('docs/assets', exist_ok=True)
    
    # Generate Controlled Synthetic Problem: Concentric Two Moons / Non-linear boundary
    np.random.seed(42)
    N = 120
    # Two clusters with non-linear overlap
    X1 = np.random.randn(N // 2, 2) * 0.4 + np.array([0.5, 0.5])
    y1 = np.ones((N // 2, 1))
    X0 = np.random.randn(N // 2, 2) * 0.4 + np.array([-0.5, -0.5])
    y0 = np.zeros((N // 2, 1))
    X = np.vstack([X1, X0])
    y = np.vstack([y1, y0])

    # Fixed Base Parameters Initialization
    np.random.seed(100)
    init_W1 = np.random.randn(2, 8) * 0.5
    init_b1 = np.zeros((1, 8))
    init_W2 = np.random.randn(8, 1) * 0.5
    init_b2 = np.zeros((1, 1))

    epochs = 150
    lr = 0.4
    gamma = 0.85

    print(f"Controlled Dataset:    {N} samples in 2D space")
    print(f"Network Topology:      2 -> 8 (Sigmoid) -> 1 (Sigmoid)")
    print(f"Initial BCE Loss:      ~0.693")
    print(f"Common Hyperparams:    Epochs={epochs}, Learning Rate={lr}, Momentum Gamma={gamma}")
    print("-" * 65)

    # 1. Train Batch GD
    mlp_bgd = ControlledMLP(init_W1, init_b1, init_W2, init_b2)
    loss_bgd = train_batch_gd(mlp_bgd, X, y, epochs=epochs, lr=lr)
    acc_bgd = np.mean((mlp_bgd.forward(X) >= 0.5) == y) * 100

    # 2. Train SGD
    mlp_sgd = ControlledMLP(init_W1, init_b1, init_W2, init_b2)
    loss_sgd = train_sgd(mlp_sgd, X, y, epochs=epochs, lr=lr, batch_size=16)
    acc_sgd = np.mean((mlp_sgd.forward(X) >= 0.5) == y) * 100

    # 3. Train Momentum
    mlp_mom = ControlledMLP(init_W1, init_b1, init_W2, init_b2)
    loss_mom = train_momentum(mlp_mom, X, y, epochs=epochs, lr=lr, gamma=gamma)
    acc_mom = np.mean((mlp_mom.forward(X) >= 0.5) == y) * 100

    # 4. Train NAG
    mlp_nag = ControlledMLP(init_W1, init_b1, init_W2, init_b2)
    loss_nag = train_nesterov(mlp_nag, X, y, epochs=epochs, lr=lr, gamma=gamma)
    acc_nag = np.mean((mlp_nag.forward(X) >= 0.5) == y) * 100

    print("OPTIMIZER BENCHMARK RESULTS (AFTER 150 EPOCHS):")
    print("-----------------------------------------------------------------")
    print(f"1. Batch Gradient Descent : Final Loss = {loss_bgd[-1]:.5f} | Accuracy = {acc_bgd:5.1f}%")
    print(f"2. Stochastic GD (Batch=16): Final Loss = {loss_sgd[-1]:.5f} | Accuracy = {acc_sgd:5.1f}%")
    print(f"3. Classical Momentum     : Final Loss = {loss_mom[-1]:.5f} | Accuracy = {acc_mom:5.1f}%")
    print(f"4. Nesterov Accelerated GD: Final Loss = {loss_nag[-1]:.5f} | Accuracy = {acc_nag:5.1f}%")

    # Assertions
    assert loss_bgd[-1] < loss_bgd[0], "BGD failed to reduce loss!"
    assert loss_mom[-1] < loss_bgd[-1], "Momentum should converge faster than standard BGD!"
    assert loss_nag[-1] < loss_bgd[-1], "NAG should converge faster than standard BGD!"
    print("\n  -> Assertions PASSED: Momentum and NAG accelerated convergence beyond standard BGD.")

    # Plot Loss vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(loss_bgd, label='Batch Gradient Descent (BGD)', color='#6B7280', linewidth=2, linestyle=':')
    plt.plot(loss_sgd, label='Mini-batch SGD (batch=16)', color='#F59E0B', linewidth=2, linestyle='--')
    plt.plot(loss_mom, label='Classical Momentum (γ=0.85)', color='#3B82F6', linewidth=2.2)
    plt.plot(loss_nag, label='Nesterov Accelerated Gradient (NAG)', color='#10B981', linewidth=2.5)

    plt.title("Step 13: Optimizer Comparison (Loss vs. Epoch on Controlled Problem)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10.5)
    plt.tight_layout()
    chart_path = 'docs/assets/optimizer_loss_comparison.png'
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\nSaved optimizer convergence chart to: {chart_path}")
    print("=" * 65)

if __name__ == '__main__':
    run_optimizer_benchmark()
