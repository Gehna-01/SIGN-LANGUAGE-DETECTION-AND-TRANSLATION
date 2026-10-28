"""
Multilayer Perceptron (MLP) Implemented Completely From Scratch
University Hackathon Project 13: Sign Language Recognition and Translation Assistant
Author: Gehna
Date: 2026-10-28

Architecture: 2 -> 4 -> 1 Feedforward Neural Network
Target Problem: Non-linearly separable XOR classification (proves overcoming single-neuron limit)
No Automatic Differentiation or Deep Learning Frameworks Used.
Every forward equation, matrix derivative, and update rule is coded explicitly.
"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(Z):
    """Numerically stable matrix sigmoid."""
    Z_clamped = np.clip(Z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-Z_clamped))

def sigmoid_derivative(A):
    """Derivative of sigmoid: d(sigma)/dz = A * (1 - A)"""
    return A * (1.0 - A)

class MultilayerPerceptron:
    """
    A 2-layer Feedforward Neural Network (1 hidden layer, 1 output layer)
    implemented from first principles using pure NumPy linear algebra.
    """
    def __init__(self, n_in: int = 2, n_hidden: int = 4, n_out: int = 1, lr: float = 1.0, seed: int = 42):
        # Hyperparameters
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr
        
        # Learnable Parameters (Weights and Biases)
        np.random.seed(seed)
        # Xavier/Glorot-like uniform initialization
        limit1 = math.sqrt(6.0 / (n_in + n_hidden))
        self.W1 = np.random.uniform(-limit1, limit1, (n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        
        limit2 = math.sqrt(6.0 / (n_hidden + n_out))
        self.W2 = np.random.uniform(-limit2, limit2, (n_hidden, n_out))
        self.b2 = np.zeros((1, n_out))
        
        # Cache for backpropagation
        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward Propagation:
        Hidden Layer:
          Z1 = X @ W1 + b1
          A1 = sigmoid(Z1)
        Output Layer:
          Z2 = A1 @ W2 + b2
          A2 = sigmoid(Z2) (predictions)
        """
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = sigmoid(Z1)
        
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = sigmoid(Z2)
        
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary Cross-Entropy Loss:
        L = - (1/N) * sum( y*ln(y_hat) + (1-y)*ln(1-y_hat) )
        """
        N = y_true.shape[0]
        eps = 1e-15
        y_pred_safe = np.clip(y_pred, eps, 1.0 - eps)
        loss = - (1.0 / N) * np.sum(y_true * np.log(y_pred_safe) + (1.0 - y_true) * np.log(1.0 - y_pred_safe))
        return float(loss)

    def backward(self, y_true: np.ndarray):
        """
        Analytical Backpropagation via Matrix Calculus (Chain Rule):
        1. Output Layer Gradients:
           delta2 = dL/dZ2 = A2 - y_true
           dW2 = (1/N) * (A1^T @ delta2)
           db2 = (1/N) * sum(delta2, axis=0, keepdims=True)
        2. Hidden Layer Gradients:
           delta1 = (delta2 @ W2^T) * sigma'(Z1) = (delta2 @ W2^T) * [A1 * (1 - A1)]
           dW1 = (1/N) * (X^T @ delta1)
           db1 = (1/N) * sum(delta1, axis=0, keepdims=True)
        """
        N = y_true.shape[0]
        X = self.cache['X']
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        
        # Output layer error
        delta2 = A2 - y_true  # (N, 1)
        dW2 = (1.0 / N) * np.dot(A1.T, delta2)  # (4, 1)
        db2 = (1.0 / N) * np.sum(delta2, axis=0, keepdims=True)  # (1, 1)
        
        # Hidden layer error
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(A1)  # (N, 4)
        dW1 = (1.0 / N) * np.dot(X.T, delta1)  # (2, 4)
        db1 = (1.0 / N) * np.sum(delta1, axis=0, keepdims=True)  # (1, 4)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def update_parameters(self, grads: dict):
        """
        Gradient Descent Update Rule:
        theta = theta - lr * dtheta
        """
        self.W1 -= self.lr * grads['dW1']
        self.b1 -= self.lr * grads['db1']
        self.W2 -= self.lr * grads['dW2']
        self.b2 -= self.lr * grads['db2']

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1500) -> list:
        """
        Executes full training loop and records epoch loss history.
        """
        history = []
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            history.append(loss)
            
            grads = self.backward(y)
            self.update_parameters(grads)
            
            if epoch in [1, 100, 500, 1000, 1500]:
                acc = np.mean((y_pred >= 0.5) == y) * 100
                print(f"  Epoch {epoch:4d} | BCE Loss: {loss:.5f} | Accuracy: {acc:5.1f}%")
        return history


def run_mlp_demo():
    print("=" * 65)
    print("STEP 11: MULTILAYER PERCEPTRON (MLP) FROM FIRST PRINCIPLES")
    print("=" * 65)
    
    os.makedirs('docs/assets', exist_ok=True)
    
    # Non-linearly separable XOR Dataset
    X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
    y = np.array([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])
                  
    print("\n[1] ARCHITECTURE & HYPERPARAMETERS")
    print(f"Network Topology:  Input({X.shape[1]}) -> Hidden(4, Sigmoid) -> Output(1, Sigmoid)")
    print("Task:              Non-linearly separable XOR logic classification")
    print("Loss Function:     Binary Cross-Entropy (BCE)")
    print("Learning Rate:     1.5 (Vanilla Gradient Descent)")
    print("Total Parameters:  (2*4 + 4) + (4*1 + 1) = 17 learnable parameters")
    
    mlp = MultilayerPerceptron(n_in=2, n_hidden=4, n_out=1, lr=1.5, seed=42)
    
    initial_loss = mlp.compute_loss(y, mlp.forward(X))
    print(f"Initial Untrained BCE Loss: {initial_loss:.5f}")
    
    print("\n[2] TRAINING PROGRESS (BACKPROPAGATION IN ACTION)")
    losses = mlp.train(X, y, epochs=1500)
    final_loss = losses[-1]
    
    # Assertions on learning
    assert final_loss < 0.05, f"Expected final loss < 0.05, got {final_loss:.5f}"
    print(f"  -> Assertion PASSED: Loss smoothly converged from {initial_loss:.4f} down to {final_loss:.4f}.")
    
    # 3. Final Predictions on XOR
    print("\n[3] FINAL TRAINED PREDICTIONS (XOR PROBLEM SOLVED)")
    y_preds = mlp.forward(X)
    print(" Input (x1, x2) | Target y | Output a2 | Predicted Class")
    print("---------------------------------------------------------")
    for i in range(len(X)):
        pred_class = 1 if y_preds[i, 0] >= 0.5 else 0
        print(f"   ({X[i, 0]:.0f}, {X[i, 1]:.0f})     |    {y[i, 0]:.0f}     |  {y_preds[i, 0]:.4f}   |        {pred_class}")
        assert pred_class == int(y[i, 0]), f"Mismatch on input {X[i]}!"
    print("  -> Assertion PASSED: 100% correct classification on non-linear XOR data.")
    
    # 4. Save Learning Curve Plot
    plt.figure(figsize=(9, 4.5))
    plt.plot(losses, color='#2563EB', linewidth=2.5, label='Training BCE Loss')
    plt.title("MLP From Scratch: XOR Learning Curve (Loss vs Epoch)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    curve_path = 'docs/assets/mlp_xor_learning_curve.png'
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"\nSaved MLP learning curve to: {curve_path}")

    print("\n" + "=" * 65)
    print("ALL MULTILAYER PERCEPTRON VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == '__main__':
    run_mlp_demo()
