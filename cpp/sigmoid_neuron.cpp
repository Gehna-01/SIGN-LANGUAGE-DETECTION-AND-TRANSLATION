/**
 * Sigmoid Neuron From Scratch in Modern C++
 * University Hackathon Project 13: Sign Language Recognition and Translation Assistant
 * Author: Gehna
 * Date: 2026-10-28
 *
 * Compiles cleanly with: g++ -O3 cpp/sigmoid_neuron.cpp -o bin/sigmoid_neuron.exe
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>

double sigmoid(double z) {
    // Clamp to prevent exp overflow
    if (z > 500.0) return 1.0;
    if (z < -500.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

double sigmoid_derivative(double a) {
    return a * (1.0 - a);
}

class SigmoidNeuron {
public:
    std::vector<double> weights;
    double bias;
    double lr;

    SigmoidNeuron(int n_features, double learning_rate = 2.0) : lr(learning_rate) {
        weights.assign(n_features, 0.1);
        bias = 0.0;
    }

    double forward(const std::vector<double>& x) const {
        double z = bias;
        for (size_t i = 0; i < weights.size(); ++i) {
            z += weights[i] * x[i];
        }
        return sigmoid(z);
    }

    double compute_bce_loss(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const {
        double total_loss = 0.0;
        double eps = 1e-15;
        for (size_t i = 0; i < X.size(); ++i) {
            double a = forward(X[i]);
            a = std::min(std::max(a, eps), 1.0 - eps);
            total_loss += -(y[i] * std::log(a) + (1.0 - y[i]) * std::log(1.0 - a));
        }
        return total_loss / (double)X.size();
    }

    void train_step(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        double N = (double)X.size();
        std::vector<double> grad_w(weights.size(), 0.0);
        double grad_b = 0.0;

        for (size_t i = 0; i < X.size(); ++i) {
            double a = forward(X[i]);
            double error = a - y[i]; // dL/dz for BCE + Sigmoid

            for (size_t j = 0; j < weights.size(); ++j) {
                grad_w[j] += error * X[i][j];
            }
            grad_b += error;
        }

        // Gradient Descent update: w = w - lr * grad
        for (size_t j = 0; j < weights.size(); ++j) {
            weights[j] -= lr * (grad_w[j] / N);
        }
        bias -= lr * (grad_b / N);
    }
};

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << " C++ SIGMOID NEURON (Zero-Dependency STL)" << std::endl;
    std::cout << " University Requirement Compliance: Native C++ Module" << std::endl;
    std::cout << "=========================================================" << std::endl;

    // 1. Sigmoid Properties
    std::cout << "\n[1] Sigmoid Function Properties:" << std::endl;
    std::cout << "    sigma(0) = " << sigmoid(0.0) << " (Expected: 0.5)" << std::endl;
    std::cout << "    sigma'(0) = " << sigmoid_derivative(sigmoid(0.0)) << " (Expected: 0.25)" << std::endl;
    assert(std::abs(sigmoid(0.0) - 0.5) < 1e-9);
    assert(std::abs(sigmoid_derivative(0.5) - 0.25) < 1e-9);

    // 2. Train on OR logic gate
    std::vector<std::vector<double>> X = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    std::vector<double> y = {0.0, 1.0, 1.0, 1.0};

    SigmoidNeuron neuron(2, 2.0);
    double initial_loss = neuron.compute_bce_loss(X, y);
    std::cout << "\n[2] Training on OR Gate:" << std::endl;
    std::cout << "    Epoch   0 | Initial BCE Loss: " << initial_loss << std::endl;

    for (int epoch = 1; epoch <= 200; ++epoch) {
        neuron.train_step(X, y);
        if (epoch % 50 == 0 || epoch == 200) {
            std::cout << "    Epoch " << std::setw(3) << epoch << " | Loss: " 
                      << neuron.compute_bce_loss(X, y) << std::endl;
        }
    }

    double final_loss = neuron.compute_bce_loss(X, y);
    assert(final_loss < 0.1);
    std::cout << "  -> Assertion PASSED: Loss converged to " << final_loss << std::endl;

    // 3. Predictions
    std::cout << "\n[3] Final Trained Predictions:" << std::endl;
    for (size_t i = 0; i < X.size(); ++i) {
        double a = neuron.forward(X[i]);
        int pred = (a >= 0.5) ? 1 : 0;
        std::cout << "    (" << X[i][0] << ", " << X[i][1] << ") -> Output: " 
                  << std::fixed << std::setprecision(4) << a << " | Predicted: " << pred << std::endl;
        assert(pred == (int)y[i]);
    }
    std::cout << "  -> Assertion PASSED: All 4 logic inputs classified correctly." << std::endl;

    std::cout << "=========================================================" << std::endl;
    std::cout << " ALL C++ SIGMOID NEURON VERIFICATIONS PASSED!" << std::endl;
    std::cout << "=========================================================" << std::endl;
    return 0;
}
