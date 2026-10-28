/**
 * Multilayer Perceptron (MLP) From Scratch in Modern C++
 * University Hackathon Project 13: Sign Language Recognition and Translation Assistant
 * Author: Gehna
 * Date: 2026-10-28
 *
 * Compiles cleanly with: g++ -O3 cpp/mlp_from_scratch.cpp -o bin/mlp_from_scratch.exe
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>

double sigmoid(double z) {
    if (z > 500.0) return 1.0;
    if (z < -500.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

double sigmoid_derivative(double a) {
    return a * (1.0 - a);
}

class MultilayerPerceptron {
public:
    int n_in, n_hidden, n_out;
    double lr;

    // Parameters (W1: 2x4, b1: 1x4, W2: 4x1, b2: 1x1)
    std::vector<std::vector<double>> W1;
    std::vector<double> b1;
    std::vector<std::vector<double>> W2;
    double b2;

    MultilayerPerceptron(int in_dim = 2, int hidden_dim = 4, int out_dim = 1, double learning_rate = 1.5)
        : n_in(in_dim), n_hidden(hidden_dim), n_out(out_dim), lr(learning_rate) {
        
        // Manual deterministic initialization
        W1 = {
            { 0.5, -0.6,  0.8, -0.4},
            {-0.7,  0.9, -0.5,  0.6}
        };
        b1 = {0.0, 0.0, 0.0, 0.0};

        W2 = {
            { 0.8},
            {-0.9},
            { 0.7},
            {-0.6}
        };
        b2 = 0.0;
    }

    void forward(
        const std::vector<std::vector<double>>& X,
        std::vector<std::vector<double>>& A1,
        std::vector<double>& A2
    ) const {
        size_t N = X.size();
        A1.assign(N, std::vector<double>(n_hidden, 0.0));
        A2.assign(N, 0.0);

        for (size_t i = 0; i < N; ++i) {
            // Hidden Layer
            for (int h = 0; h < n_hidden; ++h) {
                double z1 = b1[h];
                for (int d = 0; d < n_in; ++d) {
                    z1 += X[i][d] * W1[d][h];
                }
                A1[i][h] = sigmoid(z1);
            }

            // Output Layer
            double z2 = b2;
            for (int h = 0; h < n_hidden; ++h) {
                z2 += A1[i][h] * W2[h][0];
            }
            A2[i] = sigmoid(z2);
        }
    }

    double compute_bce_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred) const {
        double total_loss = 0.0;
        double eps = 1e-15;
        for (size_t i = 0; i < y_true.size(); ++i) {
            double p = std::min(std::max(y_pred[i], eps), 1.0 - eps);
            total_loss += -(y_true[i] * std::log(p) + (1.0 - y_true[i]) * std::log(1.0 - p));
        }
        return total_loss / (double)y_true.size();
    }

    void train_step(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        size_t N = X.size();
        std::vector<std::vector<double>> A1;
        std::vector<double> A2;
        forward(X, A1, A2);

        // Gradients
        std::vector<std::vector<double>> dW2(n_hidden, std::vector<double>(n_out, 0.0));
        double db2 = 0.0;
        std::vector<std::vector<double>> dW1(n_in, std::vector<double>(n_hidden, 0.0));
        std::vector<double> db1(n_hidden, 0.0);

        for (size_t i = 0; i < N; ++i) {
            // Output layer error: delta2 = A2 - y
            double delta2 = A2[i] - y[i];

            // dW2, db2
            for (int h = 0; h < n_hidden; ++h) {
                dW2[h][0] += delta2 * A1[i][h];
            }
            db2 += delta2;

            // Hidden layer error: delta1_h = delta2 * W2_h * sigma'(A1_h)
            for (int h = 0; h < n_hidden; ++h) {
                double delta1_h = delta2 * W2[h][0] * sigmoid_derivative(A1[i][h]);
                for (int d = 0; d < n_in; ++d) {
                    dW1[d][h] += delta1_h * X[i][d];
                }
                db1[h] += delta1_h;
            }
        }

        // Apply parameter updates: theta = theta - lr * (grad / N)
        double inv_N = 1.0 / (double)N;
        for (int h = 0; h < n_hidden; ++h) {
            W2[h][0] -= lr * dW2[h][0] * inv_N;
            b1[h] -= lr * db1[h] * inv_N;
            for (int d = 0; d < n_in; ++d) {
                W1[d][h] -= lr * dW1[d][h] * inv_N;
            }
        }
        b2 -= lr * db2 * inv_N;
    }
};

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << " C++ MULTILAYER PERCEPTRON (Zero-Dependency STL)" << std::endl;
    std::cout << " University Requirement Compliance: Native C++ Module" << std::endl;
    std::cout << "=========================================================" << std::endl;

    // XOR Dataset
    std::vector<std::vector<double>> X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<double> y = {0.0, 1.0, 1.0, 0.0};

    MultilayerPerceptron mlp(2, 4, 1, 1.5);

    std::vector<std::vector<double>> A1;
    std::vector<double> A2;
    mlp.forward(X, A1, A2);
    double initial_loss = mlp.compute_bce_loss(y, A2);
    std::cout << "\n[1] Architecture: 2 -> 4 (Sigmoid) -> 1 (Sigmoid)" << std::endl;
    std::cout << "    Initial Untrained BCE Loss: " << initial_loss << std::endl;

    // Train loop
    std::cout << "\n[2] Training on Non-Linear XOR via Backpropagation:" << std::endl;
    for (int epoch = 1; epoch <= 1500; ++epoch) {
        mlp.train_step(X, y);
        if (epoch == 1 || epoch % 300 == 0 || epoch == 1500) {
            mlp.forward(X, A1, A2);
            double loss = mlp.compute_bce_loss(y, A2);
            std::cout << "    Epoch " << std::setw(4) << epoch << " | Loss: " << loss << std::endl;
        }
    }

    mlp.forward(X, A1, A2);
    double final_loss = mlp.compute_bce_loss(y, A2);
    assert(final_loss < 0.05);
    std::cout << "  -> Assertion PASSED: Loss converged from " << initial_loss << " to " << final_loss << std::endl;

    // Predictions
    std::cout << "\n[3] Final Trained Predictions on XOR:" << std::endl;
    for (size_t i = 0; i < X.size(); ++i) {
        int pred = (A2[i] >= 0.5) ? 1 : 0;
        std::cout << "    (" << X[i][0] << ", " << X[i][1] << ") -> Output: "
                  << std::fixed << std::setprecision(4) << A2[i] << " | Predicted: " << pred << std::endl;
        assert(pred == (int)y[i]);
    }
    std::cout << "  -> Assertion PASSED: XOR problem 100% solved by C++ MLP." << std::endl;

    std::cout << "=========================================================" << std::endl;
    std::cout << " ALL C++ MULTILAYER PERCEPTRON VERIFICATIONS PASSED!" << std::endl;
    std::cout << "=========================================================" << std::endl;
    return 0;
}
