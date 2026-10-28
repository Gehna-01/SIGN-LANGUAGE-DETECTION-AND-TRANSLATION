/**
 * Numerical Finite-Difference Gradient Checking on MLP in Modern C++
 * University Hackathon Project 13: Sign Language Recognition and Translation Assistant
 * Author: Gehna
 * Date: 2026-10-28
 *
 * Compiles cleanly with: g++ -O3 cpp/gradient_checking.cpp -o bin/gradient_checking.exe
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

class MLPForGradCheck {
public:
    int n_in = 2, n_hidden = 4, n_out = 1;

    // 17 Parameters
    std::vector<std::vector<double>> W1; // 2x4 (8)
    std::vector<double> b1;              // 1x4 (4)
    std::vector<std::vector<double>> W2; // 4x1 (4)
    double b2;                           // 1x1 (1)

    MLPForGradCheck() {
        W1 = {
            { 0.5, -0.6,  0.8, -0.4},
            {-0.7,  0.9, -0.5,  0.6}
        };
        b1 = {0.1, -0.2, 0.3, -0.1};

        W2 = {
            { 0.8},
            {-0.9},
            { 0.7},
            {-0.6}
        };
        b2 = 0.2;
    }

    double forward_and_loss(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        std::vector<std::vector<double>>* out_A1 = nullptr,
        std::vector<double>* out_A2 = nullptr
    ) const {
        size_t N = X.size();
        std::vector<std::vector<double>> A1(N, std::vector<double>(n_hidden, 0.0));
        std::vector<double> A2(N, 0.0);

        for (size_t i = 0; i < N; ++i) {
            for (int h = 0; h < n_hidden; ++h) {
                double z1 = b1[h];
                for (int d = 0; d < n_in; ++d) z1 += X[i][d] * W1[d][h];
                A1[i][h] = sigmoid(z1);
            }
            double z2 = b2;
            for (int h = 0; h < n_hidden; ++h) z2 += A1[i][h] * W2[h][0];
            A2[i] = sigmoid(z2);
        }

        if (out_A1) *out_A1 = A1;
        if (out_A2) *out_A2 = A2;

        double total_loss = 0.0;
        double eps = 1e-15;
        for (size_t i = 0; i < N; ++i) {
            double p = std::min(std::max(A2[i], eps), 1.0 - eps);
            total_loss += -(y[i] * std::log(p) + (1.0 - y[i]) * std::log(1.0 - p));
        }
        return total_loss / (double)N;
    }

    std::vector<double> compute_analytical_gradients(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y
    ) const {
        size_t N = X.size();
        std::vector<std::vector<double>> A1;
        std::vector<double> A2;
        forward_and_loss(X, y, &A1, &A2);

        std::vector<std::vector<double>> dW2(n_hidden, std::vector<double>(n_out, 0.0));
        double db2 = 0.0;
        std::vector<std::vector<double>> dW1(n_in, std::vector<double>(n_hidden, 0.0));
        std::vector<double> db1(n_hidden, 0.0);

        for (size_t i = 0; i < N; ++i) {
            double delta2 = A2[i] - y[i];
            for (int h = 0; h < n_hidden; ++h) dW2[h][0] += delta2 * A1[i][h];
            db2 += delta2;

            for (int h = 0; h < n_hidden; ++h) {
                double delta1_h = delta2 * W2[h][0] * sigmoid_derivative(A1[i][h]);
                for (int d = 0; d < n_in; ++d) dW1[d][h] += delta1_h * X[i][d];
                db1[h] += delta1_h;
            }
        }

        double inv_N = 1.0 / (double)N;
        std::vector<double> flat_grads;
        // W1 (8)
        for (int d = 0; d < n_in; ++d)
            for (int h = 0; h < n_hidden; ++h)
                flat_grads.push_back(dW1[d][h] * inv_N);
        // b1 (4)
        for (int h = 0; h < n_hidden; ++h)
            flat_grads.push_back(db1[h] * inv_N);
        // W2 (4)
        for (int h = 0; h < n_hidden; ++h)
            flat_grads.push_back(dW2[h][0] * inv_N);
        // b2 (1)
        flat_grads.push_back(db2 * inv_N);

        return flat_grads;
    }

    // Helper to get and set 1D parameter array
    std::vector<double> get_params() const {
        std::vector<double> p;
        for (int d = 0; d < n_in; ++d)
            for (int h = 0; h < n_hidden; ++h)
                p.push_back(W1[d][h]);
        for (int h = 0; h < n_hidden; ++h) p.push_back(b1[h]);
        for (int h = 0; h < n_hidden; ++h) p.push_back(W2[h][0]);
        p.push_back(b2);
        return p;
    }

    void set_params(const std::vector<double>& p) {
        size_t idx = 0;
        for (int d = 0; d < n_in; ++d)
            for (int h = 0; h < n_hidden; ++h)
                W1[d][h] = p[idx++];
        for (int h = 0; h < n_hidden; ++h) b1[h] = p[idx++];
        for (int h = 0; h < n_hidden; ++h) W2[h][0] = p[idx++];
        b2 = p[idx++];
    }
};

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << " C++ FINITE-DIFFERENCE GRADIENT CHECKING ON MLP" << std::endl;
    std::cout << " University Requirement Compliance: Native C++ Module" << std::endl;
    std::cout << "=========================================================" << std::endl;

    std::vector<std::vector<double>> X = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    std::vector<double> y = {0.0, 1.0, 1.0, 0.0};

    MLPForGradCheck mlp;
    double eps = 1e-7;

    // 1. Analytical Gradients
    std::vector<double> anal_grads = mlp.compute_analytical_gradients(X, y);

    // 2. Numerical Gradients
    std::vector<double> theta = mlp.get_params();
    std::vector<double> num_grads(theta.size(), 0.0);

    for (size_t i = 0; i < theta.size(); ++i) {
        std::vector<double> theta_plus = theta;
        theta_plus[i] += eps;
        mlp.set_params(theta_plus);
        double l_plus = mlp.forward_and_loss(X, y);

        std::vector<double> theta_minus = theta;
        theta_minus[i] -= eps;
        mlp.set_params(theta_minus);
        double l_minus = mlp.forward_and_loss(X, y);

        num_grads[i] = (l_plus - l_minus) / (2.0 * eps);
    }
    mlp.set_params(theta); // Restore

    // 3. Relative Error
    double diff_sq = 0.0, anal_sq = 0.0, num_sq = 0.0;
    std::cout << " Param Index | Analytical Grad | Numerical Grad | Difference" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    for (size_t i = 0; i < theta.size(); ++i) {
        double d = std::abs(anal_grads[i] - num_grads[i]);
        diff_sq += (anal_grads[i] - num_grads[i]) * (anal_grads[i] - num_grads[i]);
        anal_sq += anal_grads[i] * anal_grads[i];
        num_sq += num_grads[i] * num_grads[i];
        std::cout << "   Param " << std::setw(2) << i << "  |   " 
                  << std::scientific << std::setprecision(6) << anal_grads[i] << "  |   "
                  << num_grads[i] << "  |  " << d << std::endl;
    }

    double rel_error = std::sqrt(diff_sq) / (std::sqrt(anal_sq) + std::sqrt(num_sq) + 1e-15);
    std::cout << "\nCalculated Relative Error: " << std::scientific << rel_error << std::endl;

    assert(rel_error < 1e-7);
    std::cout << "  -> Assertion PASSED: C++ Relative Error < 1e-7 (Zero Bugs in Backprop!)." << std::endl;

    std::cout << "=========================================================" << std::endl;
    std::cout << " ALL C++ GRADIENT CHECKING VERIFICATIONS PASSED!" << std::endl;
    std::cout << "=========================================================" << std::endl;
    return 0;
}
