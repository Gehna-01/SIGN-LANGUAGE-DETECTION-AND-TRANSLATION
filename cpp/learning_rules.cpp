/**
 * Foundational Neural Learning Rules in Modern C++
 * University Hackathon Project 13: Sign Language Recognition and Translation Assistant
 * Author: Gehna
 * Date: 2026-10-28
 *
 * Compiles cleanly with: g++ -O3 cpp/learning_rules.cpp -o bin/learning_rules.exe
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>

// 1. Classical Hebbian Learning
void test_hebbian() {
    std::cout << "\n[1] Classical Hebbian Learning:" << std::endl;
    std::vector<double> x = {1.0, 1.0};
    std::vector<double> w = {0.2, 0.2};
    double eta = 0.1;

    for (int step = 1; step <= 5; ++step) {
        double y = w[0]*x[0] + w[1]*x[1];
        w[0] += eta * x[0] * y;
        w[1] += eta * x[1] * y;
    }
    std::cout << "    Final weights after 5 steps: [" << w[0] << ", " << w[1] << "] (Unbounded growth verified)." << std::endl;
    assert(w[0] > 0.45);
}

// 2. Modified Hebbian Learning (Oja's Rule)
void test_ojas_rule() {
    std::cout << "\n[2] Modified Hebbian Learning (Oja's Rule):" << std::endl;
    // Data with principal component along [2, 1] / sqrt(5) = [0.8944, 0.4472]
    std::vector<double> w = {0.1, 0.9};
    double norm_init = std::sqrt(w[0]*w[0] + w[1]*w[1]);
    w[0] /= norm_init; w[1] /= norm_init;
    double eta = 0.05;

    std::vector<std::vector<double>> data = {
        {2.0, 1.0}, {-2.0, -1.0}, {1.9, 0.95}, {-1.9, -0.95},
        {2.1, 1.05}, {-2.1, -1.05}
    };

    for (int epoch = 0; epoch < 50; ++epoch) {
        for (const auto& x : data) {
            double y = w[0]*x[0] + w[1]*x[1];
            w[0] += eta * y * (x[0] - y * w[0]);
            w[1] += eta * y * (x[1] - y * w[1]);
        }
    }

    double final_norm = std::sqrt(w[0]*w[0] + w[1]*w[1]);
    double cos_sim = std::abs(w[0]*0.894427 + w[1]*0.447214);
    std::cout << "    Final weight: [" << w[0] << ", " << w[1] << "] | Norm: " << final_norm << " | Cosine Sim to PCA: " << cos_sim << std::endl;
    assert(std::abs(final_norm - 1.0) < 0.01);
    assert(cos_sim > 0.98);
    std::cout << "  -> Assertion PASSED: Oja's rule extracted PCA with unit norm." << std::endl;
}

// 3. Competitive Learning
void test_competitive() {
    std::cout << "\n[3] Competitive Learning (Winner-Take-All):" << std::endl;
    std::vector<std::vector<double>> samples = {
        {-2.0, -2.1}, {-1.9, -2.0}, {-2.1, -1.9},
        { 2.0,  2.1}, { 1.9,  2.0}, { 2.1,  1.9}
    };

    std::vector<std::vector<double>> proto = {{-0.1, 0.1}, {0.1, -0.1}};
    double eta = 0.1;

    for (int epoch = 0; epoch < 20; ++epoch) {
        for (const auto& x : samples) {
            double d0 = std::sqrt((x[0]-proto[0][0])*(x[0]-proto[0][0]) + (x[1]-proto[0][1])*(x[1]-proto[0][1]));
            double d1 = std::sqrt((x[0]-proto[1][0])*(x[0]-proto[1][0]) + (x[1]-proto[1][1])*(x[1]-proto[1][1]));
            int winner = (d0 < d1) ? 0 : 1;

            proto[winner][0] += eta * (x[0] - proto[winner][0]);
            proto[winner][1] += eta * (x[1] - proto[winner][1]);
        }
    }

    std::cout << "    Prototype 1: [" << proto[0][0] << ", " << proto[0][1] << "]" << std::endl;
    std::cout << "    Prototype 2: [" << proto[1][0] << ", " << proto[1][1] << "]" << std::endl;
    assert((proto[0][0] < 0 && proto[1][0] > 0) || (proto[0][0] > 0 && proto[1][0] < 0));
    std::cout << "  -> Assertion PASSED: Prototypes converged to opposite clusters." << std::endl;
}

// 4. Error-Correction (Delta Rule)
void test_error_correction() {
    std::cout << "\n[4] Error-Correction Learning (Delta Rule):" << std::endl;
    std::vector<std::vector<double>> X = {{1.0, 0.0}, {0.0, 1.0}, {2.0, 1.0}, {1.0, 2.0}};
    std::vector<double> d = {3.0, -1.5, 4.5, 0.0}; // target: w1=3.0, w2=-1.5

    std::vector<double> w = {0.0, 0.0};
    double eta = 0.1;

    for (int epoch = 0; epoch < 40; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            double y = w[0]*X[i][0] + w[1]*X[i][1];
            double err = d[i] - y;
            w[0] += eta * err * X[i][0];
            w[1] += eta * err * X[i][1];
        }
    }

    std::cout << "    Learned weights: [" << w[0] << ", " << w[1] << "] (Target: [3.0, -1.5])" << std::endl;
    assert(std::abs(w[0] - 3.0) < 0.01);
    assert(std::abs(w[1] - (-1.5)) < 0.01);
    std::cout << "  -> Assertion PASSED: Delta rule recovered exact linear targets." << std::endl;
}

// 5. Boltzmann / Energy-Based Model
void test_boltzmann() {
    std::cout << "\n[5] Boltzmann / Energy-Based Learning:" << std::endl;
    double data_corr = 0.80; // target correlated co-activation
    double w12 = 0.0;
    double eta = 0.5;
    double T = 1.0;

    for (int step = 0; step < 10; ++step) {
        double p_active = 1.0 / (1.0 + std::exp(-w12 / T));
        double dw12 = eta * (data_corr - p_active);
        w12 += dw12;
    }
    double energy_11 = -w12;
    std::cout << "    Final w12: " << w12 << " | Target State Energy E(1,1): " << energy_11 << std::endl;
    assert(w12 > 0.5);
    assert(energy_11 < -0.5);
    std::cout << "  -> Assertion PASSED: Boltzmann energy of correlated state lowered." << std::endl;
}

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << " C++ NEURAL LEARNING RULES (Zero-Dependency STL)" << std::endl;
    std::cout << " University Requirement Compliance: Native C++ Module" << std::endl;
    std::cout << "=========================================================" << std::endl;

    test_hebbian();
    test_ojas_rule();
    test_competitive();
    test_error_correction();
    test_boltzmann();

    std::cout << "=========================================================" << std::endl;
    std::cout << " ALL C++ LEARNING RULES VERIFICATIONS PASSED!" << std::endl;
    std::cout << "=========================================================" << std::endl;
    return 0;
}
