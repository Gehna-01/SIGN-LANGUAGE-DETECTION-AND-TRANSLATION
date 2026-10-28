/**
 * ============================================================================
 * ACTIVATION FUNCTIONS COMPARISON AND VANISHING GRADIENT SIMULATION
 * ============================================================================
 * Pure Modern C++11 Implementation of Core Neural Network Activation Functions:
 * 1. Heaviside Step Function
 * 2. Sigmoid (Logistic) Function
 * 3. Hyperbolic Tangent (Tanh) Function
 * 4. Rectified Linear Unit (ReLU)
 * 5. Leaky Rectified Linear Unit (LeakyReLU)
 * 
 * Computes forward activations, exact analytical derivatives, and simulates
 * deep 10-layer gradient backpropagation to demonstrate vanishing gradients.
 * ============================================================================
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>

namespace nn {

// 1. Heaviside Step Function
inline double step_function(double z) {
    return (z >= 0.0) ? 1.0 : 0.0;
}

inline double step_derivative(double /*z*/) {
    // Derivative is zero almost everywhere; Dirac delta at 0.
    // In practice, gradient descent is completely impossible with step functions.
    return 0.0;
}

// 2. Logistic Sigmoid Function
inline double sigmoid(double z) {
    if (z > 45.0) return 1.0;
    if (z < -45.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

inline double sigmoid_derivative(double z) {
    double s = sigmoid(z);
    return s * (1.0 - s); // Max value is 0.25 at z = 0
}

// 3. Hyperbolic Tangent Function
inline double tanh_fn(double z) {
    return std::tanh(z);
}

inline double tanh_derivative(double z) {
    double t = std::tanh(z);
    return 1.0 - (t * t); // Max value is 1.0 at z = 0
}

// 4. Rectified Linear Unit (ReLU)
inline double relu(double z) {
    return (z > 0.0) ? z : 0.0;
}

inline double relu_derivative(double z) {
    // Constant gradient 1.0 for positive preactivations (non-saturating).
    // Zero gradient for negative preactivations (can cause 'Dying ReLU').
    return (z > 0.0) ? 1.0 : 0.0;
}

// 5. Leaky Rectified Linear Unit (LeakyReLU)
inline double leaky_relu(double z, double alpha = 0.01) {
    return (z > 0.0) ? z : (alpha * z);
}

inline double leaky_relu_derivative(double z, double alpha = 0.01) {
    // Non-zero gradient alpha for negative inputs prevents dead-neuron traps.
    return (z > 0.0) ? 1.0 : alpha;
}

} // namespace nn

int main() {
    std::cout << "==========================================================================================\n";
    std::cout << "  STEP 15: ACTIVATION FUNCTIONS & DERIVATIVES COMPARISON (C++11 FROM SCRATCH)\n";
    std::cout << "==========================================================================================\n\n";

    std::vector<double> test_points = {-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0};

    // Table 1: Forward Activations f(z)
    std::cout << "TABLE 1: FORWARD ACTIVATION OUTPUTS f(z)\n";
    std::cout << std::string(90, '-') << "\n";
    std::cout << std::left << std::setw(12) << "Input (z)"
              << std::setw(12) << "Step"
              << std::setw(14) << "Sigmoid"
              << std::setw(14) << "Tanh"
              << std::setw(14) << "ReLU"
              << std::setw(18) << "LeakyReLU (0.01)" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (double z : test_points) {
        std::cout << std::left << std::fixed << std::setprecision(4)
                  << std::setw(12) << z
                  << std::setw(12) << nn::step_function(z)
                  << std::setw(14) << nn::sigmoid(z)
                  << std::setw(14) << nn::tanh_fn(z)
                  << std::setw(14) << nn::relu(z)
                  << std::setw(18) << nn::leaky_relu(z, 0.01) << "\n";
    }
    std::cout << std::string(90, '-') << "\n\n";

    // Table 2: Analytical Derivatives f'(z)
    std::cout << "TABLE 2: ANALYTICAL DERIVATIVES f'(z)\n";
    std::cout << std::string(90, '-') << "\n";
    std::cout << std::left << std::setw(12) << "Input (z)"
              << std::setw(12) << "Step d/dz"
              << std::setw(14) << "Sigmoid d/dz"
              << std::setw(14) << "Tanh d/dz"
              << std::setw(14) << "ReLU d/dz"
              << std::setw(18) << "LeakyReLU d/dz" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (double z : test_points) {
        std::cout << std::left << std::fixed << std::setprecision(4)
                  << std::setw(12) << z
                  << std::setw(12) << nn::step_derivative(z)
                  << std::setw(14) << nn::sigmoid_derivative(z)
                  << std::setw(14) << nn::tanh_derivative(z)
                  << std::setw(14) << nn::relu_derivative(z)
                  << std::setw(18) << nn::leaky_relu_derivative(z, 0.01) << "\n";
    }
    std::cout << std::string(90, '-') << "\n\n";

    // Table 3: 10-Layer Vanishing Gradient Simulation
    std::cout << "TABLE 3: 10-LAYER BACKPROPAGATION GRADIENT RETENTION (MILD PREACTIVATION REGIME)\n";
    std::cout << "Simulating: prod_{l=1}^L f'(z_l) along backpropagation path across 10 layers\n";
    std::cout << std::string(90, '-') << "\n";
    std::cout << std::left << std::setw(10) << "Layer"
              << std::setw(18) << "Sigmoid Grad"
              << std::setw(18) << "Tanh Grad"
              << std::setw(18) << "ReLU Grad"
              << std::setw(20) << "LeakyReLU Grad" << "\n";
    std::cout << std::string(90, '-') << "\n";

    std::vector<double> preacts = {0.5, 0.8, 1.0, 0.4, 0.7, 1.2, 0.3, 0.6, 0.9, 1.1};
    double cum_sig = 1.0;
    double cum_tanh = 1.0;
    double cum_relu = 1.0;
    double cum_lrelu = 1.0;

    for (size_t l = 0; l < 10; ++l) {
        double z = preacts[l % preacts.size()];
        cum_sig   *= nn::sigmoid_derivative(z);
        cum_tanh  *= nn::tanh_derivative(z);
        cum_relu  *= nn::relu_derivative(z);
        cum_lrelu *= nn::leaky_relu_derivative(z, 0.01);

        std::cout << std::left
                  << "Layer " << std::setw(4) << (l + 1)
                  << std::scientific << std::setprecision(3)
                  << std::setw(18) << cum_sig
                  << std::setw(18) << cum_tanh
                  << std::setw(18) << cum_relu
                  << std::setw(20) << cum_lrelu << "\n";
    }
    std::cout << std::string(90, '-') << "\n\n";

    std::cout << "ARCHITECTURAL INSIGHTS FOR SIGN RECOGNITION PIPELINE:\n";
    std::cout << "1. Why CNN Backbones Use ReLU / LeakyReLU:\n";
    std::cout << "   - Constant unit gradient (f'(z) = 1.0) prevents vanishing gradients across deep conv layers.\n";
    std::cout << "   - Encourages sparse representations (zeros out non-informative background/ambient pixels).\n";
    std::cout << "   - Computationally ultra-efficient (single max(0, z) threshold comparison without exp()).\n\n";
    std::cout << "2. Why Recurrent Gates (LSTM / GRU) Use Sigmoid and Tanh:\n";
    std::cout << "   - Sigmoid outputs (0, 1) model strictly probabilistic 'gates' (how much past sign memory to retain/forget).\n";
    std::cout << "   - Tanh outputs (-1, 1) regulate candidate cell states and hidden states symmetrically around zero.\n\n";

    return 0;
}
