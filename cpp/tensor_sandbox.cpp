/**
 * Tensor Mathematics Sandbox (From Scratch in Modern C++)
 * University Hackathon Project 13: Sign Language Recognition and Translation Assistant
 * Author: Gehna
 * Date: 2026-10-28
 *
 * Compiles cleanly with: g++ -O3 cpp/tensor_sandbox.cpp -o bin/tensor_sandbox.exe
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <cassert>

// =====================================================================
// 1. SIMPLE TENSOR CLASS IN C++
// =====================================================================
class SimpleTensor {
public:
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<double> data;

    SimpleTensor(const std::vector<int>& sh, double fill_val = 0.0) : shape(sh) {
        int total = 1;
        for (int s : shape) total *= s;
        data.assign(total, fill_val);
        compute_strides();
    }

    void compute_strides() {
        if (shape.empty()) return;
        strides.assign(shape.size(), 1);
        for (int i = (int)shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    int get_flat_index(const std::vector<int>& idx) const {
        int flat = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            flat += idx[i] * strides[i];
        }
        return flat;
    }

    double get(const std::vector<int>& idx) const {
        return data[get_flat_index(idx)];
    }

    void set(const std::vector<int>& idx, double val) {
        data[get_flat_index(idx)] = val;
    }

    SimpleTensor reshape(const std::vector<int>& new_shape) const {
        int total_new = 1;
        for (int s : new_shape) total_new *= s;
        if (total_new != (int)data.size()) {
            throw std::runtime_error("Cannot reshape: element count mismatch");
        }
        SimpleTensor res(new_shape);
        res.data = this->data;
        return res;
    }
};

// =====================================================================
// 2. VECTOR DOT & MATRIX MULTIPLICATION
// =====================================================================
double vector_dot(const std::vector<double>& u, const std::vector<double>& v) {
    if (u.size() != v.size()) throw std::runtime_error("Vector length mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < u.size(); ++i) sum += u[i] * v[i];
    return sum;
}

std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B
) {
    int M = A.size();
    int K = A[0].size();
    int K2 = B.size();
    int N = B[0].size();
    if (K != K2) throw std::runtime_error("Matmul inner dimension mismatch");

    std::vector<std::vector<double>> C(M, std::vector<double>(N, 0.0));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) {
                s += A[i][k] * B[k][j];
            }
            C[i][j] = s;
        }
    }
    return C;
}

// =====================================================================
// 3. SVD FOR 2x2 MATRIX
// =====================================================================
void svd_2x2(
    const std::vector<std::vector<double>>& A,
    std::vector<std::vector<double>>& U,
    std::vector<double>& Sigma,
    std::vector<std::vector<double>>& Vt
) {
    // AtA = A^T * A
    double a = A[0][0]*A[0][0] + A[1][0]*A[1][0];
    double b = A[0][0]*A[0][1] + A[1][0]*A[1][1];
    double c = b;
    double d = A[0][1]*A[0][1] + A[1][1]*A[1][1];

    double tr = a + d;
    double det = a * d - b * c;
    double disc = std::sqrt(std::max(0.0, tr * tr - 4.0 * det));
    double lambda1 = (tr + disc) / 2.0;
    double lambda2 = (tr - disc) / 2.0;

    Sigma.resize(2);
    Sigma[0] = std::sqrt(std::max(0.0, lambda1));
    Sigma[1] = std::sqrt(std::max(0.0, lambda2));

    double vx1 = -b, vy1 = a - lambda1;
    double norm1 = std::sqrt(vx1 * vx1 + vy1 * vy1);
    if (norm1 < 1e-9) { vx1 = 1.0; vy1 = 0.0; }
    else { vx1 /= norm1; vy1 /= norm1; }

    double vx2 = -vy1, vy2 = vx1;

    Vt = {{vx1, vy1}, {vx2, vy2}};

    U.assign(2, std::vector<double>(2, 0.0));
    if (Sigma[0] > 1e-9) {
        U[0][0] = (A[0][0]*vx1 + A[0][1]*vy1) / Sigma[0];
        U[1][0] = (A[1][0]*vx1 + A[1][1]*vy1) / Sigma[0];
    }
    if (Sigma[1] > 1e-9) {
        U[0][1] = (A[0][0]*vx2 + A[0][1]*vy2) / Sigma[1];
        U[1][1] = (A[1][0]*vx2 + A[1][1]*vy2) / Sigma[1];
    }
}

// =====================================================================
// 4. NUMERICAL FINITE-DIFFERENCE GRADIENT CHECKING
// =====================================================================
double objective_function(const std::vector<double>& w) {
    // f(w1, w2) = 3*w1^2 + 2*w1*w2 + 4*w2^3
    return 3.0 * (w[0] * w[0]) + 2.0 * w[0] * w[1] + 4.0 * (w[1] * w[1] * w[1]);
}

double check_gradients(const std::vector<double>& w, double eps = 1e-5) {
    // Analytical: df/dw1 = 6*w1 + 2*w2, df/dw2 = 2*w1 + 12*w2^2
    std::vector<double> anal_grad = {
        6.0 * w[0] + 2.0 * w[1],
        2.0 * w[0] + 12.0 * (w[1] * w[1])
    };

    std::vector<double> num_grad(w.size());
    for (size_t i = 0; i < w.size(); ++i) {
        std::vector<double> w_plus = w;
        std::vector<double> w_minus = w;
        w_plus[i] += eps;
        w_minus[i] -= eps;
        num_grad[i] = (objective_function(w_plus) - objective_function(w_minus)) / (2.0 * eps);
    }

    double diff_sq = 0.0, anal_sq = 0.0, num_sq = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
        diff_sq += (anal_grad[i] - num_grad[i]) * (anal_grad[i] - num_grad[i]);
        anal_sq += anal_grad[i] * anal_grad[i];
        num_sq += num_grad[i] * num_grad[i];
    }

    double rel_error = std::sqrt(diff_sq) / (std::sqrt(anal_sq) + std::sqrt(num_sq) + 1e-15);
    return rel_error;
}

// =====================================================================
// MAIN ENTRYPOINT
// =====================================================================
int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << " C++ TENSOR MATHEMATICS SANDBOX (Zero-Dependency STL)" << std::endl;
    std::cout << " University Requirement Compliance: Native C++ Module" << std::endl;
    std::cout << "=========================================================" << std::endl;

    // 1. Tensor Reshape
    SimpleTensor t({2, 2, 2, 2}, 1.0);
    std::cout << "[1] 4D Tensor created with elements: " << t.data.size() << std::endl;
    SimpleTensor t_mat = t.reshape({4, 4});
    std::cout << "    Reshaped to 2D matrix shape: (4, 4) -> Strides: (" 
              << t_mat.strides[0] << ", " << t_mat.strides[1] << ")" << std::endl;

    // 2. Vector Dot
    std::vector<double> u = {1.0, 2.0, 3.0};
    std::vector<double> v = {4.0, 5.0, 6.0};
    double dot = vector_dot(u, v);
    std::cout << "[2] Vector Dot Product: " << dot << " (Expected: 32.0)" << std::endl;
    assert(std::abs(dot - 32.0) < 1e-9);

    // 3. Matrix Multiplication
    std::vector<std::vector<double>> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<std::vector<double>> B = {{7.0, 8.0}, {9.0, 1.0}, {2.0, 3.0}};
    auto C = matrix_multiply(A, B);
    std::cout << "[3] Matmul C[0][0]: " << C[0][0] << " (Expected: 31.0), C[1][1]: " 
              << C[1][1] << " (Expected: 55.0)" << std::endl;
    assert(std::abs(C[0][0] - 31.0) < 1e-9);

    // 4. SVD 2x2
    std::vector<std::vector<double>> S_mat = {{3.0, 2.0}, {2.0, 3.0}};
    std::vector<std::vector<double>> U, Vt;
    std::vector<double> Sigma;
    svd_2x2(S_mat, U, Sigma, Vt);
    std::cout << "[4] SVD Singular Values: sigma1=" << Sigma[0] 
              << ", sigma2=" << Sigma[1] << " (Expected: 5.0, 1.0)" << std::endl;
    assert(std::abs(Sigma[0] - 5.0) < 1e-6);

    // 5. Gradient Check
    double rel_err = check_gradients({2.0, 1.5});
    std::cout << "[5] Numerical Gradient Check Relative Error: " << std::scientific 
              << rel_err << std::endl;
    assert(rel_err < 1e-7);
    std::cout << "    Assertion PASSED: Relative Error < 1e-7." << std::endl;

    std::cout << "=========================================================" << std::endl;
    std::cout << " ALL C++ TENSOR MATHEMATICS VERIFICATIONS PASSED!" << std::endl;
    std::cout << "=========================================================" << std::endl;
    return 0;
}
