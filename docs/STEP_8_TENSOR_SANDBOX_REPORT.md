# Step 8: Tensor Mathematics Sandbox (From First Principles)

**Project Title:** Sign Language Recognition and Translation Assistant  
**Project Number:** Project 13  
**Domain:** Assistive Technology  
**Milestone:** Phase I — Step 8 (Tensor Mathematics Fundamentals)  
**Execution Date:** October 28, 2026  
**Implementation Languages:** Python (NumPy Verification) & Modern C++11 (Zero-Dependency STL)  

---

## 1. Pedagogical Objective & Language Architecture
Section 1 of the project guidelines mandates demonstrating core neural-network fundamentals from first principles without relying on black-box deep-learning frameworks (such as PyTorch autograd or TensorFlow) to hide the underlying linear algebra and calculus.

To satisfy both transparent educational verification and the university expectation for a native C/C++ application, we implemented the Tensor Mathematics Sandbox in two synchronized implementations:
1. **Python Module ([`src/fundamentals/tensor_sandbox.py`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/src/fundamentals/tensor_sandbox.py)):** Built from scratch using standard loops and lists, cross-verified against NumPy's linear algebra engine.
2. **Native C++ Module ([`cpp/tensor_sandbox.cpp`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/cpp/tensor_sandbox.cpp)):** Built in pure C++11 using standard library vectors (`std::vector`) and compiled via MinGW `g++ -O3`.

---

## 2. Mathematical Concepts Implemented

### A. Tensors, Memory Layout & Strides
A tensor of rank $D$ with shape $(S_0, S_1, \dots, S_{D-1})$ is stored contiguously in 1D row-major memory. The flat index for a multi-dimensional coordinate $(i_0, i_1, \dots, i_{D-1})$ is computed using precomputed strides:
$$\text{flat\_index} = \sum_{k=0}^{D-1} i_k \cdot \text{stride}_k, \quad \text{where } \text{stride}_k = \prod_{j=k+1}^{D-1} S_j$$
- **Reshaping:** Reinterprets the dimensional view of the contiguous buffer without reallocating underlying data, provided total element count $\prod S_k = \prod S'_k$ is conserved.

### B. Broadcasting Engine
Implements standard tensor broadcasting semantics:
1. Trailing dimensions are aligned from right to left.
2. Two dimensions are compatible if:
   - They are equal ($S_{a,k} = S_{b,k}$), or
   - One of them is 1 ($S_{a,k} = 1$ or $S_{b,k} = 1$).
3. The resulting dimension is $\max(S_{a,k}, S_{b,k})$, stretching the unit dimension across the axis.

### C. Inner Product & Matrix Multiplication
- **Vector Dot Product:**
  $$u \cdot v = \sum_{i=1}^n u_i v_i$$
- **Matrix Multiplication:** Triple-loop standard multiplication for $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$:
  $$C_{ij} = \sum_{k=1}^K A_{ik} B_{kj}, \quad \text{Complexity: } \mathcal{O}(M \cdot K \cdot N)$$

### D. Singular Value Decomposition (SVD)
Decomposes any real matrix $A \in \mathbb{R}^{M \times N}$ into:
$$A = U \Sigma V^T$$
Where:
- $U \in \mathbb{R}^{M \times M}$ contains the orthonormal left singular vectors (eigenvectors of $A A^T$).
- $\Sigma \in \mathbb{R}^{M \times N}$ is a diagonal matrix containing non-negative singular values $\sigma_i = \sqrt{\lambda_i}$.
- $V \in \mathbb{R}^{N \times N}$ contains the orthonormal right singular vectors (eigenvectors of $A^T A$).
- **Reconstruction Verification:** $A_{\text{reconstructed}} = U \Sigma V^T \approx A$ with error $< 10^{-7}$.

### E. Finite-Difference Numerical Gradient Checking
Gradient checking verifies that analytical derivatives computed during backpropagation are exact:
$$\frac{\partial f}{\partial w_i} \approx \frac{f(w + \epsilon e_i) - f(w - \epsilon e_i)}{2\epsilon}$$
- **Relative Error Metric:**
  $$\text{Relative Error} = \frac{\|g_{\text{analytical}} - g_{\text{numerical}}\|_2}{\|g_{\text{analytical}}\|_2 + \|g_{\text{numerical}}\|_2 + 10^{-15}}$$
- **Evaluation Standard:** $\text{Relative Error} \le 10^{-7}$ signifies verified mathematical precision.

---

## 3. Empirical Verification Results

### Python vs. NumPy Verification Suite
```
[1] TENSOR CREATION & RESHAPING
4D Tensor Shape: (2, 2, 2, 2) -> Strides: (8, 4, 2, 1)
Reshaped 4D Tensor into 2D Matrix: (4, 4)
  -> Assertion PASSED: Manual row-major reshape verified.

[2] MANUAL BROADCASTING ENGINE
Tensor A shape: (3, 1), Tensor B shape: (1, 4) -> Broadcasted (A + B) shape: (3, 4)
  -> Assertion PASSED: Manual broadcasting exactly matches NumPy.

[3] INNER PRODUCT & MATRIX MULTIPLICATION
Vector u . v = 32.0 (Matches NumPy: 32.0)
Matrix A (2x3) x Matrix B (3x2) -> Matrix C (2x2)
  -> Assertion PASSED: Matrix multiplication exactly matches NumPy.

[4] SINGULAR VALUE DECOMPOSITION (SVD)
Original Matrix A: [[3. 2.], [2. 3.]]
Manual Singular Values Sigma: [[5. 0.], [0. 1.]]
Reconstructed Matrix (U * Sigma * V^T): [[3. 2.], [2. 3.]]
  -> Assertion PASSED: SVD reconstruction error < 1e-7.

[5] NUMERICAL GRADIENT CHECKING (FINITE DIFFERENCES)
Analytical Gradient:        [15.0, 31.0]
Finite-Difference Gradient: [15.0, 31.0]
Calculated Relative Error:  4.15e-12
  -> Assertion PASSED: Relative Error (4.15e-12) < 1e-7 (Verified mathematically correct!).
```

### Native C++ Execution Verification
Compiled via MinGW `g++ -O3 cpp/tensor_sandbox.cpp -o bin/tensor_sandbox.exe`:
```
=========================================================
 C++ TENSOR MATHEMATICS SANDBOX (Zero-Dependency STL)
 University Requirement Compliance: Native C++ Module
=========================================================
[1] 4D Tensor created with elements: 16 -> Reshaped to (4, 4) Strides: (4, 1)
[2] Vector Dot Product: 32 (Expected: 32.0)
[3] Matmul C[0][0]: 31 (Expected: 31.0), C[1][1]: 55 (Expected: 55.0)
[4] SVD Singular Values: sigma1=5, sigma2=1 (Expected: 5.0, 1.0)
[5] Numerical Gradient Check Relative Error: 8.871870e-012
    Assertion PASSED: Relative Error < 1e-7.
=========================================================
 ALL C++ TENSOR MATHEMATICS VERIFICATIONS PASSED!
=========================================================
```

---

## 4. Summary Table

| Operation | Implementation Type | Test Case | Relative Error / Tolerance | Verification Status |
|---|---|---|---|---|
| **Tensor Reshape** | Row-Major Indexing | $(2, 2, 2, 2) \to (4, 4)$ | Exact match ($0.0$) | **PASSED** |
| **Broadcasting** | Unit Dimension Stretch | $(3, 1) + (1, 4) \to (3, 4)$ | Exact match ($0.0$) | **PASSED** |
| **Vector Dot** | Sum of Products | $[1,2,3] \cdot [4,5,6] = 32$ | Exact match ($0.0$) | **PASSED** |
| **Matrix Matmul** | $O(M \times K \times N)$ Loops | $(2 \times 3) \times (3 \times 2) \to (2 \times 2)$ | Exact match ($0.0$) | **PASSED** |
| **SVD Decomposition** | Eigen-Analysis $A^T A$ | Sym $2 \times 2$ Matrix | Error $< 10^{-7}$ | **PASSED** |
| **Gradient Check** | Central Difference | Multi-variate Polynomial | **$4.15 \times 10^{-12}$ (Python), $8.87 \times 10^{-12}$ (C++)** | **PASSED** |
