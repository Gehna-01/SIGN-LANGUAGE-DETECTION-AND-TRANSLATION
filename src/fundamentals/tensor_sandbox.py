"""
Tensor Mathematics Sandbox (Implemented From Scratch)
University Hackathon Project 13: Sign Language Recognition and Translation Assistant
Author: Gehna
Date: 2026-10-28

Demonstrates foundational tensor algebra from first principles:
- Scalar, vector, matrix, and 3D/4D tensor representations
- Manual reshaping, stride calculation, and row-major layout
- Manual broadcasting engine
- Inner products, dot products, and matrix multiplication
- Tensor reduction operations, trace, Frobenius norm
- Singular Value Decomposition (SVD) via power iteration / eigen-analysis
- Finite-difference numerical gradient checking
"""

import math
import numpy as np

# =====================================================================
# 1. TENSOR CREATION, INDEXING & SHAPES
# =====================================================================

class SimpleTensor:
    """
    A transparent, educational tensor class implemented from first principles.
    Stores multi-dimensional data in a flat 1D list using row-major indexing.
    """
    def __init__(self, data, shape=None):
        if shape is None:
            # Infer shape and flatten nested Python lists
            self.shape, self.data = self._infer_shape_and_flatten(data)
        else:
            self.shape = tuple(shape)
            if isinstance(data, list):
                self.data = list(data)
            else:
                self.data = [float(data)] * self._total_elements(self.shape)
        
        self.strides = self._compute_strides(self.shape)

    def _infer_shape_and_flatten(self, data):
        if not isinstance(data, list):
            return (), [float(data)]
        
        shape = []
        curr = data
        while isinstance(curr, list):
            shape.append(len(curr))
            curr = curr[0] if len(curr) > 0 else 0
            
        def _flatten(nested):
            flat = []
            for item in nested:
                if isinstance(item, list):
                    flat.extend(_flatten(item))
                else:
                    flat.append(float(item))
            return flat
            
        return tuple(shape), _flatten(data)

    @staticmethod
    def _total_elements(shape):
        if len(shape) == 0:
            return 1
        prod = 1
        for s in shape:
            prod *= s
        return prod

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    def _get_flat_index(self, indices):
        if len(indices) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)} indices, got {len(indices)}")
        flat_idx = 0
        for i, (idx, dim, stride) in enumerate(zip(indices, self.shape, self.strides)):
            if idx < 0 or idx >= dim:
                raise IndexError(f"Index {idx} out of bounds for axis {i} with size {dim}")
            flat_idx += idx * stride
        return flat_idx

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) == len(self.shape):
            return self.data[self._get_flat_index(indices)]
        else:
            raise NotImplementedError("Arbitrary multi-axis slicing simplified to scalar access for transparency.")

    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)
        self.data[self._get_flat_index(indices)] = float(value)

    def reshape(self, new_shape):
        if self._total_elements(new_shape) != len(self.data):
            raise ValueError(f"Cannot reshape tensor of size {len(self.data)} into shape {new_shape}")
        return SimpleTensor(list(self.data), shape=new_shape)

    def to_numpy(self):
        return np.array(self.data).reshape(self.shape)

    def __repr__(self):
        return f"SimpleTensor(shape={self.shape}, data={self.to_numpy().tolist()})"

# =====================================================================
# 2. MANUAL BROADCASTING ENGINE
# =====================================================================

def manual_broadcast_shapes(shape_a, shape_b):
    """
    Computes resulting broadcast shape following standard NumPy/PyTorch rules:
    Align trailing dimensions; dimensions match if they are equal or one of them is 1.
    """
    len_a = len(shape_a)
    len_b = len(shape_b)
    max_len = max(len_a, len_b)
    
    pad_a = (1,) * (max_len - len_a) + shape_a
    pad_b = (1,) * (max_len - len_b) + shape_b
    
    res_shape = []
    for da, db in zip(pad_a, pad_b):
        if da == db:
            res_shape.append(da)
        elif da == 1:
            res_shape.append(db)
        elif db == 1:
            res_shape.append(da)
        else:
            raise ValueError(f"Shapes {shape_a} and {shape_b} are not broadcastable")
    return tuple(res_shape)

def manual_elementwise_add(t_a, t_b):
    """
    Elementwise addition supporting manual broadcasting.
    """
    out_shape = manual_broadcast_shapes(t_a.shape, t_b.shape)
    out_tensor = SimpleTensor(0.0, shape=out_shape)
    
    # Expand coordinates for each element
    total = SimpleTensor._total_elements(out_shape)
    out_strides = SimpleTensor._compute_strides(out_shape)
    
    pad_a = (1,) * (len(out_shape) - len(t_a.shape)) + t_a.shape
    pad_b = (1,) * (len(out_shape) - len(t_b.shape)) + t_b.shape
    
    for flat_idx in range(total):
        # Convert flat index to multi-index
        coords = []
        rem = flat_idx
        for s in out_strides:
            coords.append(rem // s)
            rem %= s
            
        # Map to t_a coords (mod 1 if dim was expanded)
        coords_a = tuple(c if dim > 1 else 0 for c, dim in zip(coords, pad_a))[-len(t_a.shape):]
        coords_b = tuple(c if dim > 1 else 0 for c, dim in zip(coords, pad_b))[-len(t_b.shape):]
        
        val_a = t_a[coords_a] if len(coords_a) > 0 else t_a.data[0]
        val_b = t_b[coords_b] if len(coords_b) > 0 else t_b.data[0]
        
        out_tensor.data[flat_idx] = val_a + val_b
        
    return out_tensor

# =====================================================================
# 3. VECTOR & MATRIX OPERATIONS (FROM SCRATCH)
# =====================================================================

def manual_vector_dot(u, v):
    """
    Inner product of two vectors: u . v = sum(u_i * v_i)
    """
    if len(u) != len(v):
        raise ValueError("Vectors must have equal length for dot product")
    dot = 0.0
    for a, b in zip(u, v):
        dot += a * b
    return dot

def manual_matrix_multiply(A, B):
    """
    Triple-loop standard matrix multiplication: C[i, j] = sum_k(A[i, k] * B[k, j])
    A: (M, K), B: (K, N) -> C: (M, N)
    """
    M = len(A)
    K = len(A[0])
    K2 = len(B)
    N = len(B[0])
    
    if K != K2:
        raise ValueError(f"Incompatible shapes for matmul: ({M},{K}) and ({K2},{N})")
        
    C = [[0.0 for _ in range(N)] for _ in range(M)]
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def manual_matrix_transpose(A):
    """
    Matrix transpose: A^T[j, i] = A[i, j]
    """
    M = len(A)
    N = len(A[0])
    return [[A[i][j] for i in range(M)] for j in range(N)]

def manual_frobenius_norm(A):
    """
    Frobenius norm: ||A||_F = sqrt(sum(A_ij^2))
    """
    s = 0.0
    for row in A:
        for val in row:
            s += val * val
    return math.sqrt(s)

# =====================================================================
# 4. SINGULAR VALUE DECOMPOSITION (SVD) FROM SCRATCH
# =====================================================================

def manual_svd_2x2(A):
    """
    Closed-form and power-iteration educational SVD for a 2x2 matrix:
    A = U * Sigma * V^T
    1. Compute A^T A (symmetric 2x2)
    2. Solve characteristic equation to find eigenvalues lambda_1, lambda_2
    3. Singular values are sigma_i = sqrt(lambda_i)
    4. Compute right singular vectors V and left singular vectors U = A * V / sigma
    """
    # A^T A
    At = manual_matrix_transpose(A)
    AtA = manual_matrix_multiply(At, A)
    
    # Characteristic polynomial for 2x2: det(AtA - lambda*I) = 0
    # lambda^2 - tr(AtA)*lambda + det(AtA) = 0
    a, b = AtA[0][0], AtA[0][1]
    c, d = AtA[1][0], AtA[1][1]
    
    tr = a + d
    det = a * d - b * c
    
    disc = math.sqrt(max(0.0, tr * tr - 4 * det))
    lambda1 = (tr + disc) / 2.0
    lambda2 = (tr - disc) / 2.0
    
    sigma1 = math.sqrt(max(0.0, lambda1))
    sigma2 = math.sqrt(max(0.0, lambda2))
    
    # Eigenvector for lambda1: (AtA - lambda1*I) * v1 = 0
    # (a - lambda1)*x + b*y = 0 -> v1 = [-b, a - lambda1]
    if abs(b) > 1e-9:
        v1 = [-b, a - lambda1]
    elif abs(c) > 1e-9:
        v1 = [d - lambda1, -c]
    else:
        v1 = [1.0, 0.0]
    norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
    if norm1 > 0:
        v1 = [v1[0]/norm1, v1[1]/norm1]
        
    # v2 is orthogonal to v1
    v2 = [-v1[1], v1[0]]
    
    V = [[v1[0], v2[0]],
         [v1[1], v2[1]]]
    
    # U = A * V * Sigma^-1
    AV = manual_matrix_multiply(A, V)
    U = [[0.0, 0.0], [0.0, 0.0]]
    if sigma1 > 1e-9:
        U[0][0] = AV[0][0] / sigma1
        U[1][0] = AV[1][0] / sigma1
    if sigma2 > 1e-9:
        U[0][1] = AV[0][1] / sigma2
        U[1][1] = AV[1][1] / sigma2
        
    Sigma = [[sigma1, 0.0], [0.0, sigma2]]
    return U, Sigma, manual_matrix_transpose(V)

# =====================================================================
# 5. NUMERICAL GRADIENT CHECKING (FROM FIRST PRINCIPLES)
# =====================================================================

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-5):
    """
    Computes numerical gradient using finite differences:
    grad_num[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2 * eps)
    
    Compares against analytical gradient using relative error:
    rel_error = ||g_anal - g_num|| / (||g_anal|| + ||g_num|| + 1e-15)
    """
    numerical_grad = []
    x_orig = list(x)
    
    for i in range(len(x)):
        x_plus = list(x_orig)
        x_minus = list(x_orig)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        loss_plus = f(x_plus)
        loss_minus = f(x_minus)
        
        g_i = (loss_plus - loss_minus) / (2.0 * epsilon)
        numerical_grad.append(g_i)
        
    # Compute relative error
    diff_norm = math.sqrt(sum((a - n)**2 for a, n in zip(analytical_grad, numerical_grad)))
    anal_norm = math.sqrt(sum(a**2 for a in analytical_grad))
    num_norm  = math.sqrt(sum(n**2 for n in numerical_grad))
    
    rel_error = diff_norm / (anal_norm + num_norm + 1e-15)
    return numerical_grad, rel_error

# =====================================================================
# 6. DEMONSTRATION & NUMERICAL VERIFICATION SUITE
# =====================================================================

def run_tensor_sandbox_demo():
    print("=" * 65)
    print("STEP 8: TENSOR MATHEMATICS SANDBOX (FROM FIRST PRINCIPLES)")
    print("=" * 65)
    
    # 1. Tensor creation, shapes, reshaping
    print("\n[1] TENSOR CREATION & RESHAPING")
    t4d_data = [[[[float(i + j + k + l) for l in range(2)] for k in range(2)] for j in range(2)] for i in range(2)]
    t4d = SimpleTensor(t4d_data)
    print(f"4D Tensor Shape (e.g. Batch x Channels x Height x Width): {t4d.shape}")
    print(f"Total elements: {len(t4d.data)}")
    print(f"Strides in row-major layout: {t4d.strides}")
    
    t_reshaped = t4d.reshape((4, 4))
    print(f"Reshaped 4D Tensor into 2D Matrix: {t_reshaped.shape}")
    assert t_reshaped.shape == (4, 4)
    print("  -> Assertion PASSED: Manual row-major reshape verified.")

    # 2. Manual Broadcasting
    print("\n[2] MANUAL BROADCASTING ENGINE")
    # A: (3, 1), B: (1, 4) -> Output: (3, 4)
    A = SimpleTensor([[1.0], [2.0], [3.0]])
    B = SimpleTensor([[10.0, 20.0, 30.0, 40.0]])
    C_manual = manual_elementwise_add(A, B)
    
    # Verify with NumPy
    np_A = np.array([[1.0], [2.0], [3.0]])
    np_B = np.array([[10.0, 20.0, 30.0, 40.0]])
    np_C = np_A + np_B
    
    print(f"Tensor A shape: {A.shape}")
    print(f"Tensor B shape: {B.shape}")
    print(f"Broadcasted (A + B) shape: {C_manual.shape}")
    print(f"Manual Output:\n{C_manual.to_numpy()}")
    np.testing.assert_allclose(C_manual.to_numpy(), np_C)
    print("  -> Assertion PASSED: Manual broadcasting exactly matches NumPy.")

    # 3. Vector Dot Product & Matrix Multiplication
    print("\n[3] INNER PRODUCT & MATRIX MULTIPLICATION")
    u = [1.0, 2.0, 3.0]
    v = [4.0, 5.0, 6.0]
    dot_manual = manual_vector_dot(u, v)
    dot_numpy = np.dot(u, v)
    print(f"Vector u: {u}, Vector v: {v}")
    print(f"Manual Dot Product: {dot_manual} (NumPy: {dot_numpy})")
    assert abs(dot_manual - dot_numpy) < 1e-9

    M1 = [[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]]
    M2 = [[7.0, 8.0],
          [9.0, 1.0],
          [2.0, 3.0]]
    C_mat = manual_matrix_multiply(M1, M2)
    np_mat = np.matmul(np.array(M1), np.array(M2))
    print(f"Matrix A (2x3) x Matrix B (3x2) -> Matrix C (2x2):")
    print(f"Manual Result:\n{np.array(C_mat)}")
    np.testing.assert_allclose(np.array(C_mat), np_mat)
    print("  -> Assertion PASSED: Matrix multiplication exactly matches NumPy.")

    # 4. SVD Decomposition
    print("\n[4] SINGULAR VALUE DECOMPOSITION (SVD)")
    A_svd = [[3.0, 2.0],
             [2.0, 3.0]]
    U, Sigma, Vt = manual_svd_2x2(A_svd)
    
    # Reconstruct: A_rec = U * Sigma * Vt
    US = manual_matrix_multiply(U, Sigma)
    A_reconstructed = manual_matrix_multiply(US, Vt)
    
    print(f"Original Matrix A:\n{np.array(A_svd)}")
    print(f"Manual Singular Values Sigma:\n{np.array(Sigma)}")
    print(f"Reconstructed Matrix (U * Sigma * V^T):\n{np.array(A_reconstructed)}")
    np.testing.assert_allclose(np.array(A_reconstructed), np.array(A_svd), atol=1e-7)
    print("  -> Assertion PASSED: SVD reconstruction error < 1e-7.")

    # 5. Numerical Gradient Checking
    print("\n[5] NUMERICAL GRADIENT CHECKING (FINITE DIFFERENCES)")
    # Objective function: f(w1, w2) = 3*w1^2 + 2*w1*w2 + 4*w2^3
    def f_obj(w):
        return 3.0 * (w[0]**2) + 2.0 * w[0] * w[1] + 4.0 * (w[1]**3)

    # Analytical gradients:
    # df/dw1 = 6*w1 + 2*w2
    # df/dw2 = 2*w1 + 12*w2^2
    w_eval = [2.0, 1.5]
    analytical_grad = [
        6.0 * w_eval[0] + 2.0 * w_eval[1],
        2.0 * w_eval[0] + 12.0 * (w_eval[1]**2)
    ]
    
    num_grad, rel_err = numerical_gradient_check(f_obj, w_eval, analytical_grad, epsilon=1e-5)
    print(f"Evaluation Point w:          {w_eval}")
    print(f"Analytical Gradient:         {analytical_grad}")
    print(f"Finite-Difference Gradient:  {[round(g, 6) for g in num_grad]}")
    print(f"Calculated Relative Error:   {rel_err:.2e}")
    assert rel_err < 1e-7, f"Gradient check failed! Relative error: {rel_err}"
    print(f"  -> Assertion PASSED: Relative Error ({rel_err:.2e}) < 1e-7 (Verified mathematically correct!).")

    print("\n" + "=" * 65)
    print("ALL TENSOR MATHEMATICS SANDBOX TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == '__main__':
    run_tensor_sandbox_demo()
