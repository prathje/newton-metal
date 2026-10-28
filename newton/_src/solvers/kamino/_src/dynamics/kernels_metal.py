# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Metal variants of the dense Delassus build and preconditioning kernels.

The CUDA/CPU kernels launch one thread per element of the *capacity* of each world's
matrix (``max_ncts * (max_ncts + 1) / 2`` resp. ``max_ncts^2`` threads per world) and
return early beyond the active size. A world's capacity is sized for its maximum
number of contacts, so most of those threads are empty: anymal_d allocates 1464 rows
and uses ~200, i.e. one million threads per world for ~21k elements. On Metal empty
threads are not free (128 worlds: 7.4 ms per Delassus build, 4.8 ms per
preconditioning). These variants launch ``stride`` threads per world that loop over the
active elements. Every element is computed by exactly the same arithmetic as in the
CUDA/CPU kernel, so the results are bit-identical. Kept in their own module so the
CUDA/CPU module is unaffected.
"""

from __future__ import annotations

import warp as wp

from .delassus import upper_triangular_indices_from_index

wp.set_module_options({"enable_backward": False})

__all__ = [
    "apply_dual_preconditioner_to_matrix_strided",
    "build_delassus_elementwise_dense_strided",
    "metal_stride",
]


def metal_stride(num_worlds: int, capacity: int, total_threads: int = 1 << 18, min_stride: int = 256) -> int:
    """Threads per world: about ``total_threads`` in all, at least ``min_stride``, at most the capacity."""
    stride = max(min_stride, total_threads // max(num_worlds, 1))
    stride = ((stride + 31) // 32) * 32
    return max(32, min(stride, ((capacity + 31) // 32) * 32))


@wp.kernel
def build_delassus_elementwise_dense_strided(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    model_bodies_inv_m_i: wp.array[wp.float32],
    data_bodies_inv_I_i: wp.array[wp.mat33f],
    jacobians_cts_offset: wp.array[wp.int32],
    jacobians_cts_data: wp.array[wp.float32],
    delassus_dim: wp.array[wp.int32],
    delassus_mio: wp.array[wp.int32],
    stride: wp.int32,
    # Outputs:
    delassus_D: wp.array[wp.float32],
):
    """``_build_delassus_elementwise_dense`` with ``stride`` threads per world looping over the upper triangle."""
    wid, lane = wp.tid()

    bio = model_info_bodies_offset[wid]
    nb = model_info_bodies_offset[wid + 1] - bio
    ncts = delassus_dim[wid]
    if ncts == 0:
        return
    dmio = delassus_mio[wid]
    cjmio = jacobians_cts_offset[wid]
    nbd = 6 * nb
    total = ncts * (ncts + 1) // 2

    for tid in range(lane, total, stride):
        i, j = upper_triangular_indices_from_index(tid, ncts)
        if i < 0 or i >= ncts or j < i or j >= ncts:
            continue

        Jv_i = wp.vec3f(0.0)
        Jv_j = wp.vec3f(0.0)
        Jw_i = wp.vec3f(0.0)
        Jw_j = wp.vec3f(0.0)
        D_ij = wp.float32(0.0)

        for k in range(nb):
            bid_k = bio + k
            dio_k = 6 * k
            jio_ik = cjmio + nbd * i + dio_k
            jio_jk = cjmio + nbd * j + dio_k
            for d in range(3):
                Jv_i[d] = jacobians_cts_data[jio_ik + d]
                Jw_i[d] = jacobians_cts_data[jio_ik + d + 3]
                Jv_j[d] = jacobians_cts_data[jio_jk + d]
                Jw_j[d] = jacobians_cts_data[jio_jk + d + 3]
            inv_m_k = model_bodies_inv_m_i[bid_k]
            inv_I_k = data_bodies_inv_I_i[bid_k]
            D_ij += inv_m_k * wp.dot(Jv_i, Jv_j) + wp.dot(Jw_i, inv_I_k @ Jw_j)

        delassus_D[dmio + ncts * i + j] = D_ij
        if i != j:
            delassus_D[dmio + ncts * j + i] = D_ij


@wp.kernel
def apply_dual_preconditioner_to_matrix_strided(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_mio: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_P: wp.array[wp.float32],
    stride: wp.int32,
    # Outputs:
    X: wp.array[wp.float32],
):
    """``_apply_dual_preconditioner_to_matrix`` with ``stride`` threads per world looping over the matrix."""
    wid, lane = wp.tid()
    ncts = problem_dim[wid]
    if ncts == 0:
        return
    mio = problem_mio[wid]
    vio = problem_vio[wid]
    for tid in range(lane, ncts * ncts, stride):
        i = tid // ncts
        j = tid % ncts
        m_ij = mio + ncts * i + j
        X_ij = X[m_ij]
        P_i = problem_P[vio + i]
        P_j = problem_P[vio + j]
        X[m_ij] = P_i * (P_j * X_ij)
