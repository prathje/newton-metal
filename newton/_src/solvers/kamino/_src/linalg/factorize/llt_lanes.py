# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: Linear Algebra: lane-parallel dense LLT (Cholesky) kernels for Apple Metal.

Warp's Metal backend runs the tile kernels of :mod:`.llt_blocked` with one thread per
matrix, so the blocked factorization and the triangular solves execute serially on the GPU.
For the Delassus systems of a single robot (``n`` ~ 100-150) that costs about 5 ms per
solve and 40 ms per factorization, and a PADMM solve calls the triangular solve once per
iteration.

The kernels here run one 32-lane SIMD group per matrix instead. Rows and vector entries are
owned by lane ``index % block_dim``; dot products are split across lanes and combined with a
block-wide ``tile_reduce``. The triangular solves only read entries their own lane wrote, so
they need no barrier. The factorization reads the pivot row written by another lane and
synchronizes once per column.

The same code is valid at any ``block_dim`` (one lane on CPU), which the unit tests use as a
reference. Only :class:`LLTBlockedSolver` on Metal devices launches these kernels.
"""

import warp as wp

###
# Module interface
###

__all__ = [
    "LLT_LANES_BLOCK_DIM",
    "llt_lanes_factorize",
    "llt_lanes_factorize_kernel",
    "llt_lanes_solve",
    "llt_lanes_solve_inplace",
    "llt_lanes_solve_kernel",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})

LLT_LANES_BLOCK_DIM = 32
"""Threads per matrix: one Apple SIMD group."""


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


###
# Kernels
###


@wp.kernel
def llt_lanes_factorize_kernel(
    # Inputs:
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    A: wp.array[wp.float32],
    # Outputs:
    L: wp.array[wp.float32],
):
    """Column-by-column (Cholesky-Crout) factorization ``A = L L^T`` of each ``n x n`` block.

    Writes the lower triangle of ``L`` and zeros its strict upper triangle.
    """
    wid, lane = wp.tid()
    num_lanes = wp.block_dim()
    n = dim[wid]
    m0 = mio[wid]
    for k in range(n):
        rk = m0 + k * n
        # Pivot: L[k, k] = sqrt(A[k, k] - sum_j L[k, j]^2), split over the lanes.
        s = wp.float32(0.0)
        for j in range(lane, k, num_lanes):
            s += L[rk + j] * L[rk + j]
        # NOTE: the reduction must stay in the kernel body (not a wp.func) for Metal's lane lowering.
        d = wp.sqrt(A[rk + k] - wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(s)), 0))
        # Column k below the pivot: each lane updates the rows it owns.
        first = k + 1 + ((lane - (k + 1)) % num_lanes + num_lanes) % num_lanes
        for i in range(first, n, num_lanes):
            ri = m0 + i * n
            acc = A[ri + k]
            for j in range(k):
                acc -= L[ri + j] * L[rk + j]
            L[ri + k] = acc / d
        if lane == k % num_lanes:
            L[rk + k] = d
            for j in range(k + 1, n):
                L[rk + j] = wp.float32(0.0)
        # Row k + 1 (owned by another lane) is read by every lane in the next column.
        _sync_threads()


@wp.kernel
def llt_lanes_solve_kernel(
    # Inputs:
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    L: wp.array[wp.float32],
    b: wp.array[wp.float32],
    # Outputs:
    y: wp.array[wp.float32],
    x: wp.array[wp.float32],
):
    """Solves ``L y = b`` then ``L^T x = y`` per block. ``b`` may alias ``x``.

    Entry ``i`` of ``b``, ``y`` and ``x`` is only read and written by lane ``i % block_dim``.
    """
    wid, lane = wp.tid()
    num_lanes = wp.block_dim()
    n = dim[wid]
    m0 = mio[wid]
    v0 = vio[wid]
    # Forward substitution: y[i] = (b[i] - L[i, :i] . y[:i]) / L[i, i]
    for i in range(n):
        ri = m0 + i * n
        s = wp.float32(0.0)
        for j in range(lane, i, num_lanes):
            s += L[ri + j] * y[v0 + j]
        tot = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(s)), 0)
        if lane == i % num_lanes:
            y[v0 + i] = (b[v0 + i] - tot) / L[ri + i]
    # Backward substitution: x[i] = (y[i] - L[i+1:, i] . x[i+1:]) / L[i, i]
    for ii in range(n):
        i = n - 1 - ii
        first = i + 1 + ((lane - (i + 1)) % num_lanes + num_lanes) % num_lanes
        s = wp.float32(0.0)
        for j in range(first, n, num_lanes):
            s += L[m0 + j * n + i] * x[v0 + j]
        tot = wp.tile_extract(wp.tile_reduce(wp.add, wp.tile(s)), 0)
        if lane == i % num_lanes:
            x[v0 + i] = (y[v0 + i] - tot) / L[m0 + i * n + i]


###
# Launchers
###


def llt_lanes_factorize(
    num_blocks: int,
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    A: wp.array[wp.float32],
    L: wp.array[wp.float32],
    block_dim: int = LLT_LANES_BLOCK_DIM,
):
    """Launches :func:`llt_lanes_factorize_kernel` with one thread block per matrix."""
    wp.launch_tiled(
        kernel=llt_lanes_factorize_kernel,
        dim=num_blocks,
        inputs=[dim, mio, A, L],
        block_dim=block_dim,
        device=A.device,
    )


def llt_lanes_solve(
    num_blocks: int,
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    L: wp.array[wp.float32],
    b: wp.array[wp.float32],
    y: wp.array[wp.float32],
    x: wp.array[wp.float32],
    block_dim: int = LLT_LANES_BLOCK_DIM,
):
    """Launches :func:`llt_lanes_solve_kernel` with one thread block per matrix."""
    wp.launch_tiled(
        kernel=llt_lanes_solve_kernel,
        dim=num_blocks,
        inputs=[dim, mio, vio, L, b, y, x],
        block_dim=block_dim,
        device=L.device,
    )


def llt_lanes_solve_inplace(
    num_blocks: int,
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    L: wp.array[wp.float32],
    y: wp.array[wp.float32],
    x: wp.array[wp.float32],
    block_dim: int = LLT_LANES_BLOCK_DIM,
):
    """In-place variant of :func:`llt_lanes_solve`: ``x`` holds ``b`` on entry."""
    llt_lanes_solve(num_blocks, dim, mio, vio, L, x, y, x, block_dim=block_dim)
