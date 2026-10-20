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
block-wide ``tile_reduce``. The factorization reads the pivot row written by another lane and
synchronizes once per column. The triangular solves work on panels of ``block_dim`` rows (see
:func:`llt_lanes_solve_kernel`).

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

_vec8f = wp.types.vector(length=8, dtype=wp.float32)


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
        # Column k below the pivot: each lane updates the rows it owns. The loads of
        # L[i, :k] . L[k, :k] are issued eight at a time (see llt_lanes_solve_kernel); the
        # subtraction stays sequential in j (masked entries subtract an exact zero).
        first = k + 1 + ((lane - (k + 1)) % num_lanes + num_lanes) % num_lanes
        for i in range(first, n, num_lanes):
            ri = m0 + i * n
            acc = A[ri + k]
            for j in range(0, k, 8):
                lv = _vec8f()
                kv = _vec8f()
                for q in range(8):
                    jq = wp.min(j + q, k - 1)
                    lv[q] = wp.where(j + q < k, L[ri + jq], 0.0)
                    kv[q] = L[rk + jq]
                for q in range(8):
                    acc -= lv[q] * kv[q]
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

    Works on panels of ``block_dim`` rows; lane ``r`` owns row ``p0 + r`` of panel ``p0``.
    Each lane first applies the already-solved part as one dot product (loads issued eight at a
    time, so a single SIMD group is not bound by one memory round trip per row), then the
    panel's triangular block is solved by broadcasting each solved entry across the lanes.
    Panels exchange their results through ``y``/``x``, with one barrier per panel.
    """
    wid, lane = wp.tid()
    num_lanes = wp.block_dim()
    n = dim[wid]
    m0 = mio[wid]
    v0 = vio[wid]
    last = n - 1

    # Forward substitution: y[i] = (b[i] - L[i, :i] . y[:i]) / L[i, i]
    for p0 in range(0, n, num_lanes):
        i = p0 + lane
        ic = wp.min(i, last)
        ri = m0 + ic * n
        diag = L[ri + ic]
        res = b[v0 + ic]
        # Rows above the panel: res -= L[i, :p0] . y[:p0]
        s = wp.float32(0.0)
        for j in range(0, p0, 8):
            lv = _vec8f()
            yv = _vec8f()
            for k in range(8):
                jk = wp.min(j + k, p0 - 1)
                lv[k] = wp.where(j + k < p0, L[ri + jk], 0.0)
                yv[k] = y[v0 + jk]
            s += wp.dot(lv, yv)
        res -= s
        # Triangular block of the panel
        for jb in range(0, num_lanes, 8):
            lv = _vec8f()
            for k in range(8):
                lv[k] = L[ri + wp.min(p0 + jb + k, last)]
            for k in range(8):
                jj = jb + k
                if jj < num_lanes:
                    # Lane jj's entry is final at step jj (rows above it are applied).
                    yb = wp.tile_extract(wp.tile(res / diag), jj)
                    if lane > jj:
                        res -= lv[k] * yb
        if i < n:
            y[v0 + i] = res / diag
        _sync_threads()

    # Backward substitution: x[i] = (y[i] - L[i+1:, i] . x[i+1:]) / L[i, i]
    num_panels = (n + num_lanes - 1) // num_lanes
    for pp in range(num_panels):
        p0 = (num_panels - 1 - pp) * num_lanes
        i = p0 + lane
        ic = wp.min(i, last)
        diag = L[m0 + ic * n + ic]
        res = y[v0 + ic]
        # Rows below the panel: res -= L[p0 + num_lanes:, i] . x[p0 + num_lanes:]
        s = wp.float32(0.0)
        for j in range(p0 + num_lanes, n, 8):
            lv = _vec8f()
            xv = _vec8f()
            for k in range(8):
                jk = wp.min(j + k, last)
                lv[k] = wp.where(j + k < n, L[m0 + jk * n + ic], 0.0)
                xv[k] = x[v0 + jk]
            s += wp.dot(lv, xv)
        res -= s
        # Triangular block of the panel, bottom row first
        for jb in range(0, num_lanes, 8):
            lv = _vec8f()
            for k in range(8):
                jj = num_lanes - 1 - (jb + k)
                lv[k] = L[m0 + wp.clamp(p0 + jj, 0, last) * n + ic]
            for k in range(8):
                jj = num_lanes - 1 - (jb + k)
                if jj >= 0 and p0 + jj < n:
                    xb = wp.tile_extract(wp.tile(res / diag), jj)
                    if lane < jj:
                        res -= lv[k] * xb
        if i < n:
            x[v0 + i] = res / diag
        _sync_threads()


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
