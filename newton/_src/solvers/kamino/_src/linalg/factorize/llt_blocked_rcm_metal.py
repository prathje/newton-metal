# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Metal variants of the RCM blocked Cholesky factorize and solve kernels.

Warp's Metal backend keeps a tile in each thread's private memory and runs a
tile kernel with one working thread per block, so the 32x32-tile kernels of
:mod:`.llt_blocked_rcm` spend most of their time copying spilled tiles. The
kernels here hold no tiles. They work in place in ``L`` (device memory) and
use one ``width``-thread threadgroup per matrix with barriers between
dependent steps (Warp's wide-block model: ``wp.block_dim()`` plus a native
``__syncthreads()``, no tile ops).

They keep the tile kernels' tile-pattern skips, outputs and arithmetic order:

- each tile update is a dot product over the tile's 32 columns in ascending
  order, subtracted once per tile with tiles in ascending order (Warp's scalar
  tile GEMM: ``C = alpha * sum + beta * C``);
- the diagonal tile uses the scalar left-looking Cholesky of Warp's tile
  Cholesky (``s -= r * r``, ``L[i, j] = s_i * (1 / s)``);
- triangular solves divide by the pivot, as Warp's scalar tile solves do.

Results match the CPU tile kernels up to FMA contraction (Metal's compiler may
fuse a multiply and an add that the CPU build keeps separate) and the sign of
exact zeros; see ``metal-port/r8_schur``. The kernels take the arguments of the tile
kernels they replace, so :class:`.LLTBlockedRCMSolver` swaps them in on Metal
and launches them through the same ``wp.launch_tiled`` wrappers.
"""

from __future__ import annotations

import warp as wp

from .llt_blocked_rcm import _sync_threads

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32


@wp.func_native("""
#if defined(__METAL_VERSION__)
return metal::fma(a, b, c);
#else
return fmaf(a, b, c);
#endif
""")
def _fma(a: float32, b: float32, c: float32) -> float32:
    """Fused ``a * b + c``, as Warp's scalar tile GEMM accumulates (``muladd`` is ``fmaf``)."""
    ...


# Threads per matrix (the solver's factorize / solve block dims on Metal).
METAL_FACTORIZE_BLOCK_DIM = 256
METAL_SOLVE_BLOCK_DIM = 64


def _make_factorize_kernel(block_size: int, width: int):
    """In-place RCM blocked Cholesky, one ``width``-thread threadgroup per matrix."""
    BS = block_size

    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def llt_blocked_rcm_factorize_metal(
        dim: wp.array[int32],
        ld: wp.array[int32],
        mio: wp.array[int32],
        tpo: wp.array[int32],
        A: wp.array[float32],
        tile_pattern: wp.array[int32],
        clear_skipped: bool,
        L: wp.array[float32],
    ):
        b, lane = wp.tid()
        block = int32(wp.block_dim())
        nthreads = int32(width)
        if block == int32(1):
            # Serial block model (Warp's Metal wide blocks disabled): lane 0 works alone.
            if lane != int32(0):
                return
            nthreads = int32(1)
        elif block != int32(width):
            # Any other block model would split one factorization: poison it instead.
            if lane == int32(0):
                for e in range(dim[b]):
                    L[mio[b] + e * ld[b] + e] = wp.nan
            return

        n = dim[b]
        stride = ld[b]
        off = mio[b]
        tp = tpo[b]
        n_tiles = (n + int32(BS - 1)) / int32(BS)

        for tk in range(n_tiles):
            k = tk * int32(BS)
            kv = wp.min(int32(BS), n - k)

            # 1. Diagonal tile: S = A_kk - sum_j L_kj L_kj^T (lower part) in place in L,
            #    zero upper part, identity padding rows/columns beyond n.
            e = lane
            while e < int32(BS * BS):
                r = e / int32(BS)
                c = e - r * int32(BS)
                if k + r < stride and k + c < stride:
                    value = float32(0.0)
                    if r < kv and c < kv:
                        if r >= c:
                            value = A[off + (k + r) * stride + k + c]
                            for tj in range(tk):
                                if tile_pattern[tp + tk * n_tiles + tj] != int32(0):
                                    j = tj * int32(BS)
                                    row_r = off + (k + r) * stride + j
                                    row_c = off + (k + c) * stride + j
                                    s = float32(0.0)
                                    for q in range(BS):
                                        s = _fma(L[row_r + q], L[row_c + q], s)
                                    value = -s + value
                    elif r == c:
                        value = float32(1.0)
                    L[off + (k + r) * stride + k + c] = value
                e += nthreads
            _sync_threads()

            # 2. Scalar left-looking Cholesky of the valid diagonal block, column by
            #    column. Every thread computes the pivot; lane 0 stores it one column
            #    late (nobody reads a stored pivot while the tile is being factored).
            pivot = float32(0.0)
            for jc in range(kv):
                if jc > int32(0) and lane == int32(0):
                    L[off + (k + jc - int32(1)) * stride + k + jc - int32(1)] = pivot
                row_j = off + (k + jc) * stride + k
                s = L[row_j + jc]
                for kk in range(jc):
                    rv = L[row_j + kk]
                    s -= rv * rv
                s = wp.sqrt(s)
                inv_s = float32(1.0) / s
                pivot = s
                i = jc + int32(1) + lane
                while i < kv:
                    row_i = off + (k + i) * stride + k
                    s_i = L[row_i + jc]
                    for kk in range(jc):
                        s_i -= L[row_i + kk] * L[row_j + kk]
                    L[row_i + jc] = s_i * inv_s
                    i += nthreads
                _sync_threads()
            if kv > int32(0) and lane == int32(0):
                L[off + (k + kv - int32(1)) * stride + k + kv - int32(1)] = pivot
            _sync_threads()

            # 3. Panel tiles below the diagonal: each thread owns one row of one tile,
            #    updates it and solves X L_kk^T = A_ik row-wise, in place in L.
            panel_rows = (n_tiles - tk - int32(1)) * int32(BS)
            t = lane
            while t < panel_rows:
                ti = tk + int32(1) + t / int32(BS)
                row = ti * int32(BS) + t % int32(BS)
                row_base = off + row * stride
                if row < stride:
                    if tile_pattern[tp + ti * n_tiles + tk] == int32(0):
                        if clear_skipped:
                            for c in range(BS):
                                L[row_base + k + c] = float32(0.0)
                    elif row >= n:
                        # Identity-padded rows beyond n hold zeros in this panel.
                        for c in range(BS):
                            L[row_base + k + c] = float32(0.0)
                    else:
                        for c in range(BS):
                            col_base = off + (k + c) * stride
                            value = A[row_base + k + c]
                            for tj in range(tk):
                                if tile_pattern[tp + ti * n_tiles + tj] != int32(0) and tile_pattern[
                                    tp + tk * n_tiles + tj
                                ] != int32(0):
                                    j = tj * int32(BS)
                                    s = float32(0.0)
                                    for q in range(BS):
                                        s = _fma(L[row_base + j + q], L[col_base + j + q], s)
                                    value = -s + value
                            for jj in range(c):
                                value -= L[col_base + k + jj] * L[row_base + k + jj]
                            diagonal = L[col_base + k + c]
                            if diagonal != float32(0.0):
                                value = value / diagonal
                            L[row_base + k + c] = value
                t += nthreads
            _sync_threads()

    return llt_blocked_rcm_factorize_metal


def _make_solve_kernels(block_size: int, width: int, forward: bool, backward: bool):
    """RCM blocked triangular solves, one ``width``-thread threadgroup per matrix.

    Returns the ``solve`` kernel (gathers ``b`` through ``P`` and scatters the
    solution to ``x``) and the ``solve_inplace`` kernel (``x`` holds the permuted
    right-hand side and receives the permuted solution), with the signatures of
    the tile kernels they replace.
    """
    BS = block_size

    @wp.func
    def solve_block(
        wid: int32,
        lane: int32,
        nthreads: int32,
        permuted: bool,
        dim: wp.array[int32],
        ld: wp.array[int32],
        mio: wp.array[int32],
        vio: wp.array[int32],
        tpo: wp.array[int32],
        P: wp.array[int32],
        L: wp.array[float32],
        tile_pattern: wp.array[int32],
        b: wp.array[float32],
        y: wp.array[float32],
        x_hat: wp.array[float32],
        x: wp.array[float32],
    ):
        n = dim[wid]
        stride = ld[wid]
        off = mio[wid]
        v = vio[wid]
        tp = tpo[wid]
        n_tiles = (n + int32(BS - 1)) / int32(BS)

        if wp.static(forward):
            row = lane
            while row < n:
                if permuted:
                    y[v + row] = b[v + P[v + row]]
                else:
                    y[v + row] = b[v + row]
                row += nthreads
            _sync_threads()
            for ti in range(n_tiles):
                i = ti * int32(BS)
                kv = wp.min(int32(BS), n - i)
                # Off-diagonal tiles: one thread per row, tiles and columns ascending.
                r = lane
                while r < kv:
                    row_base = off + (i + r) * stride
                    value = y[v + i + r]
                    for tj in range(ti):
                        if tile_pattern[tp + ti * n_tiles + tj] != int32(0):
                            j = tj * int32(BS)
                            s = float32(0.0)
                            for q in range(BS):
                                s = _fma(L[row_base + j + q], y[v + j + q], s)
                            value = -s + value
                    y[v + i + r] = value
                    r += nthreads
                _sync_threads()
                # Diagonal tile: sequential forward substitution (Warp's vector-RHS tile solve).
                if lane == int32(0):
                    for c in range(kv):
                        row_base = off + (i + c) * stride + i
                        value = y[v + i + c]
                        for jj in range(c):
                            value -= L[row_base + jj] * y[v + i + jj]
                        diagonal = L[row_base + c]
                        if diagonal != float32(0.0):
                            value = value / diagonal
                        y[v + i + c] = value
                _sync_threads()

        if wp.static(backward):
            for reverse_tile in range(n_tiles):
                ti = n_tiles - int32(1) - reverse_tile
                i = ti * int32(BS)
                kv = wp.min(int32(BS), n - i)
                r = lane
                while r < kv:
                    value = y[v + i + r]
                    for tj in range(ti + int32(1), n_tiles):
                        if tile_pattern[tp + tj * n_tiles + ti] != int32(0):
                            j = tj * int32(BS)
                            s = float32(0.0)
                            # Rows beyond n read as zero in the tile kernel; their terms vanish.
                            q_end = wp.min(int32(BS), n - j)
                            for q in range(q_end):
                                s = _fma(L[off + (j + q) * stride + i + r], x_hat[v + j + q], s)
                            value = -s + value
                    x_hat[v + i + r] = value
                    r += nthreads
                _sync_threads()
                if lane == int32(0):
                    for reverse_c in range(kv):
                        c = kv - int32(1) - reverse_c
                        value = x_hat[v + i + c]
                        for jj in range(c + int32(1), kv):
                            value -= L[off + (i + jj) * stride + i + c] * x_hat[v + i + jj]
                        diagonal = L[off + (i + c) * stride + i + c]
                        if diagonal != float32(0.0):
                            value = value / diagonal
                        x_hat[v + i + c] = value
                _sync_threads()
            if permuted:
                row = lane
                while row < n:
                    x[v + P[v + row]] = x_hat[v + row]
                    row += nthreads

    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def llt_blocked_rcm_solve_metal(
        dim: wp.array[int32],
        ld: wp.array[int32],
        mio: wp.array[int32],
        vio: wp.array[int32],
        tpo: wp.array[int32],
        P: wp.array[int32],
        L: wp.array[float32],
        tile_pattern: wp.array[int32],
        b: wp.array[float32],
        y: wp.array[float32],
        x_hat: wp.array[float32],
        x: wp.array[float32],
    ):
        wid, lane = wp.tid()
        block = int32(wp.block_dim())
        nthreads = int32(width)
        if block == int32(1):
            # Serial block model (Warp's Metal wide blocks disabled): lane 0 works alone.
            if lane != int32(0):
                return
            nthreads = int32(1)
        elif block != int32(width):
            # Any other block model would split one solve: poison it instead.
            if lane == int32(0):
                for row in range(dim[wid]):
                    y[vio[wid] + row] = wp.nan
                    x_hat[vio[wid] + row] = wp.nan
            return
        solve_block(wid, lane, nthreads, True, dim, ld, mio, vio, tpo, P, L, tile_pattern, b, y, x_hat, x)

    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def llt_blocked_rcm_solve_inplace_metal(
        dim: wp.array[int32],
        ld: wp.array[int32],
        mio: wp.array[int32],
        vio: wp.array[int32],
        tpo: wp.array[int32],
        L: wp.array[float32],
        tile_pattern: wp.array[int32],
        y: wp.array[float32],
        x: wp.array[float32],
    ):
        wid, lane = wp.tid()
        block = int32(wp.block_dim())
        nthreads = int32(width)
        if block == int32(1):
            if lane != int32(0):
                return
            nthreads = int32(1)
        elif block != int32(width):
            if lane == int32(0):
                for row in range(dim[wid]):
                    x[vio[wid] + row] = wp.nan
            return
        solve_block(wid, lane, nthreads, False, dim, ld, mio, vio, tpo, dim, L, tile_pattern, x, y, x, x)

    return llt_blocked_rcm_solve_metal, llt_blocked_rcm_solve_inplace_metal


_KERNELS: dict = {}


def factorize_kernel(block_size: int, width: int):
    """Metal replacement for :func:`.make_llt_blocked_rcm_factorize_kernel` at one
    block width (one kernel object per width: Warp fixes a kernel's block model at
    its first Metal launch)."""
    key = ("factorize", block_size, width)
    if key not in _KERNELS:
        _KERNELS[key] = _make_factorize_kernel(block_size, width)
    return _KERNELS[key]


def solve_kernels(block_size: int, width: int, forward: bool = True, backward: bool = True):
    """Metal replacements for :func:`.make_llt_blocked_rcm_solve_kernel` and
    :func:`.make_llt_blocked_rcm_solve_inplace_kernel` at one block width."""
    key = ("solve", block_size, width, bool(forward), bool(backward))
    if key not in _KERNELS:
        _KERNELS[key] = _make_solve_kernels(block_size, width, bool(forward), bool(backward))
    return _KERNELS[key]
