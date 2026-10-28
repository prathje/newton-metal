# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: Linear Algebra: threadgroup-wide dense Cholesky factorization for Apple Metal.

:mod:`.llt_lanes` factorizes each matrix with one 32-lane SIMD group, column by column:
for a ~200-row Delassus matrix that is ~1.9 ms per factorization, once per Kamino step.
The kernel here factorizes each matrix with one threadgroup (left-looking, 32-column
panels):

1. the panel's columns ``W = A[:, panel] - L[:, :p0] L[panel, :p0]^T`` (the update with
   all previous panels): lane c of each SIMD group owns column c of 4 rows, the previous
   panels are processed in 32-column chunks staged in threadgroup memory;
2. SIMD group 0 factorizes the panel's 32x32 diagonal block (Crout, in threadgroup
   memory);
3. one thread per row below the block solves its row of ``L21 L11^T = W21``.

It writes the lower triangle of ``L`` and zeros the strict upper triangle, like
:func:`llt_lanes_factorize`; a non-positive pivot gives NaN like there. The products
are those of the column-by-column algorithm, summed in another order.

The kernel is MSL source dispatched through Warp's Metal dispatcher, so it is recorded
into captured graphs like any Warp launch. Nothing here runs on CPU or CUDA.
"""

from __future__ import annotations

import os

import warp as wp

__all__ = [
    "LLT_WIDE_TG_SIZE",
    "llt_wide_enabled",
    "llt_wide_factorize",
]

LLT_WIDE_TG_SIZE = int(os.environ.get("NEWTON_KAMINO_METAL_LLT_TG", "0"))
"""Threads per matrix (threadgroup size) of the factorization; 0 picks 512 for up to 32
matrices (one per GPU core) and 256 beyond (more threadgroups resident per core)."""


def llt_wide_enabled() -> bool:
    """Whether the threadgroup-wide factorization is used (``NEWTON_KAMINO_METAL_WIDE_LLT``, default on)."""
    return os.environ.get("NEWTON_KAMINO_METAL_WIDE_LLT", "1") not in ("0", "false", "False", "off")


_FACTORIZE_MSL = r"""
#include <metal_stdlib>
using namespace metal;

#ifndef TG
#define TG 512
#endif
#define NSG (TG / 32)

// a / b from the reciprocal ib = 1 / b with one Newton correction (Markstein): rounds
// like IEEE division (see test_fused_division_is_ieee).
static inline float div_rn(float a, float b, float ib)
{
    const float q = a * ib;
    return fma(fma(-q, b, a), ib, q);
}

kernel void kamino_llt_factorize_wide(device const int* dims [[buffer(0)]],
                                      device const int* mios [[buffer(1)]],
                                      device const float* Aall [[buffer(2)]],
                                      device float* Lall [[buffer(3)]],
                                      uint w [[threadgroup_position_in_grid]],
                                      uint t [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float T[32 * 33];  // the panel's diagonal block (row stride 33)
    threadgroup float P[32 * 33];  // L[panel rows, 32-column chunk], transposed
    const int n = dims[w];
    if (n <= 0) {
        return;
    }
    const long m0 = mios[w];
    device const float* A = Aall + m0;
    device float* L = Lall + m0;

    for (int p0 = 0; p0 < n; p0 += 32) {
        const int pc = min(32, n - p0);
        const int nrows = n - p0;  // rows p0 .. n - 1
        // 1. W[i, c] = A[i, p0 + c] - L[i, :p0] . L[p0 + c, :p0] for the rows i >= p0, stored in
        //    L[i, p0 + c] (entries above the diagonal are zeroed at the end). Lane c of a SIMD
        //    group owns column c; each SIMD group takes groups of 4 rows; the previous panels'
        //    columns are processed in chunks of 32 with L[panel rows, chunk] staged (transposed)
        //    in threadgroup memory.
        const int ngroups = (nrows + 3) / 4;
        const int c = int(lane);
        for (int g = int(sg); g < ngroups; g += NSG) {
            for (int u = 0; u < 4; ++u) {
                const int i = p0 + 4 * g + u;
                if (i < n && c < pc) {
                    L[(long)i * n + p0 + c] = A[(long)i * n + p0 + c];
                }
            }
        }
        for (int j0 = 0; j0 < p0; j0 += 32) {
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            for (int e = int(t); e < 32 * 32; e += TG) {
                const int cc = e >> 5;
                const int jj = e & 31;
                P[jj * 33 + cc] = cc < pc ? L[(long)(p0 + cc) * n + j0 + jj] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            for (int g = int(sg); g < ngroups; g += NSG) {
                float li[4];
                float acc[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int i = min(p0 + 4 * g + u, n - 1);
                    li[u] = L[(long)i * n + j0 + c];  // lane c holds L[i, j0 + c] (j0 + c < p0)
                    acc[u] = 0.0f;
                }
#pragma unroll
                for (int jj = 0; jj < 32; ++jj) {
                    const float pj = P[jj * 33 + c];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        acc[u] += simd_shuffle(li[u], ushort(jj)) * pj;
                    }
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int i = p0 + 4 * g + u;
                    if (i < n && c < pc) {
                        L[(long)i * n + p0 + c] -= acc[u];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // 2. Diagonal block (Crout) by SIMD group 0: lane r holds row r in registers x0..x31
        //    (straight-line code generated below); row k's entries come from lane k by shuffles.
        if (sg == 0) {
            const int r = int(lane);
            device const float* Lr = L + (long)(p0 + min(r, pc - 1)) * n + p0;
@DIAG_CODE@
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // 3. Rows below the block: L[i, panel] L11^T = W[i, panel]. Lane c of a SIMD group owns
        //    column c of 4 rows at a time; L[i, p0 + k] is broadcast with a shuffle once final.
        {
            const float dc = T[c * 33 + c];
            const float idc = 1.0f / dc;
            const int nbelow = n - p0 - 32;
            for (int g = int(sg); 4 * g < nbelow; g += NSG) {
                float x[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int i = min(p0 + 32 + 4 * g + u, n - 1);
                    x[u] = c < pc ? L[(long)i * n + p0 + c] : 0.0f;
                }
#pragma unroll
                for (int k = 0; k < 32; ++k) {
                    if (k < pc) {
                        const float lck = T[c * 33 + k];
                        const float dk = T[k * 33 + k];
                        const float idk = 1.0f / dk;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const float yk = div_rn(simd_shuffle(x[u], ushort(k)), dk, idk);
                            if (c > k) {
                                x[u] -= lck * yk;
                            }
                        }
                    }
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int i = p0 + 32 + 4 * g + u;
                    if (i < n && c < pc) {
                        L[(long)i * n + p0 + c] = div_rn(x[u], dc, idc);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    // Strict upper triangle.
    for (int i = int(sg); i < n; i += NSG) {
        for (int j = i + 1 + int(lane); j < n; j += 32) {
            L[(long)i * n + j] = 0.0f;
        }
    }
}
"""

def _diag_code() -> str:
    """Straight-line MSL for the 32x32 diagonal-block Crout factorization (no arrays, so
    nothing can spill to stack memory). Lane r holds row r in x0..x31 and publishes each
    finished entry to T, from where row k is read as a threadgroup-memory broadcast."""
    lines = []
    for k in range(32):
        lines.append(f"            float x{k} = ({k} < pc && r < pc) ? Lr[{k}] : 0.0f;")
    for k in range(32):
        lines.append(f"            if ({k} < pc) {{")
        # Four partial sums (shorter dependency chains), combined in a fixed order.
        lines.append("                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;")
        for j in range(k):
            lines.append(f"                s{j % 4} += x{j} * T[{k * 33 + j}];")
        lines.append(f"                const float s = x{k} - ((s0 + s1) + (s2 + s3));")
        lines.append(f"                const float d = sqrt(simd_shuffle(s, ushort({k})));")
        lines.append(f"                x{k} = (r == {k}) ? d : ((r > {k}) ? s / d : x{k});")
        lines.append(f"                T[r * 33 + {k}] = x{k};")
        lines.append("                simdgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append("            }")
    for k in range(32):
        lines.append(f"            if (r < pc && {k} <= r) {{ L[(long)(p0 + r) * n + p0 + {k}] = x{k}; }}")
    return "\n".join(lines)


_PSO_CACHE: dict = {}


def _pso(tg: int):
    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    pso = _PSO_CACHE.get(tg)
    if pso is None:
        source = f"#define TG {tg}\n" + _FACTORIZE_MSL.replace("@DIAG_CODE@", _diag_code())
        pso = get_dispatcher().compile(source, "kamino_llt_factorize_wide")
        _PSO_CACHE[tg] = pso
    return pso


def _slice(arr: wp.array):
    from warp._src.context import _metal_find_buffer  # noqa: PLC0415
    from warp._src.metal_dispatch import BufferSlice  # noqa: PLC0415

    buf, off = _metal_find_buffer(arr.ptr)
    if buf is None:
        raise RuntimeError("Kamino wide LLT: array is not a Metal allocation")
    return BufferSlice(buf, off)


def llt_wide_factorize(
    num_blocks: int,
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    A: wp.array[wp.float32],
    L: wp.array[wp.float32],
    tg: int = LLT_WIDE_TG_SIZE,
) -> None:
    """Factorize ``A = L L^T`` per block with one threadgroup per block (Metal only)."""
    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    if tg <= 0:
        tg = 512 if num_blocks <= 32 else 256
    if tg < 32 or tg % 32 != 0:
        raise ValueError(f"Kamino wide LLT: threadgroup size must be a multiple of 32, got {tg}")
    get_dispatcher().dispatch(
        _pso(tg),
        [_slice(dim), _slice(mio), _slice(A), _slice(L)],
        (tg * num_blocks, 1, 1),
        (tg, 1, 1),
        binding_modes=["r", "r", "r", "rw"],
    )
