# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Metal variants of the dense Delassus build and preconditioning kernels.

The CUDA/CPU kernels launch one thread per element of the *capacity* of each world's
matrix (``max_ncts * (max_ncts + 1) / 2`` resp. ``max_ncts^2`` threads per world) and
return early beyond the active size. A world's capacity is sized for its maximum
number of contacts, so most of those threads are empty: anymal_d allocates 1464 rows
and uses ~200, i.e. one million threads per world for ~21k elements. On Metal empty
threads are not free (128 worlds: 7.4 ms per Delassus build, 4.8 ms per
preconditioning). Here the preconditioning launches ``stride`` threads per world that loop
over the active elements, and the build runs in 32x32 tiles of the active upper triangle
with the tiles' Jacobian rows staged in threadgroup memory (MSL, dispatched through Warp's
Metal dispatcher): 128 worlds 7.4 -> 0.8 ms. Every element is computed by exactly the same
operations as in the CUDA/CPU kernel, so the results are bit-identical (tested). Kept in
their own module so the CUDA/CPU module is unaffected.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

__all__ = [
    "apply_dual_preconditioner_to_matrix_strided",
    "build_delassus_tiled",
    "metal_stride",
]


def metal_stride(num_worlds: int, capacity: int, total_threads: int = 1 << 18, min_stride: int = 256) -> int:
    """Threads per world: about ``total_threads`` in all, at least ``min_stride``, at most the capacity."""
    stride = max(min_stride, total_threads // max(num_worlds, 1))
    stride = ((stride + 31) // 32) * 32
    return max(32, min(stride, ((capacity + 31) // 32) * 32))


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


###
# Tiled Delassus build (MSL)
###

_DELASSUS_TILED_MSL = r"""
#include <metal_stdlib>
using namespace metal;

#define TG 256
#define BCH 8                  // bodies per chunk staged in threadgroup memory
#define RW (6 * BCH)           // floats per staged row chunk

struct DelassusParams { int tiles_per_world; int groups_per_world; int max_tiles_1d; int pad; };

// D[i, j] = sum_k inv_m_k dot(Jv_ik, Jv_jk) + dot(Jw_ik, inv_I_k Jw_jk) for i <= j, mirrored,
// exactly as build_delassus_elementwise_dense (same operations, bodies in the same order).
// One threadgroup computes 32x32 tiles of the upper triangle with the tile's 32 + 32
// Jacobian rows staged in threadgroup memory, BCH bodies at a time.
kernel void kamino_build_delassus_tiled(device const int* bodies_offset [[buffer(0)]],
                                        device const float* inv_m [[buffer(1)]],
                                        device const float* inv_I [[buffer(2)]],
                                        device const int* J_off [[buffer(3)]],
                                        device const float* J [[buffer(4)]],
                                        device const int* dims [[buffer(5)]],
                                        device const int* mios [[buffer(6)]],
                                        device float* D [[buffer(7)]],
                                        constant DelassusParams& p [[buffer(8)]],
                                        uint tgid [[threadgroup_position_in_grid]],
                                        uint t [[thread_index_in_threadgroup]])
{
    threadgroup float Ji[32 * RW];
    threadgroup float Jj[32 * RW];
    const int w = int(tgid) / p.groups_per_world;
    const int g = int(tgid) - w * p.groups_per_world;
    const int n = dims[w];
    if (n == 0) {
        return;
    }
    const int bio = bodies_offset[w];
    const int nb = bodies_offset[w + 1] - bio;
    const int nbd = 6 * nb;
    const long cj = J_off[w];
    const long dm = mios[w];
    const int nt = (n + 31) / 32;
    const int ntiles = nt * (nt + 1) / 2;
    for (int tile = g; tile < ntiles; tile += p.groups_per_world) {
        // tile -> (bi, bj), bi <= bj, row-major over the upper triangle of tiles
        int bi = 0, rem = tile;
        while (rem >= nt - bi) {
            rem -= nt - bi;
            ++bi;
        }
        const int bj = bi + rem;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k0 = 0; k0 < nb; k0 += BCH) {
            const int kc = min(BCH, nb - k0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = int(t); e < 32 * RW; e += TG) {
                const int r = e / RW;
                const int q = e - r * RW;
                const int ii = bi * 32 + r;
                const int jj = bj * 32 + r;
                const bool ok = q < 6 * kc;
                Ji[e] = (ok && ii < n) ? J[cj + (long)nbd * ii + 6 * k0 + q] : 0.0f;
                Jj[e] = (ok && jj < n) ? J[cj + (long)nbd * jj + 6 * k0 + q] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int u = 0; u < 4; ++u) {
                const int e = int(t) + u * TG;
                const int r = e >> 5;
                const int c = e & 31;
                for (int kk = 0; kk < kc; ++kk) {
                    const int bid = bio + k0 + kk;
                    threadgroup const float* a = Ji + r * RW + 6 * kk;
                    threadgroup const float* b = Jj + c * RW + 6 * kk;
                    const float3 Jv_i = float3(a[0], a[1], a[2]);
                    const float3 Jw_i = float3(a[3], a[4], a[5]);
                    const float3 Jv_j = float3(b[0], b[1], b[2]);
                    const float3 Jw_j = float3(b[3], b[4], b[5]);
                    device const float* I = inv_I + (long)bid * 9;
                    const float3x3 M = float3x3(float3(I[0], I[3], I[6]), float3(I[1], I[4], I[7]), float3(I[2], I[5], I[8]));
                    const float t1 = inv_m[bid] * metal::dot(Jv_i, Jv_j);
                    const float t2 = metal::dot(Jw_i, M * Jw_j);
                    acc[u] = acc[u] + (t1 + t2);
                }
            }
        }
        for (int u = 0; u < 4; ++u) {
            const int e = int(t) + u * TG;
            const int i = bi * 32 + (e >> 5);
            const int j = bj * 32 + (e & 31);
            if (i < n && j < n && i <= j) {
                D[dm + (long)n * i + j] = acc[u];
                if (i != j) {
                    D[dm + (long)n * j + i] = acc[u];
                }
            }
        }
    }
}
"""

_TILED_PSO = {}


def _msl_slice(arr):
    from warp._src.context import _metal_find_buffer  # noqa: PLC0415
    from warp._src.metal_dispatch import BufferSlice  # noqa: PLC0415

    buf, off = _metal_find_buffer(arr.ptr)
    if buf is None:
        raise RuntimeError("Kamino tiled Delassus build: array is not a Metal allocation")
    return BufferSlice(buf, off)


def build_delassus_tiled(
    num_worlds, max_dim, bodies_offset, inv_m, inv_I, J_off, J, dim, mio, D, total_groups: int = 2048
) -> None:
    """Metal: the dense Delassus build in 32x32 tiles (``kamino_build_delassus_tiled``)."""
    import struct  # noqa: PLC0415

    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    disp = get_dispatcher()
    pso = _TILED_PSO.get("p")
    if pso is None:
        pso = _TILED_PSO["p"] = disp.compile(_DELASSUS_TILED_MSL, "kamino_build_delassus_tiled")
    nt = (int(max_dim) + 31) // 32
    max_tiles = nt * (nt + 1) // 2
    groups = max(1, min(max_tiles, total_groups // max(num_worlds, 1)))
    params = struct.pack("<4i", max_tiles, groups, nt, 0)
    disp.dispatch(
        pso,
        [_msl_slice(a) for a in (bodies_offset, inv_m, inv_I, J_off, J, dim, mio, D)] + [(params, len(params))],
        (256 * groups * num_worlds, 1, 1),
        (256, 1, 1),
        binding_modes=["r", "r", "r", "r", "r", "r", "r", "rw", None],
    )


###
# Zeroing only the written part of per-world dense blocks
###


@wp.kernel
def zero_written_blocks(
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    stride: wp.int32,
    extent: wp.array[wp.int32],
    X: wp.array[wp.float32],
):
    """Zero ``X[mio[w] : mio[w] + e * e]`` with ``e = max(extent[w], dim[w])`` and record ``e``.

    Every writer of a world's block writes inside ``[0, dim^2)`` (row stride ``dim``), so
    ``extent`` (the largest ``dim`` since allocation) bounds everything ever written: the
    result equals zeroing the whole capacity. All threads of a world store the same ``e``.
    """
    wid, lane = wp.tid()
    e = wp.max(extent[wid], dim[wid])
    m0 = mio[wid]
    for k in range(lane, e * e, stride):
        X[m0 + k] = 0.0
    if lane == 0:
        extent[wid] = e


def zero_written(num_worlds: int, max_dim: int, dim, mio, extent, X) -> None:
    """Metal: zero the written part of each world's dense ``max_dim``-capacity block of ``X``."""
    stride = metal_stride(num_worlds, max_dim * max_dim)
    wp.launch(zero_written_blocks, dim=(num_worlds, stride), inputs=[dim, mio, stride, extent, X], device=X.device)
