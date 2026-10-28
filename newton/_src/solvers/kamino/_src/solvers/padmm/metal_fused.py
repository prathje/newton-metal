# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: PADMM on Apple Metal: the whole accelerated iteration loop in one dispatch.

On CUDA the PADMM loop is a ``wp.capture_while`` over three kernels per iteration
(De Saxce correction + velocity bias, the Cholesky triangular solve, and the
projection / residual / acceleration update). Warp's Metal backend replays such a
loop as an unrolled program: per iteration one control dispatch plus three dependent
kernels, each paying the dispatch and barrier latency, and the triangular solve runs
in a single 32-lane SIMD group. For one robot that is ~100+ us per iteration, and a
contact-rich solve runs its full 200 iterations every substep.

This module runs the same iteration sequence for one world per threadgroup inside a
single hand-written Metal kernel:

* every PADMM iteration is the same arithmetic as ``_step_accel`` (De Saxce
  correction and velocity bias, ``D x = v`` with the Cholesky factor that
  :class:`LLTBlockedSolver` computed, projection, dual update, residual maxima and
  sums, convergence test, restart / Nesterov acceleration, penalty update, cache of
  the previous iterates);
* the world loops until it converged or reached ``max_iterations``. With equal
  ``max_iterations`` in every world this is exactly what the global loop computes:
  the iterations a world sits out after convergence while other worlds continue are
  idempotent (its ``v`` is frozen, so ``x``, ``x_p``, ``y_p``, ``z_p``, ``a_p`` are
  rewritten with the values they already hold);
* the triangular solves use the whole threadgroup: panel dot products are spread
  over all SIMD groups (``simd_sum``), each panel's 32x32 triangle is solved by one
  SIMD group with shuffles, and the solution vector lives in threadgroup memory.

The only differences to the per-kernel path are summation orders (the triangular
solve's dot products and the residual sums ``r_dx``, ``r_dy``, ``r_dz``), i.e. float
rounding. Residual maxima are exact.

The kernel is MSL source dispatched through Warp's Metal dispatcher, so it is
recorded into captured graphs like any Warp launch. Nothing here runs on CPU or CUDA.
"""

from __future__ import annotations

import os
import struct

import numpy as np
import warp as wp

from .types import PADMMConfigStruct, PADMMPenalty, PADMMPenaltyUpdate, PADMMStatus

__all__ = [
    "FUSED_TG_SIZE",
    "fused_padmm_enabled",
    "fused_padmm_supported",
    "launch_fused_padmm_accel",
    "launch_wide_llt_solve",
    "probe_div_rn",
]

FUSED_TG_SIZE = int(os.environ.get("NEWTON_KAMINO_METAL_PADMM_TG", "0"))
"""Threads per world (threadgroup size) of the fused kernel; 0 picks 512 for up to 32 worlds
(one world per GPU core, latency-bound) and 256 beyond (throughput-bound)."""


def _tg_for(num_worlds: int, tg: int) -> int:
    if tg > 0:
        return tg
    return 512 if num_worlds <= 32 else 256


_LBS = 2112
"""Floats per staged block of L (must match ``LBS`` in the MSL source)."""


def fused_padmm_enabled() -> bool:
    """Whether the fused Metal PADMM loop is enabled (``NEWTON_KAMINO_METAL_FUSED_PADMM``, default on)."""
    return os.environ.get("NEWTON_KAMINO_METAL_FUSED_PADMM", "1") not in ("0", "false", "False", "off")


###
# MSL source
###


def _msl_struct(name: str, wp_struct) -> str:
    """MSL declaration matching the memory layout of a Warp struct of 4-byte scalars."""
    dt = np.dtype(wp_struct.numpy_dtype())
    fields = []
    offset = 0
    for fname in dt.names:
        ftype, foff = dt.fields[fname][:2]
        if foff != offset or ftype.itemsize != 4:
            raise TypeError(f"{name}: field {fname!r} is not a packed 4-byte scalar")
        if ftype.kind == "f":
            ctype = "float"
        elif ftype.kind == "i":
            ctype = "int"
        else:
            raise TypeError(f"{name}: field {fname!r} has unsupported dtype {ftype}")
        fields.append(f"    {ctype} {fname};")
        offset += 4
    if offset != dt.itemsize:
        raise TypeError(f"{name}: unexpected padding ({dt.itemsize} bytes)")
    return (
        f"struct {name} {{\n"
        + "\n".join(fields)
        + f'\n}};\nstatic_assert(sizeof({name}) == {dt.itemsize}, "layout");\n'
    )


# Argument table: device addresses of every array plus scalars (bound with setBytes).
_ARG_PTRS = (
    # problem
    "dim",
    "mio",
    "vio",
    "nbc",
    "nl",
    "nc",
    "bcio",
    "cio",
    "bcgo",
    "lcgo",
    "ccgo",
    "mu",
    "v_f",
    "bound_lower",
    "bound_upper",
    "P",
    # factor
    "L",
    # solver
    "config",
    "penalty",
    "status",
    "done",
    "atol",
    # state
    "x",
    "x_p",
    "y",
    "z",
    "y_hat",
    "z_hat",
    "y_p",
    "z_p",
    "v",
    "a",
    "a_p",
    "a_factor",
)

_ARG_FMT = "<" + "Q" * len(_ARG_PTRS) + "II"


def _msl_source(tg: int, ncap: int) -> str:
    ptr_decl = "\n".join(f"    ulong {p};" for p in _ARG_PTRS)
    return (
        """
#include <metal_stdlib>
using namespace metal;

"""
        + _msl_struct("PadmmConfig", PADMMConfigStruct)
        + _msl_struct("PadmmStatus", PADMMStatus)
        + _msl_struct("PadmmPenalty", PADMMPenalty)
        + f"""
#define TG {tg}
#define NSG (TG / 32)
#define NCAP {ncap}
#define PENALTY_BALANCED {int(PADMMPenaltyUpdate.BALANCED)}
#define FLOAT32_EPS {float(np.finfo(np.float32).eps)!r}f

struct KArgs {{
{ptr_decl}
    uint has_atol;
    uint pad;
}};
"""
        + r"""
#define DPTR(T, name) ((device T*)(A.name))

// a / b from the reciprocal ib = 1 / b with one Newton correction (Markstein): the
// quotient comes out correctly rounded like IEEE division, in two fma instead of the
// longer precise-division sequence on the triangular solve's critical path.
static inline float div_rn(float a, float b, float ib)
{
    const float q = a * ib;
    return fma(fma(-q, b, a), ib, q);
}

#define PF_ROWS 4   // rows of L per thread prefetched one panel ahead for the look-ahead work
#define LBS 2112    // floats per staged block of L (64 x 33 / 32 x 64)

// Forward block of panel q: L[p0 : p0 + 64, p0 : p0 + 32] (the diagonal block and the
// block below it), row stride 33 (no bank conflicts when lane r reads row r).
static void stage_fwd(device const float* Lm, int n, int q, threadgroup float* LB, int tid, int nthr)
{
    const int p0 = q * 32;
    for (int e = tid; e < 64 * 32; e += nthr) {
        const int rr = e >> 5;
        const int cc = e & 31;
        const int row = p0 + rr;
        const int col = p0 + cc;
        LB[rr * 33 + cc] = (row < n && col < n) ? Lm[(long)row * n + col] : 0.0f;
    }
}

// Backward block of panel q: L[p0 : p0 + 32, p0 - 32 : p0 + 32] (the block left of the
// diagonal block, then the diagonal block), row stride 64.
static void stage_bwd(device const float* Lm, int n, int q, threadgroup float* LB, int tid, int nthr)
{
    const int p0 = q * 32;
    for (int e = tid; e < 32 * 64; e += nthr) {
        const int k = e >> 6;
        const int cc = e & 63;
        const int row = p0 + k;
        const int col = p0 - 32 + cc;
        LB[k * 64 + cc] = (row < n && col >= 0 && col < n) ? Lm[(long)row * n + col] : 0.0f;
    }
}

// Solve L L^T x = b in place on the threadgroup vector X[0, n) (X holds b on entry).
// L is the row-major n x n Cholesky factor at Lm (lower triangle). LB is threadgroup
// scratch of 2 * LBS floats, SP of 2 * TG floats. Needs TG >= 64.
//
// Panels of 32 rows, software-pipelined, one barrier per panel:
// * SIMD group 0 solves panel q's 32x32 triangle (lane r owns row r; each solved entry
//   is broadcast with a shuffle) and in the same steps accumulates the panel's
//   contribution to the next panel ("carry"), with L read from threadgroup memory;
// * meanwhile SIMD groups 1.. stage the next panel's blocks of L into threadgroup
//   memory and do the remaining dot products: forward, the right-looking update of every
//   later row with the previous panel; backward, the next panel's partial dot products
//   with the rows already solved (left-looking, one partial per SIMD group in SP).
// Every entry is a sum of the same products as plain substitution, in another order.
static void llt_solve_tg(device const float* Lm, int n, threadgroup float* X,
                         threadgroup float* LB, threadgroup float* SP, uint t, uint lane, uint sg)
{
    const int npanels = (n + 31) / 32;
    const int nsg_o = NSG - 1;
    const int sg_o = int(sg) - 1;
    const int tid_o = int(t) - 32;
    const int nthr_o = TG - 32;
    const int r = int(lane);
    float pf[PF_ROWS];   // SIMD groups 1..: L prefetched for the next panel

    // ---- Forward substitution: L w = b
    if (sg != 0) {
        stage_fwd(Lm, n, 0, LB, tid_o, nthr_o);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float carry = 0.0f;  // SIMD group 0: L[row of panel q, panel q-1] . w[panel q-1]
    for (int q = 0; q < npanels; ++q) {
        const int p0 = q * 32;
        const int rows = min(32, n - p0);
        threadgroup float* Lcur = LB + (q & 1) * LBS;
        if (sg != 0) {
            if (q + 1 < npanels) {
                stage_fwd(Lm, n, q + 1, LB + ((q + 1) & 1) * LBS, tid_o, nthr_o);
            }
            if (q >= 1) {
                // X[i] -= L[i, panel q-1] . w[panel q-1] for every row after panel q (panel q
                // itself gets it through the carry). The first PF_ROWS rows of this SIMD group
                // were loaded one panel earlier.
                const int c = p0 - 32 + r;
                const float wc = X[c];
                const int i0 = p0 + 32 + sg_o;
                float d[PF_ROWS];
#pragma unroll
                for (int u = 0; u < PF_ROWS; ++u) {
                    d[u] = pf[u] * wc;
                    const int iu = i0 + 32 + u * nsg_o;  // prefetch for panel q + 1
                    pf[u] = iu < n ? Lm[(long)iu * n + c + 32] : 0.0f;
                }
#pragma unroll
                for (int u = 0; u < PF_ROWS; ++u) {
                    d[u] = simd_sum(d[u]);
                }
                if (lane == 0) {
#pragma unroll
                    for (int u = 0; u < PF_ROWS; ++u) {
                        const int iu = i0 + u * nsg_o;
                        if (iu < n) {
                            X[iu] -= d[u];
                        }
                    }
                }
                for (int iu = i0 + PF_ROWS * nsg_o; iu < n; iu += nsg_o) {
                    const float du = simd_sum(Lm[(long)iu * n + c] * wc);
                    if (lane == 0) {
                        X[iu] -= du;
                    }
                }
            } else {
#pragma unroll
                for (int u = 0; u < PF_ROWS; ++u) {
                    const int iu = 64 + sg_o + u * nsg_o;
                    pf[u] = iu < n ? Lm[(long)iu * n + r] : 0.0f;
                }
            }
        } else {
            const int rr = min(r, rows - 1);
            const int i = p0 + rr;
            // Fully unrolled loops (constant trip counts) keep these in registers.
            float lrow[32];
            float lnext[32];
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                lrow[k] = Lcur[rr * 33 + k];
                lnext[k] = Lcur[(32 + r) * 33 + k];
            }
            const float diag = Lcur[rr * 33 + rr];
            const float idiag = 1.0f / diag;
            float res = X[i] - carry;
            float cn = 0.0f;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                if (k < rows) {
                    // Lane k's entry is final at step k (rows above it are applied).
                    const float yk = simd_shuffle(div_rn(res, diag, idiag), ushort(k));
                    if (r > k) {
                        res -= lrow[k] * yk;
                    }
                    cn += lnext[k] * yk;
                }
            }
            if (r < rows) {
                X[i] = div_rn(res, diag, idiag);
            }
            carry = cn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Backward substitution: L^T x = w
    // SP[(it & 1) * TG + g * 32 + r]: SIMD group g + 1's partial of
    // L[rows after panel q + 1, column r of panel q] . x for the panel q solved at step it.
    carry = 0.0f;  // SIMD group 0: L[panel q+1, column of panel q] . x[panel q+1]
    if (sg != 0) {
        stage_bwd(Lm, n, npanels - 1, LB, tid_o, nthr_o);
        const int c = (npanels - 2) * 32 + r;
#pragma unroll
        for (int u = 0; u < PF_ROWS; ++u) {
            const int ju = npanels * 32 + sg_o + u * nsg_o;
            pf[u] = (ju < n && c >= 0) ? Lm[(long)ju * n + c] : 0.0f;
        }
    } else {
        for (int g = 0; g < nsg_o; ++g) {
            SP[g * 32 + r] = 0.0f;
            SP[TG + g * 32 + r] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int q = npanels - 1, it = 0; q >= 0; --q, ++it) {
        const int p0 = q * 32;
        const int rows = min(32, n - p0);
        threadgroup float* Lcur = LB + (it & 1) * LBS;
        if (sg != 0) {
            if (q >= 1) {
                stage_bwd(Lm, n, q - 1, LB + ((it + 1) & 1) * LBS, tid_o, nthr_o);
                // Columns of the next (lower) panel: partial dots with the rows after panel q
                // (solved). The first PF_ROWS rows of this SIMD group were loaded earlier.
                const int c = p0 - 32 + r;
                const int j0 = p0 + 32 + sg_o;
                float acc = 0.0f;
#pragma unroll
                for (int u = 0; u < PF_ROWS; ++u) {
                    const int ju = j0 + u * nsg_o;
                    if (ju < n) {
                        acc += pf[u] * X[ju];
                    }
                    const int jn = ju - 32;  // prefetch for panel q - 1
                    pf[u] = (jn < n && c >= 32) ? Lm[(long)jn * n + c - 32] : 0.0f;
                }
                for (int j = j0 + PF_ROWS * nsg_o; j < n; j += nsg_o) {
                    acc += Lm[(long)j * n + c] * X[j];
                }
                SP[((it + 1) & 1) * TG + sg_o * 32 + r] = acc;
            }
        } else {
            const int rr = min(r, rows - 1);
            const int i = p0 + rr;
            float lcol[32];
            float lprev[32];
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                lcol[k] = Lcur[k * 64 + 32 + rr];
                lprev[k] = Lcur[k * 64 + r];
            }
            const float diag = Lcur[rr * 64 + 32 + rr];
            const float idiag = 1.0f / diag;
            float sp = 0.0f;
            for (int g = 0; g < nsg_o; ++g) {
                sp += SP[(it & 1) * TG + g * 32 + r];
            }
            float res = X[i] - sp - carry;
            float cn = 0.0f;
#pragma unroll
            for (int k = 31; k >= 0; --k) {
                if (k < rows) {
                    const float xk = simd_shuffle(div_rn(res, diag, idiag), ushort(k));
                    if (r < k) {
                        res -= lcol[k] * xk;
                    }
                    cn += lprev[k] * xk;
                }
            }
            if (r < rows) {
                X[i] = div_rn(res, diag, idiag);
            }
            carry = cn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Standalone solve (tests / non-fused fallback): one threadgroup per block, x may alias b.
kernel void kamino_llt_solve_wide(constant KArgs& A [[buffer(0)]],
                                  uint w [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float X[NCAP];
    threadgroup float LB[2 * LBS];
    threadgroup float SP[2 * TG];
    const int n = DPTR(int, dim)[w];
    if (n <= 0) {
        return;
    }
    const int m0 = DPTR(int, mio)[w];
    const int v0 = DPTR(int, vio)[w];
    // b is passed in "v", x in "x".
    for (int i = int(t); i < n; i += TG) {
        X[i] = DPTR(float, v)[v0 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    llt_solve_tg(DPTR(float, L) + m0, n, X, LB, SP, t, lane, sg);
    for (int i = int(t); i < n; i += TG) {
        DPTR(float, x)[v0 + i] = X[i];
    }
}

// Test probe for div_rn: x[i] = v[i] / L[i] for i < dim[0].
kernel void kamino_div_rn_probe(constant KArgs& A [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    if (int(i) < DPTR(int, dim)[0]) {
        const float b = DPTR(float, L)[i];
        DPTR(float, x)[i] = div_rn(DPTR(float, v)[i], b, 1.0f / b);
    }
}

static PadmmPenalty update_penalty(PadmmConfig config, PadmmPenalty pen, int iterations, float r_p, float r_d)
{
    if (config.penalty_update_method == PENALTY_BALANCED) {
        const int freq = config.penalty_update_freq;
        if (freq > 0 && iterations > 0 && iterations % freq == 0) {
            float rho = pen.rho;
            if (r_p > config.alpha * r_d) {
                rho = rho * config.tau;
            } else if (r_d > config.alpha * r_p) {
                rho = max(rho / config.tau, config.rho_min);
            }
            pen.rho = rho;
            pen.num_updates += 1;
        }
    }
    return pen;
}

static float3 project_to_coulomb_cone(float3 x, float mu)
{
    const float xn = x[2];
    const float xt_norm = sqrt(x[0] * x[0] + x[1] * x[1]);
    float3 y = float3(0.0f);
    if (mu * xt_norm > -xn) {
        if (xt_norm <= mu * xn) {
            y = x;
        } else {
            const float ys = (mu * xt_norm + xn) / (mu * mu + 1.0f);
            const float yts = mu * ys / xt_norm;
            y[0] = yts * x[0];
            y[1] = yts * x[1];
            y[2] = ys;
        }
    }
    return y;
}

#define CTRL_RESTART 1
#define CTRL_CONVERGED 2
#define CTRL_FROZEN 4

// The accelerated PADMM loop (``PADMMSolver._step_accel`` repeated until the world
// converged or exhausted ``max_iterations``), one threadgroup per world.
kernel void kamino_padmm_accel_dense(constant KArgs& A [[buffer(0)]],
                                     uint w [[threadgroup_position_in_grid]],
                                     uint t [[thread_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float X[NCAP];
    threadgroup float LB[2 * LBS];
    threadgroup float SP[2 * TG];
    threadgroup float RED[6 * NSG];
    threadgroup int ctrl_sh;
    threadgroup float a_factor_sh;
    threadgroup float rho_sh;

    const int ncts = DPTR(int, dim)[w];
    const int vio = DPTR(int, vio)[w];
    const int m0 = DPTR(int, mio)[w];
    const int nbc = DPTR(int, nbc)[w];
    const int nl = DPTR(int, nl)[w];
    const int nc = DPTR(int, nc)[w];
    const int lcgo = DPTR(int, lcgo)[w];
    const int bcgo = DPTR(int, bcgo)[w];
    const int ccgo = DPTR(int, ccgo)[w];
    const int cio = DPTR(int, cio)[w];
    const int bcio = DPTR(int, bcio)[w];
    const PadmmConfig config = DPTR(PadmmConfig, config)[w];
    const float eta = config.eta;

    device const float* Lm = DPTR(float, L) + m0;
    device const float* mu = DPTR(float, mu);
    device const float* v_f = DPTR(float, v_f);
    device const float* bound_lower = DPTR(float, bound_lower);
    device const float* bound_upper = DPTR(float, bound_upper);
    device const float* P = DPTR(float, P);
    device float* sx = DPTR(float, x);
    device float* sx_p = DPTR(float, x_p);
    device float* sy = DPTR(float, y);
    device float* sz = DPTR(float, z);
    device float* sy_hat = DPTR(float, y_hat);
    device float* sz_hat = DPTR(float, z_hat);
    device float* sy_p = DPTR(float, y_p);
    device float* sz_p = DPTR(float, z_p);
    device float* sv = DPTR(float, v);

    // Thread 0 owns the world's status, penalty and acceleration scalars.
    PadmmStatus status = DPTR(PadmmStatus, status)[w];
    PadmmPenalty pen = DPTR(PadmmPenalty, penalty)[w];
    float st_a = DPTR(float, a)[w];
    float st_a_p = DPTR(float, a_p)[w];
    float st_a_factor = DPTR(float, a_factor)[w];
    float atol_val = A.has_atol ? DPTR(float, atol)[w] : 0.0f;
    bool wrote_atol = false;

    float rho = pen.rho;
    bool frozen = status.converged != 0 || status.iterations >= config.max_iterations;

    while (!frozen) {
        // --- De Saxce correction and velocity bias: v = -v_f - s + eta x_p + rho y_hat + z_hat
        for (int i = int(t); i < ncts; i += TG) {
            const int k = vio + i;
            float s = 0.0f;
            if (nc > 0) {
                const int lo = i - ccgo;
                if (lo >= 0 && lo < 3 * nc) {
                    const int cid = lo / 3;
                    const int comp = lo - 3 * cid;
                    if (comp == 2) {
                        const int kk = vio + ccgo + 3 * cid;
                        const float vtx = sz_hat[kk];
                        const float vty = sz_hat[kk + 1];
                        s = mu[cio + cid] * sqrt(vtx * vtx + vty * vty);
                    }
                }
            }
            const float vi = -v_f[k] - s + eta * sx_p[k] + rho * sy_hat[k] + sz_hat[k];
            sv[k] = vi;
            X[i] = vi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Unconstrained solution: D x = v
        llt_solve_tg(Lm, ncts, X, LB, SP, t, lane, sg);

        // --- Projection, dual update, residuals
        const float inv_rho = 1.0f / rho;
        float r_p_local = 0.0f, r_d_local = 0.0f, r_c_local = 0.0f;
        float r_dx_local = 0.0f, r_dy_local = 0.0f, r_dz_local = 0.0f;
        for (int local_id = int(t); local_id < ncts; local_id += TG) {
            const int thread_offset = vio + local_id;
            if (nc > 0 && local_id >= ccgo && local_id < ccgo + 3 * nc) {
                const int local_offset = local_id - ccgo;
                const int cid = local_offset / 3;
                const int component = local_offset - 3 * cid;
                if (component == 0) {
                    const int jl = ccgo + 3 * cid;
                    const int ccio_j = vio + jl;
                    const float y0 = X[jl] - inv_rho * sz_hat[ccio_j];
                    const float y1 = X[jl + 1] - inv_rho * sz_hat[ccio_j + 1];
                    const float y2 = X[jl + 2] - inv_rho * sz_hat[ccio_j + 2];
                    const float3 y_proj = project_to_coulomb_cone(float3(y0, y1, y2), mu[cio + cid]);
                    const float3 x_c = float3(X[jl], X[jl + 1], X[jl + 2]);
                    float3 z_c = float3(0.0f);
                    for (int comp = 0; comp < 3; ++comp) {
                        const int idx = ccio_j + comp;
                        const float x = X[jl + comp];
                        const float y = y_proj[comp];
                        const float x_p = sx_p[idx];
                        const float y_p = sy_hat[idx];
                        const float z_p = sz_hat[idx];
                        const float p_i = P[idx];
                        const float z = z_p + rho * (y - x);
                        z_c[comp] = z;
                        sy[idx] = y;
                        sz[idx] = z;
                        const float r_p = p_i * (x - y);
                        const float r_d = (1.0f / p_i) * (eta * (x - x_p) + rho * (y - y_p));
                        const float r_dx = p_i * (x - x_p);
                        const float r_dy = p_i * (y - y_p);
                        const float r_dz = (1.0f / p_i) * (z - z_p);
                        r_p_local = max(r_p_local, abs(r_p));
                        r_d_local = max(r_d_local, abs(r_d));
                        r_dx_local += r_dx * r_dx;
                        r_dy_local += r_dy * r_dy;
                        r_dz_local += r_dz * r_dz;
                    }
                    r_c_local = max(r_c_local, abs(x_c[0] * z_c[0] + x_c[1] * z_c[1] + x_c[2] * z_c[2]));
                }
            } else {
                const float x = X[local_id];
                const float z_p = sz_hat[thread_offset];
                float y = x - inv_rho * z_p;
                const bool is_bounded = nbc > 0 && local_id >= bcgo && local_id < bcgo + nbc;
                const int bio = bcio + local_id - bcgo;
                const bool is_limit = nl > 0 && local_id >= lcgo && local_id < lcgo + nl;
                if (is_bounded) {
                    y = min(max(bound_lower[bio], y), bound_upper[bio]);
                } else if (is_limit) {
                    y = max(y, 0.0f);
                }
                const float x_p = sx_p[thread_offset];
                const float y_p = sy_hat[thread_offset];
                const float z_prev = sz_hat[thread_offset];
                const float p_i = P[thread_offset];
                const float z = z_prev + rho * (y - x);
                sy[thread_offset] = y;
                sz[thread_offset] = z;
                const float r_p = p_i * (x - y);
                const float r_d = (1.0f / p_i) * (eta * (x - x_p) + rho * (y - y_p));
                const float r_dx = p_i * (x - x_p);
                const float r_dy = p_i * (y - y_p);
                const float r_dz = (1.0f / p_i) * (z - z_prev);
                r_p_local = max(r_p_local, abs(r_p));
                r_d_local = max(r_d_local, abs(r_d));
                r_dx_local += r_dx * r_dx;
                r_dy_local += r_dy * r_dy;
                r_dz_local += r_dz * r_dz;
                if (is_bounded) {
                    const float lo_b = bound_lower[bio];
                    const float up_b = bound_upper[bio];
                    const float r_box = (x - lo_b) * max(z, 0.0f) + (up_b - x) * max(-z, 0.0f);
                    r_c_local = max(r_c_local, abs(r_box));
                } else if (is_limit) {
                    r_c_local = max(r_c_local, abs(x * z));
                }
            }
        }
        // Reduce to world-level metrics (maxima exact; sums in a fixed order).
        r_p_local = simd_max(r_p_local);
        r_d_local = simd_max(r_d_local);
        r_c_local = simd_max(r_c_local);
        r_dx_local = simd_sum(r_dx_local);
        r_dy_local = simd_sum(r_dy_local);
        r_dz_local = simd_sum(r_dz_local);
        if (lane == 0) {
            RED[0 * NSG + sg] = r_p_local;
            RED[1 * NSG + sg] = r_d_local;
            RED[2 * NSG + sg] = r_c_local;
            RED[3 * NSG + sg] = r_dx_local;
            RED[4 * NSG + sg] = r_dy_local;
            RED[5 * NSG + sg] = r_dz_local;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        if (t == 0) {
            float r_p_max = RED[0], r_d_max = RED[NSG], r_c_max = RED[2 * NSG];
            float r_dx_l2_sum = RED[3 * NSG], r_dy_l2_sum = RED[4 * NSG], r_dz_l2_sum = RED[5 * NSG];
            for (int g = 1; g < NSG; ++g) {
                r_p_max = max(r_p_max, RED[g]);
                r_d_max = max(r_d_max, RED[NSG + g]);
                r_c_max = max(r_c_max, RED[2 * NSG + g]);
                r_dx_l2_sum += RED[3 * NSG + g];
                r_dy_l2_sum += RED[4 * NSG + g];
                r_dz_l2_sum += RED[5 * NSG + g];
            }
            status.iterations += 1;
            status.r_p = r_p_max;
            status.r_d = r_d_max;
            status.r_c = r_c_max;
            if (A.has_atol && config.linear_solver_tolerance_ratio > 0.0f) {
                atol_val = max(config.linear_solver_tolerance_ratio * r_p_max, FLOAT32_EPS);
                wrote_atol = true;
            }
            status.r_dx = sqrt(r_dx_l2_sum);
            status.r_dy = sqrt(r_dy_l2_sum);
            status.r_dz = sqrt(r_dz_l2_sum);
            status.r_a = rho * r_dy_l2_sum + (1.0f / rho) * r_dz_l2_sum;
            if (status.iterations > 1 && r_p_max <= config.primal_tolerance && r_d_max <= config.dual_tolerance
                && r_c_max <= config.compl_tolerance) {
                status.converged = 1;
            }
            const bool done_now = status.converged != 0 || status.iterations >= config.max_iterations;
            if (done_now) {
                atomic_fetch_sub_explicit((device atomic_int*)(A.done), 1, memory_order_relaxed);
            }
            if (status.converged) {
                status.restart = 0;
            } else if (status.r_a < config.restart_tolerance * status.r_a_p) {
                status.restart = 0;
                const float a_p = st_a_p;
                const float a = (1.0f + sqrt(1.0f + 4.0f * a_p * a_p)) / 2.0f;
                st_a = a;
                st_a_factor = (a_p - 1.0f) / a;
            } else {
                status.restart = 1;
                status.num_restarts += 1;
                status.r_a = status.r_a_p / config.restart_tolerance;
                st_a = float(config.a_0);
                st_a_factor = 0.0f;
            }
            status.r_a_pp = status.r_a_p;
            status.r_a_p = status.r_a;
            pen = update_penalty(config, pen, status.iterations, r_p_max, r_d_max);
            ctrl_sh = CTRL_RESTART * status.restart + CTRL_CONVERGED * status.converged + (done_now ? CTRL_FROZEN : 0);
            a_factor_sh = st_a_factor;
            rho_sh = pen.rho;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int control = ctrl_sh;
        const float a_factor = a_factor_sh;
        const bool restarted = (control & CTRL_RESTART) != 0;
        const bool converged = (control & CTRL_CONVERGED) != 0;
        frozen = (control & CTRL_FROZEN) != 0;
        rho = rho_sh;

        // --- Acceleration and previous-iterate cache
        for (int i = int(t); i < ncts; i += TG) {
            const int vid = vio + i;
            const float x = X[i];
            const float y = sy[vid];
            const float z = sz[vid];
            const float y_p = sy_p[vid];
            const float z_p = sz_p[vid];
            if (!converged) {
                if (!restarted) {
                    sy_hat[vid] = y + a_factor * (y - y_p);
                    sz_hat[vid] = z + a_factor * (z - z_p);
                } else {
                    sy_hat[vid] = y_p;
                    sz_hat[vid] = z_p;
                }
            }
            sx[vid] = x;
            sx_p[vid] = x;
            sy_p[vid] = y;
            sz_p[vid] = z;
        }
        if (t == 0) {
            st_a_p = st_a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    // Terminal state (the vectors were written by the last iteration).
    if (t == 0) {
        DPTR(PadmmStatus, status)[w] = status;
        DPTR(PadmmPenalty, penalty)[w] = pen;
        DPTR(float, a)[w] = st_a;
        DPTR(float, a_p)[w] = st_a_p;
        DPTR(float, a_factor)[w] = st_a_factor;
        if (wrote_atol) {
            DPTR(float, atol)[w] = atol_val;
        }
    }
}
"""
    )


###
# Launchers
###

_PSO_CACHE: dict = {}


def _pso(entry: str, tg: int, ncap: int):
    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    key = (entry, tg, ncap)
    pso = _PSO_CACHE.get(key)
    if pso is None:
        pso = get_dispatcher().compile(_msl_source(tg, ncap), entry)
        _PSO_CACHE[key] = pso
    return pso


def _ncap(max_dim: int) -> int:
    return max(32, ((int(max_dim) + 31) // 32) * 32)


def _buffer(arr: wp.array):
    """``(MTLBuffer, gpu address)`` of a Metal Warp array's data."""
    from warp._src.context import _metal_find_buffer  # noqa: PLC0415

    buf, off = _metal_find_buffer(arr.ptr)
    if buf is None:
        raise RuntimeError("Kamino fused PADMM: array is not a Metal allocation")
    return buf, int(buf.gpuAddress()) + int(off)


def _dispatch(entry: str, num_worlds: int, max_dim: int, arrays: dict, modes: dict, has_atol: bool, tg: int):
    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    if tg < 64 or tg % 32 != 0:
        raise ValueError(f"Kamino fused PADMM: threadgroup size must be a multiple of 32 and >= 64, got {tg}")

    addrs = []
    resources = {}
    for name in _ARG_PTRS:
        arr = arrays.get(name)
        if arr is None or arr.ptr is None or arr.size == 0:
            addrs.append(0)
            continue
        buf, addr = _buffer(arr)
        addrs.append(addr)
        mode = modes.get(name, "r")
        prev = resources.get(id(buf))
        if prev is not None and prev[1] != mode:
            mode = "rw"
        resources[id(buf)] = (buf, mode)
    packed = struct.pack(_ARG_FMT, *addrs, int(has_atol), 0)
    get_dispatcher().dispatch(
        _pso(entry, tg, _ncap(max_dim)),
        [(packed, len(packed))],
        (tg * num_worlds, 1, 1),
        (tg, 1, 1),
        binding_modes=[None],
        extra_resources=list(resources.values()),
    )


def fused_padmm_supported(solver, problem) -> bool:
    """Whether :func:`launch_fused_padmm_accel` can replace the PADMM iteration loop.

    Requires the accelerated dense path with the Metal lane-parallel Cholesky factor,
    no per-iteration info collection, the same ``max_iterations >= 1`` in every world
    (see the module docstring), and a constraint count whose solve vector fits in
    threadgroup memory.
    """
    if not fused_padmm_enabled():
        return False
    if not getattr(solver.device, "is_metal", False):
        return False
    if not solver._use_acceleration or solver._collect_info or problem.sparse:
        return False
    delassus = getattr(problem, "_delassus", None)
    lin = getattr(delassus, "_solver", None)
    from ...linalg.linear import LLTBlockedSolver  # noqa: PLC0415

    if not isinstance(lin, LLTBlockedSolver) or not lin._use_lanes():
        return False
    iters = {int(c.max_iterations) for c in solver._config}
    if len(iters) != 1 or min(iters) < 1:
        return False
    tg = _tg_for(solver._size.num_worlds, FUSED_TG_SIZE)
    return _tg_bytes(tg, _ncap(solver._size.max_of_max_total_cts)) <= _max_tg_memory()


def _tg_bytes(tg: int, ncap: int) -> int:
    """Static threadgroup memory of :func:`kamino_padmm_accel_dense` (floats/ints of 4 bytes)."""
    return 4 * (ncap + 2 * _LBS + 2 * tg + 6 * (tg // 32) + 3)


def _max_tg_memory() -> int:
    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    return int(get_dispatcher()._device.maxThreadgroupMemoryLength())


def launch_fused_padmm_accel(solver, problem, tg: int = FUSED_TG_SIZE) -> None:
    """Run the accelerated PADMM iterations of every world in one dispatch."""
    tg = _tg_for(solver._size.num_worlds, tg)
    d = problem.data
    sd = solver._data
    st = sd.state
    lin = problem._delassus._solver
    arrays = {
        "dim": d.dim,
        "mio": d.mio,
        "vio": d.vio,
        "nbc": d.nbc,
        "nl": d.nl,
        "nc": d.nc,
        "bcio": d.bcio,
        "cio": d.cio,
        "bcgo": d.bcgo,
        "lcgo": d.lcgo,
        "ccgo": d.ccgo,
        "mu": d.mu,
        "v_f": d.v_f,
        "bound_lower": d.bound_lower,
        "bound_upper": d.bound_upper,
        "P": d.P,
        "L": lin._L,
        "config": sd.config,
        "penalty": sd.penalty,
        "status": sd.status,
        "done": st.done,
        "atol": sd.linear_solver_atol,
        "x": st.x,
        "x_p": st.x_p,
        "y": st.y,
        "z": st.z,
        "y_hat": st.y_hat,
        "z_hat": st.z_hat,
        "y_p": st.y_p,
        "z_p": st.z_p,
        "v": st.v,
        "a": st.a,
        "a_p": st.a_p,
        "a_factor": st.a_factor,
    }
    writes = ("penalty", "status", "done", "atol", "x", "x_p", "y", "z", "y_hat", "z_hat", "y_p", "z_p", "v")
    modes = dict.fromkeys((*writes, "a", "a_p", "a_factor"), "rw")
    _dispatch(
        "kamino_padmm_accel_dense",
        solver._size.num_worlds,
        solver._size.max_of_max_total_cts,
        arrays,
        modes,
        sd.linear_solver_atol is not None,
        tg,
    )


def launch_wide_llt_solve(
    num_blocks: int,
    max_dim: int,
    dim: wp.array,
    mio: wp.array,
    vio: wp.array,
    L: wp.array,
    b: wp.array,
    x: wp.array,
    tg: int = FUSED_TG_SIZE,
) -> None:
    """Solve ``L L^T x = b`` per block with one threadgroup per block (``x`` may alias ``b``)."""
    tg = _tg_for(num_blocks, tg)
    arrays = {"dim": dim, "mio": mio, "vio": vio, "L": L, "v": b, "x": x}
    modes = {"x": "rw"}
    _dispatch("kamino_llt_solve_wide", num_blocks, max_dim, arrays, modes, False, tg)


def probe_div_rn(a: wp.array, b: wp.array, out: wp.array, count: wp.array) -> None:
    """Test helper: ``out[i] = a[i] / b[i]`` with the solve's division (``count[0]`` entries)."""
    n = a.shape[0]
    tg = 256
    arrays = {"dim": count, "v": a, "L": b, "x": out}
    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    addrs = []
    resources = {}
    for name in _ARG_PTRS:
        arr = arrays.get(name)
        if arr is None:
            addrs.append(0)
            continue
        buf, addr = _buffer(arr)
        addrs.append(addr)
        resources[id(buf)] = (buf, "rw")
    packed = struct.pack(_ARG_FMT, *addrs, 0, 0)
    get_dispatcher().dispatch(
        _pso("kamino_div_rn_probe", 64, 32),
        [(packed, len(packed))],
        (((n + tg - 1) // tg) * tg, 1, 1),
        (tg, 1, 1),
        binding_modes=[None],
        extra_resources=list(resources.values()),
    )
