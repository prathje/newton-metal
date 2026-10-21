# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Metal variant of the dense colored PGS kernel.

The CUDA kernel spreads each color's dense velocity update over (slot, row)
tasks and accumulates with float atomics, so on more than one lane the
summation order (and the rounding) changes from run to run. Here every row
has one owner thread that adds the color's contributions in slot order, which
reproduces the one-lane (CPU) summation sequence exactly, and reads a
per-world transposed copy of the operator so neighbouring threads (rows) load
neighbouring addresses. The kernel has no tile ops, so Warp's Metal backend
runs it as one ``block_dim``-thread threadgroup per world with real barriers
(wide-block model; one SIMD group per world is too slow: a lone SIMD group
gets little issue throughput and no latency hiding). The CUDA-only adaptive
early exit is left out. Everything else is the CUDA/CPU kernel unchanged.
Kept in its own module so the CUDA/CPU module is unaffected.
"""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from .kernels import (
    _FUSED_BILATERAL_BLOCK,
    _FUSED_INEQUALITY_BLOCK,
    _contact_friction_normal_load,
    _project_box_update,
    _project_contact_normal_update,
    _project_contact_tangent_update,
    _sync_threads,
)
from .types import DVIConfigStruct, DVIStatus

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32


def _make_dense_pgs_rows_kernel(width: int):
    """The row-owned dense PGS kernel for one block width.

    One kernel object per width: Warp's Metal backend fixes a kernel's block
    model (and width) at its first launch, and solvers of different sizes in
    one process pick different widths.
    """

    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def _solve_dvi_inequalities_colored_pgs_rows(
        problem_dim: wp.array[int32],
        problem_mio: wp.array[int32],
        problem_vio: wp.array[int32],
        problem_nbc: wp.array[int32],
        problem_nl: wp.array[int32],
        problem_nc: wp.array[int32],
        problem_bcgo: wp.array[int32],
        problem_lcgo: wp.array[int32],
        problem_ccgo: wp.array[int32],
        problem_bcio: wp.array[int32],
        problem_cio: wp.array[int32],
        problem_uio: wp.array[int32],
        problem_mu: wp.array[float32],
        problem_bound_lower: wp.array[float32],
        problem_bound_upper: wp.array[float32],
        problem_D: wp.array[float32],
        problem_D_t: wp.array[float32],
        problem_P: wp.array[float32],
        problem_v_b: wp.array[float32],
        block_iteration: int32,
        inequality_num_colors: wp.array[int32],
        inequality_ids_by_color: wp.array[int32],
        inequality_color_starts: wp.array[int32],
        solver_config: wp.array[DVIConfigStruct],
        enable_adaptive: wp.bool,
        solver_status: wp.array[DVIStatus],
        state_delta: wp.array[float32],
        state_v_aug: wp.array[float32],
        solution_lambdas: wp.array[float32],
    ):
        """Metal variant of :func:`_solve_dvi_inequalities_colored_pgs` (see the module docstring).

        Runs the full sweep budget: the adaptive early exit (CUDA only, ``enable_adaptive``)
        needs block reductions, which a barrier-only wide block does not have. Metal never
        enables it; ``enable_adaptive`` only selects whether the status is written.
        """
        # The launch is dim=num_worlds*width, block_dim=width. Warp runs it as one
        # threadgroup per world (wide-block model, block_dim() == width) or, if it
        # keeps the serial model (wp.config.metal_coop_tiles off), as independent
        # threads with block_dim() == 1: then lane 0 sweeps the whole world alone
        # (barriers are no-ops), which is the one-lane kernel exactly.
        tid = wp.tid()
        lane = tid % width
        wid = tid / width
        block = int32(wp.block_dim())
        threads_per_world = int32(width)
        if block == int32(1):
            if lane != int32(0):
                return
            threads_per_world = int32(1)
        elif block != int32(width):
            # Any other block model would silently skip or split the sweep: poison
            # the world's impulses instead.
            if lane == int32(0):
                for row in range(problem_dim[wid]):
                    solution_lambdas[problem_vio[wid] + row] = wp.nan
            return
        cfg = solver_config[wid]
        if block_iteration >= int32(0) and block_iteration >= cfg.max_alternating_iterations:
            return

        nbc = problem_nbc[wid]
        nl = problem_nl[wid]
        nc = problem_nc[wid]
        nu = nbc + nl + nc
        if nu == 0:
            if enable_adaptive and lane == int32(0):
                status = solver_status[wid]
                status.iterations = int32(1)
                solver_status[wid] = status
            return
        ncts = problem_dim[wid]
        mio = problem_mio[wid]
        vio = problem_vio[wid]
        bcgo = problem_bcgo[wid]
        lcgo = problem_lcgo[wid]
        ccgo = problem_ccgo[wid]
        bcio = problem_bcio[wid]
        cio = problem_cio[wid]
        uio = problem_uio[wid]
        schedule_offset = uio + wid
        contact_end = ccgo + int32(3) * nc
        sweep_count = cfg.inequality_sweeps_per_iteration
        first_tangent_sweep = int32(0)
        if block_iteration == int32(_FUSED_BILATERAL_BLOCK):
            sweep_count *= cfg.max_alternating_iterations
        elif block_iteration == int32(_FUSED_INEQUALITY_BLOCK):
            tangent_sweep_count = sweep_count * cfg.max_alternating_iterations / int32(2)
            sweep_count = (sweep_count + int32(1)) * cfg.max_alternating_iterations
            first_tangent_sweep = sweep_count - tangent_sweep_count
        completed_sweeps = sweep_count
        for _sweep in range(sweep_count):
            phase_count = int32(2)
            if block_iteration == int32(_FUSED_INEQUALITY_BLOCK) and _sweep < first_tangent_sweep:
                # Establish the support load before friction in inequality-only solves.
                phase_count = int32(1)
            for phase in range(phase_count):
                # Symmetric tangent ordering reduces load bias in redundant sticking patches.
                reverse_colors = phase == int32(1) and _sweep % int32(2) != int32(0)
                num_colors = inequality_num_colors[wid]
                for color_index in range(num_colors):
                    color = color_index
                    if reverse_colors:
                        color = num_colors - int32(1) - color_index
                    color_start = inequality_color_starts[schedule_offset + color]
                    color_end = inequality_color_starts[schedule_offset + color + int32(1)]
                    color_slot = color_start + lane
                    while color_slot < color_end:
                        uid = inequality_ids_by_color[uio + color_slot]
                        delta_0 = float32(0.0)
                        delta_1 = float32(0.0)
                        column = bcgo + uid
                        column_count = int32(1)
                        active = int32(1)
                        if uid < nbc:
                            if phase == int32(1):
                                active = int32(0)
                            else:
                                vec_idx = vio + column
                                bio = bcio + uid
                                lambda_bound_old = solution_lambdas[vec_idx]
                                diagonal = wp.abs(problem_D[mio + ncts * column + column])
                                lambda_bound_new = _project_box_update(
                                    lambda_bound_old,
                                    state_v_aug[vec_idx],
                                    diagonal,
                                    cfg.regularization,
                                    cfg.omega,
                                    problem_bound_lower[bio],
                                    problem_bound_upper[bio],
                                )
                                solution_lambdas[vec_idx] = lambda_bound_new
                                delta_0 = lambda_bound_new - lambda_bound_old
                        elif uid < nbc + nl:
                            column = lcgo + (uid - nbc)
                            if phase == int32(1):
                                active = int32(0)
                            else:
                                vec_idx = vio + column
                                lambda_limit_old = solution_lambdas[vec_idx]
                                diagonal = wp.abs(problem_D[mio + ncts * column + column])
                                lambda_limit_new = lambda_limit_old
                                if diagonal > FLOAT32_EPS:
                                    lambda_limit_new = wp.max(
                                        float32(0.0),
                                        lambda_limit_old
                                        - cfg.omega * state_v_aug[vec_idx] / (diagonal + cfg.regularization),
                                    )
                                solution_lambdas[vec_idx] = lambda_limit_new
                                delta_0 = lambda_limit_new - lambda_limit_old
                        else:
                            cid = uid - nbc - nl
                            column = ccgo + int32(3) * cid
                            if phase == int32(0):
                                column += int32(2)
                                vec_idx = vio + column
                                lambda_n_old = solution_lambdas[vec_idx]
                                diagonal_n = wp.abs(problem_D[mio + ncts * column + column])
                                lambda_n_new = _project_contact_normal_update(
                                    lambda_n_old,
                                    state_v_aug[vec_idx],
                                    diagonal_n,
                                    cfg.regularization,
                                    cfg.omega,
                                )
                                solution_lambdas[vec_idx] = lambda_n_new
                                delta_0 = lambda_n_new - lambda_n_old
                            else:
                                column_count = int32(2)
                                vec_idx = vio + column
                                lambda_t0_old = solution_lambdas[vec_idx]
                                lambda_t1_old = solution_lambdas[vec_idx + int32(1)]
                                diagonal_t0 = wp.abs(problem_D[mio + ncts * column + column])
                                diagonal_t1 = wp.abs(problem_D[mio + ncts * (column + int32(1)) + column + int32(1)])
                                lambda_t_old = wp.vec2f(lambda_t0_old, lambda_t1_old)
                                lambda_t_new = _project_contact_tangent_update(
                                    lambda_t_old,
                                    wp.vec2f(state_v_aug[vec_idx], state_v_aug[vec_idx + int32(1)]),
                                    wp.vec2f(diagonal_t0, diagonal_t1),
                                    problem_D[mio + ncts * column + column + int32(1)],
                                    cfg.regularization,
                                    cfg.omega,
                                    problem_mu[cio + cid]
                                    * _contact_friction_normal_load(
                                        solution_lambdas[vec_idx + int32(2)],
                                        problem_v_b[vec_idx + int32(2)],
                                        problem_P[vec_idx + int32(2)],
                                        wp.abs(problem_D[mio + ncts * (column + int32(2)) + column + int32(2)]),
                                        cfg.regularization,
                                        cfg.omega,
                                    ),
                                )
                                solution_lambdas[vec_idx] = lambda_t_new.x
                                solution_lambdas[vec_idx + int32(1)] = lambda_t_new.y
                                delta_0 = lambda_t_new.x - lambda_t_old.x
                                delta_1 = lambda_t_new.y - lambda_t_old.y

                        if active != int32(0):
                            state_delta[vio + column] = delta_0
                            if column_count == int32(2):
                                state_delta[vio + column + int32(1)] = delta_1
                        color_slot += threads_per_world

                    # Row-owned dense velocity update: lane ``r % threads_per_world``
                    # owns row ``r`` and adds the color's contributions in slot
                    # order, the sequence the one-lane kernel's atomic adds follow,
                    # so the result does not depend on the lane count.
                    _sync_threads()
                    row_count = contact_end - bcgo
                    color_size = color_end - color_start
                    row_task = lane
                    while row_task < row_count:
                        row = bcgo + row_task
                        velocity = state_v_aug[vio + row]
                        for local_color_slot in range(color_size):
                            uid = inequality_ids_by_color[uio + color_start + local_color_slot]
                            active = int32(0)
                            if uid >= nbc + nl or phase == int32(0):
                                active = int32(1)
                            if active != int32(0):
                                column = bcgo + uid
                                column_count = int32(1)
                                if uid >= nbc + nl:
                                    cid = uid - nbc - nl
                                    column = ccgo + int32(3) * cid
                                    if phase == int32(0):
                                        column += int32(2)
                                    else:
                                        column_count = int32(2)
                                elif uid >= nbc:
                                    column = lcgo + uid - nbc
                                # problem_D_t[mio + ncts * c + r] == problem_D[mio + ncts * r + c]
                                col_mio = mio + ncts * column + row
                                dv = problem_D_t[col_mio] * state_delta[vio + column]
                                if column_count == int32(2):
                                    dv += problem_D_t[col_mio + ncts] * state_delta[vio + column + int32(1)]
                                velocity = velocity + dv
                        state_v_aug[vio + row] = velocity
                        row_task += threads_per_world
                    _sync_threads()

        if enable_adaptive and lane == int32(0):
            status = solver_status[wid]
            status.iterations = completed_sweeps
            solver_status[wid] = status

    return _solve_dvi_inequalities_colored_pgs_rows


_DENSE_PGS_ROWS_KERNELS: dict = {}


def dense_pgs_rows_kernel(width: int):
    """Cached :func:`_make_dense_pgs_rows_kernel` for ``width``."""
    kernel = _DENSE_PGS_ROWS_KERNELS.get(width)
    if kernel is None:
        kernel = _make_dense_pgs_rows_kernel(width)
        _DENSE_PGS_ROWS_KERNELS[width] = kernel
    return kernel


@wp.kernel
def _transpose_dense_world_blocks(
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    src: wp.array[float32],
    dst: wp.array[float32],
):
    """``dst[mio + n * j + i] = src[mio + n * i + j]`` for each world's ``n x n`` block."""
    wid, tid = wp.tid()
    n = problem_dim[wid]
    if tid >= n * n:
        return
    i = tid / n
    j = tid - i * n
    mio = problem_mio[wid]
    dst[mio + n * j + i] = src[mio + n * i + j]


@wp.func
def _response_tile_skipped(
    use_pattern: bool, tile_pattern: wp.array[int32], tile_offset: int32, tiles: int32, row: int32, col: int32
) -> bool:
    """Whether factor tile ``(row, col)`` is outside the filled RCM tile mask (exactly zero)."""
    if not use_pattern:
        return False
    return tile_pattern[tile_offset + row * tiles + col] == int32(0)


@wp.kernel
def _solve_bilateral_unilateral_response_columns(
    problem_dim: wp.array[int32],
    problem_njc: wp.array[int32],
    bilateral_mio: wp.array[int32],
    bilateral_ld: wp.array[int32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    bilateral_L: wp.array[float32],
    bilateral_permutation: wp.array[int32],
    use_permutation: bool,
    response_mio: wp.array[int32],
    response_stride: wp.array[int32],
    coupling: wp.array[float32],
    response_factor: wp.array[float32],
    response: wp.array[float32],
    use_pattern: bool,
    tile_pattern_offsets: wp.array[int32],
    tile_pattern: wp.array[int32],
    block_size: int32,
):
    """Metal variant of ``_solve_bilateral_unilateral_response``: one thread per response column.

    The CPU/non-CUDA kernel walks a world's columns with ``wp.block_dim()``
    lanes (one on CPU); columns are independent, so here each thread solves
    one column with the same arithmetic in the same order, leaving out only
    terms that are exactly zero: factor tiles outside the RCM tile mask
    (``use_pattern``; the factorization clears them), rows above the column's
    first nonzero coupling entry (their forward solution is zero) and tiles
    whose partial solution is still all zero. Removing ``x - 0 * y`` terms
    keeps every value; at most the sign of a zero result can differ.
    """
    wid, unilateral = wp.tid()
    njc = problem_njc[wid]
    nu = problem_dim[wid] - njc
    if unilateral >= nu:
        return
    factor = bilateral_mio[wid]
    ld = bilateral_ld[wid]
    bvio = bilateral_vio[wid]
    offset = response_mio[wid]
    unilateral_stride = response_stride[wid]
    bs = wp.max(block_size, int32(1))
    tiles = (njc + bs - int32(1)) / bs
    tile_offset = int32(0)
    if use_pattern:
        tile_offset = tile_pattern_offsets[wid]

    first_row = njc
    for row in range(njc):
        original_row = row
        if use_permutation:
            original_row = bilateral_permutation[bvio + row]
        if first_row == njc and coupling[offset + original_row * unilateral_stride + unilateral] != float32(0.0):
            first_row = row
    if first_row == njc:
        for row in range(njc):
            original_row = row
            if use_permutation:
                original_row = bilateral_permutation[bvio + row]
            response_factor[offset + row * unilateral_stride + unilateral] = float32(0.0)
            response[offset + original_row * unilateral_stride + unilateral] = float32(0.0)
        return

    # Tiles holding a nonzero entry of the partial solution (up to 64 tiles): terms
    # against an all-zero tile are exactly zero and are left out.
    use_mask = use_pattern and tiles <= int32(64)
    nonzero_tiles = wp.uint64(0)
    for row in range(first_row):
        response_factor[offset + row * unilateral_stride + unilateral] = float32(0.0)
    for row in range(first_row, njc):
        original_row = row
        if use_permutation:
            original_row = bilateral_permutation[bvio + row]
        value = bilateral_P[bvio + original_row] * coupling[offset + original_row * unilateral_stride + unilateral]
        row_tile = row / bs
        for col_tile in range(first_row / bs, row_tile + int32(1)):
            if use_mask and ((nonzero_tiles >> wp.uint64(col_tile)) & wp.uint64(1)) == wp.uint64(0):
                continue
            if not _response_tile_skipped(use_pattern, tile_pattern, tile_offset, tiles, row_tile, col_tile):
                k_begin = wp.max(first_row, col_tile * bs)
                k_end = wp.min(row, (col_tile + int32(1)) * bs)
                for k in range(k_begin, k_end):
                    value -= (
                        bilateral_L[factor + ld * row + k]
                        * response_factor[offset + k * unilateral_stride + unilateral]
                    )
        solved = value / bilateral_L[factor + ld * row + row]
        response_factor[offset + row * unilateral_stride + unilateral] = solved
        if solved != float32(0.0):
            nonzero_tiles = nonzero_tiles | (wp.uint64(1) << wp.uint64(row_tile))

    nonzero_tiles = wp.uint64(0)
    for reverse_row in range(njc):
        row = njc - int32(1) - reverse_row
        value = response_factor[offset + row * unilateral_stride + unilateral]
        col_tile = row / bs
        for row_tile in range(col_tile, tiles):
            if use_mask and ((nonzero_tiles >> wp.uint64(row_tile)) & wp.uint64(1)) == wp.uint64(0):
                continue
            if not _response_tile_skipped(use_pattern, tile_pattern, tile_offset, tiles, row_tile, col_tile):
                k_begin = wp.max(row + int32(1), row_tile * bs)
                k_end = wp.min(njc, (row_tile + int32(1)) * bs)
                for k in range(k_begin, k_end):
                    value -= (
                        bilateral_L[factor + ld * k + row]
                        * response_factor[offset + k * unilateral_stride + unilateral]
                    )
        solved = value / bilateral_L[factor + ld * row + row]
        response_factor[offset + row * unilateral_stride + unilateral] = solved
        if solved != float32(0.0):
            nonzero_tiles = nonzero_tiles | (wp.uint64(1) << wp.uint64(col_tile))

    for row in range(njc):
        original_row = row
        if use_permutation:
            original_row = bilateral_permutation[bvio + row]
        response[offset + original_row * unilateral_stride + unilateral] = (
            bilateral_P[bvio + original_row] * response_factor[offset + row * unilateral_stride + unilateral]
        )
