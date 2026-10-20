# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Metal variant of the dense colored PGS kernel.

The CUDA kernel spreads each color's dense velocity update over (slot, row)
tasks and accumulates with float atomics, so on more than one lane the
summation order (and the rounding) changes from run to run. On Metal the
kernel runs as one 32-lane SIMD group per world (Warp's coop-block lowering);
here every row has one owner lane that adds the color's contributions in slot
order, which reproduces the one-lane (CPU) summation sequence exactly.
Everything else is the CUDA/CPU kernel unchanged. Kept in its own module so
the CUDA/CPU module is unaffected.
"""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from .kernels import (
    _FUSED_BILATERAL_BLOCK,
    _FUSED_INEQUALITY_BLOCK,
    _contact_friction_normal_load,
    _dense_unilateral_terminal_residuals,
    _project_box_update,
    _project_contact_normal_update,
    _project_contact_tangent_update,
    _sync_threads,
)
from .types import DVIConfigStruct, DVIStatus

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32
vec3f = wp.vec3f


@wp.kernel
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
    """Metal variant of :func:`_solve_dvi_inequalities_colored_pgs` with a row-owned velocity update."""
    tid = wp.tid()
    threads_per_world = int32(wp.block_dim())
    lane = tid % threads_per_world
    wid = tid / threads_per_world
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
    adaptive = enable_adaptive and cfg.tolerance > float32(0.0)
    completed_sweeps = sweep_count
    pair_update = float32(0.0)
    for _sweep in range(sweep_count):
        sweep_update = float32(0.0)
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
                        if adaptive:
                            sweep_update = wp.max(sweep_update, wp.max(wp.abs(delta_0), wp.abs(delta_1)))
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
                    row_mio = mio + ncts * row
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
                            dv = problem_D[row_mio + column] * state_delta[vio + column]
                            if column_count == int32(2):
                                dv += problem_D[row_mio + column + int32(1)] * state_delta[vio + column + int32(1)]
                            velocity = velocity + dv
                    state_v_aug[vio + row] = velocity
                    row_task += threads_per_world
                _sync_threads()

        if adaptive:
            sweep_update_max = wp.tile_max(wp.tile(sweep_update))[0]
            pair_update = wp.max(pair_update, sweep_update_max)
            if (_sweep + int32(1)) % int32(2) == int32(0):
                if pair_update <= cfg.tolerance:
                    local_residuals = _dense_unilateral_terminal_residuals(
                        nl,
                        nc,
                        lcgo,
                        ccgo,
                        cio,
                        vio,
                        lane,
                        threads_per_world,
                        problem_mu,
                        problem_P,
                        state_v_aug,
                        solution_lambdas,
                    )
                    r_p = wp.tile_max(wp.tile(local_residuals[0]))[0]
                    r_d = wp.tile_max(wp.tile(local_residuals[1]))[0]
                    r_c = wp.tile_max(wp.tile(local_residuals[2]))[0]
                    if r_p <= cfg.tolerance and r_d <= cfg.tolerance and r_c <= cfg.tolerance:
                        completed_sweeps = _sweep + int32(1)
                        break
                pair_update = float32(0.0)

    if enable_adaptive and lane == int32(0):
        status = solver_status[wid]
        status.iterations = completed_sweeps
        solver_status[wid] = status
