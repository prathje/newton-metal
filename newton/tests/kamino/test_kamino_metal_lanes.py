# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Metal-only comparisons for Kamino's lane-parallel Metal code paths.

- :class:`LLTBlockedSolver` uses the lane-parallel ``llt_lanes`` kernels on Metal; its factor and
  solution must match the blocked tile kernels on CPU.
- The accelerated PADMM projection kernel runs with 32 lanes per world on Metal. With more constraint
  rows than lanes each lane handles several rows; the result must match a one-lane launch (the
  serial sweep the CPU runs) and the CPU solve.
- The DVI dense colored PGS kernel runs as one threadgroup of 64..1024 threads per world on Metal
  (row-owned velocity update, ``dvi/kernels_metal.py``). On one problem it must reproduce the one-lane
  CUDA/CPU kernel bit for bit, and match the CPU solve within tolerance.
"""

import unittest

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.linear import LLTBlockedSolver
from newton._src.solvers.kamino._src.solvers.common import WarmStartMode
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.dvi.kernels import _solve_dvi_inequalities_colored_pgs
from newton._src.solvers.kamino._src.solvers.padmm import PADMMSolver, PADMMWarmStartMode
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.test_kamino_solvers_padmm import TestSetup
from newton.tests.kamino.test_kamino_solvers_dvi import _build_five_box_stack, _build_reduced_kapla_tower
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.kamino.utils.rand import RandomProblemLLT
from newton.tests.utils import basics


def _metal_device():
    for d in wp.get_devices():
        if getattr(d, "is_metal", False):
            return d
    return None


def _llt_on(device, dims, maxdims):
    problem = RandomProblemLLT(
        dims=dims, maxdims=maxdims, seed=5, np_dtype=np.float32, wp_dtype=wp.float32, device=device
    )
    opinfo = DenseSquareMultiLinearInfo()
    opinfo.finalize(dimensions=problem.maxdims, dtype=wp.float32, device=device)
    operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
    llt = LLTBlockedSolver(operator=operator, device=device)
    operator.info.dim.assign(dims)
    llt.compute(A=problem.A_wp)
    x = wp.zeros_like(problem.b_wp)
    llt.solve(b=problem.b_wp, x=x)
    return llt.L.numpy(), x.numpy(), problem.b_wp.numpy()


def _padmm_builder():
    # Four-bar linkage plus two box stacks in one world: more than 32 constraint rows.
    builder = basics.build_boxes_fourbar()
    basics.build_boxes_stacked_on_plane(builder=builder, z_offset=2.0, ground=False, new_world=False)
    basics.build_boxes_stacked_on_plane(builder=builder, z_offset=4.0, ground=False, new_world=False)
    return builder


def _padmm_solve(device, block_dim=None):
    orig = PADMMSolver._project_dual_convergence_accel_block_dim
    if block_dim is not None:
        PADMMSolver._project_dual_convergence_accel_block_dim = lambda self: block_dim
    try:
        test = TestSetup(builder_fn=_padmm_builder, max_world_contacts=64, device=device)
        config = PADMMSolver.Config()
        config.primal_tolerance = config.dual_tolerance = config.compl_tolerance = 1e-6
        config.max_iterations = 300
        solver = PADMMSolver(model=test.model, config=config, warmstart=PADMMWarmStartMode.NONE, use_acceleration=True)
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        return (
            solver.data.status.numpy(),
            solver.data.solution.lambdas.numpy(),
            int(test.problem.data.dim.numpy().max()),
        )
    finally:
        PADMMSolver._project_dual_convergence_accel_block_dim = orig


def _dvi_pgs_builders():
    def stacks():
        builder = newton.ModelBuilder()
        builder.replicate(builder=_build_five_box_stack(), world_count=3)
        return builder

    def kapla():
        return _build_reduced_kapla_tower(layer_count=3)[0]

    # name -> (builder, forced block width or None for the solver's choice)
    return {
        "five_box_stacks": (stacks, None),
        "fourbar_and_boxes": (_padmm_builder, None),
        # 64 threads for a few hundred rows: every thread owns several rows.
        "kapla_tower_64_threads": (kapla, 64),
    }


def _dvi_pgs_problem(device, builder_fn):
    """A dense DVI problem with contacts (and, for the four-bar, bilateral joint rows)."""
    model = ModelKamino.from_newton(builder_fn().finalize(device=device))
    model, data, state, limits, detector, jacobians = make_containers(
        model=model, max_world_contacts=256, sparse=False, dt=1.0e-3
    )
    update_containers(model, data, state, limits, detector, jacobians)
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=detector.contacts,
        jacobians=jacobians,
        solver=LLTBlockedSolver,
        sparse=False,
    )
    problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)
    solver = DVISolver(
        model=model,
        data=data,
        limits=limits,
        contacts=detector.contacts,
        jacobians=jacobians,
        config=kamino_config.DVISolverConfig(
            unilateral_solver="pgs",
            max_alternating_iterations=40,
            inequality_sweeps_per_iteration=2,
            tolerance=1e-6,
            regularization=1e-6,
        ),
        warmstart=WarmStartMode.NONE,
    )
    return problem, solver


def _dvi_pgs_solve(problem, solver, kernel=None, threads_per_world=None):
    """Solve with an overridden dense PGS kernel / block width; returns (lambdas, v_aug, kernel launches)."""
    launches = [0]
    if threads_per_world is not None:
        solver._dense_pgs_threads_per_world = lambda: threads_per_world
    chosen = kernel or solver._dense_pgs_kernel()

    def _kernel():
        launches[0] += 1
        return chosen

    solver._dense_pgs_kernel = _kernel
    if chosen is _solve_dvi_inequalities_colored_pgs:
        # The CUDA/CPU kernel reads the operator directly (no transposed copy).
        solver._dense_pgs_operator = lambda _problem, _mio, D: (D,)
    try:
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        return solver.data.solution.lambdas.numpy(), solver.data.state.v_aug.numpy(), launches[0]
    finally:
        del solver._dense_pgs_kernel
        solver.__dict__.pop("_dense_pgs_threads_per_world", None)
        solver.__dict__.pop("_dense_pgs_operator", None)


class TestKaminoMetalLanes(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.metal = _metal_device()
        if self.metal is None:
            self.skipTest("needs a Metal device")

    def test_llt_blocked_solver_metal_matches_cpu(self):
        dims = [33, 64, 140, 5]
        maxdims = [40, 64, 150, 8]
        L_m, x_m, b_m = _llt_on(self.metal, dims, maxdims)
        L_c, x_c, b_c = _llt_on("cpu", dims, maxdims)
        np.testing.assert_array_equal(b_m, b_c)  # same problem data
        np.testing.assert_allclose(L_m, L_c, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(x_m, x_c, rtol=1e-4, atol=1e-5)

    def test_padmm_projection_with_more_rows_than_lanes(self):
        st_c, lam_c, ncts = _padmm_solve("cpu")
        st_1, lam_1, _ = _padmm_solve(self.metal, block_dim=1)
        st_32, lam_32, ncts_m = _padmm_solve(self.metal)
        self.assertEqual(ncts, ncts_m)
        self.assertGreater(ncts, 32, "the test needs several rows per lane")
        self.assertTrue(bool(st_32["converged"][0]))
        self.assertTrue(bool(st_1["converged"][0]))
        # simd_sum reorders the r_dx/r_dy/r_dz sums: equal within tolerance, not bitwise.
        self.assertEqual(int(st_32["iterations"][0]), int(st_1["iterations"][0]))
        np.testing.assert_allclose(lam_32, lam_1, rtol=1e-4, atol=1e-6)
        for field in ("r_p", "r_d", "r_c", "r_dx", "r_dy", "r_dz"):
            np.testing.assert_allclose(st_32[field], st_1[field], rtol=1e-3, atol=1e-9, err_msg=field)
        # CPU runs the same algorithm with different rounding (iteration counts may differ).
        self.assertTrue(bool(st_c["converged"][0]))
        np.testing.assert_allclose(lam_32, lam_c, rtol=1e-3, atol=1e-5)

    def test_dvi_dense_pgs_lanes_match_one_lane_bitwise(self):
        for name, (builder_fn, width) in _dvi_pgs_builders().items():
            with self.subTest(name):
                problem, solver = _dvi_pgs_problem(self.metal, builder_fn)
                ncts = problem.data.dim.numpy()
                self.assertGreater(int(ncts.max()), 32, "the test needs several rows per lane")
                # One problem on Metal, solved by the CUDA/CPU kernel with one lane and by the Metal
                # kernel with one threadgroup per world (row-owned velocity update: same summation
                # order).
                lam_ref, v_ref, n_ref = _dvi_pgs_solve(problem, solver, _solve_dvi_inequalities_colored_pgs, 1)
                lam_32, v_32, n_32 = _dvi_pgs_solve(problem, solver, threads_per_world=width)
                self.assertGreater(n_32, 0)
                # Ran as one real threadgroup per world (Warp's wide-block model), with more
                # threads than one SIMD group and fewer than the rows.
                width = width or solver._dense_pgs_threads_per_world()
                self.assertGreater(width, 32)
                if name.startswith("kapla"):
                    self.assertGreater(int(ncts.max()), 4 * width)
                from newton._src.solvers.kamino._src.solvers.dvi.kernels_metal import dense_pgs_rows_kernel

                self.assertEqual(getattr(dense_pgs_rows_kernel(width).adj, "_metal_coop_block_n", 0), width)
                self.assertEqual(n_ref, n_32)
                self.assertGreater(np.abs(lam_ref).max(), 0.0)
                np.testing.assert_array_equal(lam_32, lam_ref)
                np.testing.assert_array_equal(v_32, v_ref)
                # CPU: same algorithm and rows, but contacts may come in another order and redundant
                # contact patches split their load non-uniquely; compare each world's total impulse.
                problem_c, solver_c = _dvi_pgs_problem("cpu", builder_fn)
                np.testing.assert_array_equal(problem_c.data.dim.numpy(), ncts)
                lam_c, _v_c, _ = _dvi_pgs_solve(problem_c, solver_c)
                vio = problem.data.vio.numpy()
                for w, (o, n) in enumerate(zip(vio, ncts)):
                    np.testing.assert_allclose(
                        lam_32[o : o + n].sum(), lam_c[o : o + n].sum(), rtol=2e-3, err_msg=f"world {w}"
                    )

if __name__ == "__main__":
    unittest.main(verbosity=2)
