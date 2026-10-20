# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Metal-only comparisons for Kamino's lane-parallel Metal code paths.

- :class:`LLTBlockedSolver` uses the lane-parallel ``llt_lanes`` kernels on Metal; its factor and
  solution must match the blocked tile kernels on CPU.
- The accelerated PADMM projection kernel runs with 32 lanes per world on Metal. With more constraint
  rows than lanes each lane handles several rows; the result must match a one-lane launch (the
  serial sweep the CPU runs) and the CPU solve.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.linear import LLTBlockedSolver
from newton._src.solvers.kamino._src.solvers.padmm import PADMMSolver, PADMMWarmStartMode
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.test_kamino_solvers_padmm import TestSetup
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
