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
- The bilateral-unilateral response (sparse Schur DVI) runs one thread per column on Metal and skips
  exactly-zero terms (RCM tile mask, zero coupling rows, all-zero solution tiles); it must match the
  one-lane kernel bit for bit and the CPU kernel within tolerance.
- :class:`LLTBlockedRCMSolver` uses tile-free factorize/solve kernels on Metal
  (``llt_blocked_rcm_metal``) and an in-order RCM traversal (``rcm_batch_metal``); permutation, factor
  and solutions are bit-identical to CPU.
- The general sparse PGS kernel runs as one 256-thread threadgroup per world on Metal for a few worlds
  (wide blocks),
  the contact-only one with 64 threads; lanes accumulate with atomics, so both must match one lane and
  CPU within tolerance.
"""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm_solver import LLTBlockedRCMSolver
from newton._src.solvers.kamino._src.linalg.linear import LLTBlockedSolver
from newton._src.solvers.kamino._src.solvers.common import WarmStartMode
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.dvi import sparse as dvi_sparse
from newton._src.solvers.kamino._src.solvers.dvi.kernels import (
    _solve_bilateral_unilateral_response,
    _solve_dvi_inequalities_colored_pgs,
)
from newton._src.solvers.kamino._src.solvers.dvi.kernels_metal import _solve_bilateral_unilateral_response_columns
from newton._src.solvers.kamino._src.solvers.padmm import PADMMSolver, PADMMWarmStartMode
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino import test_kamino_solver_kamino_joint_friction as joint_friction
from newton.tests.kamino.test_kamino_solvers_dvi import _build_five_box_stack, _build_reduced_kapla_tower
from newton.tests.kamino.test_kamino_solvers_padmm import TestSetup
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

    def mixed_worlds():
        # Worlds of 60 / 56 / 240 rows (not powers of two) in one launch.
        builder = newton.ModelBuilder()
        builder.add_world(_build_five_box_stack())
        builder.add_world(_padmm_builder())
        builder.add_world(_build_reduced_kapla_tower(layer_count=2)[0])
        return builder

    # name -> (builder, forced block width or None for the solver's choice, check CPU world totals,
    #          Warp coop tiles on)
    return {
        "five_box_stacks": (stacks, None, True, True),
        "fourbar_and_boxes": (_padmm_builder, None, True, True),
        # 64 threads for a few hundred rows: every thread owns several rows.
        "kapla_tower_64_threads": (kapla, 64, True, True),
        # CPU emits world 0's contacts in another order, so its 40-iteration PGS stops at a
        # different unconverged iterate (0.1547 vs 0.1487; both reach 0.14715 = 15 g dt with
        # 1000+ iterations): no CPU totals check here.
        "mixed_worlds": (mixed_worlds, None, False, True),
        # Warp keeps the serial model (block_dim() == 1): lane 0 must sweep the world alone.
        "five_box_stacks_coop_tiles_off": (stacks, 256, True, False),
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


def _response_problem(device, block_sparse=False, failed=False):
    """Two worlds of bilateral-unilateral response inputs.

    ``block_sparse``: the factor is zero outside a random 8x8-tile mask (as the
    RCM factorization leaves it) and some coupling columns are zero or start
    with zero rows; the Metal kernel then skips those terms.
    ``failed``: world 0 has NaN pivots (a failed factorization), world 1 a zero
    pivot; the NaNs must reach the entries they reach in the one-lane kernel.
    """
    rng = np.random.default_rng(11)
    njc, nu = [37, 9], [50, 5]
    ld, stride, bs = 40, 56, 8
    L = np.zeros((2, ld, ld), np.float32)
    patterns = []
    for w in range(2):
        n = njc[w]
        L[w, :n, :n] = np.tril(rng.uniform(-0.3, 0.3, (n, n)))
        tiles = (n + bs - 1) // bs
        pattern = np.ones((tiles, tiles), np.int32)
        if block_sparse:
            pattern = np.tril((rng.random((tiles, tiles)) < 0.4).astype(np.int32), -1)
            pattern[np.arange(tiles), np.arange(tiles)] = 1
            mask = np.kron(pattern, np.ones((bs, bs)))[:n, :n]
            L[w, :n, :n] *= mask
        patterns.append(pattern.ravel())
        L[w, np.arange(n), np.arange(n)] = rng.uniform(1.0, 2.0, n)
    if failed:
        L[0, 20:, 20] = np.nan
        L[1, 3, 3] = 0.0
    perm = np.concatenate([np.pad(rng.permutation(n), (0, ld - n)) for n in njc]).astype(np.int32)
    coupling = rng.uniform(-1.0, 1.0, (2, ld, stride)).astype(np.float32)
    if block_sparse:
        # Columns 0-9: zero; columns 10-29: nonzero only from a random permuted row on.
        coupling[:, :, :10] = 0.0
        for w in range(2):
            for col in range(10, 30):
                first = rng.integers(0, njc[w])
                coupling[w, perm[w * ld : w * ld + first], col] = 0.0
    arrays = {
        "dim": np.array([a + b for a, b in zip(njc, nu, strict=True)], np.int32),
        "njc": np.array(njc, np.int32),
        "mio": np.array([0, ld * ld], np.int32),
        "ld": np.array([ld, ld], np.int32),
        "vio": np.array([0, ld], np.int32),
        "P": rng.uniform(0.5, 2.0, 2 * ld).astype(np.float32),
        "L": L.ravel(),
        "perm": perm,
        "rmio": np.array([0, ld * stride], np.int32),
        "stride": np.array([stride, stride], np.int32),
        "coupling": coupling.ravel(),
        "tpo": np.array([0, patterns[0].size], np.int32),
        "pattern": np.concatenate(patterns).astype(np.int32),
    }
    a = {k: wp.array(v, device=device) for k, v in arrays.items()}
    factor = wp.zeros(2 * ld * stride, dtype=wp.float32, device=device)
    response = wp.zeros(2 * ld * stride, dtype=wp.float32, device=device)
    inputs = [
        a["dim"],
        a["njc"],
        a["mio"],
        a["ld"],
        a["vio"],
        a["P"],
        a["L"],
        a["perm"],
        True,
        a["rmio"],
        a["stride"],
        a["coupling"],
        factor,
        response,
    ]
    pattern_inputs = [block_sparse, a["tpo"], a["pattern"], bs]
    return inputs, pattern_inputs, factor, response, max(nu)


def _rcm_matrices(seed=3):
    """Shuffled SPD blocks: block-diagonal (Kamino joint rows), banded, random sparse, dense and 1x1.

    Banded and random sparse graphs have BFS levels with several vertices, whose order the
    atomic GPU traversal used to change from run to run.
    """
    rng = np.random.default_rng(seed)

    def shuffled(A):
        p = rng.permutation(A.shape[0])
        return A[np.ix_(p, p)].astype(np.float32)

    def block_diagonal(n, block):
        A = np.zeros((n, n))
        for s in range(0, n, block):
            e = min(n, s + block)
            m = rng.standard_normal((e - s, e - s))
            A[s:e, s:e] = m @ m.T + (e - s) * np.eye(e - s)
        return shuffled(A)

    band = np.tril(np.triu(rng.standard_normal((70, 70)), -6), 6)
    sparse = rng.standard_normal((80, 80)) * (rng.random((80, 80)) < 0.04)
    dense = rng.standard_normal((40, 40))
    return [
        block_diagonal(90, 5),
        block_diagonal(37, 5),
        shuffled(band @ band.T + 7.0 * np.eye(70)),
        shuffled(sparse @ sparse.T + 16.0 * np.eye(80)),
        (dense @ dense.T + 40 * np.eye(40)).astype(np.float32),
        np.eye(1, dtype=np.float32),
    ]


def _rcm_solve(device, mats, capacity_stride):
    """Factor and solve with :class:`LLTBlockedRCMSolver`; returns (P, L, x from solve, x from solve_inplace)."""
    dims = [m.shape[0] for m in mats]
    caps = [(d + 31) // 32 * 32 for d in dims] if capacity_stride else dims
    info = DenseSquareMultiLinearInfo()
    info.finalize(dimensions=caps, dtype=wp.float32, device=device)
    info.dim.assign(np.array(dims, np.int32))
    if capacity_stride:
        flat = np.concatenate(
            [np.pad(m, ((0, c - m.shape[0]), (0, c - m.shape[0]))).ravel() for m, c in zip(mats, caps, strict=True)]
        )
    else:
        flat = np.concatenate([m.ravel() for m in mats])
    A = wp.array(flat, dtype=wp.float32, device=device)
    solver = LLTBlockedRCMSolver(
        operator=DenseLinearOperatorData(info=info, mat=A),
        capacity_stride=capacity_stride,
        failed_pivot_shift=1.0e-4,
        device=device,
    )
    solver.compute(A)
    rhs = np.random.default_rng(5).standard_normal(sum(caps)).astype(np.float32)
    x = wp.zeros(sum(caps), dtype=wp.float32, device=device)
    solver.solve(wp.array(rhs, device=device), x)
    x_inplace = wp.array(rhs, device=device)
    solver.solve_inplace(x_inplace)
    return solver.P.numpy(), solver.L.numpy(), x.numpy(), x_inplace.numpy()


def _compact_schur_rollout(device, steps=4):
    """A short sparse compact-Schur DVI rollout (the joint-friction test's model); returns joint_qd per step."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    joint_counts = (40, 9)
    frictions = np.resize(np.array([0.02, 0.2, 2.0, 20.0]), sum(joint_counts))
    joint_index = 0
    for count in joint_counts:
        builder.begin_world()
        for _ in range(count):
            body = builder.add_link(
                mass=joint_friction._BODY_MASS,
                inertia=np.eye(3).ravel().tolist(),
                com=wp.vec3f(joint_friction._BODY_COM_X, 0.0, 0.0),
                lock_inertia=True,
            )
            joint = builder.add_joint_revolute(-1, body, axis=newton.Axis.Y, friction=float(frictions[joint_index]))
            joint_index += 1
            builder.add_articulation([joint])
        builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    config = SolverKamino.Config(
        dynamics_solver="dvi",
        sparse_dynamics=True,
        sparse_jacobian=True,
        use_collision_detector=False,
        use_fk_solver=False,
        dvi=kamino_config.DVISolverConfig(
            tolerance=1.0e-5,
            use_schur_complement=True,
            bilateral_solver_type="LLTBRCM",
            max_alternating_iterations=8,
            bilateral_solve_interval=8,
            inequality_sweeps_per_iteration=2,
        ),
    )
    solver = SolverKamino(model, config)
    initial = np.resize(np.array([-0.04, 0.008, 0.0, -0.008, 0.04], dtype=np.float32), sum(joint_counts))
    model.joint_qd.assign(initial * frictions / 2.0)
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    control.joint_f.assign((np.resize([0.5, -1.5, 1.0001], sum(joint_counts)) * frictions).astype(np.float32))
    out = []
    for _ in range(steps):
        solver.step(state, next_state, control=control, contacts=None, dt=joint_friction.DT)
        state, next_state = next_state, state
        out.append(state.joint_qd.numpy().copy())
    return np.array(out)


def _sparse_kapla_rollout(device, steps=12):
    """A short sparse DVI rollout of a reduced kapla tower (contact-only sparse PGS); returns body_q per step."""
    builder, _bodies, _ = _build_reduced_kapla_tower(layer_count=3)
    model = builder.finalize(device=device)
    config = SolverKamino.Config(
        dynamics_solver="dvi",
        use_collision_detector=True,
        sparse_dynamics=True,
        sparse_jacobian=True,
        collision_detector=kamino_config.CollisionDetectorConfig(
            max_contacts=256, max_contacts_per_world=256, max_contacts_per_pair=8
        ),
    )
    solver = SolverKamino(model, config=config)
    state_0, state_1 = model.state(), model.state()
    impulse, qd = [], []
    for _ in range(steps):
        solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-3)
        state_0, state_1 = state_1, state_0
        # Total impulse: independent of the (device-dependent) contact order.
        impulse.append(float(solver._solver_kamino.solver_fd.data.solution.lambdas.numpy().astype(np.float64).sum()))
        qd.append(state_0.body_qd.numpy().copy())
    return np.array(impulse), np.array(qd)


def _failed_pivot_blocks(shift):
    """Healthy and slightly indefinite SPD-like blocks (a ragged tile and the DR Legs size)."""
    rng = np.random.default_rng(97)
    blocks, failing = [], []
    for n in (40, 192):
        basis = rng.normal(size=(n, n))
        blocks.append((basis @ basis.T / n + np.eye(n)).astype(np.float32))
        failing.append(False)
        # One slightly negative eigenvalue mimics round-off on a singular block.
        eigenvectors = np.linalg.qr(rng.normal(size=(n, n)))[0]
        eigenvalues = np.linspace(1.0, 2.0, n)
        eigenvalues[0] = -0.2 * shift
        blocks.append((eigenvectors * eigenvalues @ eigenvectors.T).astype(np.float32))
        failing.append(True)
    return blocks, failing


def _failed_pivot_factor(device, solver_cls, blocks, shift, captured, **kwargs):
    """Factor ``blocks`` with ``failed_pivot_shift``; returns (factor, matrix after compute, mio)."""
    info = DenseSquareMultiLinearInfo()
    info.finalize(dimensions=[b.shape[0] for b in blocks], dtype=wp.float32, device=device)
    source = np.concatenate([b.ravel() for b in blocks])
    matrix = wp.array(source, dtype=wp.float32, device=device)
    solver = solver_cls(
        operator=DenseLinearOperatorData(info=info, mat=matrix), failed_pivot_shift=shift, device=device, **kwargs
    )
    solver.compute(matrix)
    if captured:
        with wp.ScopedCapture(device=device) as capture:
            solver.compute(matrix)
        wp.capture_launch(capture.graph)
    P = solver.P.numpy() if hasattr(solver, "P") else None
    return solver.L.numpy(), matrix.numpy(), info.mio.numpy(), P


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
        from newton._src.solvers.kamino._src.solvers.dvi.kernels_metal import dense_pgs_rows_kernel  # noqa: PLC0415

        for name, (builder_fn, width, check_cpu, coop_tiles) in _dvi_pgs_builders().items():
            with self.subTest(name):
                problem, solver = _dvi_pgs_problem(self.metal, builder_fn)
                ncts = problem.data.dim.numpy()
                self.assertGreater(int(ncts.max()), 32, "the test needs several rows per lane")
                # One problem on Metal, solved by the CUDA/CPU kernel with one lane and by the Metal
                # kernel with one threadgroup per world (row-owned velocity update: same summation
                # order).
                lam_ref, v_ref, n_ref = _dvi_pgs_solve(problem, solver, _solve_dvi_inequalities_colored_pgs, 1)
                saved = wp.config.metal_coop_tiles
                wp.config.metal_coop_tiles = coop_tiles
                try:
                    lam_m, v_m, n_m = _dvi_pgs_solve(problem, solver, threads_per_world=width)
                finally:
                    wp.config.metal_coop_tiles = saved
                self.assertGreater(n_m, 0)
                launch_width = width or solver._dense_pgs_threads_per_world()
                self.assertGreater(launch_width, 32)
                if name.startswith("kapla"):
                    self.assertGreater(int(ncts.max()), 4 * launch_width)
                # Wide-block model (one real threadgroup per world), or the serial model when
                # coop tiles are off.
                expected_block = launch_width if coop_tiles else 0
                self.assertEqual(
                    getattr(dense_pgs_rows_kernel(launch_width).adj, "_metal_coop_block_n", 0), expected_block
                )
                self.assertEqual(n_ref, n_m)
                self.assertGreater(np.abs(lam_ref).max(), 0.0)
                np.testing.assert_array_equal(lam_m, lam_ref)
                np.testing.assert_array_equal(v_m, v_ref)
                if not check_cpu:
                    continue
                # CPU: same algorithm and rows, but contacts may come in another order and redundant
                # contact patches split their load non-uniquely; compare each world's total impulse.
                problem_c, solver_c = _dvi_pgs_problem("cpu", builder_fn)
                np.testing.assert_array_equal(problem_c.data.dim.numpy(), ncts)
                lam_c, _v_c, _ = _dvi_pgs_solve(problem_c, solver_c)
                vio = problem.data.vio.numpy()
                for w, (o, n) in enumerate(zip(vio, ncts, strict=True)):
                    np.testing.assert_allclose(
                        lam_m[o : o + n].sum(), lam_c[o : o + n].sum(), rtol=2e-3, err_msg=f"world {w}"
                    )

    def test_bilateral_response_columns_match_one_lane(self):
        for block_sparse, failed in ((False, False), (True, False), (True, True)):
            results = {}
            for name, device, kernel in (
                ("cpu", "cpu", _solve_bilateral_unilateral_response),
                ("metal_1", self.metal, _solve_bilateral_unilateral_response),
                ("metal_columns", self.metal, _solve_bilateral_unilateral_response_columns),
            ):
                inputs, pattern_inputs, factor, response, max_nu = _response_problem(device, block_sparse, failed)
                if kernel is _solve_bilateral_unilateral_response:
                    wp.launch(kernel, dim=2, inputs=inputs, device=device, block_dim=1)
                else:
                    wp.launch(kernel, dim=(2, max_nu), inputs=inputs + pattern_inputs, device=device, block_dim=256)
                results[name] = (factor.numpy(), response.numpy())
            if failed:
                # Both worlds fail; a skipped 0 * NaN term would leave finite entries.
                self.assertTrue(np.isnan(results["cpu"][1]).any())
            else:
                self.assertGreater(np.abs(results["cpu"][1]).max(), 0.1)
            # Skipped terms are exact zeros: equal values (a zero may differ in sign), NaNs in
            # the same places.
            for a, b in zip(results["metal_columns"], results["metal_1"], strict=True):
                np.testing.assert_array_equal(a, b, err_msg=f"block_sparse={block_sparse} failed={failed}")
            for a, b in zip(results["metal_columns"], results["cpu"], strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_failed_pivot_retry_on_metal(self):
        # failed_pivot_shift (#4839) refactorizes blocks with a non-finite pivot after a diagonal
        # shift. LLTB runs the lane-parallel kernels on Metal and LLTBRCM the tile-free ones;
        # both must take the retry (no NaN factors) and leave healthy blocks untouched.
        shift = 1.0e-5
        blocks, failing = _failed_pivot_blocks(shift)
        source = np.concatenate([b.ravel() for b in blocks])
        for solver_cls in (LLTBlockedSolver, LLTBlockedRCMSolver):
            for captured in (False, True):
                with self.subTest(solver=solver_cls.__name__, captured=captured):
                    L, matrix, mio, P = _failed_pivot_factor(self.metal, solver_cls, blocks, shift, captured)
                    L_cpu, _, _, _ = _failed_pivot_factor("cpu", solver_cls, blocks, shift, False)
                    self.assertTrue(np.isfinite(L).all())
                    if solver_cls is LLTBlockedSolver:
                        # The shift is removed from the caller's matrix again.
                        np.testing.assert_allclose(matrix, source, rtol=0.0, atol=1.0e-6)
                    vio = 0
                    for index, (A, fails) in enumerate(zip(blocks, failing, strict=True)):
                        n = A.shape[0]
                        entries = slice(mio[index], mio[index] + n * n)
                        factor = np.tril(L[entries].reshape(n, n))
                        np.testing.assert_allclose(factor, np.tril(L_cpu[entries].reshape(n, n)), rtol=0.0, atol=2.0e-3)
                        A64 = A.astype(np.float64)
                        if P is not None:
                            order = P[vio : vio + n]
                            A64 = A64[np.ix_(order, order)]
                        vio += n
                        expected = np.linalg.cholesky(A64 + (shift if fails else 0.0) * np.eye(n))
                        np.testing.assert_allclose(factor, expected, rtol=0.0, atol=2.0e-3)

    def test_lox_solver_rejected_on_metal(self):
        # LOX's kernels cannot be compiled (or would race) on Metal: construction must fail loudly.
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        body = builder.add_link(mass=1.0, inertia=np.eye(3).ravel().tolist())
        builder.add_articulation([builder.add_joint_revolute(-1, body, axis=newton.Axis.Y)])
        model = builder.finalize(device=self.metal)
        with self.assertRaisesRegex(NotImplementedError, "LOX dynamics solver is not supported on Metal"):
            SolverKamino(model, SolverKamino.Config(dynamics_solver="lox", use_collision_detector=False))

    def test_rcm_blocked_cholesky_matches_cpu_bitwise(self):
        # Metal traverses the RCM graph in the CPU's order, and its factorize/solve kernels
        # reproduce the CPU tile kernels' arithmetic order (fmaf tile dot products, scalar
        # Cholesky, dividing solves): permutation, factor and both solves are bit-identical.
        mats = _rcm_matrices()
        for capacity_stride in (True, False):
            cpu = _rcm_solve("cpu", mats, capacity_stride)
            metal = _rcm_solve(self.metal, mats, capacity_stride)
            np.testing.assert_array_equal(metal[0], cpu[0])
            for a, b in zip(metal[1:], cpu[1:], strict=True):
                self.assertGreater(np.abs(b).max(), 0.01)
                np.testing.assert_array_equal(a, b, err_msg=f"capacity_stride={capacity_stride}")

    def test_sparse_pgs_wide_block_matches_one_lane(self):
        # One lane first: a kernel object keeps the block model it gets at 256 threads.
        with mock.patch.object(dvi_sparse, "_metal_sparse_inequality_pgs_threads", lambda num_worlds: 1):
            qd_1 = _compact_schur_rollout(self.metal)
        qd_256 = _compact_schur_rollout(self.metal)
        qd_cpu = _compact_schur_rollout("cpu")
        kernel = dvi_sparse._solve_dvi_sparse_inequalities_pgs
        self.assertEqual(getattr(kernel._metal_artifact, "coop_block_n", 0), 256)
        self.assertGreater(np.abs(qd_cpu).max(), 1e-3)
        # Lanes accumulate shared updates with atomics: equal within tolerance, not bitwise.
        np.testing.assert_allclose(qd_256, qd_1, rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(qd_256, qd_cpu, rtol=0.0, atol=2e-6)

    def test_sparse_contact_pgs_wide_block_matches_one_lane(self):
        with mock.patch.object(dvi_sparse, "_metal_sparse_contact_pgs_threads", lambda: 1):
            imp_1, qd_1 = _sparse_kapla_rollout(self.metal)
        imp_64, qd_64 = _sparse_kapla_rollout(self.metal)
        imp_cpu, qd_cpu = _sparse_kapla_rollout("cpu")
        kernel = dvi_sparse._solve_dvi_sparse_contacts_pgs
        self.assertEqual(getattr(kernel._metal_artifact, "coop_block_n", 0), 64)
        self.assertGreater(imp_cpu.min(), 0.0)
        # Lanes reorder atomic updates: equal within tolerance, not bitwise.
        np.testing.assert_allclose(imp_64, imp_1, rtol=1e-4)
        np.testing.assert_allclose(imp_64, imp_cpu, rtol=1e-4)
        np.testing.assert_allclose(qd_64, qd_1, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(qd_64, qd_cpu, rtol=0.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
