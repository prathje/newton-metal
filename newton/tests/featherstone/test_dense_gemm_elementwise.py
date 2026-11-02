# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.featherstone.kernels import (
    eval_dense_gemm_batched,
    eval_dense_gemm_batched_elementwise,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_dense_gemm_elementwise_matches_serial(test, device):
    """The per-element gemm (Metal's Featherstone path) equals the per-articulation kernel bitwise."""
    rng = np.random.default_rng(3)
    # (m, n, p) per batch entry, including an empty one.
    dims = [(6, 3, 6), (12, 7, 12), (18, 27, 18), (0, 4, 0), (24, 14, 24)]
    for trans_a, trans_b in ((False, False), (True, False), (False, True), (True, True)):
        m = np.array([d[0] for d in dims], np.int32)
        n = np.array([d[1] for d in dims], np.int32)
        p = np.array([d[2] for d in dims], np.int32)
        a_size, b_size, c_size = m * p, p * n, m * n
        a_start = np.concatenate(([0], np.cumsum(a_size)[:-1])).astype(np.int32)
        b_start = np.concatenate(([0], np.cumsum(b_size)[:-1])).astype(np.int32)
        c_start = np.concatenate(([0], np.cumsum(c_size)[:-1])).astype(np.int32)
        a = rng.standard_normal(int(a_size.sum())).astype(np.float32)
        b = rng.standard_normal(int(b_size.sum())).astype(np.float32)
        a[::7] = 0.0
        inputs = [
            wp.array(m, dtype=int, device=device),
            wp.array(n, dtype=int, device=device),
            wp.array(p, dtype=int, device=device),
            trans_a,
            trans_b,
            wp.array(a_start, dtype=int, device=device),
            wp.array(b_start, dtype=int, device=device),
            wp.array(c_start, dtype=int, device=device),
            wp.array(a, dtype=float, device=device),
            wp.array(b, dtype=float, device=device),
        ]
        c_serial = wp.full(int(c_size.sum()), np.nan, dtype=float, device=device)
        c_elem = wp.full(int(c_size.sum()), np.nan, dtype=float, device=device)
        wp.launch(eval_dense_gemm_batched, dim=len(dims), inputs=inputs, outputs=[c_serial], device=device)
        wp.launch(
            eval_dense_gemm_batched_elementwise,
            dim=(len(dims), int(c_size.max())),
            inputs=inputs,
            outputs=[c_elem],
            device=device,
        )
        expected = c_serial.numpy()
        test.assertFalse(np.isnan(expected).any())
        np.testing.assert_array_equal(c_elem.numpy().view(np.int32), expected.view(np.int32))


def test_featherstone_dense_path_without_grad(test, device):
    """Featherstone's dense (non-tile) path gives the same state with and without gradient tracking."""
    results = []
    for requires_grad in (False, True):
        builder = newton.ModelBuilder()
        builder.default_joint_cfg.armature = 0.1
        builder.add_mjcf(newton.examples.get_asset("nv_ant.xml"), ignore_names=["floor", "ground"])
        multi = newton.ModelBuilder()
        multi.replicate(builder, 2, spacing=(2.0, 2.0, 0.0))
        model = multi.finalize(device=device, requires_grad=requires_grad)
        model.joint_qd.assign(np.random.default_rng(0).uniform(-1.0, 1.0, model.joint_dof_count).astype(np.float32))
        solver = newton.solvers.SolverFeatherstone(model)
        s0, s1, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        for _ in range(3):
            solver.step(s0, s1, control, None, 1.0 / 600.0)
            s0, s1 = s1, s0
        results.append((solver.H.numpy(), s0.joint_qd.numpy()))
    for got, expected in zip(results[0], results[1], strict=True):
        np.testing.assert_array_equal(got.view(np.int32), expected.view(np.int32))


class TestFeatherstoneDenseGemmElementwise(unittest.TestCase):
    pass


add_function_test(
    TestFeatherstoneDenseGemmElementwise,
    "test_dense_gemm_elementwise_matches_serial",
    test_dense_gemm_elementwise_matches_serial,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherstoneDenseGemmElementwise,
    "test_featherstone_dense_path_without_grad",
    test_featherstone_dense_path_without_grad,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
