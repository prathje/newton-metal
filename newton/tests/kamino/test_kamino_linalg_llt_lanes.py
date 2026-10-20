# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the lane-parallel LLT kernels (linalg/factorize/llt_lanes.py).

``LLTBlockedSolver`` uses these kernels on Metal devices. They are valid at any block size,
so every test device runs them (one lane on CPU) and compares against NumPy.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.factorize import llt_lanes
from newton.tests.unittest_utils import add_function_test, get_test_devices

# Matrix sizes per world: empty, smaller than one SIMD group, not a multiple of 32,
# and the size range of a legged robot's Delassus matrix in contact.
_DIMS = [0, 5, 33, 70, 140]


def _make_problem(seed: int):
    rng = np.random.default_rng(seed)
    mats, rhs, mio, vio = [], [], [], []
    mo = vo = 0
    for n in _DIMS:
        M = rng.standard_normal((n, n))
        A = M @ M.T / max(n, 1) + np.eye(n)
        mats.append(A.astype(np.float32))
        rhs.append(rng.standard_normal(n).astype(np.float32))
        mio.append(mo)
        vio.append(vo)
        mo += n * n
        vo += n
    return mats, rhs, np.array(mio, np.int32), np.array(vio, np.int32)


def _check_world_results(test, mats, rhs, mio, vio, L_np, x_np):
    for k, n in enumerate(_DIMS):
        if n == 0:
            continue
        A = mats[k].astype(np.float64)
        L = L_np[mio[k] : mio[k] + n * n].reshape(n, n)
        np.testing.assert_allclose(L, np.linalg.cholesky(A), rtol=1e-4, atol=1e-5)
        test.assertTrue(np.all(np.triu(L, 1) == 0.0), "strict upper triangle must be zero")
        x_ref = np.linalg.solve(A, rhs[k].astype(np.float64))
        np.testing.assert_allclose(x_np[vio[k] : vio[k] + n], x_ref, rtol=1e-4, atol=1e-5)


def test_llt_lanes_factorize_and_solve(test, device):
    mats, rhs, mio, vio = _make_problem(7)
    with wp.ScopedDevice(device):
        dim = wp.array(np.array(_DIMS, np.int32), dtype=wp.int32)
        mio_wp = wp.array(mio, dtype=wp.int32)
        vio_wp = wp.array(vio, dtype=wp.int32)
        A = wp.array(np.concatenate([m.ravel() for m in mats]), dtype=wp.float32)
        L = wp.full(A.shape, 123.0, dtype=wp.float32)  # garbage the kernel must overwrite
        b = wp.array(np.concatenate(rhs), dtype=wp.float32)
        y = wp.zeros_like(b)
        x = wp.zeros_like(b)
        llt_lanes.llt_lanes_factorize(len(_DIMS), dim, mio_wp, A, L)
        llt_lanes.llt_lanes_solve(len(_DIMS), dim, mio_wp, vio_wp, L, b, y, x)
        _check_world_results(test, mats, rhs, mio, vio, L.numpy(), x.numpy())
        np.testing.assert_array_equal(b.numpy(), np.concatenate(rhs))  # b is input-only

        # In-place: x holds b on entry.
        x_in = wp.array(np.concatenate(rhs), dtype=wp.float32)
        llt_lanes.llt_lanes_solve_inplace(len(_DIMS), dim, mio_wp, vio_wp, L, y, x_in)
        np.testing.assert_array_equal(x_in.numpy(), x.numpy())


def test_llt_lanes_in_graph(test, device):
    """Repeated graph replays give the eager result (the PADMM loop replays the solve)."""
    if not device.is_cuda and not getattr(device, "is_metal", False):
        test.skipTest("graph capture needs a GPU device")
    mats, rhs, mio, vio = _make_problem(11)
    with wp.ScopedDevice(device):
        dim = wp.array(np.array(_DIMS, np.int32), dtype=wp.int32)
        mio_wp = wp.array(mio, dtype=wp.int32)
        vio_wp = wp.array(vio, dtype=wp.int32)
        A = wp.array(np.concatenate([m.ravel() for m in mats]), dtype=wp.float32)
        L = wp.zeros_like(A)
        b = wp.array(np.concatenate(rhs), dtype=wp.float32)
        y = wp.zeros_like(b)
        x = wp.zeros_like(b)
        with wp.ScopedCapture() as capture:
            llt_lanes.llt_lanes_factorize(len(_DIMS), dim, mio_wp, A, L)
            llt_lanes.llt_lanes_solve(len(_DIMS), dim, mio_wp, vio_wp, L, b, y, x)
        for _ in range(3):
            x.zero_()
            wp.capture_launch(capture.graph)
            _check_world_results(test, mats, rhs, mio, vio, L.numpy(), x.numpy())


class TestLinAlgLLTLanes(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestLinAlgLLTLanes, "test_llt_lanes_factorize_and_solve", test_llt_lanes_factorize_and_solve, devices=devices
)
add_function_test(TestLinAlgLLTLanes, "test_llt_lanes_in_graph", test_llt_lanes_in_graph, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
