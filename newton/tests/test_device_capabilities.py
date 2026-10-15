# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the internal device capability helpers in ``newton._src.utils.device``."""

import contextlib
import gc
import unittest

import numpy as np
import warp as wp

from newton._src.utils import is_graph_capture_allocation_enabled
from newton._src.utils.device import (
    active_capture_key,
    capture_while_budget,
    check_conditional_capture,
    is_capturing,
    is_metal,
    supports_conditional_graphs,
    supports_graph_capture,
)
from newton.tests.unittest_utils import add_function_test, get_metal_test_devices, get_test_devices


@wp.kernel
def _add_one(values: wp.array[float]):
    values[wp.tid()] += 1.0


@wp.kernel
def _copy(src: wp.array[float], dst: wp.array[float]):
    dst[wp.tid()] = src[wp.tid()]


@wp.kernel
def _count_down(counter: wp.array[wp.int32], steps: wp.array[wp.int32]):
    steps[0] = steps[0] + 1
    counter[0] = counter[0] - 1


def test_cpu_capabilities(test, device):
    test.assertFalse(is_metal(device))
    test.assertFalse(supports_graph_capture(device))
    test.assertFalse(supports_conditional_graphs(device))
    test.assertFalse(is_capturing(device))
    test.assertIsNone(active_capture_key(device))
    test.assertIsInstance(capture_while_budget(device, 7), contextlib.nullcontext)
    test.assertTrue(is_graph_capture_allocation_enabled(device))
    # Outside a Metal capture the check never raises.
    check_conditional_capture(device, "test")


def _metal_records_conditionals() -> bool:
    return hasattr(wp, "ScopedMetalCaptureWhileBudget")


def test_metal_capture_state(test, device):
    test.assertTrue(is_metal(device))
    test.assertFalse(is_capturing(device))
    test.assertIsNone(active_capture_key(device))
    with wp.ScopedCapture(device=device) as capture:
        test.assertTrue(is_capturing(device))
        key = active_capture_key(device)
        test.assertIsNotNone(key)
        wp.launch(_count_down, dim=1, inputs=[wp.zeros(1, dtype=wp.int32, device=device)] * 2, device=device)
    test.assertFalse(is_capturing(device))
    test.assertIsNone(active_capture_key(device))
    with wp.ScopedCapture(device=device):
        test.assertIsNot(active_capture_key(device), key)
    del capture


def test_metal_conditional_capture(test, device):
    if not _metal_records_conditionals():
        test.assertFalse(supports_conditional_graphs(device))
        with wp.ScopedCapture(device=device):
            with test.assertRaises(RuntimeError):
                check_conditional_capture(device, "test")
        return

    test.assertEqual(supports_conditional_graphs(device), supports_graph_capture(device))
    counter = wp.array([5], dtype=wp.int32, device=device)
    steps = wp.zeros(1, dtype=wp.int32, device=device)

    def body():
        wp.launch(_count_down, dim=1, inputs=[counter, steps], device=device)

    # The default budget would also cover five iterations; a small explicit
    # budget checks that capture_while_budget reaches Warp.
    with wp.ScopedCapture(device=device) as capture:
        check_conditional_capture(device, "test")
        with capture_while_budget(device, 6):
            wp.capture_while(counter, body)
    for start in (5, 3):
        counter.assign(np.array([start], dtype=np.int32))
        steps.zero_()
        wp.capture_launch(capture.graph)
        test.assertEqual(int(counter.numpy()[0]), 0)
        test.assertEqual(int(steps.numpy()[0]), start)

    # A loop that outruns its budget is reported on replay.
    counter.assign(np.array([9], dtype=np.int32))
    with test.assertRaises(RuntimeError):
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)


def test_metal_capture_allocation(test, device):
    """Arrays allocated during a Metal capture behave like CUDA graph allocations."""
    if not is_graph_capture_allocation_enabled(device):
        return

    n = 1024
    out = wp.zeros(n, dtype=float, device=device)
    with wp.ScopedCapture(device=device) as capture:
        # The zero fill of an array allocated during capture is recorded and
        # runs on every replay, as with a CUDA graph memset node.
        values = wp.zeros(n, dtype=float, device=device)
        wp.launch(_add_one, dim=n, inputs=[values], device=device)
        wp.launch(_copy, dim=n, inputs=[values, out], device=device)
    for _ in range(2):
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(values.numpy(), 1.0)

    # The graph keeps its buffers alive after the Python array is gone, and
    # later allocations do not alias them.
    del values
    gc.collect()
    others = [wp.full(n, -7.0, dtype=float, device=device) for _ in range(8)]
    out.zero_()
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(out.numpy(), 1.0)
    for other in others:
        np.testing.assert_array_equal(other.numpy(), -7.0)


class TestDeviceCapabilities(unittest.TestCase):
    pass


add_function_test(
    TestDeviceCapabilities,
    "test_cpu_capabilities",
    test_cpu_capabilities,
    devices=[d for d in get_test_devices() if d.is_cpu],
)
add_function_test(
    TestDeviceCapabilities, "test_metal_capture_state", test_metal_capture_state, devices=get_metal_test_devices()
)
add_function_test(
    TestDeviceCapabilities,
    "test_metal_conditional_capture",
    test_metal_conditional_capture,
    devices=get_metal_test_devices(),
)
add_function_test(
    TestDeviceCapabilities,
    "test_metal_capture_allocation",
    test_metal_capture_allocation,
    devices=get_metal_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
