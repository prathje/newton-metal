# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

###
# Module interface
###

__all__ = ["expect_contact_capacity_warning", "setup_tests", "skip_lox_on_metal", "test_context"]

_CONTACT_CAPACITY_WARNING_RE = r"^Warning: Kamino per-world contact capacity exceeded\..*\n?"


def expect_contact_capacity_warning(test):
    """Require the device-side contact capacity warning in ``test``'s stdout."""
    # Warp CPU printf capture is unreliable on Windows. Warp's Metal backend compiles
    # wp.printf out (it says so once per process), so Metal cannot print the warning.
    device = getattr(test, "default_device", None) or wp.get_device()
    if sys.platform == "win32" or getattr(wp.get_device(device), "is_metal", False):
        test.allowOutputRegex(_CONTACT_CAPACITY_WARNING_RE, stream="stdout")
    else:
        test.expectOutputRegex(_CONTACT_CAPACITY_WARNING_RE, stream="stdout")


def skip_lox_on_metal(test, device):
    """Skip ``test`` on a Metal ``device``, where the LOX dynamics solver is not supported yet."""
    if getattr(wp.get_device(device), "is_metal", False):
        from newton._src.solvers.kamino._src.solvers.lox.solver import LOX_METAL_UNSUPPORTED_MESSAGE  # noqa: PLC0415

        test.skipTest(LOX_METAL_UNSUPPORTED_MESSAGE)


###
# Global test context
###


@dataclass
class TestContext:
    setup_done: bool = False
    """Whether the global test setup has already run """

    verbose: bool = False
    """Global default verbosity flag to be used by unit tests """

    device: wp.DeviceLike | None = None
    """Global default device to be used by unit tests """

    output_path: Path | None = None
    """Global cache directory for tests to use, if any."""


test_context = TestContext()
""" Global object shared across unit tests, containing status & settings regarding test execution. """


###
# Functions
###


def setup_tests(verbose: bool = False, device: wp.DeviceLike | str | None = None, clear_cache: bool = False):
    # Numpy configuration
    np.set_printoptions(
        linewidth=999999, edgeitems=999999, threshold=999999, precision=10, suppress=True
    )  # Suppress scientific notation

    # Clear cache
    if clear_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()

    # Update test context
    test_context.verbose = verbose
    test_context.device = wp.get_device(device)
    test_context.setup_done = True

    # Set the cache directory for optional test output, if any
    # Data directory (contains perfprof.csv)
    test_context.output_path = Path(__file__).parent / "output"
    test_context.output_path.mkdir(parents=True, exist_ok=True)
