# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pin the reference PADMM sources that the hand-written Metal kernel mirrors.

``padmm/metal_fused.py`` re-implements the accelerated PADMM iteration (``_step_accel`` and
the kernels and functions it launches: De Saxce correction and velocity bias, the Cholesky
solve, projection, dual update, residuals, convergence, restart / Nesterov acceleration,
penalty update) in MSL. A change to any of these Python sources would silently diverge on
Metal only. This test hashes them; when it fails, re-verify ``metal_fused.py`` against the
changed function(s) (and run the Metal fused-vs-loop tests in ``test_kamino_metal_lanes``),
then update the pinned hash here.

Runs on any device (pure source inspection).
"""

import hashlib
import inspect
import textwrap
import unittest

from newton._src.solvers.kamino._src.linalg import linear
from newton._src.solvers.kamino._src.solvers.padmm import kernels, math, solver

# (module, qualified name) -> sha256 of the dedented source, recorded when metal_fused.py was
# last verified against them (metal-r10-kperf 0b10052a + review a7d26f7b).
PINNED = {
    ("solver", "PADMMSolver.solve"): "ec3121e678a528f4731520e95947e3eaffee425d3e969952bb0c51a7400772b8",
    ("solver", "PADMMSolver._step_accel"): "a1210530c99bf68f2d3e6e44bd7d31325c11e0e17ae9592ea84486f996efd34a",
    (
        "solver",
        "PADMMSolver._update_desaxce_and_velocity_bias",
    ): "2ef7584f75c2540c0e5759911ec50e779e1afa4c5a83f7d859e11e06bbd58e28",
    (
        "solver",
        "PADMMSolver._update_unconstrained_solution",
    ): "6722c6702d6d349936a1cea031926ec381f310e96312ae764d7ed7d7f2a271a8",
    (
        "solver",
        "PADMMSolver._update_projection_dual_convergence_accel",
    ): "5398d7a8650cbb44371bfe433d66aa72c51eb22090f6b0acd76b8363a9f24d3c",
    (
        "kernels",
        "make_desaxce_correction_and_velocity_bias_kernel",
    ): "5b12849c54e104542b0b4e7e1798377a8b96ccbfd9d51fad8beda712c9924893",
    (
        "kernels",
        "_make_project_dual_convergence_accel_kernel",
    ): "61c7436603e47d42cdba26288ac4b8f7f89744f611d37184108600bcb65a6d64",
    ("kernels", "_update_penalty"): "0827c8d8286682a6dd2e4afa2c7001dc2a592f701771a0d629c6408ff0824e0d",
    ("math", "project_to_coulomb_cone"): "10824f534c9173324137dbf9e6955bc9078804851e84e62d5f3bf353570250f5",
    (
        "math",
        "compute_box_complementarity_residual",
    ): "fbad3954c50f97f1e90015c6c5973fce4643a39e52bd46b1b955a1ab48313372",
    ("math", "compute_desaxce_corrections"): "e6d478f4d89a74e5e06178b7165efbb4571bf9a475b4145f32fde0c4fe0ef6ac",
    ("linear", "LLTBlockedSolver._solve_impl"): "bb9ca51b789646d8d6793d9300d75fccbab4ab33d0f0709b5a8ea1d58c19a7be",
    (
        "linear",
        "LLTBlockedSolver._solve_inplace_impl",
    ): "a250e171adc38464e8d23e35f574333003fcaf23ee4b1b80ff654419e8c8dff1",
}

_MODULES = {"solver": solver, "kernels": kernels, "math": math, "linear": linear}


def _source_hash(module_key: str, qualname: str) -> str:
    obj = _MODULES[module_key]
    for part in qualname.split("."):
        obj = getattr(obj, part)
    obj = getattr(obj, "func", obj)  # @wp.func / @wp.kernel -> the Python function
    src = textwrap.dedent(inspect.getsource(obj))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


class TestKaminoMetalFusedPadmmPin(unittest.TestCase):
    def test_reference_sources_unchanged(self):
        changed = []
        for (mod, name), pinned in PINNED.items():
            got = _source_hash(mod, name)
            if got != pinned:
                changed.append(f"  {mod}.{name}: pinned {pinned[:12] or '<none>'}, now {got[:12]} ({got})")
        self.assertFalse(
            changed,
            "The reference PADMM sources mirrored by the hand-written Metal kernel "
            "(newton/_src/solvers/kamino/_src/solvers/padmm/metal_fused.py) changed:\n"
            + "\n".join(changed)
            + "\nRe-verify metal_fused.py against these functions (same iteration arithmetic, restart / "
            "acceleration and penalty rules), run test_kamino_metal_lanes on Metal, then update PINNED in " + __file__,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
