# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mesh soft-contact capacity overflow report on Metal (``soft_contacts_mesh_metal``).

CPU and CUDA print the overflow line from ``_end_mesh_contacts`` with
``wp.printf`` during the pass. Metal cannot print from kernels; the line is
printed at the next host sync point instead, for eager launches, graph
replays, a pass right before interpreter exit, and a collected pipeline.
"""

import gc
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import warp as wp

import newton
from newton.tests.unittest_utils import StdOutCapture, add_function_test, get_test_devices

_LINE = "Mesh soft-contact capacity exceeded"
_POINTS = [(0.02, 0.02, 0.055), (-0.02, -0.02, 0.055)]


def _pipeline(device, capacity):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    mesh = newton.Mesh.create_box(0.1, 0.1, 0.05, compute_inertia=False)
    builder.add_shape_mesh(-1, mesh=mesh, cfg=builder.ShapeConfig(margin=0.0, gap=0.0))
    for point in _POINTS:
        builder.add_particle(wp.vec3(*point), wp.vec3(0.0), mass=1.0, radius=0.01)
    model = builder.finalize(device=device)
    state = model.state()
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        enable_rigid_soft_full_surface_contact=True,
        soft_contact_gap=0.02,
        soft_contact_max=capacity,
    )
    return pipeline, state, pipeline.contacts()


def _required(contacts):
    return int(contacts._soft_contact_mesh_data.contact_count.numpy()[0])


def _callbacks():
    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    return get_dispatcher()._sync_callbacks


def _captured(fn):
    capture = StdOutCapture()
    capture.begin()
    try:
        fn()
    finally:
        output = capture.end()
    return output


def test_overflow_reported_at_next_readback(test, device):
    pipeline, state, contacts = _pipeline(device, capacity=1)
    out = _captured(lambda: (pipeline.collide(state, contacts), wp.synchronize_device(device)))
    required = _required(contacts)
    test.assertGreater(required, 2)
    test.assertEqual(out.count(_LINE), 1, out)
    test.assertIn(f"soft_contact_max=1, required={required}", out)

    # Each overflowing pass followed by a host read reports again (CPU prints every pass),
    # through a plain readback of the contacts.
    def two_passes():
        for _ in range(2):
            pipeline.collide(state, contacts)
            contacts.soft_contact_count.numpy()

    test.assertEqual(_captured(two_passes).count(_LINE), 2)
    # Nothing new: a sync prints nothing.
    test.assertEqual(_captured(lambda: wp.synchronize_device(device)).count(_LINE), 0)

    # Large enough: never reported.
    big, big_state, big_contacts = _pipeline(device, capacity=required)
    out = _captured(lambda: (big.collide(big_state, big_contacts), wp.synchronize_device(device)))
    test.assertNotIn(_LINE, out)
    test.assertEqual(int(big_contacts.soft_contact_count.numpy()[0]), required)


def test_overflow_reported_after_graph_replay(test, device):
    pipeline, state, contacts = _pipeline(device, capacity=1)
    pipeline.collide(state, contacts)  # warm-up: compile outside the capture
    _captured(lambda: wp.synchronize_device(device))

    graph = None

    def record():
        nonlocal graph
        with wp.ScopedCapture(device=device) as capture:
            pipeline.collide(state, contacts)
        graph = capture.graph

    test.assertEqual(_captured(record).count(_LINE), 0)

    # Replays report at the next host read, once per sync window.
    def replays():
        for _ in range(3):
            wp.capture_launch(graph)
        contacts.soft_contact_count.numpy()

    out = _captured(replays)
    test.assertEqual(out.count(_LINE), 1, out)
    test.assertIn(f"required={_required(contacts)}", out)
    test.assertEqual(_captured(lambda: (wp.capture_launch(graph), wp.synchronize())).count(_LINE), 1)


def test_collected_pipeline_reports_once_then_unregisters(test, device):
    before = len(_callbacks())
    pipeline, state, contacts = _pipeline(device, capacity=1)
    test.assertEqual(len(_callbacks()), before + 1)

    def overflow_and_drop():
        nonlocal pipeline, state, contacts
        pipeline.collide(state, contacts)
        del pipeline, state, contacts
        gc.collect()  # no output from the finalizer itself

    test.assertNotIn(_LINE, _captured(overflow_and_drop))
    out = _captured(lambda: wp.synchronize_device(device))
    test.assertEqual(out.count(_LINE), 1, out)
    test.assertEqual(len(_callbacks()), before)
    test.assertNotIn(_LINE, _captured(lambda: wp.synchronize_device(device)))


def test_overflow_in_last_pass_reported_at_exit(test, device):
    # No sync after the last pass: the dispatcher's atexit drain reports it.
    script = textwrap.dedent(
        f"""
        import warp as wp
        wp.config.enable_metal = True
        from newton.tests.test_soft_contacts_mesh_metal import _pipeline
        pipeline, state, contacts = _pipeline("{device}", capacity=1)
        pipeline.collide(state, contacts)
        wp.synchronize_device("{device}")
        print("WARM", flush=True)
        # A fresh pipeline whose only pass overflows, then exit without a sync.
        pipeline, state, contacts = _pipeline("{device}", capacity=1)
        pipeline.collide(state, contacts)
        """
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="newton_mesh_overflow_") as f:
        f.write(script)
    try:
        env = dict(os.environ, WARP_ENABLE_METAL="1")
        r = subprocess.run(
            [sys.executable, f.name], capture_output=True, text=True, timeout=600, check=False, env=env, cwd=os.getcwd()
        )
    finally:
        os.unlink(f.name)
    test.assertEqual(r.returncode, 0, r.stdout + r.stderr)
    after_warm = r.stdout.split("WARM", 1)[-1]
    test.assertEqual(after_warm.count(_LINE), 1, r.stdout + r.stderr)


class TestSoftContactsMeshMetal(unittest.TestCase):
    pass


_metal_devices = [d for d in get_test_devices() if getattr(wp.get_device(d), "is_metal", False)]
for _fn in (
    test_overflow_reported_at_next_readback,
    test_overflow_reported_after_graph_replay,
    test_collected_pipeline_reports_once_then_unregisters,
    test_overflow_in_last_pass_reported_at_exit,
):
    add_function_test(TestSoftContactsMeshMetal, _fn.__name__, _fn, devices=_metal_devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
