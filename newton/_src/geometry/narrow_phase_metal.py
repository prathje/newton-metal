# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collision buffer-overflow reporting on Apple Metal devices.

On CUDA and CPU, :func:`~newton._src.geometry.narrow_phase.verify_narrow_phase_buffers`
reports overflowing collision buffers with ``wp.printf``. Metal kernels cannot
print, so that kernel would silently truncate contacts there. On Metal the
narrow phase launches :func:`verify_narrow_phase_buffers_metal` instead, which
records each overflow in a small device array. The host reads that array lazily,
at the start of the next :meth:`NarrowPhase.launch` that is not being captured,
and warns once per overflow kind.

The read does not force a sync. Metal arrays live in unified memory, so the
host reads whatever the GPU has finished writing: an overflow shows up one or
more steps after it happens and never stalls the pipeline. Graph replays do not
call ``launch``, so under :func:`warp.capture_launch` the report comes from the
next eager launch, from :meth:`MetalBufferOverflowMonitor.poll`, or when the
narrow phase is garbage-collected.

This module lives apart from ``narrow_phase.py`` so that the CPU/CUDA kernels of
that module (and their module hash) are unchanged.
"""

from __future__ import annotations

import ctypes
import warnings
import weakref

import numpy as np
import warp as wp

from ..utils.device import is_capturing

# Overflow kinds, in the order of ``verify_narrow_phase_buffers``'s checks.
# Each entry: (key, message template with ``{count}`` / ``{capacity}``).
_KINDS: tuple[tuple[str, str], ...] = (
    ("broad_phase", "Broad phase pair buffer overflowed {count} > {capacity}."),
    ("split_query", "Split query-result buffer overflowed {count} > {capacity}."),
    ("gjk", "GJK candidate pair buffer overflowed {count} > {capacity}."),
    ("split_gjk", "Split GJK work-item buffer overflowed {count} > {capacity}."),
    ("split_manifold", "Split manifold work-item buffer overflowed {count} > {capacity}."),
    ("mesh", "Mesh-convex shape pair buffer overflowed {count} > {capacity}."),
    ("triangle", "Triangle pair buffer overflowed {count} > {capacity}."),
    ("mesh_plane", "Mesh-plane shape pair buffer overflowed {count} > {capacity}."),
    ("mesh_mesh", "Mesh-mesh shape pair buffer overflowed {count} > {capacity}."),
    ("sdf_sdf", "SDF-SDF shape pair buffer overflowed {count} > {capacity}."),
    ("contact", "Contact buffer overflowed {count} > {capacity}."),
    (
        "reduction_ht_fill",
        "Contact reduction hashtable fill ratio exceeded {percent}% ({count} / {capacity}). "
        "Increase contact_reduction_hashtable_size_factor or max_triangle_pairs.",
    ),
    (
        "reduction_ht_insert",
        "Contact reduction hashtable insert failures {count}. "
        "Increase contact_reduction_hashtable_size_factor or max_triangle_pairs.",
    ),
    (
        "reduction_buffer",
        "Contact reduction buffer overflowed; {count} contact candidates were dropped. Increase max_triangle_pairs.",
    ),
)
NUM_OVERFLOW_KINDS = len(_KINDS)


@wp.func
def _record(flags: wp.array2d[int], kind: int, count: int, capacity: int):
    # Sticky maximum: the host may read only every few steps.
    if count > flags[kind, 0]:
        flags[kind, 0] = count
        flags[kind, 1] = capacity


@wp.kernel(enable_backward=False)
def verify_narrow_phase_buffers_metal(
    broad_phase_count: wp.array[int],
    max_broad_phase: int,
    split_query_count: wp.array[int],
    max_split_query: int,
    gjk_count: wp.array[int],
    max_gjk: int,
    split_gjk_count: wp.array[int],
    max_split_gjk: int,
    split_manifold_count: wp.array[int],
    max_split_manifold: int,
    mesh_count: wp.array[int],
    max_mesh: int,
    triangle_count: wp.array[int],
    max_triangle: int,
    mesh_plane_count: wp.array[int],
    max_mesh_plane: int,
    mesh_mesh_count: wp.array[int],
    max_mesh_mesh: int,
    sdf_sdf_count: wp.array[int],
    max_sdf_sdf: int,
    contact_count: wp.array[int],
    max_contacts: int,
    reduction_ht_active_slots: wp.array[int],
    reduction_ht_capacity: int,
    reduction_ht_insert_failures: wp.array[int],
    reduction_buffer_overflows: wp.array[int],
    reduction_ht_warn_load_percent: int,
    flags: wp.array2d[int],
):
    """Same checks as ``verify_narrow_phase_buffers``, recorded in ``flags`` (kind, [count, capacity])."""
    if broad_phase_count[0] > max_broad_phase:
        _record(flags, 0, broad_phase_count[0], max_broad_phase)
    if max_split_query >= 0 and split_query_count[0] > max_split_query:
        _record(flags, 1, split_query_count[0], max_split_query)
    if gjk_count[0] > max_gjk:
        _record(flags, 2, gjk_count[0], max_gjk)
    if max_split_gjk >= 0:
        split_count = split_gjk_count[0]
        if split_gjk_count.shape[0] > 1:
            split_count += split_gjk_count[1]
        if split_count > max_split_gjk:
            _record(flags, 3, split_count, max_split_gjk)
    if max_split_manifold >= 0 and split_manifold_count[0] > max_split_manifold:
        _record(flags, 4, split_manifold_count[0], max_split_manifold)
    if mesh_count:
        if mesh_count[0] > max_mesh:
            _record(flags, 5, mesh_count[0], max_mesh)
    if triangle_count:
        if triangle_count[0] > max_triangle:
            _record(flags, 6, triangle_count[0], max_triangle)
    if mesh_plane_count:
        if mesh_plane_count[0] > max_mesh_plane:
            _record(flags, 7, mesh_plane_count[0], max_mesh_plane)
    if mesh_mesh_count:
        if mesh_mesh_count[0] > max_mesh_mesh:
            _record(flags, 8, mesh_mesh_count[0], max_mesh_mesh)
    if sdf_sdf_count:
        if sdf_sdf_count[0] > max_sdf_sdf:
            _record(flags, 9, sdf_sdf_count[0], max_sdf_sdf)
    if contact_count[0] > max_contacts:
        _record(flags, 10, contact_count[0], max_contacts)
    if reduction_ht_capacity > 0:
        reduction_ht_active_count = reduction_ht_active_slots[reduction_ht_capacity]
        if wp.int64(reduction_ht_active_count) * wp.int64(100) >= wp.int64(reduction_ht_capacity) * wp.int64(
            reduction_ht_warn_load_percent
        ):
            _record(flags, 11, reduction_ht_active_count, reduction_ht_capacity)
        if reduction_ht_insert_failures[0] > 0:
            _record(flags, 12, reduction_ht_insert_failures[0], 0)
        if reduction_buffer_overflows[0] > 0:
            _record(flags, 13, reduction_buffer_overflows[0], 0)


def _peek(flags: wp.array) -> np.ndarray:
    """Host copy of ``flags`` without waiting for the GPU.

    Warp's Metal arrays are shared-storage buffers in unified memory, so the
    host address aliases what the GPU writes. Work still in flight may not be
    visible yet; the flags are sticky, so a later read catches up.
    """
    if not getattr(wp.config, "metal_native_dispatch", True):
        # The MLX path gives no such guarantee; read with a sync instead.
        return flags.numpy()
    n = flags.size
    raw = (ctypes.c_int32 * n).from_address(flags.ptr)
    return np.frombuffer(raw, dtype=np.int32, count=n).reshape(flags.shape).copy()


def _report_from_finalizer(flags: wp.array, warned: set[str], warn_load_percent: int) -> None:
    """Finalizer variant of :func:`_report` that never synchronizes.

    A garbage collection can run the finalizer anywhere, including inside a
    graph capture or a Warp-internal locked section, where a sync could fail
    or deadlock. The MLX path can only read the flags with a sync, so it
    skips the report there.
    """
    if not getattr(wp.config, "metal_native_dispatch", True):
        return
    _report(flags, warned, warn_load_percent)


def _report(flags: wp.array, warned: set[str], warn_load_percent: int) -> None:
    values = _peek(flags)
    for kind, (key, template) in enumerate(_KINDS):
        count, capacity = int(values[kind, 0]), int(values[kind, 1])
        if count <= 0 or key in warned:
            continue
        warned.add(key)
        warnings.warn(
            template.format(count=count, capacity=capacity, percent=warn_load_percent)
            + " (reported on a Metal device, where the collision pipeline cannot print; "
            "contacts past the buffer capacity were dropped.)",
            RuntimeWarning,
            stacklevel=3,
        )


class MetalBufferOverflowMonitor:
    """Device overflow flags for one :class:`NarrowPhase` on a Metal device."""

    def __init__(self, device, warn_load_percent: int):
        self.device = wp.get_device(device)
        self.warn_load_percent = warn_load_percent
        self.flags = wp.zeros((NUM_OVERFLOW_KINDS, 2), dtype=int, device=self.device)
        self.warned: set[str] = set()
        # Overflows recorded during graph replays (no eager launch follows) are
        # reported when the narrow phase goes away.
        self._finalizer = weakref.finalize(self, _report_from_finalizer, self.flags, self.warned, warn_load_percent)

    def poll(self, synchronize: bool = False) -> None:
        """Warn for overflow kinds recorded so far (once per kind).

        Does nothing while the device is capturing a graph. Without
        ``synchronize`` the read does not wait for in-flight GPU work.
        """
        if is_capturing(self.device):
            return
        if synchronize:
            wp.synchronize_device(self.device)
        _report(self.flags, self.warned, self.warn_load_percent)

    def launch(self, inputs: list) -> None:
        """Report earlier overflows, then launch this step's check."""
        self.poll()
        wp.launch(
            verify_narrow_phase_buffers_metal,
            dim=[1],
            inputs=[*inputs, self.flags],
            device=self.device,
            record_tape=False,
        )
