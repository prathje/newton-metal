# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mesh soft-contact capacity overflow reporting on Apple Metal devices.

On CPU and CUDA, ``_end_mesh_contacts`` in :mod:`soft_contacts_mesh` prints a
``wp.printf`` message when the full-surface mesh contacts exceed
``soft_contact_max``. Metal kernels cannot print, so on Metal the collision
pass additionally launches :func:`_record_mesh_contact_overflow`, which keeps
the largest required capacity in a small device array. The host reads it
without forcing a sync (Metal arrays live in unified memory) at the start of
the next mesh contact pass, from :meth:`MeshContactOverflowMonitor.poll`, or
when the monitor is garbage-collected, and prints the CPU message line to
stdout, once per new largest requirement. Unlike the printf, the report thus
arrives one pass (or one poll) late.

The truncated contact count itself (``soft_contact_max + 1``) is written on
every device, so the overflow is never silent in the data.

This module lives apart from ``soft_contacts_mesh.py`` so that the CPU/CUDA
kernels of that module (and their module hash) are unchanged.
"""

from __future__ import annotations

import ctypes
import weakref

import numpy as np
import warp as wp

from ..utils.device import is_capturing


@wp.kernel(enable_backward=False)
def _record_mesh_contact_overflow(source: wp.array[wp.int64], capacity: int, flags: wp.array[wp.int64]):
    # flags = [largest required count seen, capacity at that time]
    if source[0] > wp.int64(capacity) and source[0] > flags[0]:
        flags[0] = source[0]
        flags[1] = wp.int64(capacity)


def _peek(flags: wp.array) -> np.ndarray:
    """Host copy of ``flags`` without waiting for in-flight GPU work."""
    if not getattr(wp.config, "metal_native_dispatch", True):
        return flags.numpy()
    raw = (ctypes.c_int64 * flags.size).from_address(flags.ptr)
    return np.frombuffer(raw, dtype=np.int64, count=flags.size).copy()


def _report(flags: wp.array, reported: list[int]) -> None:
    required, capacity = (int(x) for x in _peek(flags))
    if required <= reported[0]:
        return
    reported[0] = required
    # The exact line ``_end_mesh_contacts`` prints with wp.printf on CPU / CUDA, on stdout.
    print(  # noqa: T201 (stands in for the kernel printf)
        f"Mesh soft-contact capacity exceeded: soft_contact_max={capacity}, required={required}. "
        f"Increase soft_contact_max to at least {required} for this collision pass; "
        "this collision result is incomplete.",
        flush=True,
    )


def _report_from_finalizer(flags: wp.array, reported: list[int]) -> None:
    # A finalizer may run anywhere (inside a capture or a locked section): never sync.
    if getattr(wp.config, "metal_native_dispatch", True):
        _report(flags, reported)


class MeshContactOverflowMonitor:
    """Device overflow record for one :class:`MeshContactData` on a Metal device."""

    def __init__(self, device):
        self.device = wp.get_device(device)
        self.flags = wp.zeros(2, dtype=wp.int64, device=self.device)
        self._reported = [0]
        self._finalizer = weakref.finalize(self, _report_from_finalizer, self.flags, self._reported)

    def poll(self, synchronize: bool = False) -> None:
        """Print the overflow message if a larger overflow than the last reported one was recorded.

        Does nothing while the device is capturing a graph. Without ``synchronize``
        the read does not wait for in-flight GPU work.
        """
        if is_capturing(self.device):
            return
        if synchronize:
            wp.synchronize_device(self.device)
        _report(self.flags, self._reported)

    def launch(self, contact_count: wp.array, capacity: int) -> None:
        """Report earlier overflows, then record this pass's required count."""
        self.poll()
        wp.launch(
            _record_mesh_contact_overflow,
            dim=1,
            inputs=[contact_count, capacity, self.flags],
            device=self.device,
            record_tape=False,
        )
