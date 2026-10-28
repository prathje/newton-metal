# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mesh soft-contact capacity overflow reporting on Apple Metal devices.

On CPU and CUDA, ``_end_mesh_contacts`` in :mod:`soft_contacts_mesh` prints a
``wp.printf`` message when the full-surface mesh contacts exceed
``soft_contact_max``. Metal kernels cannot print, so on Metal the collision
pass additionally launches :func:`_record_mesh_contact_overflow`, which stores
the latest overflowing pass's requirement and counts overflowing passes in a
small device array.

The host prints the CPU message line on stdout at the next host sync point:
Warp's Metal dispatcher calls back after every completed sync (``.numpy()``,
``wp.synchronize*``, other host reads), after eager launches and graph replays
alike, and once more when the interpreter exits, so an overflow in the last
pass of a script is reported too. At that point every earlier launch has
finished, so the read is exact. One line is printed per sync window that saw
overflowing passes (CPU prints one per pass), with the requirement of the
latest of them. A monitor that is garbage-collected stays registered until
the next sync has reported what it recorded; nothing is printed from a
finalizer.

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
    # flags = [required count of the latest overflowing pass, its capacity, number of overflowing passes]
    if source[0] > wp.int64(capacity):
        flags[0] = source[0]
        flags[1] = wp.int64(capacity)
        flags[2] = flags[2] + wp.int64(1)


def _peek(flags: wp.array) -> np.ndarray:
    """Host copy of ``flags`` without issuing a sync (the caller has just synced)."""
    if not getattr(wp.config, "metal_native_dispatch", True):
        return flags.numpy()
    raw = (ctypes.c_int64 * flags.size).from_address(flags.ptr)
    return np.frombuffer(raw, dtype=np.int64, count=flags.size).copy()


def _print_overflow(required: int, capacity: int) -> None:
    # The exact line ``_end_mesh_contacts`` prints with wp.printf on CPU / CUDA, on stdout.
    print(  # noqa: T201 (stands in for the kernel printf)
        f"Mesh soft-contact capacity exceeded: soft_contact_max={capacity}, required={required}. "
        f"Increase soft_contact_max to at least {required} for this collision pass; "
        "this collision result is incomplete.",
        flush=True,
    )


class _SyncReport:
    """Dispatcher sync callback: print once if passes overflowed since the last report."""

    def __init__(self, flags: wp.array):
        self.flags = flags
        self.passes_reported = 0
        self.retired = False

    def __call__(self) -> bool:
        required, capacity, passes = (int(x) for x in _peek(self.flags))
        if passes > self.passes_reported:
            self.passes_reported = passes
            _print_overflow(required, capacity)
        # A retired (collected) monitor has now reported everything it recorded.
        return not self.retired


def _metal_dispatcher():
    try:
        from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

        disp = get_dispatcher()
    except Exception:
        return None
    return disp if hasattr(disp, "add_sync_callback") else None


def _retire(report: _SyncReport) -> None:
    # Finalizers may run anywhere (inside a capture, a locked section): only flag it.
    report.retired = True


class MeshContactOverflowMonitor:
    """Device overflow record for one :class:`MeshContactData` on a Metal device."""

    def __init__(self, device):
        self.device = wp.get_device(device)
        self.flags = wp.zeros(3, dtype=wp.int64, device=self.device)
        self._report = _SyncReport(self.flags)
        disp = _metal_dispatcher()
        if disp is None:
            raise RuntimeError(
                "Mesh soft-contact overflow reporting on Metal needs a Warp Metal dispatcher with "
                "add_sync_callback(); this Warp build does not provide it."
            )
        disp.add_sync_callback(self._report)
        weakref.finalize(self, _retire, self._report)

    def poll(self) -> None:
        """Synchronize the device, which prints the overflow message if passes overflowed
        since the last report. Does nothing while capturing a graph."""
        if not is_capturing(self.device):
            wp.synchronize_device(self.device)

    def launch(self, contact_count: wp.array, capacity: int) -> None:
        """Record this pass's required count if it exceeds ``capacity``."""
        wp.launch(
            _record_mesh_contact_overflow,
            dim=1,
            inputs=[contact_count, capacity, self.flags],
            device=self.device,
            record_tape=False,
        )
