# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Metal variant of the batched RCM traversal in :mod:`.rcm_batch`.

Off CUDA, :func:`.rcm_batch.create_rcm_batch_launch` replays one BFS step and
one advance/seed launch per level (``max_dim`` pairs for a complete traversal).
The BFS step claims vertices with ``atomic_cas`` and appends them with
``atomic_add``, so on a GPU the order inside a level, and with it the
permutation, the factor and every downstream result, changes from run to run.
On CPU the threads run in index order, which fixes the order.

The kernel here runs that same sequence of steps with one thread per block, so
the permutation on Metal is the CPU's, run to run, in a single launch.
"""

from __future__ import annotations

from functools import cache

import warp as wp


@cache
def make_traverse_kernel(dtype):
    """One-thread-per-block replay of ``iteration_count`` (BFS step, advance-or-seed) pairs."""

    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def rcm_traverse_in_order_kernel(
        num_blocks: int,
        tol: dtype,  # type: ignore[valid-type]
        A: wp.array[dtype],  # type: ignore[valid-type]
        dims: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        ld: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        degree: wp.array[wp.int32],
        level: wp.array[wp.int32],
        order_buf: wp.array[wp.int32],
        head: wp.array[wp.int32],
        current_level: wp.array[wp.int32],
        discovered: wp.array[wp.int32],
        reorder_active: wp.array[wp.int32],
        iteration_count: int,
    ):
        b = wp.tid()
        if b >= num_blocks or reorder_active[b] == int(0):
            return
        n_b = dims[b]
        vb = vio[b]
        mb = mio[b]
        stride = ld[b]
        cur = current_level[b]
        count = head[b]
        found = discovered[b]
        for _iteration in range(iteration_count):
            # BFS step: vertices of the current level in index order, neighbours in index order.
            next_lvl = cur + int(1)
            for i in range(n_b):
                if level[vb + i] == cur:
                    base = mb + i * stride
                    for j in range(n_b):
                        if j != i:
                            av = dtype(wp.abs(A[base + j]))
                            if av > tol:
                                if level[vb + j] == int(-1):
                                    level[vb + j] = next_lvl
                                    found = int(1)
                                    order_buf[vb + count] = j
                                    count += int(1)
            # Advance to the next level, or seed the next component at a minimum degree.
            if count >= n_b:
                break
            cur = next_lvl
            if found != int(0):
                found = int(0)
                continue
            best_deg = int(2147483647)
            best_idx = int(-1)
            for i in range(n_b):
                if level[vb + i] == int(-1):
                    d = degree[vb + i]
                    if d < best_deg:
                        best_deg = d
                        best_idx = i
            if best_idx >= int(0):
                level[vb + best_idx] = cur
                order_buf[vb + count] = best_idx
                count += int(1)
        head[b] = count
        current_level[b] = cur
        discovered[b] = found

    return rcm_traverse_in_order_kernel
