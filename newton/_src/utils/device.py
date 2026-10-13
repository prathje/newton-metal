# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device capability queries shared by Newton's internal code paths.

Newton historically branched on ``device.is_cuda`` / ``device.is_cpu``. With
Warp's experimental Apple Metal backend a third device kind exists that is a
GPU (graph capture, massively parallel launches) but lacks several CUDA-only
features (streams, events, conditional graph nodes, memory pools) and runs
most tile kernels in a serial per-tile model like the CPU backend. These
helpers name the capability being tested so each call site states *why* it
branches, instead of guessing from the device kind.

All helpers accept a :class:`warp.Device` or any device identifier accepted by
:func:`warp.get_device` and return ``False`` for Metal-specific queries on
Warp builds without a Metal backend.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "active_capture_key",
    "check_conditional_capture",
    "is_capturing",
    "is_gpu",
    "is_metal",
    "supports_conditional_graphs",
    "supports_cooperative_blocks",
    "supports_graph_capture",
    "supports_streams",
]


def is_metal(device: wp.DeviceLike = None) -> bool:
    """Return whether ``device`` is an Apple Metal device.

    Warp builds without a Metal backend have no ``Device.is_metal`` attribute;
    those always report ``False``.
    """
    return bool(getattr(wp.get_device(device), "is_metal", False))


def is_gpu(device: wp.DeviceLike = None) -> bool:
    """Return whether ``device`` is a GPU (CUDA or Metal)."""
    device = wp.get_device(device)
    return device.is_cuda or is_metal(device)


def supports_graph_capture(device: wp.DeviceLike = None) -> bool:
    """Return whether Newton should capture GPU graphs on ``device``.

    True for CUDA and for Metal when Warp uses native dispatch (the only Metal
    path that records launches). Covers :class:`warp.ScopedCapture` and
    :func:`warp.capture_launch` only; conditional nodes are reported by
    :func:`supports_conditional_graphs`. CPU returns ``False`` so call sites
    that previously tested ``device.is_cuda`` keep their CPU behavior.
    """
    device = wp.get_device(device)
    if device.is_cuda:
        return True
    if is_metal(device):
        return bool(getattr(wp.config, "metal_native_dispatch", True))
    return False


def supports_conditional_graphs(device: wp.DeviceLike = None) -> bool:
    """Return whether :func:`warp.capture_while` / :func:`warp.capture_if` work on ``device``.

    Conditional graph nodes are a CUDA 12.4+ feature; Metal and CPU graphs do
    not support them.
    """
    return wp.get_device(device).is_cuda and wp.is_conditional_graph_supported()


def supports_streams(device: wp.DeviceLike = None) -> bool:
    """Return whether ``device`` supports :class:`warp.Stream` and :class:`warp.Event`.

    Only CUDA devices do; Metal raises on stream access.
    """
    return wp.get_device(device).is_cuda


def supports_cooperative_blocks(device: wp.DeviceLike = None) -> bool:
    """Return whether kernels on ``device`` observe the launch ``block_dim``.

    On CUDA, ``wp.block_dim()`` inside a kernel equals the launch block size
    and the lanes of a block cooperate on tiles. The CPU and Metal backends
    run tile kernels serially (``wp.block_dim() == 1``), so launch geometry
    and tile shapes must be sized for one lane per block there.
    """
    return wp.get_device(device).is_cuda


def is_capturing(device: wp.DeviceLike = None) -> bool:
    """Return whether a graph capture is active on ``device``.

    Equivalent to :attr:`warp.Device.is_capturing`, except that it also
    reports an active Metal capture, which the experimental Metal backend
    does not reflect in ``Device.is_capturing``. Use it where code must avoid
    host synchronization while launches are being recorded.
    """
    device = wp.get_device(device)
    if device.is_capturing:
        return True
    if is_metal(device):
        from warp._src import context as wp_context  # noqa: PLC0415

        # Private Warp state; drop once Metal reports Device.is_capturing.
        return getattr(wp_context.runtime, "_metal_capture_graph", None) is not None
    return False


def active_capture_key(device: wp.DeviceLike = None) -> object | None:
    """Return a key identifying the graph capture active on ``device``.

    CUDA returns the stream capture id. Metal returns the graph being
    recorded, which is a distinct object for every capture. Returns ``None``
    when no capture is active or the device has no capture identity.
    """
    device = wp.get_device(device)
    if device.is_cuda:
        if not device.is_capturing:
            return None
        capture = device.captures.get(device.stream)
        return None if capture is None else capture.capture_id
    if is_metal(device):
        from warp._src import context as wp_context  # noqa: PLC0415

        return getattr(wp_context.runtime, "_metal_capture_graph", None)
    return None


def check_conditional_capture(device: wp.DeviceLike, feature: str) -> None:
    """Raise if ``device`` is recording a graph that cannot hold conditional nodes.

    Outside a capture, :func:`warp.capture_while` and :func:`warp.capture_if`
    evaluate their condition on the host, which works on every device. During
    a Metal capture they would evaluate the condition once at record time and
    bake that branch into the graph, so code that relies on them must not be
    captured on Metal.

    Args:
        device: The device that launches the conditional work.
        feature: Name of the caller, used in the error message.

    Raises:
        RuntimeError: If a Metal graph capture is active on ``device``.
    """
    if is_metal(device) and is_capturing(device):
        raise RuntimeError(
            f"{feature} uses wp.capture_while/wp.capture_if, which Metal graph capture does not support. "
            "Run it outside graph capture on Metal devices."
        )
