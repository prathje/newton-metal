Add experimental support for Apple Metal devices from Warp's opt-in Metal backend (`WARP_ENABLE_METAL=1`):

  - Treat Metal as a GPU for graph capture and lane-parallel launches, and as a serial-tile device for tiled kernels.
  - Skip CUDA stream overlap on Metal.
  - On Metal builds of Warp that record `wp.capture_while`/`wp.capture_if` inside captures, capture Kamino, MuJoCo Warp, implicit MPM and the collision substep scheduler with their conditional loops, sized from each solver's iteration limits; allow allocation during capture; and build texture SDFs. Older Metal builds fall back to unrolled loops and raise a clear error for conditional captures and texture SDFs.
  - Add `newton.examples.supports_graph_capture()` and accept `--device metal` in the examples.
  - Include Metal devices in the test device lists when the backend is enabled.

CUDA and CPU behavior is unchanged.
