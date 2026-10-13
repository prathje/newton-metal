Add experimental support for Apple Metal devices from Warp's opt-in Metal backend (`WARP_ENABLE_METAL=1`):

  - Treat Metal as a GPU for graph capture and lane-parallel launches, and as a serial-tile device for tiled kernels.
  - Skip CUDA stream overlap on Metal, and raise a clear error when conditional-graph code is captured or texture SDFs are built there.
  - Disable conditional graph nodes for Kamino and MuJoCo Warp on Metal so their solver steps stay capturable.
  - Add `newton.examples.supports_graph_capture()` and accept `--device metal` in the examples.
  - Include Metal devices in the test device lists when the backend is enabled.

CUDA and CPU behavior is unchanged.
