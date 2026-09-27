"""兼容 shim：实现已迁移至 `phdnet.backends.torch_backend`（2026-09-27 目录化）。"""

from .backends.torch_backend import *          # noqa: F401,F403
from .backends.torch_backend import (TorchReadout, TorchSTDPCore,   # noqa: F401
                                     bench_readout, probe_devices,
                                     selftest_torch)
