"""Load a model exactly as oMLX serves it: ``--loader omlx`` (engines/mlx.py load_model), for models only oMLX can load.

Constructs oMLX's own VLMBatchedEngine and borrows the loaded module, so the
copy sees the same patches, fused kernels and PLE residency as the server.
Runs under oMLX's bundled Python (a job's GPU run, or scripts/omlx-python).
"""

from __future__ import annotations

import asyncio

_LOOP = None  # keep the engine's loop alive for the life of the process


def load_omlx_model(path: str):
    global _LOOP
    from omlx.engine.vlm import VLMBatchedEngine

    engine = VLMBatchedEngine(path)  # no model_settings: MTP off, default PLE mode
    _LOOP = asyncio.new_event_loop()
    _LOOP.run_until_complete(engine.start())
    # The adapter is what oMLX's scheduler calls: it builds the per-layer
    # cache types and M-RoPE position ids. Hooks find the module tree through it.
    return engine._adapter, engine.tokenizer
