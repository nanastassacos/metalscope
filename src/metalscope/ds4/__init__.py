"""metalscope for models served by ds4 (antirez/ds4, C and Metal).

ds4 itself records (its metalscope build: a request's "scope" options make it write each forward's routing, stream
norms, what each layer changed in each stream, lenses and top 10, and add steering vectors); this package reads
those records into the same traces and live stream as the oMLX integration:

    frames.py   the capture file ds4 writes, and the files it reads (steering, J-lens)
    vocab.py    the model's vocabulary, shape and output-head rows, from its GGUF
    backend.py  the served model's J-lens; following a request's capture into TraceManager (live.py)
    proxy.py    ds4-server behind a proxy that also serves /scope

Every file a request's scope names is in ``scope_dir()``: ds4-server is started with it as ``--scope-dir`` and opens
no scope file outside it (a page could post a scope to its port and name any path).
"""

from pathlib import Path

from .. import config


def scope_dir() -> Path:
    """The one directory ds4-server opens a request's scope files in (its --scope-dir): the proxy's request files
    (``runs/``), a measurement's (``engine/``) and the J-lens as ds4 reads it, under ~/.metalscope/ds4/scope. Private
    (0700): traces hold the prompts and replies."""
    d = config.home() / "ds4" / "scope"
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    if d.stat().st_mode & 0o777 != 0o700:
        d.chmod(0o700)
    return d
