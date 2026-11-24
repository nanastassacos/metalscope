# Design and development

metalscope's design rules, the repository's layout, and working on it from a checkout.

## Design rules

1. **Observe, don't reconstruct.** Hooks sit on the real modules (instance
   class-swap, see `hooks.py`), so captures include RoPE, sinks, fused
   kernels, quantisation and decode-time state exactly as served.
2. **No extra syncs.** Hooks emit lazy reductions; the caller folds them into
   the step's existing `mx.async_eval`. Host transfer happens off the hot path.
3. **Reduce on GPU.** Norms and top-k leave the GPU as small arrays.
4. **Architecture specifics live in adapters** (`adapters/`). Capture code
   is model-agnostic.
5. **Traces are the interface.** What a reply recorded is read back from its
   trace on disk, never recomputed.

## Layout

```
src/metalscope/       the package: tracing inside a server (live.py, capture.py, hooks.py, tracer.py), the /scope
                      routes (web.py), steering (steer.py), the command line (cli.py, doctor.py), traces (store.py)
  omlx/               the oMLX backend: the hook, the patches into oMLX's engine, its admin API, the word edit
  ds4/                the ds4 backend: the proxy, its capture files, the build, the ds4 patch and ds4's licence
  adapters/           per-architecture capture points
  engines/            what a measurement runs on: an MLX copy of the model, or the model ds4 serves
  concepts/           concept profiles: the texts on disk, the statistics, the sweeps
  learn/              the GPU jobs (concept learn and calibrate, jlens learn): planning, running, grading
  jlens/              the Jacobian lens: loading, choosing and fitting one
  data/concepts/      the shipped concepts
  frontends/pi/       the Pi extension (TypeScript, shipped in the package); test/ its tests, test.sh runs them
tests/                pytest; a temporary data home, never ~/.metalscope (tests/__init__.py)
scripts/              a checkout's tools: convert_lens.py, omlx-python, pi_mock_omlx.py
```

## Development

From a checkout, for development:

```bash
uv sync --python 3.11 --extra dev     # the locked versions (uv.lock); the tests pin model internals
.venv/bin/metalscope omlx install && .venv/bin/metalscope omlx restart && .venv/bin/metalscope pi install
cd src/metalscope/frontends/pi && npm install      # the Pi extension's tests and type-check (bash test.sh)
```

oMLX loads the package through the hook's link, so a server restart picks up changes to `src/metalscope/`; Pi
loads the extension when it starts, so restart Pi after changing `src/metalscope/frontends/pi/`.
A server running older code than is installed (after an upgrade or a change here) says so: Pi warns at session
start and `metalscope doctor` gives the restart (`metalscope omlx restart`; ds4: `metalscope ds4 stop`, then
`metalscope ds4`).

```bash
.venv/bin/pytest
```

`scripts/omlx-python` runs oMLX's bundled interpreter with this repo importable.
The Pi extension's tests and type-check (`bash src/metalscope/frontends/pi/test.sh`) need its
dev tools once: `cd src/metalscope/frontends/pi && npm install`.

The Pi extension can be developed without a model: `scripts/pi_mock_omlx.py`
replays recorded live streams as a fake oMLX server. Record a stream while traced replies run (the key is oMLX's,
when it has one), then:

```bash
curl -N -H "Authorization: Bearer YOUR_OMLX_API_KEY" http://127.0.0.1:8008/scope/api/live > recorded.sse
.venv/bin/python scripts/pi_mock_omlx.py recorded.sse --port 8099 &
METALSCOPE_URL=http://127.0.0.1:8099 pi --model metalscope-mock/Qwen3.8-Flash-Next-oQ4e-mtp
```

`METALSCOPE_DISABLE=1` starts oMLX without metalscope; `metalscope omlx uninstall`
removes the .pth file.
