# Backends

Installing metalscope in detail, what works on each model, the two backends, where each model's data lives, and updating the engines.

## Install in detail

```bash
uv tool install git+https://github.com/nanastassacos/metalscope
metalscope omlx install                 # the oMLX backend: hook into oMLX.app (then metalscope omlx restart)
npm install -g @earendil-works/pi-coding-agent   # Pi, if you use it
metalscope pi install                   # the Pi front end
metalscope doctor                       # check every piece

# the ds4 backend: its proxy's dependencies, then ds4 itself, fetched at the commit the patch is for and built
uv tool install --reinstall "metalscope[ds4] @ git+https://github.com/nanastassacos/metalscope"
metalscope ds4 install

# concept pull from Hugging Face (huggingface_hub)
uv tool install --reinstall "metalscope[share] @ git+https://github.com/nanastassacos/metalscope"
```

An extra is added by installing the tool again with it; name every extra you want each time (`metalscope[ds4,share]`).

`omlx install` writes `metalscope.pth` into `~/.local/lib/python3.11/site-packages` and a link to the package (with
`installed.json`) into `~/.local/share/metalscope/omlx/`. The `.pth` does nothing outside oMLX.app, and oMLX.app
itself is never modified, so the hook survives oMLX updates. `ds4 install` builds under `~/.local/share/metalscope/ds4/`
(`METALSCOPE_DS4_DIR` elsewhere). Data (traces, lenses, vectors, concepts, settings) lives in `~/.metalscope`
(`METALSCOPE_HOME` elsewhere); what belongs to a model is kept by checkpoint ([Checkpoints](#checkpoints)).

**Who can reach it.** metalscope is meant for a single-user Mac. Its API (`/scope`) and its request headers answer
only this machine's own clients, never a web page of another site; on oMLX with an API key set, the API also needs
that key, as oMLX's own does (Pi and the command line send it). The ds4 proxy has no key. The data home is private
to your account (0700): traces hold your prompts and replies. See [SECURITY.md](../SECURITY.md) to report a
vulnerability.

**Upgrade and uninstall.**

```bash
uv tool upgrade metalscope        # upgrade: then point the hook at the new package
metalscope omlx install
metalscope omlx restart

metalscope uninstall              # uninstall: back to stock engines, then the tool
uv tool uninstall metalscope
```

`metalscope uninstall` stops a ds4 server metalscope started, removes the oMLX hook (the `.pth`, its link and
manifest, `~/.local/share/metalscope/run/`) and restarts a running oMLX without it, and removes Pi's package entry; a
piece already gone is skipped. Run it first: once the tool is gone, nothing removes the hook. (A hook left behind stays
silent.) `ds4 stop`, `omlx uninstall` then `omlx restart`, and `pi remove` do the same piece by piece.
Left behind on purpose: your data in `~/.metalscope`, the ds4 build and its models in
`~/.local/share/metalscope/ds4` (hundreds of GB if you downloaded models).

## What works on each model

What works on each, from `metalscope doctor --live` run on every one:

| | Qwen3.5 2B, oMLX | Qwen3.8 27B, oMLX | Qwen3.6 35B, oMLX | Flash Next, oMLX | Flash Next, ds4 | DeepSeek V4.1 Flash, ds4 | Inkling 2 Mini, oMLX |
|---|---|---|---|---|---|---|---|
| A reply traced token by token | yes | yes | yes | yes | yes | yes | yes |
| How sure the model was at each token | yes | yes | yes | yes | yes | yes | yes |
| Logit lens | yes | yes | yes | yes | yes | yes | yes |
| Jacobian lens | yes | no (none fitted) | yes | yes | yes | no | yes (fitted with `/jlens learn`; none ships) |
| More or fewer layers read | yes | yes | yes | yes | yes | yes | yes |
| The experts each token used | no experts | no experts | yes | yes | yes | yes | yes |
| Residual stream views | spikes, largest | spikes, largest | spikes, largest | spikes and largest, each (hc) and (layers) | spikes and largest, each (hc) and (layers) | spikes and largest, each (hc) and (layers) | spikes (writes), spikes (reads) |
| Engram inspection | no engram | no engram | no engram | yes | yes | yes (disk streaming only) | no engram |
| Steering with a concept | yes | yes | yes | yes | yes | yes | yes |
| One word of a reply changed (experimental) | yes | no (needs the Jacobian lens) | yes | yes | yes | no (needs the Jacobian lens) | yes (with a fitted Jacobian lens) |

The dock's one-token edit is refused on oMLX for models oMLX reads through its own output parser (DeepSeek V4/V4.1
among them); the dock then hides its key. Expert routing is recorded with each model family's own rule
(`routing.py`); an MoE family outside those is recorded approximately and marked so.

Measured: [Inkling 2](measurements.md#inkling-2), [what it costs](measurements.md#what-it-costs), [the jobs](measurements.md#the-jobs).

## oMLX

```bash
metalscope omlx MODEL                     # start oMLX if it is not running and load MODEL (ds4 stopped first)
metalscope omlx status                    # listening?, served/loaded models, tracing mode
metalscope omlx restart --model ID        # via the admin API (~/.omlx/bin/omlx when it cannot), then load ID
metalscope omlx load ID                   # / unload ID
metalscope omlx settings ID mtp_enabled=true mtp_adaptive_max_depth=4
```

These use the API key in `~/.omlx/settings.json`, never printed. Chat with any client through oMLX as usual;
with tracing on, each response is saved as a trace, and Pi's dock follows it as it is written. A single request
can opt in or out with the header `X-Scope-Trace: on | off`, whatever the global mode (`X-Scope-Trace-Client` names the client, `X-Scope-Steer` steers it).

With tracing off nothing is recorded and no hooks are installed. With it on, a trace keeps residual and stream norms
and expert routing at every layer; the top 10 next tokens and the lens readings (the logit lens, and the J-lens
while it is on) go to the live feed. A prompt reads the lenses on its last 64 rows only. What it costs is the [bench table](measurements.md#what-it-costs). A traced or steered request
reuses no prefix cache: oMLX prefills its whole prompt on its own.

## MTP models

Models that oMLX loads with Lightning MTP (multi-token prediction; e.g.
`Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` with `mtp_enabled` in its model settings)
are traced on their own path (`mtp.py`): records are filed by position, so a rejected draft's are overwritten by
the tokens actually emitted. The MTP head's own drafts are not recorded. On a 128 GB Mac this model needs oMLX's
expert offload on (`moe_expert_offload_resident_fraction` 0.75 here); without it oMLX rejects every prompt.

## The ds4 backend

[ds4](https://github.com/antirez/ds4) (antirez; C and Metal) serves Qwen3.8
Flash Next and DeepSeek V4.1 Flash from GGUF files, V4.1 with SSD streaming on
a 128 GB Mac. metalscope works with it the way it works with oMLX: traces, the
live stream, Pi's dock and steering. ds4 records the trace itself (its metalscope patch,
`src/metalscope/ds4/metalscope.patch` here, on top of ds4 5379c1c): the readings are taken inside each forward's
own GPU work, and the steer is applied there too. `metalscope ds4 serve` starts ds4-server behind a proxy that turns
those readings into traces:

```bash
uv tool install --reinstall "metalscope[ds4] @ git+https://github.com/nanastassacos/metalscope"   # once: the extra
metalscope ds4 install                       # once: ds4 at 5379c1c with the patch, built
~/.local/share/metalscope/ds4/download_model.sh qwen38-q4k        # a model (ds4's own script; ds41f-q2 for V4.1)
metalscope model link Qwen3.8-Flash-Next-ds4-q4k Qwen3.8-Flash-Next-oQ4e-mtp   # once: the checkpoint oMLX's copy is, its data and J-lens
metalscope ds4 ~/.local/share/metalscope/ds4/gguf/Qwen3.8-Flash-Next-Q4.gguf --name Qwen3.8-Flash-Next-ds4-q4k
metalscope ds4 stop                          # or: metalscope omlx MODEL (ds4 stopped, MODEL loaded)
```

Serving a model:

- `metalscope ds4 MODEL` serves a model in the background and returns once it answers (`--no-wait`: once it is
  loading). oMLX's loaded models are unloaded first (one fits at a time) and a different ds4 model is stopped.
  `metalscope ds4 serve GGUF` is the same in the foreground (Ctrl-C stops it).
- MODEL is a name ds4 served before (`metalscope status` lists them) or a GGUF file (named `--name`, else the file
  name and -ds4); options after `--` go to ds4-server and are kept for the model. Besides `ds4 install`'s `gguf/`,
  ds4 lists the GGUFs in the folders named in `METALSCOPE_DS4_GGUF`.
- The model runs at its own maximum context (Qwen3.8 Flash Next 262,144, V4.1 1,048,576; `--ctx N` for less). The
  proxy's `/v1/models` reports `--ctx` as `max_model_len`, Pi's context window for the model.
- The global mode (`metalscope on`) and the request headers apply as on oMLX. On ds4 the lenses and the top 10 are
  read on reply tokens and a prompt chunk's last token only.
- ds4-server's log is `~/.metalscope/ds4/ds4-server.log`.

Ports and who can reach them:

- `--port` (8009) is the proxy, `--ds4-port` (8010) ds4-server's own.
- ds4-server's own port listens on localhost and takes metalscope's capture options (files it writes and reads)
  from any request, so never expose it. The proxy drops a client's, and ds4-server opens those files only inside
  `~/.metalscope/ds4/scope/` (its `--scope-dir`, which `ds4 serve` passes; started without one it refuses them).
- The proxy answers only this machine's own clients: a request a web page of another site sent is refused (403),
  and a POST to `/v1` must say `Content-Type: application/json` (415 otherwise; curl needs the `-H`).

ds4's download script reports a checksum failure for the V4.1 Q2 file even when the download is complete (it
matches the Hugging Face repository's own SHA-256, 5296e85f…); the file runs.

Sharing a model's data with its oMLX copy:

- `metalscope model link Qwen3.8-Flash-Next-ds4-q4k CKPT` makes the ds4 model a copy of CKPT, the checkpoint oMLX's
  `Qwen3.8-Flash-Next-oQ4e-mtp` is: `metalscope model show Qwen3.8-Flash-Next-oQ4e-mtp` names it, and on a fresh
  install it is that name itself ([Checkpoints](#checkpoints)).
- Link before learning anything on the ds4 name: a name with data of its own (concepts, a lens) is refused.
- The ds4 model then reads the same steering vectors, concept texts, profiles and J-lens, with nothing copied. Its
  strength table and chat calibrations are its own: trace a few unsteered replies on ds4 before steering there, and
  `/concept calibrate` there for its chat doses.
- The proxy checks the GGUF's shape against the checkpoint's when it starts; restart it after a link.
- `/concept learn` works on ds4: it measures the model ds4 serves while the proxy refuses chat. `/jlens learn`
  needs oMLX, as it loads its own MLX copy of the model.

## Checkpoints

A steering vector is a direction in a model's residual stream, so it belongs
to the model, not to the backend or quantisation serving it: directions, the
texts they were learned from, concept profiles and lenses are kept under the model's **checkpoint**
(`src/metalscope/models.py`). What depends on the engine and quantisation is
kept per **served copy**: the strength table that turns a dose into a size
(each copy's typical residual norm per layer, from its own unsteered traces,
so `dose 0.4` is the same relative push on every engine) and the chat
calibrations (measured in conversation through that copy). Nothing falls back
from one copy to another: a copy with no unsteered traces of its own has no
strength table, and steering it says so.

What was learned on a checkpoint is in its folder, including the texts each concept was learned from. Between
models only the description of a concept of yours is shared: each model writes its own texts from it.

```
~/.metalscope/models/<checkpoint>/
    checkpoint.json              {name, shape: {n_layers, width, hc, d_model}, served: [...], lens, jlens}
    experts-tally.npz            the expert pages' tally (metalscope experts tally)
    concepts/<concept>/
        concept.json, pos.txt, neg.txt, task.json, writing.json   the texts it was learned from and how they were made
        vector.safetensors (vector.json), profile.json, generation files
        <served>/                calibration.json (it fits the profile), replies.jsonl, grades.json
    served/<served>/
        norms.json               its strength table (metalscope vector norms SERVED)
        plain-chats/             the unsteered calibration conversations, reused by every concept
    lenses/<name>/               a J-lens; checkpoint.json names the one it reads
    fits/<fit id>/               a /jlens learn fit in progress (lens-<served>-<corpus id>)
~/.metalscope/concepts/<concept>/concept.json   a concept of yours: {name, description} (none for a shipped one)
~/.metalscope/trash/<YYYYMMDD-HHMMSS>-<concept>-<checkpoint>/   a deleted concept's folder on one checkpoint, texts included (concept delete, d then y in the dock)
~/.metalscope/trash/<YYYYMMDD-HHMMSS>-<concept>-definition/     a deleted description (concept delete when no model has it)
~/.metalscope/traces/, index.jsonl  per served copy: the traces
~/.metalscope/expert_layouts/<checkpoint>.json   the expert map the dock's grids are laid out by
```

Linking a served name to a checkpoint:

- A served name nothing links is its own checkpoint. `metalscope model link SERVED CKPT` makes a served name a copy
  of a checkpoint: SERVED's own folder (`served/SERVED/`) and its `/jlens learn` fits move under CKPT, and from
  then on it reads CKPT's directions, texts, profiles, J-lens and expert tally. Its strength table stays its own,
  and its chat calibrations do not move (calibrate it again under CKPT).
- A link is refused when their recorded shapes differ, while a job runs on SERVED, when SERVED holds data of its
  own (the message names the folders to move), or when CKPT has a folder by the same name already. A link that
  stopped part way finishes when run again.
- Nothing links by itself: `model show NAME` names up to three models of the same shape with data
  (`metalscope model link NAME X if it is the same model`), and the dock's Steer page offers the same
  ([Pi](dock.md#pi)). `model list` shows every checkpoint, `model show NAME` where a name's data is.
- The shape is recorded the first time an engine or the ds4 proxy opens the model, or oMLX installs its hooks on it
  (the offer needs it), and checked every time after, so a wrong link fails before a vector is applied.
- Re-profiling a concept, or a new strength table (`metalscope vector norms NAME --refresh`), makes the
  calibrations made before it out of date.

## Updating the engines

metalscope owns neither engine, so it pins the version of each it was checked against, and `metalscope doctor`
prints both pins and whether what is installed matches:

| | pinned in | on another version |
|---|---|---|
| oMLX | `TESTED_OMLX` in `omlx/install.py` (0.8.0, 0.8.2, 0.8.3) | oMLX runs untraced and loads nothing of metalscope but the hook; `omlx install`, `omlx status` and doctor say so: install a tested oMLX, or update metalscope once it supports yours |
| ds4 | `DS4_COMMIT` in `ds4/build.py` (5379c1c) and `ds4/metalscope.patch` | `ds4 install` builds the pin; doctor fails when the build is not the pin and patch |

Moving to a newer version:

1. oMLX: run `pytest tests/test_omlx_contract.py` (it reads the installed app's source, or `OMLX_APP=<bundle>`'s,
   and checks every oMLX function metalscope wraps or calls), read oMLX's diff for what those do, port what moved
   and set `TESTED_OMLX`. ds4: in a ds4 checkout carrying the metalscope commit (`ds4 install`'s build is one), rebase
   it onto the new commit, `make ds4-server ds4_test && ./ds4_test --metal-kernels`, then
   `git format-patch -1 --stdout > src/metalscope/ds4/metalscope.patch` and set `DS4_COMMIT`.
2. `.venv/bin/pytest`.
3. Serve a model with a learned concept on the new version (without one the concept steer is not checked) and run
   `metalscope doctor --live`: short traced and steered completions on
   each backend serving one, checking what the dock draws end to end. A moved internal shows up here.
