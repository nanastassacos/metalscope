# Front ends

The live stream any client can read, the Jacobian lens, Pi, and the traces they save. Pi's commands are in
[Steering and concepts](steering.md#pis-commands-steer-concept) and the [Command reference](commands.md).

**Lenses.** Each traced reply records the logit lens: the word the model would say if it stopped at each layer. A
Jacobian lens (one per model, fitted overnight or converted from a published one) reads the middle layers better and
shows words the model holds but never says.

**Pi.** Pi's extension adds a dock with each token's confidence, alternatives, lens and experts, live, and the commands `/scope`, `/steer`, `/concept` and
`/jlens`.

## Live stream

Any client can follow traced replies as they are written at `GET /scope/api/live` (server-sent events, see
`livestream.py`); `experts=true` adds each token's expert routing, `lens=true` each read layer's top 10 and
`writes=true` how much each layer changed each of its residual streams (one on most models; Inkling 2: its running
sum, and where each visit's inputs read from) and, without reads, that change's share of the residual it read.
`GET /scope/api/traces/ID/routing[?last=N]` is a saved reply's routing in the same per-token form.

One trace keeps at most 1.5 GB of records (`live.TRACE_MAX_BYTES`; Flash Next's whole 262k context fits, DeepSeek V4.1 on ds4 stops near 415k tokens). Past it the
reply goes on, the trace keeps the tokens up to the cap with `meta.capped`, and the dock says where the trace stops.

The dock's expert grids lay each layer's experts out so that experts which fire on the same tokens sit together, once
`metalscope experts layout` has built the model's map from its newest saved traces (kept in
`~/.metalscope/expert_layouts/<checkpoint>.json`; `--rebuild` builds it again). Until then the grid is in number
order and EXPERTS says so.

## Jacobian lens: what the model holds but doesn't say

With a [Jacobian lens](https://github.com/anthropics/jacobian-lens) installed
for the served model's checkpoint (`~/.metalscope/models/<checkpoint>/lenses/<name>/`, chosen with `/jlens use`; see `jlens/`), the
J-lens is read beside the logit lens. Add a lens from a checkout of this repository:

```bash
python scripts/convert_lens.py LENS.pt ~/.metalscope/models/<checkpoint>/lenses/<name> --hf-model <hf id> --checkpoint <checkpoint>  # torch env
metalscope jlens use <served-model> <name>        # not needed while it is the checkpoint's only lens
```

Published lenses for ~40 open models: `neuronpedia/jacobian-lens` on Hugging Face.
Or fit one overnight: `/jlens learn until 07:30` (`metalscope jlens learn <served-model> --until 07:30`),
then `/jlens use <its name>`. A window took about 28 to 35 min on Flash Next and 12 to 17 on Inkling 2 Mini; the job stops if none fits before the
deadline ([lens-fit scores](measurements.md#jacobian-lens-fits)). Without a J-lens the dock's LENS reads the
logit lens alone, and its edit has no word vectors. A lens belongs to the checkpoint: the same model on ds4
reads it once linked ([Checkpoints](backends.md#checkpoints)).

## Pi

[Pi](https://github.com/earendil-works/pi) is a terminal coding agent
(`npm install -g @earendil-works/pi-coding-agent`). `metalscope pi install` installs metalscope's extension as a Pi
package (`metalscope pi remove` removes it); it ships inside the package (`src/metalscope/frontends/pi/`).

In Pi, pick a model with `/model`. The extension registers the two backends as providers, `metalscope-omlx`
(oMLX's server, at its settings' address) and `metalscope-ds4` (the ds4 proxy, 127.0.0.1:8009), each with every
model it has whether its server runs or not. Picking a model makes it the one served, since one fits at a time: a
ds4 model runs `metalscope ds4 NAME`, an oMLX one `metalscope omlx NAME`. The switch starts when you pick the model,
its progress in the status line, and the first request waits for it. The live stream, the dock and the commands
follow the selected model's server. With `METALSCOPE_URL` set the extension registers that one server instead.
Each model's context window is what its server reports (131072 when it reports none), read again each time `/model`
opens. oMLX's context setting is not checked against memory: past what oMLX's memory guard admits, oMLX rejects the
prompt and Pi does not compact; start a new session with `/new`.

The first session after install says once where to look: "metalscope: replies are traced. ctrl+\ opens the dock,
/scope help the rest." (remembered in `~/.metalscope/pi.json`).

**The dock** (`ctrl+\` or `/scope dock`) is a live side panel. `ctrl+\` again steps in (the keys go to the dock),
again closes it. What it shows depends on its size:

- **The board** (a dock 100 columns by 36 lines, so a terminal 181 columns wide): a header with the steer on as a
  solid `STEER` chip (or `no steering`), a dim `untested` for a model outside the [Backends](../README.md#backends) table and a
  status only when something is wrong; then the reply strip (its title names the steer the reply ran under), then
  MAP and LENS (SETTLING over the telescope) on the left, TOKENS, EXPERTS and RESIDUAL STREAM on the right.
- **The one-panel board** (a smaller dock, down to 44 by 23): the reply's last lines and one panel at a time; `tab`
  cycles them.
- **The mini dock** (`m`, `/scope dock mini`, or under 23 rows; 50 columns, the board's right column, in a terminal
  from 100 columns, as the full dock): EXPERTS' layers, with MAP (its newest 3 lines; on a model without experts,
  the lines left), SETTLING and RESIDUAL STREAM under them, each as the board draws it; RESIDUAL STREAM's history
  takes a quarter of the dock's rows and the lines left over, 8 to 16 lines. Under 100 columns the dock is hidden.

The panels: **MAP** is the reply a token a cell, coloured by confidence. **LENS** is the token in focus through the
logit lens or the J-lens, layer by layer (`s` switches lens where both were read; `-` `+` change how many layers the
next traced reply reads). **SETTLING**, at the top of LENS, shows, for recent tokens, the layer where the lens
settles on the written word. **TOKENS** is the token in focus: its probability and the model's top 10.
**RESIDUAL STREAM** (a board at least 40 lines tall, a mini dock about 20; `5` opens it full size) is a key a layer
under the last tokens, its title naming the model's streams and the view, which `s` switches: `spikes` strikes a key
where its layer changed the residual over 2 standard deviations past what it had so far in the reply (from its 13th
token), in the colour of the stream it changed; `largest` (Flash Next and V4.1, four streams) lights each key in the colour of the stream its layer changed
most; `layers` (where the writes come with the layers' shares: every model in backends.md's table but Inkling 2) lights only the five layers that changed the residual most for the size of
the residual they read, brightest first, in one colour (layer 0, which reads the embedding, never); `reads` (Inkling 2, one stream) strikes a key where its layer visit leaned on one of its five sources far more
than usual. A steered layer's change includes the steer's push. **EXPERTS** lights the experts each layer used; `4`,
then enter, opens a layer's page and from it an expert's page, with the tokens each expert is most distinctive for,
counted from your traced chats (`metalscope experts tally`; kept in
`~/.metalscope/models/<checkpoint>/experts-tally.npz`, never shared). `6` opens the Steer page: the concepts ranked
by the chat test, each one's test chats, enter to steer. On a model with nothing learned whose shape matches a model
with concepts or a J-lens, the page lists those models and `l` links it to one ([Checkpoints](backends.md#checkpoints)); Pi's
pick list puts "Keep it separate" first. Where this model has no J-lens and such a model has one, LENS points to the
page. `g` opens engram inspection on a finished reply where the model has one (Flash Next; V4.1 with disk
streaming): which tokens the model's n-gram lookup helped. `?` explains the panel in focus where the key line was.
The keys are in the [Command reference](commands.md).

While a GPU job runs, a widget above the editor shows its steps and the time left. `/scope on` and `/scope off` set
the level; `/scope` alone shows what is on. Pi's requests are traced by header, independent of the global mode.

Untraced requests run at full speed even while tracing is on. On oMLX a traced or steered request prefills its whole
prompt on its own: no prefix reuse, and never in a forward shared with other prompts (oMLX's packed prefill) or with
another Mac (oMLX's Prefill Accelerator). Traces live in `~/.metalscope/traces/<date>/<id>/` (`arrays.npz`,
`meta.json`; the top choices and lens read-outs go out on the live stream only), indexed by
`~/.metalscope/index.jsonl`. Turns of one chat are grouped into a conversation by fingerprinting the start of the
history.

## Traces and their size budget

A traced reply takes about 8 MB on disk, a long Pi reply with thinking 30 MB or more. Traces are kept under a size
budget, 10 GB unless you set another (`metalscope traces keep 20GB`). After each saved trace the server checks the
total in the background and, past the budget, removes the oldest automated runs first (calibrations, ladders,
anything with a client other than Pi's), then the oldest chats. It never removes pinned traces
(`metalscope traces pin ID`) or each model's 20 newest unsteered traces, which steering sizes its doses from.
`metalscope traces auto off` turns the automatic cleanup off: then nothing goes until you prune, and doctor warns
when the traces are over the budget.
`metalscope traces` shows the total and the budget; `metalscope traces prune --dry-run` says what would go,
`metalscope traces prune` frees the space now. A trace folder a day old that the index does not list (a save
stopped part way, a lost line) is listed again first and counted like any other. `--older-than 7d` removes every trace older than a week and `--all`
every trace, whatever the budget (pins and those 20 kept). `metalscope doctor` shows the total and how much is
older than a week.
