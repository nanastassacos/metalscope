# Steering and concepts

Steering a reply with a concept, learning and calibrating concepts, and sharing them.

**Steering.** A request with the `X-Scope-Steer` header gets a direction added to the model's inner state at one
layer, on every token of the prompt and the reply, thinking included. The dose is a fraction of that layer's typical
size on this model, measured from its own unsteered traces: that is why a few traced replies come first.

**Concepts.** A concept is a set of texts: matched pairs (the same sentence with and without the concept) and
multiple-choice questions it should sway. `concept learn` turns them into a direction on your model and picks the
layer to steer at; for a new name the model writes the texts first. Then it calibrates: 3 conversations of 6
messages at several doses, graded blind by the model itself. The dose with the most good replies of the 18 (the
lower on a tie) becomes the concept's default. Each model learns from its own copy of the texts, kept in its own
folder: a shipped concept's texts are copied in, and for a concept of yours the description is kept, so another
model writes its own texts from it.
The model grades the replies itself (and writes a new concept's texts), so a concept it misread can still score well.

**Sharing.** `concept pack` writes a model's concepts into a folder to upload, like a LoRA adapter: it names the
model it was made on. `concept pull` places one for a model you serve, after checking it has the same shape.
Calibrations stay with the machine that ran them: calibrate on yours.

## Pi's commands: steer, concept

- **`/steer <concept> [dose]`** adds the concept's direction to the residual on every following request (the
  `X-Scope-Steer` header), at its reader layer and, by default, its chat dose from `/concept calibrate` on the
  served model. A concept not calibrated there has no default: give a dose, or calibrate it. A library concept not
  learned on the model yet says so, with the `/concept learn NAME` that learns it. `/steer angry + robotic` (or
  `angry@0.4 + robotic@0.8`) steers with several concepts at once, their vectors summed. Steering is the full dose
  at every token, prompt and reply, thinking included, until `/steer off`. `/steer again` answers your last message
  again under the steer on (a reply that ran tools is refused). A word is changed in the dock's LENS edit, one
  token only; `/steer` takes concepts.
  Steering is enabled on the server the first time (`metalscope steering off` reverses it). A line above the editor
  shows what is on; `/steer` itself says only what that line cannot (a dose not tried, a missing calibration),
  `/steer status` all of it.
- **`/steer replay`** shows your last reply beside the same message answered unsteered, as a transcript entry the
  model never sees; `/steer replay angry@0.6` answers it under that steer instead. The fresh answer sees what your
  reply saw, without tools, at most 2,048 tokens, at Pi's temperature (`temp:0` for greedy).
- **`/concept`** is the concept library: `list` (the concepts learned on the model ranked by their good replies in
  conversation, the top 5 marked, then the library's concepts not learned there yet), `NAME` (one concept in a few
  lines: who wrote its texts, its good replies at each dose tried, the layer it steers at), and the
  two jobs below. One job runs at a time; a widget shows its steps and the time left. When it ends, the result lands
  in the transcript as those steps, frozen (the model's unload and reload left out), and the one command that goes
  further (the dose `/steer NAME` uses);
  `/scope jobs` shows it again and `/scope jobs cancel` (or `option+x`) stops it.
- **`/concept learn NAME [<what it means>]`** makes a concept (`metalscope concept learn`) and asks first, because
  its GPU part holds the model. It learns from the model's own texts when it has them (a re-run), else from
  `--corpus DIR` (texts written elsewhere: concept.json, pos.txt, neg.txt, task.json), else from a shipped concept's
  texts, else the served model writes its own from the description typed now or the one saved when the concept was
  first made. The texts it learns from are kept in the model's folder; another model's are never used. A GPU job
  then measures the
  concept and picks the layer to steer at, and the chat calibration follows. On oMLX the served model is unloaded
  while the job runs its own copy (26 minutes in all on Qwen3.6 35B). On ds4 it measures the model ds4 serves and
  the proxy refuses chat meanwhile (on DeepSeek V4.1 1 to 1.5 hours, then 3 to 6 hours calibrating). While its GPU
  part runs, loading or switching models is refused. It stops early, with the reason, when the concept can't be
  steered.
- **`/concept calibrate NAME`** tests steering the concept in conversation: 3 conversations of 6 messages behind a
  plain system prompt, steered at doses 0.2, 0.4, 0.6 and 0.8 (the stronger ones skipped once a dose's replies break or run away in length; a runaway dose is graded only when fewer than two others are), then 0.1 either side of the best, and graded blind
  by the served model for style, coherence and usefulness. A reply is good with all three 2 or more of 3, on its
  first try. The dose with the most good replies of the 18 (the lower on a tie) becomes `/steer NAME`'s default,
  shown as `12/18 good @0.5`. Chat works meanwhile. It is written to the served model's `calibration.json`, with
  every reply in `replies.jsonl` and each grading in `grades.json` ([Checkpoints](backends.md#checkpoints)). Its steered
  replies and its grading requests are not traced (nor are the texts a learn job has the model write); its plain
  conversations are.
  `metalscope concept calibrate CONCEPT NAME` is the same from the command line.
- **`/scope help`** puts a one-screen overview into the transcript; `/<command> help` gives one command's forms
  (/scope's: `/scope help scope`), and `/scope help <topic>` what a feature shows and how to read it.

## Steering (writing to the residual)

A request can add a vector to the residual at chosen layers while it is served
(`steer.py`). Off by default:

```bash
metalscope steering on                         # allow the header (settings.json)
metalscope vector norms Qwen3.6-35B-A3B-MLX-4bit       # what dose 1.0 means per layer
```

Then on any chat request:

```
X-Scope-Steer: humorous                      # its calibrated dose and layer on this model: what /steer humorous uses
X-Scope-Steer: angry@0.5                     # half the layer's typical residual norm, at angry's steer layer
X-Scope-Steer: angry@0.5;layers=31;positions=decode
X-Scope-Steer: random~angry@0.5               # the control: a random direction with the same norm profile, same layer
X-Scope-Steer: angry@0.4/24+robotic@0.8/24   # two vectors summed, each with its own strength and layers
```

The grammar is `PART[+PART…][;option=value…]` (at most 8 parts, each name once) with
`PART = [random~]NAME@DOSE[/LAYERS]`, `NAME` alone (a 400 when it has no calibration there) or `remove~NAME/LAYERS`
(the dock's edit's: no dose, NAME's direction projected out of the residual before any add there; a word vector cut
to its first token is refused), `LAYERS` a comma list of layers or ranges `a-b`. A word's vector (made for the edit)
steers one token only: any other steer naming one is refused (`a word changes one token (the dock's LENS edit), not
a whole reply`). The options after
`;` are the request's, given once: `layers=` (for parts that name none; without it, a concept's steer layer, the
one `NAME` alone uses; a 400 when learn found none; a vector with no profile, such as a word's, at all its layers),
`positions=all|decode` (default all). `random~` is per
part, and each control part draws its own Gaussian base. Every part is constant: the same vector at
every position it covers.

Strength is relative to the median residual norm of that layer on the model's
own unsteered traces, capped at 4. The response carries `X-Scope-Steer-Accepted` (a part with the layer it was
given, `angry@0.5/24`), and each part is saved with the trace (`meta.steer.parts`). A steer that cannot be built (no
strength table yet, a layer its vector lacks) is refused with a 400 that says what to do, and one the server cannot
set up with a 503, on both engines: never answered unsteered. If steering fails once the request is being served,
the reply goes on unsteered from that point (on ds4, all of it) and its trace is marked `FAILED`. Vectors live
in the concept's folder on the checkpoint, `~/.metalscope/models/<checkpoint>/concepts/<name>/vector.safetensors`
([Checkpoints](backends.md#checkpoints)).

## Concept profiles: writer, carrier, reader

`metalscope concept learn` (`learn/`, on `concepts/`) strings the
read and write sides together on one concept. A concept is a matched contrast
corpus (the same sentences with only the concept swapped) plus a forced-choice
task the concept should move. Its profile finds the **writer** (the first layer
where the diff-of-means direction's held-out d' is significant: matched pairs
held out together, p < 0.01 in a permutation test that flips the labels within
pairs), the
**carrier** (the band where adjacent layers' directions stay aligned) and the **reader** (where
injecting the direction moves the choice beyond the norm-matched random
control at a dose that keeps the answer well formed). 33 concepts ship with their texts (`concept list`): 20 written by hand
(by Claude, among them the emotion set, fluffy and four philosophical contrasts), 12 written by the models that
learned them first (DeepSeek V4.1, Qwen3.8 Flash Next, Qwen3.6) and one by Claude Opus; concept.json's `learned` says which. Any model learns one with
`concept learn NAME MODEL`, from a copy of its texts. Its chat calibration (`/concept calibrate`) gives its good replies of 18 at its best dose.
`concept delete CONCEPT NAME` moves its folder on NAME's checkpoint (its texts, vector, profile and every served
copy's calibration) to `~/.metalscope/trash/` by a rename; the saved description stays, so another learn writes new
texts. When no model has the concept, it moves the description instead. It asks first (`--yes` skips that) and is
refused while a learn or calibrate of it runs.

```bash
metalscope concept learn sleepy Qwen3.6-35B-A3B-MLX-4bit --description "drowsy and slow"   # the model writes the texts
metalscope concept learn sleepy Qwen3.8-Flash-Next-oQ4e-mtp  # another model writes its own, from the saved description
metalscope concept learn fluffy Qwen3.6-35B-A3B-MLX-4bit      # a shipped concept: its texts copied in
metalscope concept show fluffy Qwen3.6-35B-A3B-MLX-4bit       # its summary
```

Every measurement runs on an **engine** (`src/metalscope/engines/`): `--engine mlx` loads a copy of the checkpoint
in the job's own process, `--engine ds4` measures the model ds4 serves (`auto` picks it when ds4 serves the model).

## Sharing concepts

A package is a copy of concept folders, shared like a LoRA adapter: it declares its base model, you name the model
you serve, and only the shape is checked (`src/metalscope/share.py`).

- **`concept pack CKPT OUT_DIR --base-model ID --license ID`** writes a package ready for one Hugging Face repo per
  base model: `manifest.json`, a `README.md` model card (front matter with `base_model` as the base model's HF id,
  `base_model_relation: adapter`, the licence, tags; then each concept's texts' author, how it did on your copy, setup),
  a `LICENSE` (`--license-file PATH` copies the text in; else a note naming the licence; `--license other` needs
  `--license-name` and `--license-file`) and the concept folders, each copy's calibration kept under `<checkpoint>/reference/<served>/`
  as a reference (no paths from your disk go in). It packs the concepts that ship with metalscope; yours need naming
  (or `--all`), and it says which it left out. The texts packed are the checkpoint's own; a concept with none there
  is packed without them, and pack names it. Upload it yourself (`hf upload OWNER/REPO OUT_DIR`).
- **`concept pull DIR|OWNER/REPO[@REV] SERVED`** (a repo needs `huggingface_hub`: the `share` extra, [Install in detail](backends.md#install-in-detail); `@REV` a branch, tag or commit) refuses another shape (serve SERVED
  once first), checked on each vector file's tensors too (safetensors only), a package with texts inside a
  `<concept>/<checkpoint>/` folder, and a `vector.json` that is not a JSON object. It places each concept's texts,
  vector and profile in the concept's folder on SERVED's checkpoint, whole or not at all; the repo, revision and commit are kept in the vector's `source`. A concept the model already
  has (its texts, vector or profile) is kept, and pull says whether it was yours or another package's; a newer
  version of the package replaces its own. A pulled concept is listed by `concept list SERVED`; `concept list` alone
  ends with how many concepts only the models have. A calibration belongs to the copy that ran it: the packer's are
  references, `concept calibrate CONCEPT SERVED` finds your own dose (it needs the concept's texts on the model).
