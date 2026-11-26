# Command reference

The CLI (`metalscope --help` prints a shorter list; `scripts/omlx-python -m
metalscope.cli ...` runs the model-loading commands under oMLX's Python):

```
# backends: oMLX, ds4 (one serves a model at a time; starting one's model frees the other's)
metalscope status [--json]               # each backend: running or not, its models and their context, what is loaded,
                                         # then how to switch, stop and get help (metalscope alone: the same)
metalscope help [COMMAND ...]            # this list (--help), or one command's options: metalscope help ds4 start
metalscope --version                     # the installed version
metalscope omlx [MODEL] | omlx start [MODEL] | omlx stop     # MODEL: ds4 stopped first, MODEL loaded
metalscope ds4 [MODEL|GGUF] | ds4 start [MODEL|GGUF] [--name ID] [--ctx N] [--no-wait] [-- DS4-SERVER OPTIONS] | ds4 stop
                                         # in the background, at the model's full context; oMLX's model unloaded first
metalscope omlx install [--app PATH] | uninstall [--app PATH]
metalscope uninstall                     # back to stock: ds4 stopped, the oMLX hook and Pi package removed; data kept
metalscope omlx status | restart [--model ID] | load ID | unload ID | settings ID key=value ...   [--url URL]
metalscope ds4 install [--dir DIR] [--source URL|PATH] [--force]
metalscope ds4 serve GGUF [--name ID] [--port 8009] [--ds4-dir DIR] [--ds4-port 8010] [-- DS4-SERVER OPTIONS]

# front ends: this command line, Pi
metalscope pi install | remove

# every day
metalscope on | off
metalscope doctor [--app PATH] [--live]   # every piece; a table of the models: tested, J-lens, concepts, edit (info only)
                                         # --live: a traced and a steered completion on each backend; exits 1 when a check fails
metalscope bench [--backend omlx|ds4] [--model ID] [--quick] [--rest 60] [--repeats 3] [--max-tokens 300] [--contexts 1000,4000,16000] [--json FILE]
metalscope steering on | off
metalscope traces | traces status         # saved traces: total, count, size budget (Traces and their size budget)
metalscope traces keep SIZE              # the budget, e.g. 20GB (default 10GB)
metalscope traces auto [on|off]          # automatic cleanup past the budget (on by default); off: only prune removes
metalscope traces prune [--dry-run] [--older-than AGE | --all]  # down to the budget now; or every trace older than
                                         # AGE (7d, 2w, 12h), or all unprotected; --dry-run: say what would go
metalscope traces pin | unpin ID         # never prune this trace, or let it go again
metalscope jobs [cancel [JOB]]
metalscope model list | show NAME | link SERVED CHECKPOINT
metalscope experts tally [NAME] [--rebuild|--clear]   # the expert pages' tally from your chats: its size, count it afresh, delete it
metalscope experts layout [NAME] [--rebuild]          # the expert map the dock's grids are laid out by: build it, or again
metalscope jlens status [NAME]
metalscope jlens learn NAME [PATH] --until HH:MM | --windows N [--fresh] [--model PATH] [--loader auto|mlx_lm|omlx] [--need-gb GB]
           [--job ID] [--origin cli|pi]
metalscope jlens use NAME LENS
metalscope jlens on | off NAME
metalscope concept list [NAME]
metalscope concept show CONCEPT NAME
metalscope concept learn CONCEPT NAME [--description TEXT] [--corpus DIR] [--engine auto|mlx|ds4] [--url URL] [--model PATH]
           [--loader auto|mlx_lm|omlx] [--writer MODEL] [--need-gb GB] [--sweep-layers ..] [--strengths 0.1,..] [--seed 0]
           [--job ID] [--origin cli|pi]
metalscope concept calibrate CONCEPT NAME [--max-tokens 4000] [--url URL] [--job ID] [--origin cli|pi]
metalscope concept delete CONCEPT NAME [--yes]                    # its folder on NAME's checkpoint, texts included, to the trash
metalscope concept pack CKPT OUT_DIR [CONCEPT ... | --all] --base-model ID --license ID   # a package to share (Sharing concepts)
           [--license-name NAME] [--license-file PATH]
metalscope concept pull DIR|OWNER/REPO[@REV] SERVED                  # a package's concepts for SERVED

# research (measuring vectors and concepts by hand; Pi does not need these)
metalscope vector list NAME
metalscope vector norms NAME [--refresh]
```

`metalscope bench` times the model a backend serves now: writing speed with tracing off and at each lens step, the
dock's one-word edit, and engram inspection after conversations of each `--contexts` length (`""`: none). It is best
effort: other requests to the server skew the numbers. Each measurement starts after a `--rest` (60 s; the Mac slows
itself under sustained load) and tracing off is measured again after the lens steps to show any drift. It marks a
part the model or server cannot do as n/a with the reason; `--json` keeps every repeat.

In Pi:

```
/scope on | off | dock [mini | full] | jobs [cancel] | help [topic] | status
                                                          (alone: status; dock: open, step in, close, as ctrl+\ does;
                                                          mini: lights, MAP, SETTLING, m in the dock; jobs: the running or
                                                          last job, then the last ten; help scope: these forms)
/concept list | NAME | learn NAME [<what it means> | --corpus DIR]
       | calibrate NAME | help [SUB]                    (alone: list; SUB help: one subcommand; learn asks first: on oMLX it
                                                        unloads the model (Qwen3.6: 26 min with calibration), on ds4 refuses chat while measuring
                                                        (V4.1: 1-1.5 h, then 3-6 h calibrating);
                                                        learn NAME alone: a library concept, or one with a saved description)
/steer <concept>[@dose] [dose] [+ <concept> …] | again | off | status | replay [<steer>] [temp:T] | help     (alone: what is on)
       a bare number or dose:D is the dose (/steer fluffy 0.5 = /steer fluffy@0.5 = /steer fluffy dose:0.5);
       replay: your reply beside the same message unsteered, or under the steer given (at most 2,048 tokens)
/jlens | status | on | off | learn [PATH] until HH:MM [fresh] | learn [PATH] windows N [fresh] | use NAME | help
                                                       (alone: status; learn unloads the model until the deadline
                                                       and asks first)
```

One word, one job: `/scope` looks at what the model does and at the GPU jobs, `/concept` is the
library of concepts and the jobs that make and calibrate them, `/steer`
changes the model, `/jlens` is the J-lens the readouts go through.
A new `/steer` replaces the one on; `/steer off` stops all of it.

The dock's keys (`?` in the dock explains the panel in focus):

| Where | Keys |
|---|---|
| Anywhere in Pi | `ctrl+\` opens the dock, again steps in, again closes it; `alt+i` steps straight in (where the terminal sends Option as Alt); `option+x` (`alt+x`) stops the running job, asked first |
| Stepped in | ←/→ or `h`/`l` a token, shift ←/→ ten, alt or cmd ←/→ or `H`/`L` a sentence, page up/down 25, home the first, `f` (or `end`) the newest, followed; `u`/`U` the next or previous unsure token |
| | `tab` the next panel, `?` what it shows, `s` the other lens (RESIDUAL STREAM: its view), `m` the mini dock, `6` the Steer page, `g` engram inspection, esc back, `q` out at once; the key line names the numbers that do something as one, `1-6 panel` |
| The board | `1` MAP, `2` LENS, `3` TOKENS, `4` EXPERTS, `5` RESIDUAL STREAM open that panel over the board; the same number again, the board |
| The one-panel board | `tab` cycles the panels; a number picks one |
| LENS | ↑/↓ a layer, `-` `+` fewer or more layers read, enter edits a word (experimental), `x` steering off |
| EXPERTS | `[`/`]` its layers; `4`, then enter a layer's page, then an expert's; `a` all chats or this session |
| RESIDUAL STREAM | `s` the next view (spikes, largest, layers, reads: what the model records) |
| TOKENS | ↑/↓ the next or previous unsure token |
| Steer page | ↑/↓ a concept, ←/→ a dose, enter steers, `x` off, `d` then `y` deletes it (to the trash), `n`/`N` a turn of its test chats, `6` the board; with nothing learned, `l` links a same-shape model |
| Mini dock | `[`/`]` more layers, `s` the next RESIDUAL STREAM view, `m` the whole dock |

Editing a word on LENS (experimental): enter puts the cursor on the word the model wrote; ⌫ removes it, letters or
↑/↓ type another over it, enter answers the reply again from that token with the edit at that one position, and
the new reply replaces the old in Pi's transcript. The newest reply only, and not one that ran tools; it needs the
server's `edit_at` (oMLX and ds4 have it) and a model the server takes the edit for.

`g` on a finished reply, where the model has an engram (Flash Next; V4.1 with disk streaming), reads the reply
again with and without it and marks in blue the tokens it helped; TOKENS shows each token's top choices without and
with it. `g` while it reads stops it; `g` again closes it. The reading holds the GPU, and on ds4 chat is refused
meanwhile. On V4.1 ds4 reads the reply a token at a time (about 0.07 s a token each reading, on top of the
conversation's read), and an inspection expected to take over 10 minutes is refused.
