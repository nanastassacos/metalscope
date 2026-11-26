# Changelog

## Unreleased

- The dock's board is laid out again: TOKENS, MAP and LENS on the left, EXPERTS over RESIDUAL STREAM on the right
  (`{` `}` move the divider between them). On a model without experts TOKENS sits over RESIDUAL STREAM. The panel
  keys follow the order: `1` TOKENS, `2` MAP, `3` LENS, `4` EXPERTS, `5` RESIDUAL STREAM, `6` the Steer page.
- The lenses read at three sizes, by name: small (4 layers), medium (a third of the depth, at most 16; the default)
  and large (half, at most 24), spread evenly with the last layer always read. A number saved by 0.1.0 is mapped
  once. LENS grows with its size: small is compact, medium fills its panel, large takes MAP's spare lines.
- The residual stream is recorded on models with one stream too (Qwen3.5 2B, Qwen3.8 27B, Qwen3.6 35B).
- RESIDUAL STREAM's views are named by what they measure and over what: spikes or largest, by hyper-connection
  stream or by layer (Inkling 2: spikes by writes or by reads). Largest is drawn by each layer's share of the change,
  not as a top five. The panel fills its width, and `?` explains the view on screen.
- LENS's speed line per size counts only replies read at that size's number of layers.
- Engram inspection keeps the written token's row however short TOKENS is.

## 0.1.0 (2026-11-24)

First public version.

- Tracing inside the server: a hook for oMLX 0.8.0, 0.8.2 and 0.8.3 (the app is not modified) and a patch and proxy for ds4 at
  5379c1c. Each token's confidence, alternatives, logit lens, experts and residual stream are recorded.
- Engram inspection: a second pass over a finished reply with the model's n-gram lookup off, on models that have one.
- Jacobian lens: fit one per model, or convert a published one.
- Word swap (experimental): one word of a reply removed or swapped in the dock, through the Jacobian lens.
- Steering by request header, thinking included, dosed from the model's own traces.
- Concepts: learn a direction from texts (33 ship; for a new name the model writes its own), calibrate it in
  conversation, `concept pack` and `concept pull` to share.
- Everything learned on a model is kept in that model's folder, `~/.metalscope/models/<model>/`.
- Pi extension: a live dock and the `/scope`, `/steer`, `/concept` and `/jlens` commands.
- `metalscope doctor` checks the install and the engine versions.
- `metalscope uninstall` puts the engines back to stock in one step.
- Run on Qwen3.5 2B, Qwen3.8 27B, Qwen3.6 35B, Qwen3.8 Flash Next and Inkling 2 Mini (oMLX), and on Qwen3.8 Flash
  Next and DeepSeek V4.1 Flash (ds4).
- Inkling 2 Mini needs oMLX 0.8.3 (an older oMLX's Inkling 2 is refused). It is traced over its 48 layer visits; its
  Jacobian lens is fitted on each visit's state (`/jlens learn`), and concepts steer it over a block of 6 visits.
