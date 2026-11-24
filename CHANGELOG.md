# Changelog

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
