"""The jobs Pi and the CLI start (job.py's docstring walks through them):

- job.py: the GPU window and the kinds: ``concept learn``, ``concept calibrate``, ``jlens learn``;
- state.py: their state files, one job at a time, cancel; spawn.py: the server starting one, detached;
- writer.py: the served model writes a new concept's spec, 20 matched pairs and about 60 two-option task items (and
  the instructions it was sent, writing.json); supplied, shipped and a model's own texts are read and checked here;
- gpu.py: ``concept learn``'s GPU child, every measurement on one model load;
- decide.py: the items the push is tested on, the push both ways, the early stops ("can't be steered") and the
  nearest existing concepts;
- calibrate.py: the chat dose from conversations behind a plain system prompt, graded by the served model; chat
  requests only;
- lens_fit_gpu.py: ``jlens learn``'s GPU child.
"""
