"""Concept profiles: where a concept is written, carried and read.

A concept is a matched contrast corpus (``pos.txt`` against ``neg.txt``: the same sentences with only the concept
swapped) and a forced-choice task it should move (``task.json``). The **writer** is the first layer whose
diff-of-means direction is present, p < WRITER_ALPHA: held-out d' on paired folds (a pair's two sentences are held
out together, or the twin's shared wording reads the held-out one low) against the paired null (labels flipped
within pairs). The **carrier** is the band where adjacent layers' directions stay aligned. The **reader** is where
injecting the direction moves the choice most over a norm-matched random control at a dose that keeps the answer
well formed. ``profile`` measures the writer and the carrier; ``concept learn``'s GPU stage (learn/gpu.py) the
reader, the dose ladder and the test.

Everything runs on an engine (engines/), one prompt per call: a single-row forward is deterministic, and a batched
row is not (on Flash Next the row count picks another bf16 matmul kernel, and routing turns that rounding into
other experts at 9 of 64 positions by L0).

The modules, each building on the ones before: ``corpus`` (concepts on disk, the task as the model sees it),
``stats`` (contrast statistics, the writer rule, the carrier band), ``measure`` (sweep and reader, open
generation), ``summary`` (headline numbers, terminal lines), ``run`` (``profile`` and
``write_profile``). The names the CLI, server and tests use are
re-exported here.
"""

from __future__ import annotations

from .corpus import (Concept, Item, Task, definition, list_concepts, load_concept, option_letters, prompts_for,
                     render_item, shipped_dir, texts_dir, texts_line)
from .measure import (DISTINCT_DROP, DISTINCT_WINDOW, EXPLAIN_LONG, LADDER, LADDER_TOKENS, SHORT_TOKENS,
                      choice_scores, fluency_rungs, generation, reader_layer, sweep, target_chosen)
from .run import attach_generation, profile, write_profile
from .stats import CARRIER_COS, FOLD_REPEATS, carrier_band, contrast_stats, floor_of, peak_layer, writer_layer
from .summary import (calibration_of, current_calibrations, generation_dose, steer_dose, steer_layer, summarize,
                      summary_lines)
