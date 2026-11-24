# Measurements

What was measured: Inkling 2's lenses, steering and word edit, what metalscope costs, the jobs' times and lens fits.

## Inkling 2

Inkling 2 (oMLX 0.8.3 on) runs its middle layers three times a token and has no single residual stream: its layers
here are its layer visits (48 on the Mini: layers 0 to 5, then 6 to 17 three times, then 18 to 23). The logit lens
reads the head's own mix at each visit: what the final norm and head would read had the model stopped there. The
Jacobian lens reads the state at each visit: the finished blocks, the running sum, and in a later pass the previous
pass's messages; not the first block, which is the token's own embedding and reads as the token just read. A lens fitted on 8 windows of the default text (13 to 17 min a window)
had the model's own next token in its top 10 at 0.31 of held-out positions against the logit lens's 0.26, the mean
of 47 visits (as the top word 0.14 against 0.12; at the 7 visits the job scores, 0.41 against 0.31; fitted on to 16
windows, 0.44). Which reads
better depends on the visit: the Jacobian lens through the second pass and the start of the third (0.41 against 0.15
in the second), the logit lens at the first 6 visits and in the last visits of each pass, where it climbs steeply.
Inside a pass the Jacobian lens changes little from visit to visit: most of its state is finished blocks and the
previous pass's messages, which stay as they are until the next block.

Concepts steer whole replies on Inkling 2, thinking included (`/concept learn`, `/steer NAME`). A concept's direction
is learned on each visit's running sum, and a push at a visit spans the rest of its block of 6 visits, each visit with
its own direction and dose. A dose is a multiple of the running sum's norm. The steer layer is one of two, picked per
concept by `concept learn`: the model's first visit, which every pass restarts from, or the first visit of the last
pass (30 on the Mini). Measured offline (replies sampled as the model's config asks): a voice
(Shakespearean) takes at the start at doses 0.2 to 0.3 with the task kept, and loses the answer when pushed late; a
topic or a mood (fluffy, joyful, oceans) takes late, over visits 30 to 35, at 0.15 to 0.6, and breaks replies at the
start from 0.3. A random direction of the same size does nothing at 0.3 in either place; the middle passes and the
later blocks do nothing, lose the answer or loop.

One word of a reply can be changed on Inkling 2 (the dock's LENS edit), as on the other models. The edit acts on
the running sum alone, over the last block's visits (42 to 46 on the Mini),
with the word's lens direction cut to the sum's part of the state. Measured on 80 next words (20 one-word answers, 60
words inside replies): removing the model's word there changed it in 74% (96% of the 52 it was less than
0.95 sure of), with 4.0 of its next 5 candidates still in the new top 5, and removing another word's direction
changed it in 3%. Over visits 18 to 44, the band the other models use, 62%; out of the whole state instead of the sum,
85% but with 3.1 of 5 kept, and junk tokens on the one-word answers. A swap (the new word pushed in at visit 44, the dose a
multiple of the running sum's norm there) made the new word the top one in 100% at the dock's dose of 0.7, with 2.3
of the model's next 5 candidates kept (94% and 3.4 kept at 0.35). Held on every token of a reply the same push has no dose
that works: nothing up to 0.25 (0 of 5 replies took the word), and at 0.35 one reply took it, two looped and two
ignored it: a word changes one token, on every model.

## What it costs

What it costs, from `metalscope bench` (an M5 Max with 128 GB; a 300-token reply, the median of 3, each
setting after a minute's rest; tokens a second). Tracing reads the lenses at one of five steps of layers, and that
is most of its cost (`-` `+` on LENS in the dock): lightest is the fewest layers, the default the middle step,
fullest every layer the lens has.

| | tracing off | on, lenses off | on, lightest | on, default | on, fullest |
|---|---|---|---|---|---|
| Qwen3.5 2B, oMLX | 288 | 264 | 209 | 218 | 183 |
| Qwen3.6 35B, oMLX | 205 | 192 | 169 | 158 | 112 |
| Qwen3.8 27B (MTP), oMLX | 57 | 58 | 54 | 48 | 40 |
| Flash Next, oMLX | 44 | 41 | 34 | 21 | 11 |
| Inkling 2 Mini (MTP), oMLX | 74 | 64 | 53 | 28 | 11 |
| Flash Next, ds4 | 69 | 63 | 55 | 40 | 18 |
| DeepSeek V4.1 Flash, ds4 | 16 | 16 | 16 | 15 | 14 |

Tracing off measured again after the lens steps was within 6% of the first on every model. The oMLX rows were
measured on oMLX 0.8.3 (on 0.8.0 the 2B, the 35B and Flash Next wrote the same within 4%, the 27B 54, 54,
51, 44 and 38). Inkling 2 Mini has a Jacobian lens fitted on it (none ships), which reads each visit's whole state;
with no Jacobian lens the same steps wrote 53, 49 and 42 on oMLX 0.8.2 (66 with tracing off). On the 2B the lightest
step writes slower than the default, the same in three runs: not explained. These are speeds from rest: under
sustained load the Mac slows itself, traced or not (Flash Next on ds4, untraced, fell from 69 to 42 tokens a second
over 2.5 minutes of back-to-back replies and was back after 90 s idle), so a long agent run writes slower than this.

The two things done on request, in seconds: one word of a 300-token reply changed (until the new word shows), and
the engram inspection of a reply after about 1,000, 4,000 and 16,000 tokens of conversation.

| | one word changed | engram inspection |
|---|---|---|
| Qwen3.5 2B, oMLX | 0.1 | |
| Qwen3.6 35B, oMLX | 0.6 | |
| Qwen3.8 27B (MTP), oMLX | 0.6 (a concept at one token: no Jacobian lens on it) | |
| Inkling 2 Mini (MTP), oMLX | 0.4 | |
| Flash Next, oMLX | 0.7 | 8, 29, then refused (oMLX's memory guard, at 107 GB) |
| Flash Next, ds4 | 0.8 | 2, 6, 25 |
| DeepSeek V4.1 Flash, ds4 | 17 (a concept at one token; the model is read from disk) | 40, 77, 108 (a token at a time) |

## The jobs

The jobs, as run on a fresh install on that Mac: one concept learned with its calibration, and a
Jacobian lens fitted. Flash Next under oMLX leaves little memory free; its jobs fit because a job's own requests are
not traced.

| | Qwen3.6 35B, oMLX | Qwen3.8 27B (MTP), oMLX | Flash Next, oMLX | Flash Next, ds4 |
|---|---|---|---|---|
| `concept learn` with its calibration | 25 to 43 min | 33 to 68 min | under 25 min, then 51 min to calibrate | 1 h 18 min to 2 h 28 min |
| `jlens learn`, 4 windows | 2 h 37 min | not run | not run | needs oMLX |

## Jacobian lens fits

More windows stop helping: the held-out score of Flash Next's lens rose to 0.27 by its 12th window and was
0.28 at its 42nd; Inkling 2 Mini's was 0.41 at 8 windows and 0.44 at 16, the last four within 0.01 of each other.
