"""Convert a reference jlens .pt file to metalscope's safetensors lens format.

Needs torch, so run it with a PyTorch environment, not oMLX's Python:

    python scripts/convert_lens.py LENS.pt ~/.metalscope/models/Qwen3.5-2B-4bit/lenses/NAME --hf-model Qwen/Qwen3.5-2B \
        --source "neuronpedia/jacobian-lens qwen3.5-2b" --checkpoint Qwen3.5-2B-4bit

then ``metalscope jlens use SERVED NAME`` (a checkpoint with only this lens reads it without).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

ap = argparse.ArgumentParser()
ap.add_argument("pt")
ap.add_argument("out")
ap.add_argument("--hf-model", required=True)
ap.add_argument("--source", default="")
ap.add_argument("--checkpoint", required=True, help="the checkpoint the lens is for (metalscope model list)")
args = ap.parse_args()

ckpt = torch.load(args.pt, map_location="cpu", weights_only=True)
J = {int(l): t.to(torch.float16).numpy() for l, t in ckpt["J"].items()}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
save_file({f"J.{l}": np.ascontiguousarray(m) for l, m in J.items()}, str(out / "lens.safetensors"))
meta = dict(source=args.source, hf_model=args.hf_model, n_prompts=int(ckpt["n_prompts"]),
            d_model=int(ckpt["d_model"]), layers=sorted(J), checkpoint=args.checkpoint)
(out / "lens.json").write_text(json.dumps(meta, indent=1))
print(f"{len(J)} layers, d_model {meta['d_model']}, n_prompts {meta['n_prompts']} -> {out}")
