"""Run a decider checkpoint's own /v1/systemone server on Apple silicon (MPS).

The published server (decider/serve.py + decider/engine.py) targets CUDA: CUDA graphs, torch.compile and fp8
_scaled_mm. This launcher changes only the device plumbing and leaves prompt construction, batching, slot read-out,
temperature and the wire format as published:
  - Engine runs on device "mps" with use_graphs=False (no CUDA graphs on MPS).
  - compile and fp8 are off (DECIDER_COMPILE=0, DECIDER_FP8=0, set by the caller); conv_patch follows compile, so the
    stock transformers causal conv path runs.
  - warmup runs a few eager forwards and synchronizes MPS instead of capturing 72 CUDA graphs.
Usage: python -I serve_mps.py <model_dir> <port>   (DECIDER_MODEL is set from model_dir)
"""
import os, sys, time

model_dir, port = sys.argv[1], int(sys.argv[2])
os.environ["DECIDER_MODEL"] = model_dir
sys.path.insert(0, model_dir)

import torch
import decider.engine as E

_orig_init = E.Engine.__init__


def _init(self, path, device="mps", dtype=torch.bfloat16, use_graphs=True, max_ctx_tokens=1536, compile=True, fp8=False, conv_patch=True):
    _orig_init(self, path, device="mps", dtype=dtype, use_graphs=False, max_ctx_tokens=max_ctx_tokens, compile=compile, fp8=fp8,
               conv_patch=conv_patch)


def _warmup(self, shapes=None):
    t = time.time()
    for B, T in ((1, 128), (1, 512), (4, 256)):
        self.logits_all(torch.full((B, T), self.tok.pad_token_id, dtype=torch.long, device=self.dev))
    torch.mps.synchronize()
    return time.time() - t


E.Engine.__init__ = _init
E.Engine.warmup = _warmup

import uvicorn
import decider.serve as serve

print(f"[serve_mps] torch {torch.__version__} mps={torch.backends.mps.is_available()} model={model_dir} "
      f"compile={serve.COMPILE} fp8={serve.FP8}", flush=True)
uvicorn.run(serve.app, host="127.0.0.1", port=port, log_level="warning")
