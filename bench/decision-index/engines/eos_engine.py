"""Decision Index engine wrapper for vllm-sr/Decision-2.0-Eos-0.8B.

Thin adapter over the model repository's own ``decision2`` runtime package
(``decision2.Decision2.from_pretrained`` / ``system_one``). The author's runtime verifies the
package against ``MODEL_MANIFEST.json``, renders the checkpoint's own prompt, runs one prefill per
question and normalizes the candidate logits into a distribution; this wrapper only translates the
kit request/response and calls ``system_one``. It re-implements no prompt, no read-out and no
calibration.

Load with::

    --engine eos_engine:EosEngine --option model_dir=/path/to/model
"""

import sys

from decision_index.engines.base import Engine, Unsupported

MAX_OPTIONS = 255       # decision2/_vendor/dev2model/data.py:15


class EosEngine(Engine):
    name = "decision-2.0-eos-0.8b"
    latency = "Device-synchronized in-process request wall time including the author's prompt encoding and prefill; excludes model loading and package verification."

    def __init__(self, model_dir=None, device="mps", bf16_resident=True, graphs=True,
                 kernels=True, base_path=None, threads=None, share_context=False, **options):
        super().__init__(**options)
        if not model_dir:
            raise ValueError("EosEngine needs model_dir")
        import torch

        sys.path.insert(0, str(model_dir))
        from decision2 import Decision2

        self.torch = torch
        self.device = device
        self.model = Decision2.from_pretrained(
            model_dir,
            device=device,
            base_path=base_path,
            threads=threads,
            bf16_resident=bf16_resident,
            graphs=graphs,
            kernels=kernels,
            share_context=share_context,
        )
        self.model_id = self.model.model_name
        self.max_input_tokens = self.model.max_input_tokens
        self.provenance = {
            "kind": "author-inference",
            "repo": "vllm-sr/Decision-2.0-Eos-0.8B",
            "entry": "decision2/api.py:Decision2.system_one",
            "device": device,
            "bf16_resident": bf16_resident,
            "graphs": graphs,
            "kernels": kernels,
            "share_context": share_context,
            "context_limit_tokens": self.max_input_tokens,
            "max_options": MAX_OPTIONS,
            "policy": "state and questions passed unchanged to system_one(); the author's own package verification, prompt, read-out and per-kind temperatures; over-budget questions refused as unsupported, never truncated.",
        }

    def runtime(self):
        return {"torch": self.torch.__version__, "device": self.device}

    def synchronize(self):
        if str(self.device).startswith("mps"):
            self.torch.mps.synchronize()

    def __call__(self, state, questions):
        for key, question in questions.items():
            if question["type"] not in ("choice", "noul"):
                raise Unsupported("Unsupported question type " + str(question["type"]))
            options = question.get("criteria") or {}
            if len(options) > MAX_OPTIONS:
                raise Unsupported(f"question {key!r} has {len(options)} options; the author's runtime accepts at most {MAX_OPTIONS} (data.py:15)")

        result = self.model.system_one(state=state, questions={k: dict(q) for k, q in questions.items()})

        answers = {}
        for key, question in questions.items():
            answer = result["answers"][key]
            if answer.get("error"):
                if answer["error"] == "max_length_exceeded":
                    raise Unsupported(f"question {key!r} exceeds the {self.max_input_tokens}-token input limit (qwen.py:337-349)")
                raise ValueError(f"question {key!r}: {answer['error']}")
            if question["type"] == "choice":
                probabilities = answer["probabilities"]
                choice = answer.get("choice") or max(probabilities, key=probabilities.get)
                answers[key] = {"type": "choice", "choice": choice, "probabilities": probabilities}
            else:
                answers[key] = {"type": "noul", "noul": answer["noul"]}
        response = {"model": result.get("model", self.model_id), "answers": answers,
                    "usage": result.get("usage", {})}
        return response, result
