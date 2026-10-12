"""Decision Index engine wrapper for internlm/Intern-Decision-0.8B.

Thin adapter over the model repository's own ``inference.py`` (``DecisionEngine``). The author's
module compiles the request into one masked assistant skeleton, runs one causal forward pass and
reads the logits immediately before each ``<decision>`` marker; this wrapper only translates the
kit request/response and calls ``DecisionEngine.predict``. It re-implements no prompt, no read-out
and no calibration.

Load with::

    --engine intern_decision_engine:InternDecisionEngine --option model_dir=/path/to/model
"""

import copy
import string
import sys

from decision_index.engines.base import Engine, Unsupported

STOCK_QUESTIONS = 16    # inference.py:115  "Supply 1-16 questions."
MAX_LENGTH = 262144     # the board's lift of inference.py:191,253 (default 8192) to the backbone window


class InternDecisionEngine(Engine):
    name = "intern-decision-0.8b"
    latency = "Device-synchronized in-process request wall time including the author's prompt compilation and one forward pass; excludes model loading."

    def __init__(self, model_dir=None, device="mps", dtype="bfloat16", max_length=MAX_LENGTH,
                 temperature=None, attn_implementation="sdpa", media_root=".", board_lifts=True, **options):
        super().__init__(**options)
        if not model_dir:
            raise ValueError("InternDecisionEngine needs model_dir")
        import torch

        sys.path.insert(0, str(model_dir))
        import inference
        from inference import DEFAULT_TEMPERATURE, DecisionEngine

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.engine = DecisionEngine(
            checkpoint=str(model_dir),
            temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
            max_length=max_length,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            media_root=media_root,
        )
        self.board_lifts = bool(board_lifts)
        self.stock_symbols = inference.ANSWER_SYMBOLS
        self.symbols = tuple(inference.ANSWER_SYMBOLS)
        if self.board_lifts:
            # The board's documented lifts of two serving guards (methodology notes.lifts, intern-decision-0.8b):
            # the 1-16 questions check runs over chunks of 16 (whole row still one skeleton, one forward), and the
            # 62-symbol alphabet is extended with the two-letter labels AA-ZZ that encode to one token.
            tok = self.engine.tokenizer
            doubles = [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
            doubles = [d for d in doubles if len(tok.encode(d, add_special_tokens=False)) == 1]
            self.symbols = tuple(inference.ANSWER_SYMBOLS) + tuple(doubles)
            inference.ANSWER_SYMBOLS = self.symbols
            stock_validate = inference.validate_request

            def validate_in_chunks(request):
                qs = request.get("questions") if isinstance(request, dict) else None
                if not isinstance(qs, dict) or len(qs) <= STOCK_QUESTIONS:
                    return stock_validate(request)
                items, clean = list(qs.items()), {}
                for i in range(0, len(items), STOCK_QUESTIONS):
                    clean.update(stock_validate({**request, "questions": dict(items[i:i + STOCK_QUESTIONS])})["questions"])
                row = {"state": copy.deepcopy(request["state"]), "questions": clean}
                if request.get("images"):
                    row["images"] = list(request["images"])
                inference.compile_row(row)
                return row

            inference.validate_request = validate_in_chunks
        self.model_id = "Intern-Decision-0.8B"
        self.provenance = {
            "kind": "author-inference",
            "repo": "internlm/Intern-Decision-0.8B",
            "entry": "inference.py:DecisionEngine.predict",
            "device": device,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "context_limit_tokens": max_length,
            "board_lifts": self.board_lifts,
            "answer_symbols": len(self.symbols),
            "policy": "state and questions passed unchanged to the author's predict(); author's own temperature calibration; the board's documented lifts (max_length 262144, 16-question check over chunks, two-letter one-token labels) unless board_lifts=False; remaining limits refused as unsupported, never truncated.",
        }

    def runtime(self):
        return {"torch": self.torch.__version__, "device": self.device}

    def synchronize(self):
        if self.device == "mps":
            self.torch.mps.synchronize()

    def __call__(self, state, questions):
        if not self.board_lifts and len(questions) > STOCK_QUESTIONS:
            raise Unsupported(f"{len(questions)} questions; the author's engine accepts at most {STOCK_QUESTIONS} (inference.py:115)")
        for key, question in questions.items():
            if question["type"] not in ("choice", "noul"):
                raise Unsupported("Unsupported question type " + str(question["type"]))
            options = question.get("criteria") or {}
            if question["type"] == "choice" and len(options) > len(self.symbols):
                raise Unsupported(f"question {key!r} has {len(options)} options; {len(self.symbols)} one-token answer labels (inference.py:87)")

        request = {"state": state, "questions": {k: dict(q) for k, q in questions.items()}}
        try:
            result = self.engine.predict(request)
        except ValueError as exc:
            if "truncation is forbidden" in str(exc):
                raise Unsupported(str(exc)) from exc
            raise

        answers = {}
        for key, question in questions.items():
            answer = result["answers"][key]
            if question["type"] == "choice":
                answers[key] = {"type": "choice", "choice": answer["choice"], "probabilities": answer["probabilities"]}
            else:
                answers[key] = {"type": "noul", "noul": answer["noul"]}
        response = {"model": result.get("model", self.model_id), "answers": answers,
                    "usage": result.get("usage", {})}
        return response, result
