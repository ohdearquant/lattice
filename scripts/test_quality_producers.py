import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import quality_hf_dump as q
from quality_compare import compare


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"schema_version": 1, "max_new_tokens": 2, "cases": [{"id": "a", "prompt": "hello"}]}

    def dump(self, **kwargs):
        args = {"tokenize": lambda _: [4, 5], "generate": lambda ids, n: [6, 7], "eos_control": lambda: True}
        args.update(kwargs)
        return q.produce(self.manifest, "hf", **args)

    def test_produces_comparable_dump(self):
        dump = self.dump()
        self.assertEqual(dump["cases"][0]["prompt_ids"], [4, 5])
        self.assertEqual(compare(dump, dump)["cases"][0]["divergence_kind"], "identical")

    def test_rejects_failed_eos_control_before_generation(self):
        def fail(*_):
            self.fail("generation should not run")
        with self.assertRaisesRegex(ValueError, "EOS"):
            self.dump(eos_control=lambda: False, generate=fail)

    def test_rejects_incomplete_output_and_empty_tokens(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.dump(generate=lambda ids, n: [6])
        with self.assertRaisesRegex(ValueError, "prompt"):
            self.dump(tokenize=lambda _: [])

    def test_rejects_duplicate_and_unknown_fields(self):
        self.manifest["cases"].append({"id":"a", "prompt":"different"})
        with self.assertRaises(ValueError):
            self.dump()
        self.manifest["cases"].pop()
        self.manifest["extra"] = True
        with self.assertRaises(ValueError):
            self.dump()

    def test_atomic_publish_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "result.json"
            dump = self.dump()
            q.publish(path, dump)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                q.publish(path, dump)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(list(Path(d).iterdir())), 1)

    def test_edited_protocol_file_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(q.__file__).resolve().parent.parent / "crates/inference/quality_protocol_v1.json"
            path = Path(d) / "protocol.json"
            protocol = json.loads(source.read_text(encoding="utf-8"))
            protocol["temperature"] = 0.5
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "effective settings differ"):
                self.dump(protocol_path=path)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.json"
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                q.load_json(path)

    def test_hf_real_model_does_not_inherit_generation_constraints(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
            self.skipTest("torch and transformers are required")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=32, n_positions=32, n_ctx=32, n_embd=16, n_layer=1, n_head=1,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )).cpu().eval()
        model.generation_config.eos_token_id = 2
        model.generation_config.suppress_tokens = [5]
        model.generation_config.forced_eos_token_id = 2

        class Tokenizer:
            def __call__(self, prompt, *, add_special_tokens, truncation):
                assert not add_special_tokens and not truncation
                return {"input_ids": [1, 3]}

        observed = []
        original_logits = model._get_logits_processor
        original_stops = model._get_stopping_criteria

        def logits_observer(*args, **kwargs):
            config = kwargs["generation_config"]
            processors = original_logits(*args, **kwargs)
            observed.append(("logits", config.eos_token_id, config.suppress_tokens, config.forced_eos_token_id,
                             [type(item).__name__ for item in processors]))
            return processors

        def stops_observer(*args, **kwargs):
            config = kwargs["generation_config"]
            criteria = original_stops(*args, **kwargs)
            observed.append(("stops", config.eos_token_id, config.suppress_tokens, config.forced_eos_token_id,
                             [type(item).__name__ for item in criteria]))
            return criteria

        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            output = Path(directory) / "output.json"
            manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
            with patch.object(AutoTokenizer, "from_pretrained", return_value=Tokenizer()), \
                 patch.object(AutoModelForCausalLM, "from_pretrained", return_value=model), \
                 patch.object(model, "_get_logits_processor", side_effect=logits_observer), \
                 patch.object(model, "_get_stopping_criteria", side_effect=stops_observer):
                q.main(["quality_hf_dump.py", "local-model", str(manifest), str(output), "hf"])
            dump = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(dump["cases"][0]["prompt_ids"], [1, 3])
        self.assertEqual(len(dump["cases"][0]["generated_ids"]), 2)
        self.assertEqual([item[0] for item in observed], ["logits", "stops", "logits", "stops"])
        for _kind, eos, suppressed, forced_eos, components in observed:
            self.assertIsNone(eos)
            self.assertIsNone(suppressed)
            self.assertIsNone(forced_eos)
            self.assertNotIn("SuppressTokensLogitsProcessor", components)
            self.assertNotIn("ForcedEOSTokenLogitsProcessor", components)
            self.assertNotIn("EosTokenCriteria", components)
        self.assertIn("ForceEos", observed[0][4])
        self.assertNotIn("ForceEos", observed[2][4])
        self.assertIn("MaxLengthCriteria", observed[1][4])
        self.assertIn("MaxLengthCriteria", observed[3][4])


if __name__ == "__main__":
    unittest.main()
