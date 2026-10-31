import builtins
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import importlib.util
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "embedding_st_adapter.py"
SPEC = importlib.util.spec_from_file_location("embedding_st_adapter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


class FakeBackend:
    version = "fake"

    def __init__(self, vectors=None):
        self.vectors = vectors
        self.texts = None

    def encode(self, texts):
        self.texts = texts
        return self.vectors if self.vectors is not None else [[1.0] + [0.0] * 383 for _ in texts]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        modules = [{"type": t, "path": p} for t, p in zip(
            ["sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling",
             "sentence_transformers.models.Normalize"], ["", "1_Pooling", "2_Normalize"])]
        content = {
            "modules.json": json.dumps(modules),
            "sentence_bert_config.json": json.dumps({"max_seq_length": 512, "do_lower_case": True}),
            "1_Pooling/config.json": json.dumps({"word_embedding_dimension": 384,
                                                   "pooling_mode_cls_token": True}),
        }
        for name in adapter.FILES:
            file = self.model / name
            file.parent.mkdir(exist_ok=True, parents=True)
            file.write_text(content.get(name, "fixture"), encoding="utf-8")
        self.manifest = self.root / "model.json"
        self.write_manifest()
        self.cases = self.root / "cases.jsonl"
        self.cases.write_text('{"id":"one","text":"hello"}\n{"id":"two","text":"world"}\n', encoding="utf-8")
        self.output = self.root / "vectors.jsonl"
        self.fake = FakeBackend()

    def write_manifest(self):
        hashes = {name: hashlib.sha256((self.model / name).read_bytes()).hexdigest() for name in adapter.FILES}
        self.manifest.write_text(json.dumps({"schema_version": 1, "model_id": adapter.MODEL_ID,
                                             "revision": adapter.REVISION, "files": hashes}), encoding="utf-8")

    def run_adapter(self, backend=None):
        adapter.run(self.model, self.manifest, self.cases, self.output,
                    lambda _: backend if backend is not None else self.fake)

    def test_success_produces_ordered_embedding_records_without_text(self):
        self.run_adapter()
        records = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([r["case_id"] for r in records], ["one", "two"])
        self.assertEqual(self.fake.texts, ["hello", "world"])
        self.assertEqual(records[0]["dimension"], 384)
        self.assertEqual(records[0]["model_revision"], adapter.REVISION)
        self.assertEqual(records[0]["pooling"], "cls")
        self.assertEqual(records[0]["normalization"], "l2")
        self.assertNotIn("text", records[0])
        self.assertNotIn("timing", records[0])

    def test_missing_dependency_refuses(self):
        real_import = builtins.__import__
        def failing_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=failing_import):
            with self.assertRaisesRegex(adapter.Refusal, "dependency"):
                adapter.run(self.model, self.manifest, self.cases, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_or_changed_data_refuses_without_output(self):
        (self.model / "model.safetensors").unlink()
        with self.assertRaises(OSError):
            self.run_adapter()
        self.assertFalse(self.output.exists())
        (self.model / "model.safetensors").write_text("changed")
        with self.assertRaisesRegex(adapter.Refusal, "digest mismatch"):
            self.run_adapter()
        self.assertFalse(self.output.exists())

    def test_bad_cases_refuse_without_output(self):
        for text in ('', '{"id":"a","text":"x"}\n{"id":"a","text":"y"}\n',
                     '{"id":"a","id":"b","text":"x"}\n', '{"id":"a","text":""}\n'):
            with self.subTest(text=text):
                self.cases.write_text(text)
                with self.assertRaises(adapter.Refusal):
                    self.run_adapter()
                self.assertFalse(self.output.exists())

    def test_bad_vectors_refuse_without_output(self):
        good = [1.0] + [0.0] * 383
        for vectors in ([good], [good, [1.0]], [good, [float("nan")] * 384],
                        [good, [0.0] * 384]):
            with self.subTest(vectors=vectors):
                with self.assertRaises(adapter.Refusal):
                    self.run_adapter(FakeBackend(vectors))
                self.assertFalse(self.output.exists())

    def test_bad_model_configuration_refuses_even_when_hashes_match(self):
        p = self.model / "1_Pooling/config.json"
        for pooling in ({"pooling_mode_cls_token": False},
                        {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": True}):
            with self.subTest(pooling=pooling):
                self.output.unlink(missing_ok=True)
                p.write_text(json.dumps({"word_embedding_dimension": 384, **pooling}))
                self.write_manifest()
                with self.assertRaisesRegex(adapter.Refusal, "CLS"):
                    self.run_adapter()
                self.assertFalse(self.output.exists())

    def test_existing_output_is_never_replaced(self):
        self.output.write_text("existing")
        with self.assertRaises(FileExistsError):
            self.run_adapter()
        self.assertEqual(self.output.read_text(), "existing")

    def test_real_backend_pins_cpu_and_encode_settings_and_refuses_truncation(self):
        class Transformer:
            pass
        class Pooling:
            pooling_mode_cls_token = True
        class Normalize:
            pass
        class Model:
            device = "cpu"
            max_seq_length = 512
            _modules = {"0": Transformer(), "1": Pooling(), "2": Normalize()}
            def __init__(self, directory, **kwargs):
                self.kwargs = kwargs
                self.token_count = 2
                self.encode_kwargs = None
            def get_sentence_embedding_dimension(self):
                return 384
            def tokenizer(self, text, **kwargs):
                self.tokenizer_kwargs = kwargs
                return {"input_ids": [1] * self.token_count}
            def encode(self, texts, **kwargs):
                self.encode_kwargs = kwargs
                return [[1.0] + [0.0] * 383 for _ in texts]
        module = types.SimpleNamespace(__version__="fake", SentenceTransformer=Model)
        with patch.dict("sys.modules", {"sentence_transformers": module}):
            backend = adapter.Backend(self.model)
            self.assertEqual(backend.model.kwargs["device"], "cpu")
            self.assertTrue(backend.model.kwargs["local_files_only"])
            self.assertFalse(backend.model.kwargs["trust_remote_code"])
            self.assertTrue(backend.model.kwargs["model_kwargs"]["use_safetensors"])
            backend.encode(["hello"])
            self.assertFalse(backend.model.tokenizer_kwargs["truncation"])
            self.assertEqual(backend.model.encode_kwargs["prompt"], "")
            self.assertEqual(backend.model.encode_kwargs["device"], "cpu")
            self.assertFalse(backend.model.encode_kwargs["normalize_embeddings"])
            backend.model.token_count = 513
            with self.assertRaisesRegex(adapter.Refusal, "token limit"):
                backend.encode(["too long"])

    def test_wrong_revision_refused(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["revision"] = "wrong"
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(adapter.Refusal, "identity"):
            self.run_adapter()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
