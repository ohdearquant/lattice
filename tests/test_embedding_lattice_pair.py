import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import embedding_lattice_pair as pair


class Backend:
    version = "fixture"

    def __init__(self, _directory=None):
        self.model = types.SimpleNamespace(tokenizer=lambda text, **_kwargs: {
            "input_ids": [101, len(text), 102], "attention_mask": [1, 1, 1]})


class Producer:
    def __init__(self, _executable=None, _directory=None):
        self.calls = []
        self.finished = False
        self.aborted = False
        self.overrides = {}

    def encode(self, texts):
        self.calls.append(list(texts))
        rows = [self.overrides.get(text, {}) for text in texts]
        return {"ids": [row.get("ids", [101, len(text), 102]) for row, text in zip(rows, texts)],
                "vectors": [row.get("vector", [1.0] + [0.0] * 383) for row in rows]}

    def finish(self):
        self.finished = True

    def abort(self):
        self.aborted = True


class PairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = [{"id": "one", "text": "one"}, {"id": "two", "text": "second"}]
        self.hashes = {"model.safetensors": "a" * 64}
        self.backend = Backend()
        self.producer = Producer()
        self.refs = pair.ref.produce(self.cases, self.hashes, types.SimpleNamespace(
            version="fixture", encode=lambda texts: [[1.0] + [0.0] * 383 for _ in texts]))
        self.output = self.root / "out.jsonl"
        self.source = self.root / "ref.jsonl"
        self.source.write_text("".join(json.dumps(row) + "\n" for row in self.refs))
        self.case_path = self.root / "cases.jsonl"
        self.case_path.write_text("".join(json.dumps(row) + "\n" for row in self.cases))
        self.revision = "b" * 40

    def run_pair(self, size=1):
        with patch.object(pair.ref, "verify_snapshot", return_value=self.hashes):
            pair.run(self.root, self.root / "manifest", self.case_path, self.source,
                     self.root / "producer", self.output, self.revision, size,
                     lambda _: self.backend, lambda *_: self.producer)

    def test_order_batch_long_lived_and_output(self):
        self.run_pair()
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(self.producer.calls, [["one"], ["second"]])
        self.assertTrue(self.producer.finished)
        self.assertEqual([row["case_id"] for row in rows], ["one", "two"])
        self.assertEqual(rows[0]["engine_version"], self.revision)
        self.assertIn("not captured", rows[0]["token_preflight"])
        self.assertNotIn("text", rows[0])

    def test_reference_identity_shape_and_case_mismatch_refuse(self):
        for key, value in (("model_revision", "wrong"), ("case_id", "other"),
                           ("text_sha256", "wrong"), ("engine_version", "other"),
                           ("files_sha256", {}), ("embedding", [1.0])):
            with self.subTest(key=key):
                changed = [dict(row) for row in self.refs]
                changed[0][key] = value
                self.source.write_text("".join(json.dumps(row) + "\n" for row in changed))
                with self.assertRaises(pair.ref.Refusal):
                    self.run_pair()
                self.assertFalse(self.output.exists())

    def assert_producer_refusal(self, override, message):
        self.producer.overrides = {"one": override}
        with self.assertRaisesRegex(pair.ref.Refusal, message):
            self.run_pair()
        self.assertFalse(self.output.exists())
        self.assertTrue(self.producer.aborted)

    def test_wrong_ids_alone_refuse_with_tokenization_error(self):
        self.assert_producer_refusal({"ids": [101, 4, 102]}, "tokenization preflight mismatch")

    def test_orthogonal_unit_embedding_refuses_with_parity_error(self):
        self.assert_producer_refusal(
            {"vector": [0.0, 1.0] + [0.0] * 382}, "embedding parity failed for one")

    def test_cosine_threshold_refuses_near_boundary(self):
        delta = 0.0048
        cosine = math.sqrt(1.0 - delta * delta)
        self.assertLess(cosine, 0.99999)
        self.assertLess(delta, 0.005)
        self.assert_producer_refusal(
            {"vector": [cosine, delta] + [0.0] * 382}, "embedding parity failed for one")

    def test_invalid_embedding_shapes_and_values_refuse(self):
        for value, message in (([1.0], "embedding dimension mismatch"),
                               ([math.nan] * 384, "invalid embedding value")):
            with self.subTest(message=message):
                self.assert_producer_refusal({"vector": value}, message)

    def install_distinct_batch_cases(self):
        self.cases = [{"id": "one", "text": "one"}, {"id": "two", "text": "second"},
                      {"id": "three", "text": "third"}]
        vectors = {text: [1.0 if i == index else 0.0 for i in range(384)]
                   for index, text in enumerate(case["text"] for case in self.cases)}
        self.refs = pair.ref.produce(self.cases, self.hashes, types.SimpleNamespace(
            version="fixture", encode=lambda texts: [vectors[text] for text in texts]))
        self.producer.overrides = {text: {"vector": vector} for text, vector in vectors.items()}
        self.source.write_text("".join(json.dumps(row) + "\n" for row in self.refs))
        self.case_path.write_text("".join(json.dumps(row) + "\n" for row in self.cases))
        return vectors

    def test_distinct_vectors_remain_aligned_across_full_and_partial_batches(self):
        vectors = self.install_distinct_batch_cases()
        self.run_pair(size=2)
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(self.producer.calls, [["one", "second"], ["third"]])
        self.assertEqual([row["case_id"] for row in rows], ["one", "two", "three"])
        for case, row in zip(self.cases, rows):
            self.assertEqual(row["embedding"], vectors[case["text"]])
            self.assertAlmostEqual(row["reference_cosine"], 1.0)
            self.assertEqual(row["reference_max_abs"], 0.0)

    def test_swapped_batch_vectors_refuse_for_first_case(self):
        vectors = self.install_distinct_batch_cases()
        self.producer.overrides["one"]["vector"] = vectors["second"]
        self.producer.overrides["second"]["vector"] = vectors["one"]
        with self.assertRaisesRegex(pair.ref.Refusal, "embedding parity failed for one"):
            self.run_pair(size=2)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.producer.aborted)

    def test_oversized_serialized_request_is_not_sent(self):
        producer = object.__new__(pair.Producer)
        producer.process = types.SimpleNamespace(stdin=types.SimpleNamespace(
            write=lambda _text: self.fail("oversized batch was sent")))
        with self.assertRaisesRegex(pair.ref.Refusal, "serialized batch exceeds producer request limit"):
            producer.encode(["\x01" * 65_536] * 6)

    def test_truncation_and_invalid_batch_refuse(self):
        self.backend.model.tokenizer = lambda *_args, **_kwargs: {"input_ids": [1] * 513, "attention_mask": [1] * 513}
        with self.assertRaises(pair.ref.Refusal):
            self.run_pair()
        self.assertFalse(self.output.exists())
        with self.assertRaises(pair.ref.Refusal):
            self.run_pair(0)

    def test_existing_output_is_never_replaced(self):
        self.output.write_text("original")
        with self.assertRaises(FileExistsError):
            self.run_pair()
        self.assertEqual(self.output.read_text(), "original")

    def test_snapshot_changed_after_encoding_refuses_publication(self):
        with patch.object(pair.ref, "verify_snapshot", side_effect=[self.hashes, {}]):
            with self.assertRaises(pair.ref.Refusal):
                pair.run(self.root, self.root / "manifest", self.case_path, self.source,
                         self.root / "producer", self.output, self.revision, 1,
                         lambda _: self.backend, lambda *_: self.producer)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
