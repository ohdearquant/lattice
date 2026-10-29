import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
