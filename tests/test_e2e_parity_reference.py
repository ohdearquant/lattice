import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/e2e_parity_check.py"
SPEC = importlib.util.spec_from_file_location("e2e_parity_check", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)


def output(tokens, margins):
    return {
        "generated_ids": tokens,
        "logit_margins": margins,
        "text": "",
        "elapsed_s": 1.0,
        "tok_per_sec": len(tokens),
    }


class RegenerationDeterminismTest(unittest.TestCase):
    def test_accepts_matching_thread_outputs_and_finds_global_minimum(self):
        one = {
            "first": output([1, 2], [4.0, 2.0]),
            "second": output([3, 4], [3.0, 1.5]),
        }
        four = {
            "first": output([1, 2], [3.5, 2.5]),
            "second": output([3, 4], [2.5, 1.75]),
        }

        result = PARITY.validate_regeneration_outputs(
            {1: one, 4: four}, ["first", "second"], 1.0
        )

        self.assertEqual(result["global_minimum_margin"], 1.5)
        self.assertEqual(result["global_minimum_prompt"], "second")
        self.assertEqual(result["global_minimum_position"], 1)

    def test_names_prompt_and_position_on_token_disagreement(self):
        one = {"prompt": output([10, 20, 30], [3.0, 2.0, 1.0])}
        four = {"prompt": output([10, 21, 30], [3.0, 2.0, 1.0])}

        with self.assertRaisesRegex(
            RuntimeError, "prompt 'prompt'.*token position 1.*20.*21"
        ):
            PARITY.validate_regeneration_outputs(
                {1: one, 4: four}, ["prompt"], 0.5
            )

    def test_rejects_non_finite_or_below_floor_margin(self):
        for margin, message in (
            (math.nan, "non-finite"),
            (math.inf, "non-finite"),
            (0.25, "below refusal floor"),
        ):
            with self.subTest(margin=margin):
                runs = {
                    1: {"prompt": output([10], [margin])},
                    4: {"prompt": output([10], [1.0])},
                }
                with self.assertRaisesRegex(RuntimeError, message):
                    PARITY.validate_regeneration_outputs(
                        runs, ["prompt"], 0.5
                    )


if __name__ == "__main__":
    unittest.main()
