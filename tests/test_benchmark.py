from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "dmpod" / "lib"))

from dmpod_benchmark import NanoGPTBackend, macro_f1, parse_benchmarks, polish_example


class FakeTokenizer:
    name = "test-tokenizer"

    def __init__(self, encoded: dict[str, list[int]]) -> None:
        self.encoded = encoded

    def encode(self, text: str) -> list[int]:
        return self.encoded[text]


class BenchmarkTests(unittest.TestCase):
    def test_score_pairs_aligns_padded_and_truncated_continuations(self) -> None:
        backend = NanoGPTBackend.__new__(NanoGPTBackend)
        backend.device = torch.device("cpu")
        backend.precision = torch.float32
        backend.batch_size = 2
        backend.block_size = 3
        backend.vocab_size = 8
        backend.eot_token_id = 0
        backend.truncated_requests = 0
        backend.tokenizer = FakeTokenizer(
            {
                "A": [1],
                "ABC": [1, 2, 3],
                "D": [4],
                "DE": [4, 5],
                "BC": [2, 3],
                "ABCDE": [1, 2, 3, 4, 5],
                " A": [1],
            }
        )

        def full_logits(input_ids: torch.Tensor) -> torch.Tensor:
            logits = torch.full((*input_ids.shape, 8), -5.0)
            logits.scatter_(2, ((input_ids + 1) % 8).unsqueeze(-1), 5.0)
            return logits

        backend._full_logits = full_logits
        progress = mock.Mock(tqdm=lambda values, desc: values)
        with mock.patch.dict(sys.modules, {"tqdm": progress}):
            scores = backend.score_pairs(
                [
                    ("A", "BC"),
                    ("D", "E"),
                    ("", "A"),
                    (" ", "A"),
                    ("ABC", "DE"),
                ],
                "test likelihoods",
            )

        token_score = 5.0 - math.log(math.exp(5.0) + 7 * math.exp(-5.0))
        self.assertEqual([greedy for _, greedy, _ in scores], [True] * 5)
        self.assertEqual([length for _, _, length in scores], [2, 1, 1, 1, 2])
        for result, expected_tokens in zip(scores, [2, 1, 1, 1, 2], strict=True):
            self.assertAlmostEqual(result[0], token_score * expected_tokens, places=5)
        self.assertEqual(backend.truncated_requests, 1)

    def test_selection_aliases_and_order(self) -> None:
        self.assertEqual(
            parse_benchmarks(["polish", "blimp"]),
            ["blimp", "8tags", "polemo2-in", "polemo2-out"],
        )
        with self.assertRaisesRegex(ValueError, "Unknown benchmark"):
            parse_benchmarks(["unknown"])

    def test_macro_f1(self) -> None:
        self.assertAlmostEqual(macro_f1([0, 1, 1, 2], [0, 1, 2, 2], 3), 7 / 9)

    def test_polish_protocol_maps_dataset_labels(self) -> None:
        prompt, choices, target = polish_example(
            "8tags", {"sentence": "Przykladowy tekst", "label": "4"}
        )
        self.assertIn("Kategoria:", prompt)
        self.assertEqual(choices[target], "motoryzacja")

        prompt, choices, target = polish_example(
            "polemo2-in",
            {
                "sentence": "Przykladowa recenzja",
                "target": "__label__meta_plus_m",
            },
        )
        self.assertIn("Wydźwięk:", prompt)
        self.assertEqual(choices[target], "pozytywny")


if __name__ == "__main__":
    unittest.main()
