"""Unit tests for sed.py."""
import logging
import unittest

import numpy as np

from trans.actions import Sub, Ins
from trans import sed
from trans import test_optimal_expert_substitutions


class TestTransducer(unittest.TestCase):
    def setUp(self) -> None:

        self.source_alphabet1 = list("abcdefg")
        self.target_alphabet1 = list("fghijk")

        self.smart_sed = sed.StochasticEditDistance.build_sed(
            self.source_alphabet1, self.target_alphabet1
        )

    def test_sed_random_initialization(self):

        sed_ = sed.StochasticEditDistance.build_sed(
            self.source_alphabet1, self.target_alphabet1, copy_probability=None
        )
        eos_weight = sed_.delta_eos

        for weight_dict in ("delta_del", "delta_ins", "delta_sub"):
            for weight in getattr(sed_, weight_dict).values():
                self.assertTrue(np.isclose(eos_weight, weight))

    def test_sed_copy_biased_initialization(self):

        sed_ = sed.StochasticEditDistance.build_sed(
            self.source_alphabet1, self.target_alphabet1
        )
        eos_weight = sed_.delta_eos

        for weight_dict in ("delta_del", "delta_ins"):
            for weight in getattr(sed_, weight_dict).values():
                self.assertTrue(np.isclose(eos_weight, weight))

        for (x, y), weight in sed_.delta_sub.items():
            if x == y:
                self.assertFalse(np.isclose(eos_weight, weight))
            else:
                self.assertTrue(np.isclose(eos_weight, weight))

    def test_viterbi_decoding(self):

        best_edits, distance = self.smart_sed.viterbi_distance(
            source="affa", target="iffig", with_alignment=True
        )

        expected_edits = [
            Sub(old="a", new="i"),
            Sub(old="f", new="f"),
            Sub(old="f", new="f"),
            Ins(new="i"),
            Sub(old="a", new="g"),
        ]

        self.assertTrue(np.isclose(-26.7633, distance))
        self.assertListEqual(expected_edits, best_edits)

    def test_stochastic_decoding(self):

        distance = self.smart_sed.stochastic_distance(
            source="affa", target="iffig"
        )

        self.assertTrue(np.isclose(-26.05741, distance))

    def test_em(self):

        input_pairs = [
            ("abby", "a b i"),
            ("abidjan", "a b i d ʒ ɑ"),
            ("abject", "a b ʒ ɛ k t"),
            ("abolir", "a b ɔ l i ʁ"),
            ("abonnement", "a b ɔ n m ɑ"),
        ]

        sources, targets = zip(*input_pairs)

        source_alphabet = {c for source in sources for c in source}
        target_alphabet = {c for target in targets for c in target}

        sed_ = sed.StochasticEditDistance.build_sed(
            source_alphabet, target_alphabet
        )

        o = sed_.stochastic_distance(sources[1], targets[1])
        logging.info(o)

        before_ll = sed_.log_likelihood(sources, targets)
        sed_.update_model(sources, targets, iterations=1)
        after_ll = sed_.log_likelihood(sources, targets)
        self.assertTrue(before_ll <= after_ll)

    def test_fit_from_data(self):

        input_lines = [
            "abby\ta b i",
            "abidjan\ta b i d ʒ ɑ",
            "abject\ta b ʒ ɛ k t",
            "abolir\ta b ɔ l i ʁ",
            "abonnement\ta b ɔ n m ɑ",
        ]

        data = map(test_optimal_expert_substitutions.to_sample, input_lines)
        sed_ = sed.StochasticEditDistance.fit_from_data(data, em_iterations=1)
        logging.info(sed_.params)


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", format="%(levelname)s: %(message)s")
    TestTransducer().run()
