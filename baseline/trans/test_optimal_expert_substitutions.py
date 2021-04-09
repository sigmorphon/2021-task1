"""Unit tests for optimal expert with substitution actions."""
import heapq
import os
import unittest

from trans import optimal_expert_substitutions
from trans import sed
from trans import utils
from trans.actions import Copy, Del, Ins, Sub, EndOfSequence


class OptimalSubstitutionExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        aligner = optimal_expert_substitutions.NoSubstitutionAligner()
        cls.optimal_expert = (
            optimal_expert_substitutions.OptimalSubstitutionExpert(aligner)
        )
        cls.test_fre = os.path.join(
            os.path.dirname(__file__), "test_data/fre_train.tsv"
        )

    def test_score_end(self):
        x = "walk"
        i = 4
        y = "walked"
        t = "walked"
        action_scores = self.optimal_expert.score(x, t, i, y)
        expected_action_scores = {EndOfSequence(): 0}
        self.assertEqual(expected_action_scores, action_scores)

    def test_score_empty_strings(self):
        x = ""
        i = 0
        t = ""
        y = "a"
        action_scores = self.optimal_expert.score(x, t, i, y)
        expected_action_scores = {EndOfSequence(): 0}
        self.assertEqual(expected_action_scores, action_scores)

    def test_score(self):
        x = ""
        i = 0
        t = "abbbbbbb"
        y = "bbbbbbb"
        action_scores = self.optimal_expert.score(x, t, i, y)
        expected_action_scores = {EndOfSequence(): 0, Ins("b"): 1}
        self.assertEqual(expected_action_scores, action_scores)

    def test_correct_end(self):
        x = "walk"
        i = 4
        y = "walk"
        t = "walked"
        action_scores = self.optimal_expert.score(x, t, i, y)
        expected_action_scores = {Ins("e"): 2}
        self.assertEqual(expected_action_scores, action_scores)

    def test_sed_aligner(self):

        input_lines = [
            "abba\tabba",
            "bababa\tbababa",
            "bba\tbba",
            "bbbb\tbbb",
            "bbbbb\tbbbb",
        ]

        # learns to copy even when not initialized with copy bias
        input_data = map(to_sample, input_lines)
        aligner = sed.StochasticEditDistance.fit_from_data(
            input_data, em_iterations=5
        )
        expert = optimal_expert_substitutions.OptimalSubstitutionExpert(
            aligner
        )

        x = ""
        i = 0
        t = "abbbbbbb"
        y = "bbbbbbb"
        action_scores = expert.score(x, t, i, y)
        optimal_action, _ = min_dict(action_scores)
        expected_actions = {EndOfSequence(), Ins("b")}
        self.assertSetEqual(expected_actions, set(action_scores.keys()))
        self.assertEqual(EndOfSequence(), optimal_action)

    def test_sed_aligner_real_data(self):

        verbose = False
        input_lines = []
        with open(self.test_fre) as f:
            try:
                for _ in range(100):
                    input_lines.append(to_sample(next(f)))
            except StopIteration:
                pass

        aligner = sed.StochasticEditDistance.fit_from_data(
            input_lines, em_iterations=1
        )
        expert = optimal_expert_substitutions.OptimalSubstitutionExpert(
            aligner
        )

        x = "abject"
        t = "a b ʒ ɛ k t"
        i = 3
        y = "a b ʒ e"

        optimal_actions = iter(
            (
                Sub(old="e", new=" "),
                Sub(old="c", new="k"),
                Ins(new=" "),
                Copy(old="t", new="t"),
                EndOfSequence(),
            )
        )

        while True:
            action_scores = expert.score(x, t, i, y)
            action, score = min_dict(action_scores)
            if verbose:
                print(action_scores)
                print(f"optimal action: {action, score}\n")
                print()
            if isinstance(action, EndOfSequence):
                break
            if isinstance(action, Del):
                i += 1
            elif isinstance(action, Ins):
                y += action.new
            elif isinstance(action, Sub):
                i += 1
                y += action.new
            else:
                raise ValueError(f"action: {action}")
            self.assertEqual(next(optimal_actions), action)

    def test_actions(self):

        self.assertTrue(isinstance(Copy(1, 1), Sub))
        self.assertFalse(isinstance(Sub(1, 2), Copy))


def min_dict(d):
    x = [(v, i, k) for i, (k, v) in enumerate(d.items())]
    heapq.heapify(x)
    v, _, k = heapq.heappop(x)
    return k, v


def to_sample(line: str):
    input_, target = line.rstrip().split("\t", 1)
    return utils.Sample(input_, target)


if __name__ == "__main__":
    unittest.main()
