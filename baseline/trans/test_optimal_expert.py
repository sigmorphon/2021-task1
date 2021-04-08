"""Unit tests for optimal expert for character-level string transduction."""
import unittest

import editdistance
import numpy as np

from trans import optimal_expert
from trans.actions import (
    ConditionalCopy,
    ConditionalDel,
    ConditionalIns,
    EndOfSequence,
)


class TestOptimalExpert(unittest.TestCase):
    def setUp(self) -> None:

        self.optimal_expert = optimal_expert.OptimalExpert()

    def test_edit_distance(self):
        x = "wal"
        y = "walked"
        ed = optimal_expert.levenshtein_distance(x, y)[-1, -1]
        self.assertEqual(ed, editdistance.eval(x, y))

    def test_prefix_matrix(self):
        x = "wad"
        y = "walked"
        prefix_matrix = optimal_expert.levenshtein_distance(x, y)
        expected_prefix_matrix = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6],
                [1, 0, 1, 2, 3, 4, 5],
                [2, 1, 0, 1, 2, 3, 4],
                [3, 2, 1, 1, 2, 3, 3],
            ],
            dtype=np.float_,
        )
        self.assertTrue(np.allclose(expected_prefix_matrix, prefix_matrix))

    def test_x_offset(self):
        x = "lk"
        y = "lked"
        delete_cost = optimal_expert.action_sequence_cost(
            x, y, x_offset=1, y_offset=0
        )
        self.assertEqual(3, delete_cost)

        x = "lk"
        y = "ked"
        delete_cost = optimal_expert.action_sequence_cost(
            x, y, x_offset=1, y_offset=0
        )
        self.assertEqual(2, delete_cost)

    def test_y_offset(self):
        x = "lk"
        y = "lked"
        insert_cost = optimal_expert.action_sequence_cost(
            x, y, x_offset=0, y_offset=1
        )
        self.assertEqual(3, insert_cost)

        x = "lk"
        y = "ked"
        insert_cost = optimal_expert.action_sequence_cost(
            x, y, x_offset=0, y_offset=1
        )
        self.assertEqual(4, insert_cost)

    def test_both_offsets(self):
        x = "lk"
        y = "lked"
        insert_cost = optimal_expert.action_sequence_cost(
            x, y, x_offset=1, y_offset=1
        )
        self.assertEqual(2, insert_cost)

    def test_find_prefixes(self):
        y = "wad"
        t = "walked"
        prefixes = self.optimal_expert.find_prefixes(y, t)
        expected_prefixes = [
            optimal_expert.Prefix(y, t, 2),
            optimal_expert.Prefix(y, t, 3),
        ]
        self.assertEqual(expected_prefixes, prefixes)

    def test_prefix(self):
        y = "wad"
        t = "walked"
        prefix1 = optimal_expert.Prefix(y, t, 2)
        prefix2 = optimal_expert.Prefix(y, t, 3)

        self.assertEqual("lked", prefix1.suffix)
        self.assertEqual("ked", prefix2.suffix)
        self.assertEqual("l", prefix1.leftmost_of_suffix)
        self.assertEqual("k", prefix2.leftmost_of_suffix)

    def test_edge_case_prefix(self):
        prefix1 = optimal_expert.Prefix("w", "w", 1)
        self.assertEqual("", prefix1.suffix)
        self.assertEqual(None, prefix1.leftmost_of_suffix)

    def test_find_valid_actions(self):
        x = "wal"
        y = "wad"
        t = "walked"
        prefixes = self.optimal_expert.find_prefixes(y, t)
        valid_actions = self.optimal_expert.find_valid_actions(
            x, 2, y, prefixes
        )
        expected_valid_actions = [
            optimal_expert.ActionsPrefix(
                {ConditionalCopy(), ConditionalDel(), ConditionalIns("l")},
                optimal_expert.Prefix(y, t, 2),
            ),
            optimal_expert.ActionsPrefix(
                {ConditionalDel(), ConditionalIns("k")},
                optimal_expert.Prefix(y, t, 3),
            ),
        ]
        self.assertEqual(expected_valid_actions, valid_actions)

    def test_roll_out(self):
        x = "walk"
        i = 2
        y = "wad"
        t = "walked"
        prefixes = self.optimal_expert.find_prefixes(y, t)
        valid_actions = self.optimal_expert.find_valid_actions(
            x, i, y, prefixes
        )
        action_scores = self.optimal_expert.roll_out(x, t, i, valid_actions)
        expected_action_scores = {
            ConditionalCopy(): 2,
            ConditionalDel(): 3,
            ConditionalIns("l"): 4,
            ConditionalIns("k"): 5,
        }
        self.assertEqual(expected_action_scores, action_scores)

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
        expected_action_scores = {EndOfSequence(): 0, ConditionalIns("b"): 1}
        self.assertEqual(expected_action_scores, action_scores)

    def test_correct_end(self):
        x = "walk"
        i = 4
        y = "walk"
        t = "walked"
        action_scores = self.optimal_expert.score(x, t, i, y)
        expected_action_scores = {ConditionalIns("e"): 2}
        self.assertEqual(expected_action_scores, action_scores)


if __name__ == "__main__":
    TestOptimalExpert().run()
