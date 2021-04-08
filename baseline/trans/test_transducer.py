"""Unit tests for transducer.py."""
import unittest

import dynet as dy
import numpy as np
from scipy.special import log_softmax


from trans import optimal_expert
from trans import transducer
from trans import vocabulary
from trans.actions import (
    Copy,
    ConditionalCopy,
    ConditionalDel,
    ConditionalIns,
    ConditionalSub,
    Sub,
)


np.random.seed(1)


class TransducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        model = dy.Model()
        vocabulary_ = vocabulary.Vocabularies()
        vocabulary_.encode_input("foo")
        vocabulary_.encode_actions("bar")
        expert = optimal_expert.OptimalExpert()
        cls.transducer = transducer.Transducer(
            model, vocabulary_, expert, 3, 3, 3, 1, 3, 1
        )

    def test_sample(self):
        log_probs = log_softmax([5, 4, 10, 1])
        action_code = self.transducer.sample(log_probs)
        self.assertTrue(0 <= action_code < self.transducer.number_actions)

    def test_compute_valid_actions(self):
        valid_actions = self.transducer.compute_valid_actions(3)
        self.assertTrue(self.transducer.number_actions, len(valid_actions))
        valid_actions = self.transducer.compute_valid_actions(1)
        self.assertTrue(ConditionalCopy not in valid_actions)

    def test_remap_actions(self):
        action_scores = {Copy("w", "w"): 7.0, Sub("w", "v"): 5.0}
        expected = {ConditionalCopy(): 7.0, ConditionalSub("v"): 5.0}
        remapped = self.transducer.remap_actions(action_scores)
        self.assertDictEqual(expected, remapped)

    def test_expert_rollout(self):
        optimal_actions = self.transducer.expert_rollout(
            input_="foo", target="bar", alignment=1, prediction=["b", "a"]
        )
        expected = {
            self.transducer.vocab.encode_unseen_action(a)
            for a in (ConditionalIns("r"), ConditionalDel())
        }
        self.assertSetEqual(expected, set(optimal_actions))


if __name__ == "__main__":
    TransducerTests().run()
