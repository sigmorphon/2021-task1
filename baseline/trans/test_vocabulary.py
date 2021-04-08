"""Unit tests for vocabulary.py."""
import unittest

from trans import vocabulary
from trans.actions import (
    BeginOfSequence,
    ConditionalDel,
    ConditionalCopy,
    ConditionalIns,
    ConditionalSub,
    EndOfSequence,
)
from trans.vocabulary import UNK_CHAR


class VocabularyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabulary = vocabulary.Vocabulary()
        for c in "foo":
            cls.vocabulary.encode(c)
        cls.vocabularies = vocabulary.Vocabularies()
        cls.vocabularies.encode_input("foo")
        cls.vocabularies.encode_actions("baa")

    def test_vocabulary(self):
        i2w1 = []
        i2w2 = [3, 4, 5]
        vocabulary1 = vocabulary.Vocabulary(i2w1)
        self.assertListEqual(
            [BeginOfSequence(), EndOfSequence(), UNK_CHAR], vocabulary1.i2w
        )
        self.assertDictEqual(
            {BeginOfSequence(): 0, EndOfSequence(): 1, UNK_CHAR: 2},
            vocabulary1.w2i,
        )
        vocabulary2 = vocabulary.Vocabulary(i2w2)
        self.assertListEqual(
            [BeginOfSequence(), EndOfSequence(), UNK_CHAR, 3, 4, 5],
            vocabulary2.i2w,
        )
        self.assertDictEqual(
            {
                BeginOfSequence(): 0,
                EndOfSequence(): 1,
                UNK_CHAR: 2,
                3: 3,
                4: 4,
                5: 5,
            },
            vocabulary2.w2i,
        )

    def test_actions(self):
        vocabulary1 = vocabulary.ActionVocabulary()
        expected_i2w = [
            BeginOfSequence(),
            EndOfSequence(),
            ConditionalDel(),
            ConditionalCopy(),
        ]
        expected_w2i = {
            BeginOfSequence(): 0,
            EndOfSequence(): 1,
            ConditionalDel(): 2,
            ConditionalCopy(): 3,
        }
        self.assertListEqual(expected_i2w, vocabulary1.i2w)
        self.assertDictEqual(expected_w2i, vocabulary1.w2i)

    def test_vocabulary_encode(self):
        vocabulary1 = vocabulary.Vocabulary()
        for c in "foo":
            vocabulary1.encode(c)
        self.assertEqual(5, len(vocabulary1))
        self.assertEqual(4, vocabulary1.encode("o"))

    def test_vocabulary_decode(self):
        self.assertEqual("f", self.vocabulary.decode(3))
        self.assertRaises(IndexError, self.vocabulary.decode, 10)

    def test_vocabulary_lookup(self):
        self.assertEqual(2, self.vocabulary.lookup("F"))

    def test_vocabularies_encode_input(self):
        vocabulary1 = vocabulary.Vocabularies()
        encoded_foo = vocabulary1.encode_input("foo")
        self.assertListEqual([0, 3, 4, 4, 1], encoded_foo)

    def test_vocabularies_encode_actions(self):
        vocabulary1 = vocabulary.Vocabularies()
        vocabulary1.encode_actions("baa")
        expected_i2w = [
            BeginOfSequence(),
            EndOfSequence(),
            ConditionalDel(),
            ConditionalCopy(),
            ConditionalSub("b"),
            ConditionalIns("b"),
            ConditionalSub("a"),
            ConditionalIns("a"),
        ]
        self.assertListEqual(expected_i2w, vocabulary1.actions.i2w)

    def test_vocabularies_encode_unseen_input(self):
        encoded_fox = self.vocabularies.encode_unseen_input("fox")
        self.assertListEqual([0, 3, 4, 2, 1], encoded_fox)

    def test_vocabularies_encode_unseen_actions(self):
        encoded_action = self.vocabularies.encode_unseen_action(
            ConditionalIns("b")
        )
        self.assertEqual(5, encoded_action)
        self.assertRaises(
            KeyError,
            self.vocabularies.encode_unseen_action,
            ConditionalIns("Q"),
        )

    def test_vocabularies_decode_actions(self):
        decoded_actions = [
            self.vocabularies.decode_action(i) for i in (0, 3, 4)
        ]
        expected_actions = [
            BeginOfSequence(),
            ConditionalCopy(),
            ConditionalSub("b"),
        ]
        self.assertListEqual(expected_actions, decoded_actions)
        self.assertRaises(IndexError, self.vocabularies.decode_action, 10)

    def test_vocabularies_decode_input(self):
        expected_input = self.vocabularies.decode_input([4, 4, 3])
        self.assertEqual("oof", expected_input)
        self.assertRaises(IndexError, self.vocabularies.decode_input, [5])


if __name__ == "__main__":
    VocabularyTests().run()
