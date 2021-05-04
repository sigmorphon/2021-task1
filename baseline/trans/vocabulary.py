"""Vocabularies."""
from typing import Any, List, Iterable, Optional
import pickle

from trans.actions import (
    BeginOfSequence,
    ConditionalCopy,
    ConditionalDel,
    ConditionalIns,
    ConditionalSub,
    EndOfSequence,
)


UNK_CHAR = "<UNK>"


class Vocabulary:

    EXTRAS = BeginOfSequence(), EndOfSequence(), UNK_CHAR

    def __init__(self, i2w: Optional[Iterable[Any]] = None):
        if i2w is None:
            i2w = []
        self.i2w = list(self.EXTRAS) + i2w
        self.w2i = {w: i for i, w in enumerate(self.i2w)}

    def encode(self, word: Any) -> int:
        if word in self.w2i:
            index = self.w2i[word]
        else:
            index = len(self.i2w)
            self.i2w.append(word)
            self.w2i[word] = index
        return index

    def decode(self, index: int) -> Any:
        return self.i2w[index]

    def lookup(self, word: Any) -> int:
        return self.w2i.get(word, UNK)

    def __len__(self):
        return len(self.i2w)

    def __repr__(self):
        return f"Vocabulary({str(self.w2i)})"

    def to_i2w(self):
        return self.i2w[len(self.EXTRAS) :]


class ActionVocabulary(Vocabulary):
    EXTRAS = (
        BeginOfSequence(),
        EndOfSequence(),
        ConditionalDel(),
        ConditionalCopy(),
    )

    def lookup(self, word: Any) -> int:
        return self.w2i[word]  # N.B. no UNK

    @property
    def substitutions(self):
        return [
            i for i, a in enumerate(self.i2w) if isinstance(a, ConditionalSub)
        ]

    @property
    def insertions(self):
        return [
            i for i, a in enumerate(self.i2w) if isinstance(a, ConditionalIns)
        ]


BEGIN_WORD = Vocabulary.EXTRAS.index(BeginOfSequence())
END_WORD = Vocabulary.EXTRAS.index(EndOfSequence())
UNK = Vocabulary.EXTRAS.index(UNK_CHAR)
DELETE = ActionVocabulary.EXTRAS.index(ConditionalDel())
COPY = ActionVocabulary.EXTRAS.index(ConditionalCopy())


class Vocabularies:
    """Holds encodings of input characters and edit actions."""

    def __init__(
        self,
        characters: Optional[Iterable[str]] = None,
        actions: Optional[Iterable[Any]] = None,
    ):
        self.characters = Vocabulary(characters)
        self.actions = ActionVocabulary(actions)
        self.target_characters = set()

    def encode_input(self, input_: str) -> List[int]:
        encoded_input = [BEGIN_WORD]
        encoded_input.extend(self.characters.encode(c) for c in input_)
        encoded_input.append(END_WORD)
        return encoded_input

    def encode_actions(self, target: str) -> None:
        """Encodes all writing actions for the target string."""
        for c in target:
            if c in self.target_characters:
                continue
            self.actions.encode(ConditionalSub(c))
            self.actions.encode(ConditionalIns(c))
            self.target_characters.add(c)

    def encode_unseen_input(self, input_: str) -> List[int]:
        encoded_input = [BEGIN_WORD]
        encoded_input.extend(self.characters.lookup(c) for c in input_)
        encoded_input.append(END_WORD)
        return encoded_input

    def encode_unseen_action(self, action: Any) -> int:
        return self.actions.lookup(action)

    def decode_input(self, encoded_input: Iterable[int]) -> str:
        return "".join(self.characters.decode(i) for i in encoded_input)

    def decode_action(self, encoded_action: int) -> Any:
        return self.actions.decode(encoded_action)

    def persist(self, filename: str):
        vocabularies = {
            "characters": self.characters.to_i2w(),
            "actions": self.actions.to_i2w(),
        }
        with open(filename, mode="wb") as w:
            pickle.dump(vocabularies, w)

    @property
    def substitutions(self):
        return self.actions.substitutions

    @property
    def insertions(self):
        return self.actions.insertions
