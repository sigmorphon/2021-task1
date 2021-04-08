"""Optimal expert for character-level string transduction.

Given an input string x, a target string t, alignment index i, and a partial
prediction y, it returns optimal cost-to-go for all valid edit actions."""
from typing import Any, Iterable, List, Sequence, Set
import abc
import dataclasses

import numpy as np

from trans.actions import (
    ConditionalCopy,
    ConditionalDel,
    ConditionalIns,
    Edit,
    EndOfSequence,
)


class Expert(abc.ABC):
    @abc.abstractmethod
    def score(
        self, x: Sequence[Any], t: Sequence[Any], i: int, y: Sequence[Any]
    ):
        raise NotImplementedError


def edit_distance(
    x: Sequence[Any],
    y: Sequence[Any],
    del_cost: float,
    ins_cost: float,
    sub_cost: float,
    x_offset: int,
    y_offset: int,
) -> np.ndarray:
    x_size = len(x) - x_offset + 1
    y_size = len(y) - y_offset + 1
    prefix_matrix = np.full((x_size, y_size), np.inf, dtype=np.float_)
    for i in range(x_size):
        prefix_matrix[i, 0] = i * del_cost
    for j in range(y_size):
        prefix_matrix[0, j] = j * ins_cost
    for i in range(1, x_size):
        for j in range(1, y_size):
            if x[i - 1 + x_offset] == y[j - 1 + y_offset]:
                substitution = 0.0
            else:
                substitution = sub_cost
            prefix_matrix[i, j] = min(
                prefix_matrix[i - 1, j] + del_cost,
                prefix_matrix[i, j - 1] + ins_cost,
                prefix_matrix[i - 1, j - 1] + substitution,
            )
    return prefix_matrix


def levenshtein_distance(x: Sequence[Any], y: Sequence[Any]) -> np.ndarray:
    return edit_distance(
        x, y, del_cost=1.0, ins_cost=1.0, sub_cost=1.0, x_offset=0, y_offset=0
    )


def action_sequence_cost(
    x: Sequence[Any], y: Sequence[Any], x_offset: int, y_offset: int
) -> float:
    ed = edit_distance(
        x,
        y,
        del_cost=1.0,
        ins_cost=1.0,
        sub_cost=np.inf,
        x_offset=x_offset,
        y_offset=y_offset,
    )
    return ed[-1, -1]


@dataclasses.dataclass
class Prefix:
    y: Sequence[Any]
    t: Sequence[Any]
    j: int

    @property
    def suffix(self):
        return self.t[self.j :]

    @property
    def leftmost_of_suffix(self):
        try:
            return self.t[self.j]
        except IndexError:
            return None


@dataclasses.dataclass
class ActionsPrefix:
    actions: Set[Edit]
    prefix: Prefix


class OptimalExpert(Expert):
    def __init__(self, maximum_output_length: int = 150):

        self.maximum_output_length = maximum_output_length

    def find_valid_actions(
        self,
        x: Sequence[Any],
        i: int,
        y: Sequence[Any],
        prefixes: Iterable[Prefix],
    ):
        if len(y) >= self.maximum_output_length:
            return {EndOfSequence()}
        input_not_empty = i < len(x)
        actions_prefixes: List[ActionsPrefix] = []
        for prefix in prefixes:
            prefix_insert = prefix.leftmost_of_suffix
            if prefix_insert is None:
                valid_actions = {EndOfSequence()}
            else:
                valid_actions = {ConditionalIns(prefix_insert)}
            if input_not_empty:
                if prefix_insert is not None and x[i] == prefix_insert:
                    valid_actions.add(ConditionalCopy())
                valid_actions.add(ConditionalDel())
            actions_prefix = ActionsPrefix(valid_actions, prefix)
            actions_prefixes.append(actions_prefix)
        return actions_prefixes

    @staticmethod
    def find_prefixes(y: Sequence[Any], t: Sequence[Any]) -> List[Prefix]:
        prefix_matrix = levenshtein_distance(y, t)
        ys_row = prefix_matrix[-1]
        prefixes = []
        for j in np.where(ys_row == ys_row.min())[0]:
            prefixes.append(Prefix(y, t, j))
        return prefixes

    def roll_out(
        self,
        x: Sequence[Any],
        t: Sequence[Any],
        i: int,
        actions_prefixes: Iterable[ActionsPrefix],
    ):
        costs_to_go = dict()
        for actions_prefix in actions_prefixes:
            suffix_begin = actions_prefix.prefix.j
            for action in actions_prefix.actions:
                if isinstance(action, ConditionalDel):
                    x_offset = i + 1
                    t_offset = suffix_begin
                    action_cost = 1
                elif isinstance(action, ConditionalCopy):
                    x_offset = i + 1
                    t_offset = suffix_begin + 1
                    action_cost = 0
                elif isinstance(action, EndOfSequence):
                    x_offset = i
                    t_offset = suffix_begin
                    action_cost = 0
                else:
                    x_offset = i
                    t_offset = suffix_begin + 1
                    action_cost = 1
                sequence_cost = action_sequence_cost(x, t, x_offset, t_offset)
                cost = action_cost + sequence_cost
                if action not in costs_to_go or costs_to_go[action] > cost:
                    costs_to_go[action] = cost
        return costs_to_go

    def score(
        self, x: Sequence[Any], t: Sequence[Any], i: int, y: Sequence[Any]
    ):

        prefixes = self.find_prefixes(y, t)
        valid_actions = self.find_valid_actions(x, i, y, prefixes)
        valid_action_scores = self.roll_out(x, t, i, valid_actions)
        return valid_action_scores
