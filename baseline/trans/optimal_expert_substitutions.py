"""Optimal expert that additionally uses substitution actions."""
from typing import Any, Iterable, List, Sequence

import numpy as np

from trans import actions
from trans import optimal_expert
from trans.actions import Copy, Del, Edit, EndOfSequence, Ins, Sub


class EditDistanceAligner(actions.Aligner):
    def __init__(self, del_cost=1.0, ins_cost=1.0, sub_cost=1.0):
        self.del_cost = del_cost
        self.ins_cost = ins_cost
        self.sub_cost = sub_cost

    def action_sequence_cost(
        self, x: Sequence[Any], y: Sequence[Any], x_offset: int, y_offset: int
    ) -> float:
        ed = optimal_expert.edit_distance(
            x,
            y,
            del_cost=self.del_cost,
            ins_cost=self.ins_cost,
            sub_cost=self.sub_cost,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        return ed[-1, -1]

    def action_cost(self, action: Edit):
        if isinstance(action, Copy) or isinstance(action, EndOfSequence):
            return 0
        if isinstance(action, Del):
            return self.del_cost
        if isinstance(action, Ins):
            return self.ins_cost
        if isinstance(action, Sub):
            return self.sub_cost
        raise ValueError(f"Unexpected action: {action}!")


class NoSubstitutionAligner(EditDistanceAligner):
    def __init__(self):
        super().__init__(del_cost=1.0, ins_cost=1.0, sub_cost=1.0)

    def action_cost(self, action: Edit):
        if isinstance(action, Sub):
            return np.inf
        return super().action_cost(action)


class OptimalSubstitutionExpert(optimal_expert.OptimalExpert):
    def __init__(
        self, aligner: actions.Aligner, maximum_output_length: int = 150
    ):
        super().__init__(maximum_output_length)
        self.aligner = aligner

    def find_valid_actions(
        self,
        x: Sequence[Any],
        i: int,
        y: Sequence[Any],
        prefixes: Iterable[optimal_expert.Prefix],
    ):
        if len(y) >= self.maximum_output_length:
            return {EndOfSequence()}
        input_not_empty = i < len(x)
        attention = x[i] if input_not_empty else None
        actions_prefixes: List[optimal_expert.ActionsPrefix] = []
        for prefix in prefixes:
            prefix_insert = prefix.leftmost_of_suffix
            if prefix_insert is None:
                valid_actions = {EndOfSequence()}
            else:
                valid_actions = {Ins(prefix_insert)}
            if input_not_empty:
                if prefix_insert is not None:
                    if prefix_insert == attention:
                        valid_actions.add(Copy(attention, prefix_insert))
                    else:
                        valid_actions.add(
                            Sub(old=attention, new=prefix_insert)
                        )
                valid_actions.add(Del(attention))
            actions_prefix = optimal_expert.ActionsPrefix(
                valid_actions, prefix
            )
            actions_prefixes.append(actions_prefix)
        return actions_prefixes

    def roll_out(
        self,
        x: Sequence[Any],
        t: Sequence[Any],
        i: int,
        actions_prefixes: Iterable[optimal_expert.ActionsPrefix],
    ):
        costs_to_go = dict()
        for actions_prefix in actions_prefixes:
            suffix_begin = actions_prefix.prefix.j
            for action in actions_prefix.actions:
                if isinstance(action, Del):
                    x_offset = i + 1
                    t_offset = suffix_begin
                elif isinstance(action, Ins):
                    x_offset = i
                    t_offset = suffix_begin + 1
                elif isinstance(action, Sub):
                    x_offset = i + 1
                    t_offset = suffix_begin + 1
                elif isinstance(action, EndOfSequence):
                    x_offset = i
                    t_offset = suffix_begin
                else:
                    raise ValueError(f"Unknown action: {action}")
                sequence_cost = self.aligner.action_sequence_cost(
                    x, t, x_offset, t_offset
                )
                action_cost = self.aligner.action_cost(action)
                cost = action_cost + sequence_cost
                if action not in costs_to_go or costs_to_go[action] > cost:
                    costs_to_go[action] = cost
        return costs_to_go
