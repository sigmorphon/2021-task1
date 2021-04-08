"""Classes for edit actions and aligner base class."""
from typing import Any, Sequence
import abc
import dataclasses


class Edit(abc.ABC):
    pass


@dataclasses.dataclass(frozen=True, eq=True)
class BeginOfSequence(Edit):
    def __repr__(self):
        return "\u27ea"


@dataclasses.dataclass(frozen=True, eq=True)
class EndOfSequence(Edit):
    def __repr__(self):
        return "\u27eb"


class GenerativeEdit(Edit):
    @abc.abstractmethod
    def conditional_counterpart(self):
        raise NotImplementedError


class ConditionalEdit(Edit):
    def conditional_counterpart(self):
        return self


@dataclasses.dataclass(frozen=True, eq=True)
class ConditionalSub(ConditionalEdit):
    new: Any


@dataclasses.dataclass(frozen=True, eq=True)
class ConditionalCopy(ConditionalEdit):
    pass


@dataclasses.dataclass(frozen=True, eq=True)
class ConditionalDel(ConditionalEdit):
    pass


@dataclasses.dataclass(frozen=True, eq=True)
class ConditionalIns(ConditionalEdit):
    new: Any


@dataclasses.dataclass(frozen=True, eq=True)
class Sub(GenerativeEdit):
    old: Any
    new: Any

    def conditional_counterpart(self):
        return ConditionalSub(self.new)


@dataclasses.dataclass(frozen=True, eq=True)
class Copy(Sub):
    old: Any
    new: Any

    def __post_init__(self):
        if self.old != self.new:
            raise ValueError(f"Copy: old={self.old} != new={self.new}")

    def conditional_counterpart(self):
        return ConditionalCopy()


@dataclasses.dataclass(frozen=True, eq=True)
class Del(GenerativeEdit):
    old: Any

    def conditional_counterpart(self):
        return ConditionalDel()


@dataclasses.dataclass(frozen=True, eq=True)
class Ins(GenerativeEdit):
    new: Any

    def conditional_counterpart(self):
        return ConditionalIns(self.new)


class Aligner(abc.ABC):
    @abc.abstractmethod
    def action_sequence_cost(
        self, x: Sequence[Any], y: Sequence[Any], x_offset: int, y_offset: int
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def action_cost(self, action: Edit) -> float:
        raise NotImplementedError
