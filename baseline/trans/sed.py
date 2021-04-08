"""Based on Ristad and Yianilos (1998) Learning String Edit Distance."""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import dataclasses
import logging
import os
import pickle

import numpy as np
from scipy.special import logsumexp

from trans import actions
from trans import utils
from trans.actions import Copy, Del, Edit, EndOfSequence, Ins, Sub


LARGE_NEG_CONST = -float(10 ** 6)


@dataclasses.dataclass
class ParamDict:
    delta_sub: Dict[Tuple[Any, Any], float]
    delta_del: Dict[Any, float]
    delta_ins: Dict[Any, float]
    delta_eos: float

    def sum(self) -> float:
        values = [self.delta_eos]
        for vs in (self.delta_sub, self.delta_ins, self.delta_del):
            values.extend(vs.values())
        return logsumexp(values)

    @classmethod
    def from_params(cls, other: "ParamDict"):
        return cls(
            delta_sub=dict(other.delta_sub),
            delta_del=dict(other.delta_del),
            delta_ins=dict(other.delta_ins),
            delta_eos=other.delta_eos,
        )


class StochasticEditDistance(actions.Aligner):
    """Implementation of the Stochastic Edit Distance (SED) model from
    Ristad & Yianilos 1998 "Learning String Edit Distance." The model is a
    memoryless probabilistic weighted finite-state transducer that by use of
    character edits (insertions, deletions, substitutions) turns one string to
    another. Edit weights are learned with Expectation-Maximization.

    Args:
        param: SED log weights."""

    def __init__(self, params: ParamDict, *args, **kwargs) -> None:

        self.params = params
        self.delta_sub = params.delta_sub
        self.delta_del = params.delta_del
        self.delta_ins = params.delta_ins
        self.delta_eos = params.delta_eos
        self.default = (
            LARGE_NEG_CONST  # ad-hoc fix for unseen inputs / outputs
        )

        if not np.isclose(0.0, self.params.sum()):
            raise ValueError(
                f"Parameters do not sum to 1!: {self.params.sum():.4f}."
            )

    @classmethod
    def build_sed(
        cls,
        source_alphabet: Iterable[Any],
        target_alphabet: Iterable[Any],
        copy_probability: Optional[float] = 0.9,
    ):
        """Builds a SED given a source and a target alphabets. If
        copy_probability is not None, distributes this probability mass across
        all copy actions to bias SED towards copying.

        Args:
            source_alphabet: Characters of all input strings.
            target_alphabet: Characters of all target strings.
            copy_probability: On weight init, how much mass to give to copy
                edits."""
        source_alphabet = set(source_alphabet)
        target_alphabet = set(target_alphabet)

        n = (
            len(source_alphabet) * len(target_alphabet)
            + len(source_alphabet)
            + len(target_alphabet)
            + 1
        )

        if copy_probability is None:
            uniform_weight = np.log(1 / n)
            log_copy_prob = uniform_weight  # probability of a copy action
            log_rest_prob = uniform_weight  # probability of any other action
        elif 0 < copy_probability < 1:
            # split copy mass over individual copy actions
            num_copy_edits = len(target_alphabet & source_alphabet)
            num_rest = n - num_copy_edits
            log_copy_prob = np.log(copy_probability / num_copy_edits)
            log_rest_prob = np.log((1 - copy_probability) / num_rest)
        else:
            raise ValueError(
                f"0 < copy probability={copy_probability} < 1 doesn't hold."
            )

        delta_sub = {
            (s, t): log_copy_prob if s == t else log_rest_prob
            for s in source_alphabet
            for t in target_alphabet
        }
        delta_del = {s: log_rest_prob for s in source_alphabet}
        delta_ins = {t: log_rest_prob for t in target_alphabet}
        delta_eos = log_rest_prob
        params = ParamDict(delta_sub, delta_del, delta_ins, delta_eos)
        return cls(params)

    @classmethod
    def fit_from_data(
        cls,
        lines: Iterable[utils.Sample],
        copy_probability: float = None,
        em_iterations: int = 30,
        output_path: str = None,
    ):

        source_alphabet = set()
        target_alphabet = set()
        sources = []
        targets = []
        for line in lines:
            source = line.input
            target = line.target
            source_alphabet.update(source)
            target_alphabet.update(target)
            sources.append(source)
            targets.append(target)

        sed = cls.build_sed(source_alphabet, target_alphabet, copy_probability)
        sed.update_model(
            sources, targets, iterations=em_iterations, output_path=output_path
        )
        return sed

    @classmethod
    def from_pickle(cls, path2pkl: str):
        logging.info("Loading sed channel parameters from file: ", path2pkl)
        with open(path2pkl, "rb") as w:
            params: ParamDict = pickle.load(w)
        return cls(params)

    def to_pickle(self, pkl: str):
        with open(pkl, "wb") as w:
            pickle.dump(self.params, w)

    def forward_evaluate(
        self, source: Sequence[Any], target: Sequence[Any]
    ) -> np.ndarray:
        """Computes forward probabilities.

        Computes dynamic programming table (in log real) filled with forward
        log probabilities."""
        T, V = len(source), len(target)
        alpha = np.full((T + 1, V + 1), LARGE_NEG_CONST)
        alpha[0, 0] = 0.0
        for t in range(T + 1):
            for v in range(V + 1):
                summands = [alpha[t, v]]
                if v > 0:
                    summands.append(
                        self.delta_ins.get(target[v - 1], self.default)
                        + alpha[t, v - 1]
                    )
                if t > 0:
                    summands.append(
                        self.delta_del.get(source[t - 1], self.default)
                        + alpha[t - 1, v]
                    )
                if v > 0 and t > 0:
                    summands.append(
                        self.delta_sub.get(
                            (source[t - 1], target[v - 1]), self.default
                        )
                        + alpha[t - 1, v - 1]
                    )
                alpha[t, v] = logsumexp(summands)
        alpha[T, V] += self.delta_eos
        return alpha

    def backward_evaluate(
        self, source: Sequence[Any], target: Sequence[Any]
    ) -> np.ndarray:
        """Computes backward probabilities.

        Compute dynamic programming table (in log real) filled with backward log
        probabilities (the probabilities of the suffix, i.e.
        p(source[t:], target[v:]). E.g. p("", "a") = p(ins(a))*p(#)."""
        T, V = len(source), len(target)
        beta = np.full((T + 1, V + 1), LARGE_NEG_CONST)
        beta[T, V] = self.delta_eos
        for t in range(T, -1, -1):
            for v in range(V, -1, -1):
                summands = [beta[t, v]]
                if v < V:
                    summands.append(
                        self.delta_ins.get(target[v], self.default)
                        + beta[t, v + 1]
                    )
                if t < T:
                    summands.append(
                        self.delta_del.get(source[t], self.default)
                        + beta[t + 1, v]
                    )
                if v < V and t < T:
                    summands.append(
                        self.delta_sub.get(
                            (source[t], target[v]), self.default
                        )
                        + beta[t + 1, v + 1]
                    )
                beta[t, v] = logsumexp(summands)
        return beta

    def log_likelihood(
        self,
        sources: Iterable[Sequence[Any]],
        targets: Iterable[Sequence[Any]],
    ) -> float:
        """Computes log likelihood."""

        ll = np.mean(
            [
                self.forward_evaluate(source, target)[-1, -1]
                for source, target in zip(sources, targets)
            ]
        )
        return float(ll)

    def em(
        self,
        sources: Sequence[Any],
        targets: Sequence[Any],
        iterations: int = 10,
    ) -> None:
        """Update parameters using Expectation-Maximization.

        Args:
            sources: Source strings.
            targets: Target strings.
            iterations: Number of iterations of EM."""
        logging.info(
            "Initial weighted LL=%.4f", self.log_likelihood(sources, targets)
        )

        for i in range(iterations):
            gammas = ParamDict.from_params(self.params)
            for j, (source, target) in enumerate(zip(sources, targets)):
                self.e_step(source, target, gammas)
                if j > 0 and j % 1000 == 0:
                    logging.info("\t...processed %d samples", j)
            self.m_step(gammas)
            logging.info(
                "IT_%d=%.4f", i, self.log_likelihood(sources, targets)
            )

    def e_step(
        self, source: Sequence[Any], target: Sequence[Any], gammas: ParamDict
    ) -> None:
        """Accumulates soft counts.

        Args:
            source: Source string.
            target: Target string.
            gammas: Unnormalized log weights."""
        alpha = self.forward_evaluate(source, target)
        beta = self.backward_evaluate(source, target)
        gammas.delta_eos = logsumexp([gammas.delta_eos, 0.0])
        T, V = len(source), len(target)
        for t in range(T + 1):
            for v in range(V + 1):
                rest = beta[t, v] - alpha[T, V]
                schar = source[t - 1]
                tchar = target[v - 1]
                stpair = schar, tchar
                if t > 0 and schar in gammas.delta_del:
                    gammas.delta_del[schar] = logsumexp(
                        [
                            gammas.delta_del[schar],
                            alpha[t - 1, v] + self.delta_del[schar] + rest,
                        ]
                    )
                if v > 0 and tchar in gammas.delta_ins:
                    gammas.delta_ins[tchar] = logsumexp(
                        [
                            gammas.delta_ins[tchar],
                            alpha[t, v - 1] + self.delta_ins[tchar] + rest,
                        ]
                    )
                if t > 0 and v > 0 and stpair in gammas.delta_sub:
                    gammas.delta_sub[stpair] = logsumexp(
                        [
                            gammas.delta_sub[stpair],
                            alpha[t - 1, v - 1]
                            + self.delta_sub[stpair]
                            + rest,
                        ]
                    )

    def m_step(self, gammas: ParamDict) -> None:
        """Normalizes weights and stores them."""
        denom = gammas.sum()
        gammas.delta_sub = {
            k: (v - denom) for k, v in gammas.delta_sub.items()
        }
        gammas.delta_del = {
            k: (v - denom) for k, v in gammas.delta_del.items()
        }
        gammas.delta_ins = {
            k: (v - denom) for k, v in gammas.delta_ins.items()
        }
        gammas.delta_eos -= denom

        assert np.isclose(0.0, gammas.sum()), gammas.sum()
        self.params = gammas
        self.delta_sub = gammas.delta_sub
        self.delta_del = gammas.delta_del
        self.delta_ins = gammas.delta_ins
        self.delta_eos = gammas.delta_eos

    def viterbi_distance(
        self, source: Sequence, target: Sequence, with_alignment: bool = False
    ) -> Union[float, Tuple[List, float]]:
        """Computes Viterbi edit distance.

        Viterbi edit distance \propto max_{edits} p(target, edit | source).

        Args:
            source: Source string.
            target: Target string.
            with_alignment: Whether to output the corresponding sequence of
                edits.

        Returns:
            Probability score and, optionally, the sequence of edits that gives
            this score."""
        T, V = len(source), len(target)
        alpha = np.full((T + 1, V + 1), LARGE_NEG_CONST)
        alpha[0, 0] = 0.0
        for t in range(T + 1):
            for v in range(V + 1):
                alternatives = [alpha[t, v]]
                if v > 0:
                    alternatives.append(
                        self.delta_ins.get(target[v - 1], self.default)
                        + alpha[t, v - 1]
                    )
                if t > 0:
                    alternatives.append(
                        self.delta_del.get(source[t - 1], self.default)
                        + alpha[t - 1, v]
                    )
                if v > 0 and t > 0:
                    alternatives.append(
                        self.delta_sub.get(
                            (source[t - 1], target[v - 1]), self.default
                        )
                        + alpha[t - 1, v - 1]
                    )
                alpha[t, v] = max(alternatives)
        alpha[T, V] += self.delta_eos
        optim_score = alpha[T, V]
        if not with_alignment:
            return optim_score
        # compute an optimal alignment
        alignment = []
        ind_w, ind_c = len(source), len(target)
        while ind_w >= 0 and ind_c >= 0:
            if ind_w == 0 and ind_c == 0:
                return alignment[::-1], optim_score
            if ind_w == 0:
                # can only go left, i.e. via insertions
                ind_c -= ind_c
                alignment.append(
                    Ins(target[ind_c])
                )  # minus 1 is due to offset
            elif ind_c == 0:
                # can only go up, i.e. via deletions
                ind_w -= ind_w
                alignment.append(
                    Del(source[ind_w])
                )  # minus 1 is due to offset
            else:
                # pick the smallest cost actions
                pind_w = ind_w - 1
                pind_c = ind_c - 1
                action_idx = np.argmax(
                    [
                        alpha[pind_w, pind_c],
                        alpha[ind_w, pind_c],
                        alpha[pind_w, ind_c],
                    ]
                )
                if action_idx == 0:
                    action = Sub(source[pind_w], target[pind_c])
                    ind_w = pind_w
                    ind_c = pind_c
                elif action_idx == 1:
                    action = Ins(target[pind_c])
                    ind_c = pind_c
                else:
                    action = Del(source[pind_w])
                    ind_w = pind_w
                alignment.append(action)

    def stochastic_distance(
        self, source: Sequence[Any], target: Sequence[Any]
    ) -> float:
        """Computes stochastic edit distance.

        Stochastic edit distance \propto sum_{edits} p(target, edit | source) =
        p(target | source).

        Args:
            source: Source string.
            target: Target string.

        Returns:
            Probability score.
        """
        return self.forward_evaluate(source, target)[-1, -1]

    def update_model(
        self,
        sources: Sequence[Iterable[Any]],
        targets: Sequence[Iterable[Any]],
        iterations: int = 10,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Update weights by maximizing likelihood by Expectation-Maximization.

        Args:
            sources: Source strings.
            targets: Target strings.
            iterations: Number of iterations of EM.
            output_path: Path where to write learned weights."""
        logging.info(
            "Updating model parameters by maximizing likelihood using "
            "EM (%d iterations).",
            iterations,
        )
        self.em(sources, targets, iterations)

        if output_path is not None:
            self.to_pickle(output_path)
            logging.info("Wrote latest model weights to %s.", output_path)

    def action_sequence_cost(
        self, x: Sequence[Any], y: Sequence[Any], x_offset: int, y_offset: int
    ) -> float:

        return -self.viterbi_distance(source=x[x_offset:], target=y[y_offset:])

    def action_cost(self, action: Edit) -> float:
        if isinstance(action, Del):
            return -self.delta_del.get(action.old, self.default)
        if isinstance(action, Ins):
            return -self.delta_ins.get(action.new, self.default)
        if isinstance(action, Sub):
            return -self.delta_sub.get((action.old, action.new), self.default)
        if isinstance(action, EndOfSequence):
            return -self.delta_eos
        if isinstance(action, Copy):
            return -self.delta_sub.get((action.old, action.old), self.default)
        raise ValueError(f"Unknown action!: {action}!")
