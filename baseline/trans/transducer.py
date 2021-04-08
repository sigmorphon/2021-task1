"""Defines a neural transducer."""
from typing import Any, Dict, List, Optional
import dataclasses
import functools
import heapq

import dynet as dy
import numpy as np

from trans import optimal_expert
from trans import vocabulary
from trans.actions import (
    ConditionalCopy,
    ConditionalDel,
    ConditionalIns,
    ConditionalSub,
    Edit,
    EndOfSequence,
    GenerativeEdit,
)
from trans.vocabulary import BEGIN_WORD, COPY, DELETE, END_WORD


MAX_ACTION_SEQ_LEN = 150


@functools.total_ordering
@dataclasses.dataclass
class Output:
    action_history: List[Any]
    output: str
    log_p: float
    losses: List[dy.Expression] = None

    def __lt__(self, other):
        return self.log_p < other.log_p

    def __eq__(self, other):
        return self.log_p == other.log_p


@dataclasses.dataclass
class Hypothesis:
    action_history: List[Any]
    alignment: int
    decoder: dy.RNNState
    negative_log_p: float
    output: List[str]


@functools.total_ordering
@dataclasses.dataclass
class Expansion:
    action: Any
    decoder: dy.RNNState
    from_hypothesis: Hypothesis
    negative_log_p: float

    def __lt__(self, other):
        return self.negative_log_p < other.negative_log_p

    def __eq__(self, other):
        return self.negative_log_p == other.negative_log_p


class Transducer:
    def __init__(
        self,
        model,
        vocab: vocabulary.Vocabularies,
        expert: optimal_expert.Expert,
        char_dim: int,
        action_dim: int,
        enc_hidden_dim: int,
        enc_layers: int,
        dec_hidden_dim: int,
        dec_layers: int,
        **kwargs,
    ):

        self.vocab = vocab
        self.optimal_expert = expert

        self.number_characters = len(vocab.characters)
        self.number_actions = len(vocab.actions)
        self.substitutions = self.vocab.substitutions
        self.inserts = self.vocab.insertions
        lstm = dy.CoupledLSTMBuilder

        # encoder
        self.char_lookup = model.add_lookup_parameters(
            (self.number_characters, char_dim)
        )

        self.fenc = lstm(enc_layers, char_dim, enc_hidden_dim, model)
        self.benc = lstm(enc_layers, char_dim, enc_hidden_dim, model)

        # decoder
        self.act_lookup = model.add_lookup_parameters(
            (self.number_actions, action_dim)
        )

        self.dec = lstm(
            dec_layers, enc_hidden_dim * 2 + action_dim, dec_hidden_dim, model
        )

        # classifier
        self.pW = model.add_parameters((self.number_actions, dec_hidden_dim))
        self.pb = model.add_parameters(self.number_actions)

    def input_embedding(self, input_: List[int], is_training: bool):
        """Returns a list of character embeddings for the input."""
        if is_training:
            emb = [self.char_lookup[i] for i in input_]
        else:
            # UNK is the average of trained embeddings (excluding UNK)
            unk = dy.average(
                [self.char_lookup[i] for i in range(1, self.number_characters)]
            )
            emb = [
                self.char_lookup[i] if i < self.number_characters else unk
                for i in input_
            ]
        return emb

    def bidirectional_encoding(self, embeddings: List[dy.Expression]):
        """Bidirectional LSTM encoding of the input embeddings."""
        f_init = self.fenc.initial_state()
        b_init = self.benc.initial_state()
        f_states = f_init.add_inputs(embeddings)
        b_states = reversed(b_init.add_inputs(reversed(embeddings)))
        return [
            dy.concatenate([fs.output(), bs.output()])
            for fs, bs in zip(f_states, b_states)
        ]

    def compute_valid_actions(self, length_encoder_suffix: int) -> List[int]:
        valid_actions = [END_WORD]
        valid_actions.extend(self.inserts)
        if length_encoder_suffix > 1:
            valid_actions.extend([COPY, DELETE])
            valid_actions.extend(self.substitutions)
        return valid_actions

    @staticmethod
    def sample(log_probs: np.array) -> int:
        """Samples an action from a log-probability distribution."""
        dist = np.exp(log_probs)
        rand = np.random.rand()
        for action, p in enumerate(dist):
            rand -= p
            if rand <= 0:
                break
        return action

    @staticmethod
    def remap_actions(action_scores: Dict[Any, float]) -> Dict[Any, float]:
        """Maps a generative oracle's edit to their conditional counterparts."""
        remapped_action_scores = dict()
        for action, score in action_scores.items():
            if isinstance(action, GenerativeEdit):
                remapped_action = action.conditional_counterpart()
            elif isinstance(action, Edit):
                remapped_action = action
            else:
                raise ValueError(
                    f"Unknown action: {action, score}.\n"
                    f"action_scores: {action_scores}"
                )
            remapped_action_scores[remapped_action] = score
        return remapped_action_scores

    def expert_rollout(
        self, input_: str, target: str, alignment: int, prediction: List[str]
    ) -> List[int]:
        """Rolls out wit;h optimal expert policy.

        Args:
            input_: Input string (x).
            target: Target prediction (t).
            alignment: Position of control in the input string.
            prediction: The current prediction so far (y).

        Returns:
            List of optimal actions as integer codes."""
        raw_action_scores = self.optimal_expert.score(
            input_, target, alignment, prediction
        )
        action_scores = self.remap_actions(raw_action_scores)

        optimal_value = min(action_scores.values())
        return [
            self.vocab.encode_unseen_action(action)
            for action, value in action_scores.items()
            if value == optimal_value
        ]

    def log_sum_softmax_loss(
        self,
        optimal_actions: List[int],
        logits: dy.Expression,
        valid_actions: List[int],
    ) -> dy.Expression:
        """Compute log loss similar to Riezler et al 2000."""
        log_validity = np.full(self.number_actions, -np.inf)  # invalid
        log_validity[valid_actions] = 0.0
        logits += dy.inputVector(log_validity)
        log_sum_selected_terms = dy.logsumexp(
            [dy.pick(logits, index=e) for e in optimal_actions]
        )
        normalization_term = dy.logsumexp(list(logits))
        return log_sum_selected_terms - normalization_term

    def transduce(
        self,
        input_: str,
        encoded_input: List[int],
        target: Optional[str] = None,
        rollin: Optional[float] = None,
        external_cg: bool = True,
    ):
        """Runs the transducer for dynamic-oracle training and greedy decoding.

        Args:
            input_: Input string.
            encoded_input: List of integer character codes.
            target: Target string during training, `None` during prediction.
            external_cg: Whether an external computation graph is defined.
            rollin: The probability with which an action sampled from the model
                    is executed. Used during training."""
        if not external_cg:
            dy.renew_cg()

        is_training = bool(target)
        input_emb = self.input_embedding(encoded_input, is_training)
        bidirectional_emb = self.bidirectional_encoding(input_emb)[
            1:
        ]  # drop BEGIN_WORD
        input_length = len(bidirectional_emb)
        decoder = self.dec.initial_state()

        alignment = 0
        action_history: List[int] = [BEGIN_WORD]
        output: List[str] = []
        losses: List[dy.Expression] = []
        log_p = 0.0

        while len(action_history) <= MAX_ACTION_SEQ_LEN:

            length_encoder_suffix = input_length - alignment
            valid_actions = self.compute_valid_actions(length_encoder_suffix)

            input_char_embedding = bidirectional_emb[alignment]
            previous_action_embedding = self.act_lookup[action_history[-1]]
            decoder_input = dy.concatenate(
                [input_char_embedding, previous_action_embedding]
            )
            decoder = decoder.add_input(decoder_input)

            decoder_output = decoder.output()
            logits = self.pW * decoder_output + self.pb
            log_probs = dy.log_softmax(logits, valid_actions)

            log_probs_np = log_probs.npvalue()

            if target is None:
                # argmax decoding
                action = np.argmax(log_probs_np)
            else:
                # training with dynamic oracle

                # 1. ACTIONS TO MAXIMIZE
                optim_actions = self.expert_rollout(
                    input_, target, alignment, output
                )

                loss = self.log_sum_softmax_loss(
                    optim_actions, logits, valid_actions
                )

                # 2. ACTION SPACE EXPLORATION: NEXT ACTION
                if np.random.rand() <= rollin:
                    # action is picked by sampling
                    action = self.sample(log_probs_np)
                else:
                    # action is picked from optim_actions
                    # reinforce model beliefs by picking highest probability
                    # action that is consistent with oracle
                    action = optim_actions[
                        int(
                            np.argmax([log_probs_np[a] for a in optim_actions])
                        )
                    ]
                losses.append(loss)

            log_p += log_probs_np[action]
            action_history.append(action)
            # execute the action to update the transducer state
            action = self.vocab.decode_action(action)

            if isinstance(action, ConditionalCopy):
                char_ = input_[alignment]
                alignment += 1
                output.append(char_)
            elif isinstance(action, ConditionalDel):
                alignment += 1
            elif isinstance(action, ConditionalIns):
                output.append(action.new)
            elif isinstance(action, ConditionalSub):
                alignment += 1
                output.append(action.new)
            elif isinstance(action, EndOfSequence):
                break
            else:
                raise ValueError(f"Unknown action: {action}.")

        return Output(action_history, "".join(output), log_p, losses)

    def beam_search_decode(
        self,
        input_: str,
        encoded_input: List[int],
        beam_width: int,
        external_cg: bool = True,
    ):

        if not external_cg:
            dy.renew_cg()

        input_emb = self.input_embedding(encoded_input, is_training=False)
        bidirectional_emb = self.bidirectional_encoding(input_emb)[
            1:
        ]  # drop BEGIN_WORD
        input_length = len(bidirectional_emb)
        decoder = self.dec.initial_state()

        beam: List[Hypothesis] = [
            Hypothesis(
                action_history=[BEGIN_WORD],
                alignment=0,
                decoder=decoder,
                negative_log_p=0.0,
                output=[],
            )
        ]

        hypothesis_length = 0
        complete_hypotheses = []

        while (
            beam and beam_width > 0 and hypothesis_length <= MAX_ACTION_SEQ_LEN
        ):

            expansions: List[Hypothesis] = []

            for hypothesis in beam:

                length_encoder_suffix = input_length - hypothesis.alignment
                valid_actions = self.compute_valid_actions(
                    length_encoder_suffix
                )
                # decoder
                decoder_input = dy.concatenate(
                    [
                        bidirectional_emb[hypothesis.alignment],
                        self.act_lookup[hypothesis.action_history[-1]],
                    ]
                )
                decoder = hypothesis.decoder.add_input(decoder_input)
                # classifier
                logits = self.pW * decoder.output() + self.pb
                log_probs_expr = dy.log_softmax(logits, valid_actions)
                log_probs = log_probs_expr.npvalue()

                for action in valid_actions:

                    log_p = (
                        hypothesis.negative_log_p - log_probs[action]
                    )  # min heap, so minus

                    heapq.heappush(
                        expansions,
                        Expansion(action, decoder, hypothesis, log_p),
                    )

            beam: List[Hypothesis] = []

            for _ in range(beam_width):

                expansion: Expansion = heapq.heappop(expansions)
                from_hypothesis = expansion.from_hypothesis
                action = expansion.action
                action_history = list(from_hypothesis.action_history)
                action_history.append(action)
                output = list(from_hypothesis.output)

                # execute the action to update the transducer state
                action = self.vocab.decode_action(action)

                if isinstance(action, EndOfSequence):
                    # 1. COMPLETE HYPOTHESIS, REDUCE BEAM
                    complete_hypothesis = Output(
                        action_history=action_history,
                        output="".join(output),
                        log_p=-expansion.negative_log_p,
                    )  # undo min heap minus

                    complete_hypotheses.append(complete_hypothesis)
                    beam_width -= 1
                else:
                    # 2. EXECUTE ACTION AND ADD FULL HYPOTHESIS TO NEW BEAM
                    alignment = from_hypothesis.alignment

                    if isinstance(action, ConditionalCopy):
                        char_ = input_[alignment]
                        alignment += 1
                        output.append(char_)
                    elif isinstance(action, ConditionalDel):
                        alignment += 1
                    elif isinstance(action, ConditionalIns):
                        output.append(action.new)
                    elif isinstance(action, ConditionalSub):
                        alignment += 1
                        output.append(action.new)
                    else:
                        raise ValueError(f"Unknown action: {action}.")

                    hypothesis = Hypothesis(
                        action_history=action_history,
                        alignment=alignment,
                        decoder=expansion.decoder,
                        negative_log_p=expansion.negative_log_p,
                        output=output,
                    )

                    beam.append(hypothesis)

            hypothesis_length += 1

        if not complete_hypotheses:
            # nothing found because the model is very bad
            for hypothesis in beam:

                complete_hypothesis = Output(
                    action_history=hypothesis.action_history,
                    output="".join(hypothesis.output),
                    log_p=-hypothesis.negative_log_p,
                )  # undo min heap minus

                complete_hypotheses.append(complete_hypothesis)

        complete_hypotheses.sort(reverse=True)
        return complete_hypotheses
