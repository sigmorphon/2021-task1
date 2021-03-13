"""Evaluation functions for sequence models."""

from typing import Iterator, List, Tuple


Labels = List[str]


def wer(correct: int, incorrect: int) -> float:
    """Computes WER."""
    return 100 * incorrect / (correct + incorrect)


def tsv_reader(path: str) -> Iterator[Tuple[Labels, Labels]]:
    """Reads pairs of strings from a TSV filepath."""
    with open(path, "r") as source:
        for line in source:
            (gold, hypo) = line.split("\t", 1)
            # Stripping is performed after the fact so the previous line
            # doesn't fail when `hypo` is null.
            hypo = hypo.rstrip()
            yield (gold.split(), hypo.split())
