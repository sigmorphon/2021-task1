"""Utility functions and classes."""
from typing import Any, Dict, List, Optional, TextIO
import dataclasses
import logging
import os
import time
import re
import unicodedata


@dataclasses.dataclass
class Sample:
    input: str
    target: Optional[str]
    encoded_input: Optional[List[int]] = None


@dataclasses.dataclass
class DecodingOutput:
    accuracy: float
    loss: float
    predictions: List[str]


class OpenNormalize:
    def __init__(self, filename: str, normalize: bool, mode: str = "rt"):
        self.filename = filename
        self.file: Optional[TextIO] = None
        mode_pattern = re.compile(r"[arw]t?$")
        if not mode_pattern.match(mode):
            raise ValueError(
                f"Unexpected mode {mode_pattern.pattern}: {mode}."
            )
        self.mode = mode
        if normalize:
            form = "NFD" if self.mode.startswith("r") else "NFC"
            self.normalize = lambda line: unicodedata.normalize(form, line)
        else:
            self.normalize = lambda line: line

    def __enter__(self):
        self.file = open(self.filename, mode=self.mode, encoding="utf8")
        return self

    def __iter__(self):
        for line in self.file:
            yield self.normalize(line)

    def write(self, line: str):
        if not isinstance(line, str):
            raise ValueError(
                f"Line is not a unicode string ({type(line)}): {line}"
            )
        return self.file.write(self.normalize(line))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


def write_results(
    accuracy: float,
    predictions: List[str],
    output: str,
    normalize: bool,
    dataset_name: str,
    beam_width: int = 1,
    decoding_name: Optional[str] = None,
    dargs: Dict[str, Any] = None,
):

    logging.info("%s set accuracy: %.4f.", dataset_name.title(), accuracy)

    if decoding_name is None:
        decoding_name = "greedy" if beam_width == 1 else f"beam{beam_width}"

    eval_file = os.path.join(output, f"{dataset_name}_{decoding_name}.eval")

    with open(eval_file, mode="w") as w:
        if dargs is not None:
            for key, value in dargs.items():
                w.write(f"{key}: {value}\n")
        w.write(f"{dataset_name} accuracy: {accuracy:.4f}\n")

    predictions_tsv = os.path.join(
        output, f"{dataset_name}_{decoding_name}.predictions"
    )

    with OpenNormalize(predictions_tsv, normalize, mode="w") as w:
        w.write("\n".join(predictions))


class Timer:
    def __init__(self):
        self.time = None

    def __enter__(self):
        self.time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.info("\t...finished in %.3f sec.", time.time() - self.time)
