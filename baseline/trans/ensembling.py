"""Performs majority-vote ensembling over files with predictions."""

from typing import TextIO

import argparse
import collections
import logging
import os

from trans import utils


def read_files(fileobj: TextIO):
    samples = []
    for line in fileobj:
        input_, prediction = line.rstrip("\n").split("\t", 1)
        samples.append(utils.Sample(input_, prediction))
    return samples


def main(args: argparse.Namespace):

    os.path.exists(args.output) or os.makedirs(args.output)

    logging.info(
        "Producing a majority-vote prediction file from %d system files.",
        len(args.systems),
    )

    with open(args.gold, encoding="utf8") as f:
        gold = read_files(f)

    systems = []
    for system in args.systems:
        with open(system, encoding="utf8") as f:
            systems.append(read_files(f))

    length = len(gold)
    for j, system in enumerate(systems):
        if len(system) != length:
            raise ValueError(
                f"Number of lines mismatch between gold and {j}th system file: "
                f"{length} vs {len(system)}."
            )

    line_number = 0
    correct = 0
    predictions = []
    for gold_sample, *system_samples in zip(gold, *systems):
        input_ = gold_sample.input
        sample_predictions = []
        for j, system_sample in enumerate(system_samples):
            if system_sample.input != input_:
                raise ValueError(
                    f"Input mismatch between gold and {j}th system: "
                    f"Line {line_number}: {input_} vs {system_sample.input}."
                )
            sample_predictions.append(system_sample.target)
        majority_prediction = collections.Counter(
            sample_predictions
        ).most_common(1)[0][0]
        if majority_prediction == gold_sample.target:
            correct += 1
        predictions.append(f"{input_}\t{majority_prediction}")
        line_number += 1

    accuracy = correct / length
    dataset_name = "test" if "test" in os.path.basename(args.gold) else "dev"
    decoding_name = f"{len(args.systems)}ensemble"
    utils.write_results(
        accuracy,
        predictions,
        args.output,
        normalize=False,
        dataset_name=dataset_name,
        decoding_name=decoding_name,
        dargs=args.__dict__,
    )


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="path to gold data")
    parser.add_argument(
        "--systems",
        required=True,
        nargs="+",
        help="path to system data",
    )
    parser.add_argument(
        "--output", required=True, help="output directory path"
    )
    main(parser.parse_args())
