#!/usr/bin/env python
"""Evaluates sequence model.

This script assumes the gold and hypothesis data is stored in a two-column TSV
file, one example per line."""

import argparse
import statistics

import evallib


def main(args: argparse.Namespace) -> None:
    wers = []
    for tsv_path in args.tsv_paths:
        correct = 0
        incorrect = 0
        for (gold, hypo) in evallib.tsv_reader(tsv_path):
            if gold == hypo:
                correct += 1
            else:
                incorrect += 1
        wer = evallib.wer(correct, incorrect)
        wers.append(wer)
        print(f"{tsv_path} WER:\t{wer:5.2f}")
    wer = statistics.mean(wers)
    print(f"Macro-average WER:\t{wer:5.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates sequence model")
    parser.add_argument(
        "tsv_paths", nargs="+", help="path to gold/hypo TSV file"
    )
    main(parser.parse_args())
