#!/usr/bin/env python
"""Evaluates sequence model.

This script assumes the gold and hypothesis data is stored in a two-column TSV
file, one example per line."""

import argparse

import evallib


def main(args: argparse.Namespace) -> None:
    correct = 0
    incorrect = 0
    for (gold, hypo) in evallib.tsv_reader(args.tsv_path):
        if gold == hypo:
            correct += 1
        else:
            incorrect += 1
    wer = evallib.wer(correct, incorrect)
    print(f"WER:\t{wer:5.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates sequence model")
    parser.add_argument("tsv_path", help="path to gold/hypo TSV file")
    main(parser.parse_args())
