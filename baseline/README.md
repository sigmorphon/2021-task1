# Baseline

This year's baseline is a ensembled neural transition system based on the
imitation learning paradigm introduced by Makarov & Clematide (2018).

## How to use

1.  The baseline requires Python 3.7. If your system does not use Python 3.7 by
    default (i.e., see `python --version` for your default Python), create and
    activate either:

    -   a [3.7 virtualenv](https://virtualenv.pypa.io/en/latest/):

    ```bash
    virtualenv --python=python3.7 sigmorphon
    source sigmorphon/bin/activate
    ```

    -   or a [3.7 Conda
        environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html#installing-a-different-version-of-python):

    ```bash
    conda create --name=sigmorphon python=3.7
    conda activate sigmorphon
    ```

2.  Install the requirements and the library itself:

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install .
    ```

3.  Run the sweep (may take a while):

    ```bash
    ./sweep
    ```

## License

The baseline is made available under the [Apache 2.0](LICENSE.txt) license.

## References

Makarov, P., and Clematide, S. 2018. [Imitation learning for neural
morphological string transduction](https://www.aclweb.org/anthology/D18-1314/).
In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing*, pages 2877-2882.
