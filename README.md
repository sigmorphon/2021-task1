# Task 1: Second SIGMORPHON Shared Task on Grapheme-to-Phoneme Conversions

In this task, participants will create computational models that map a sequence
of "graphemes"---characters---representing a word to a transcription of that
word's pronunciation. This task is an important part of speech technologies,
including recognition and synthesis. This is the second iteration of this task.

To sign-up for the mailing list, please click [this link](https://groups.google.com/u/1/g/sigmorphon-2021-task-1), then below the search bar, click the button that reads, "Ask to join group." You will be added shortly.

## Data

### Source

The data is extracted from the English-language portion of
[Wiktionary](https://en.wiktionary.org/) using
[WikiPron](https://github.com/kylebgorman/wikipron) (Lee et al. 2020), then
filtered and downsampled using proprietary techniques.

### Format

Training and development data are UTF-8-encoded tab-separated values files. Each
example occupies a single line and consists of a grapheme sequence---a sequence
of [NFC](https://en.wikipedia.org/wiki/Unicode_equivalence#Normal_forms) Unicode
codepoints---a tab character, and the corresponding phone sequence, a
roughly-phonemic IPA, tokenized using the
[`segments`](https://github.com/cldf/segments) library. The following shows
three lines of Romanian data:

    antonim a n t o n i m
    ploaie  p lʷ a j e
    pornește    p o r n e ʃ t e

The provided test data is of a similar format but only has the first column,
containing grapheme sequences.

### Subtasks

There are three subtasks, which will be scored separately. Participant teams may
submit as many systems as they want to as many subtasks as they want.

In all three subtasks, the data is randomly split into training (80%),
development (10%), and testing (10%) data.

#### Subtask 1: high-resource subtask

Subtask 1 consists of a roughly 41,000-word sample from a single language:
American English. Subtask participants are permitted to use all external
resources, including other pronunciation dictionaries like the [CMU Pronouncing
Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict), except Wiktionary or
WikiPron itself.

#### Subtask 2: medium-resource subtask

Subtask 2 consists of 10,000 words from the following ten languages:

1.  `arm_e`: Armenian (Eastern dialect)
2.  `bul`: Bulgarian
3.  `dut`: Dutch
4.  `fre`: French
5.  `geo`: Georgian
6.  `hbs_latn`: Serbo-Croatian (Latin script)
7.  `hun`: Hungarian
8.  `jpn_hira`: Japanese (Hiragana script)
9.  `kor`: Korean
10. `vie_hanoi`: Vietnamese (Hanoi dialect)

Subtask 2 participants are not permitted to use any external resources except
for [UniMorph](https://unimorph.github.io/) paradigms; these paradigms may be
used to lemmatize, or to look up part-of-speech tags for words, for instance.

#### Subtask 3: low-resource subtask

Subtask 3, which simulates a low-resource setting, consists of 1,000 words from
the following ten languages:

1.  `ady`: Adyghe
2.  `gre`: Modern Greek
3.  `ice`: Icelandic
4.  `ita`: Italian
5.  `khm`: Khmer
6.  `lav`: Latvian
7.  `mlt_latn`: Maltese (Latin script)
8.  `rum`: Romanian
9.  `slv`: Slovene
10. `wel_sw`: Welsh (South Wales dialect)

Subtask 3 participants are not permitted to use any external resources.

## Evaluation

The metric used to rank systems is *word error rate* (WER), the percentage of
words for which the hypothesized transcription sequence does not match the gold
transcription. This value, in accordance with common practice, is a decimal
value multiplied by 100 (e.g.: 13.53). In the medium- and low-frequency tasks,
WER is macro-averaged across all ten languages. We provide two Python scripts
for evaluation:

-   [`evaluate.py`](evaluation/evaluate.py) computes the WER for one language.
-   [`evaluate_all.py`](evaluation/evaluate_all.py) computes per-language and
    average WER across multiple languages.

## Submission

**Please submit your results in the two-column (grapheme sequence,
tab-character, tokenized phone sequence) TSV format, the same one used for the
training and development data.** If you use an internal representation other
than NFC, you must convert back before submitting.

**TBD**: Email for submission.

## Timeline

**TBD**: This is currently estimated and may change in the near future.

-   March 1, 2021: Data released.
-   March 8, 2021: Baseline code and results released.
-   May 1, 2021: Participants' submissions due.
-   May 8, 2021: Participants' draft system description papers due.
-   May 15, 2021: Participants' camera-ready system description papers due.

## Baseline

This year's baseline is a ensembled neural transition system based on the
imitation learning paradigm introduced by Makarov & Clematide (2018). A variant
of this system (Makarov & Clematide 2020) was the second-best system overall in
the 2020 shared task (Gorman et al. 2020). Code, predictions, and results for
the baseline will also be provided.

**TBD**: Add links to the baseline library.

## Comparison with the 2020 shared task

In contrast to the 2020 shared task (Gorman et al. 2020):

-   There is a new baseline.
-   There are new languages.
-   There are three subtasks.
-   There are no suprise languages.
-   The data been subjected to novel quality-assurance procedures.

## Organizers

The task is organized by members of the Computational Linguistics Lab at the
[Graduate Center, City University of New York](https://gc.cuny.edu/Home) and the
Institut für Computerlinguistik at the [University of
Zurich](https://www.uzh.ch/en.html).

## References

Gorman, K., Ashby, L. F.E., Goyzueta, A., McCarthy, A. D., Wu, S., and You, D.
2020. [The SIGMORPHON 2020 shared task on multilingual grapheme-to-phoneme
conversion](https://www.aclweb.org/anthology/2020.sigmorphon-1.2/). In *17th
SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and
Morphology*, pages 40-50.

Lee, J. L, Ashby, L. F.E., Garza, M. E., Lee-Sikka, Y., Miller, S., Wong, A.,
McCarthy, A. D., and Gorman, K. 2020. [Massively multilingual pronunciation
mining with WikiPron](https://www.aclweb.org/anthology/2020.lrec-1.521/). In
*Proceedings of the 12th Language Resources and Evaluation Conference*, pages
4223-4228.

Makarov, P., and Clematide, S. 2018. [Imitation learning for neural
morphological string transduction](https://www.aclweb.org/anthology/D18-1314/).
In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing*, pages 2877-2882.

Makarov, P., and Clematide, S. 2020. [CLUZH at SIGMORPHON 2020 shared task on
multi-lingual grapheme-to-phoneme
conversion](https://www.aclweb.org/anthology/2020.sigmorphon-1.19/). In
*Proceedings of the 17th SIGMORPHON Workshopon Computational Research in
Phonetics, Phonology, and Morphology*, pages 171-176.
