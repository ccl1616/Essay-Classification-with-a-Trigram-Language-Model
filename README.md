# Essay Classification with a Trigram Language Model

## Introduction

This project implements a trigram-based statistical language model and evaluate by a essay classification evaluator. The model calculates probabilities of word sequences and evaluates text based on its statistical structure. It supports features such as:

- Counting unigram, bigram, and trigram occurrences.
- Computing raw and smoothed probabilities.
- Generating sentences based on a trained language model.
- Evaluating sentence likelihood using perplexity.
- Classifying essays based on perplexity scores, achieved an **accuracy of 86.25%**.

<!-- This project demonstrates my ability to work with foundational natural language processing techniques and statistical models. -->

## Algorithm & Techniques Used

- **N-Gram Language Modeling**:
  - Probabilistic modeling using frequency counts of n-grams.
  - Raw probabilities for unigrams, bigrams, and trigrams.
  ![Trigram Language Model Diagram](https://miro.medium.com/v2/resize:fit:885/1*AeneIZX0g2kZ4sj9p2k7Ew.png)
    [Reference](https://www.google.com/url?sa=i&url=https%3A%2F%2Fayselaydin.medium.com%2F7-understanding-n-grams-in-nlp-03109b218113&psig=AOvVaw04tbwQmPBp6qtV8LM57HtB&ust=1734811302595000&source=images&cd=vfe&opi=89978449&ved=0CAMQjB1qFwoTCLi3mO6Rt4oDFQAAAAAdAAAAABAY)
- **Smoothing**:
  - Linear interpolation with equal weights for smoothed trigram probability computation.
- **Perplexity**:
  - Measures the quality of the language model for a given corpus.
- **Sentence Generation**:
  - Produces realistic sentences using trigram probabilities.
- **Essay Scoring Experiment**:
  - Classifies essays into "high" or "low" quality based on perplexity.

## Data Preparation

1. **Corpus Reader**: Reads text files and processes sentences into word sequences. Words not in the lexicon are replaced with a special token (`UNK`).
2. **Lexicon Construction**: Builds a lexicon of words appearing more than once in the corpus. Adds `START`, `STOP`, and `UNK` tokens for preprocessing.
3. **N-Grams**: Generates unigram, bigram, and trigram sequences with appropriate padding (`START`, `STOP`).

## Workflow

1. **Data Preparation**:
   -  **`corpus_reader`** Reads and processes the input corpus.
   -  **`get_ngrams`** Generates n-grams from sequences.
   - **`count_ngrams`** Counts occurrences of unigrams, bigrams, and trigrams. Stores counts in dictionaries for efficient lookup.
2. **Evaluation**:
   - **`raw_*_probability`** Computes raw probabilities for unigrams, bigrams, and trigrams.
   - **`smoothed_trigram_probability`** Uses linear interpolation for smoothed trigram probabilities.
   - Computes sentence log-probabilities and corpus perplexity for evaluation.
3. **Generation**:
   - **`generate_sentence`** Generates sentences based on trigram probabilities.
   - Randomly selects words using probabilities from trained data.
4. **Essay Classification**:
   - **`perplexity`** **`essay_scoring_experiment`** Evaluates essays using perplexity scores to determine their classification accuracy.
   - Perplexity score: It can be thought of as an evaluation of the model’s ability to predict uniformly among the set of specified tokens in a corpus. [Reference](https://huggingface.co/docs/transformers/en/perplexity)
       ![](https://miro.medium.com/v2/resize:fit:1400/1*nYdAKtgkpz95DQVsaHKa6A.png)
       [Reference](https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72)


## How to Run

To train and test the model, use the following command:
```bash
python trigram_model.py <training_corpus> <test_corpus>
```

## Result: Essay Classification Experiment
**Objective**
- To classify essays as "high-quality" or "low-quality" by comparing their perplexity scores under two separate trigram models trained on high- and low-quality essay corpora.

**Methodology**
* The perplexity of each essay in the test sets was computed using both models. The essay was classified as "high-quality" or "low-quality" based on the model yielding the lower perplexity score.

**Result** 
* The experiment achieved an accuracy of 86.25%, demonstrating the model's ability to distinguish between essay categories based on language patterns.

