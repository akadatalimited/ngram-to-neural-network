# Neural Model README

## Purpose

This stage begins the move from exact count-based language models to learned weight-based models.

The n-gram work stored exact observed sequences:

* word counts
* bigrams
* trigrams
* fourgrams

The neural stage changes the method.

Instead of storing exact phrase chains, the model learns numeric weights that help it predict the next character from a short fixed context.

This first neural stage stays deliberately small and readable:

* plain C
* no framework
* one hidden layer
* character-level prediction
* explicit forward pass and backpropagation

## Why start with a character model

A character-level model is a sensible first neural step because it keeps the moving parts small.

Advantages:

* simple input pipeline
* no heavy tokeniser yet
* small vocabulary
* easy to inspect training behaviour
* easy to compare with n-gram generation

The task is:

> given the last N characters, predict the next character

That is the neural equivalent of the earlier context-based prediction work.

## Files in this stage

### 1. `src/nn_char_model.c`

Trains a small character-level neural network.

It:

* reads the training text
* builds a character vocabulary
* uses one-hot encoding for the last `context_len` characters
* predicts the next character
* updates weights using backpropagation
* saves the learned model to a binary file

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 src/nn_char_model.c -lm -o out/nn_char_model
```

Example training run:

```bash
time out/nn_char_model training/Darby.verses.txt learned/nn_char_darby.bin 3 8 64 0.05
```

Larger run used later:

```bash
time out/nn_char_model training/Darby.verses.txt learned/nn_char_darby.bin 5 8 96 0.05
```

### 2. `src/nn_char_generate.c`

Loads a trained model and generates text from a seed string.

It supports:

* greedy decoding
* random decoding
* optional random seed

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 src/nn_char_generate.c -lm -o out/nn_char_generate
```

Examples:

```bash
out/nn_char_generate learned/nn_char_darby.bin "And God " 300 greedy
out/nn_char_generate learned/nn_char_darby.bin "And God " 300 random 12345
out/nn_char_generate learned/nn_char_darby.bin "And God " 300 random 54321
```

### 3. `src/nn_char_generate_temp.c`

Adds temperature-controlled decoding.

This changes generation behaviour without changing the trained model.

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 src/nn_char_generate_temp.c -lm -o out/nn_char_generate_temp
```

Examples:

```bash
out/nn_char_generate_temp learned/nn_char_darby.bin "And God " 300 0.7 random 12345
out/nn_char_generate_temp learned/nn_char_darby.bin "And God " 300 1.0 random 12345
out/nn_char_generate_temp learned/nn_char_darby.bin "And God " 300 1.3 random 12345
out/nn_char_generate_temp learned/nn_char_darby.bin "And God " 300 1.0 greedy
```

### 4. `src/nn_char_generate_markers.c`

Adds output handling for `STARTTOK` and `ENDTOK`.

Because the character model was trained on verse-bounded text, it learned those marker words as ordinary character sequences. This file cleans presentation by:

* suppressing `STARTTOK`
* turning `ENDTOK` into a newline
* allowing generation to stop after a chosen number of verse endings

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 src/nn_char_generate_markers.c -lm -o out/nn_char_generate_markers
```

Examples:

```bash
out/nn_char_generate_markers learned/nn_char_darby.bin "Thus say " 500 1.1 5 random 12345
out/nn_char_generate_markers learned/nn_char_darby.bin "Thus say " 500 0.9 5 random 16342
out/nn_char_generate_markers learned/nn_char_darby.bin "And God " 300 1.0 2 greedy
```

## Training corpus used

The neural model was trained on:

```text
training/Darby.verses.txt
```

This corpus was chosen because:

* it was already cleaned and prepared during the n-gram stage
* it includes `STARTTOK` and `ENDTOK`
* each line corresponds to a verse
* the corpus is already well understood from earlier experiments

That made it a strong first neural training set.

## What the first training runs showed

### First run

Command:

```bash
time out/nn_char_model training/Darby.verses.txt learned/nn_char_darby.bin 3 8 64 0.05
```

Observed loss:

* epoch 1: `1.276458`
* epoch 2: `1.200957`
* epoch 3: `1.186347`

Approximate runtime:

* about `3m39s`

Meaning:

* the data pipeline worked
* the network learned real predictive structure
* the loss decreased as expected

### Larger run

Command:

```bash
time out/nn_char_model training/Darby.verses.txt learned/nn_char_darby.bin 5 8 96 0.05
```

Observed loss:

* epoch 1: `1.224951`
* epoch 2: `1.135885`
* epoch 3: `1.117906`
* epoch 4: `1.108617`
* epoch 5: `1.102725`

Approximate runtime:

* about `9m8s`

Meaning:

* more capacity and more epochs improved learning
* the model became noticeably stronger
* runtime increased in proportion to the larger network and longer training

## What generation showed

### Greedy decoding

Greedy decoding repeatedly chose the highest-probability next character.

This caused collapse into repetitive corridors such as:

* `the things to the things ...`
* `the seven to the things ...`

This matches the earlier lesson from n-grams:

* greedy decoding is usually too rigid
* it reveals the strongest corridor in the model
* it is useful mainly as a diagnostic

### Random decoding

Random decoding produced much more interesting outputs.

It showed:

* recognisable biblical style
* plausible-looking phrases
* malformed spellings
* invented word-like forms
* much more variation than greedy mode

This was the first clear sign that the neural model was learning style rather than storing exact observed word sequences.

## Temperature findings

Temperature affects only generation, not training.

The trained model file stays the same. Temperature reshapes the output probabilities during sampling.

Observed behaviour on this model:

* around `0.9` to `1.2` gave the best balance of readability and variation
* `0.8` to `0.9` was tighter and safer, however more likely to repeat
* `1.4` was already pushing beyond sensible output for longer generations
* `2.0` and above became increasingly unstable
* very high values like `9.0` produced near-random character soup

Important note:

* temperature strongly affects **random** decoding
* temperature usually changes little in **greedy** mode, because greedy still picks the single top character

Also observed:

* very low temperatures such as `0.1` or `0.01` make random decoding behave almost like greedy decoding

## Markers: `STARTTOK` and `ENDTOK`

Because the network is character-level, it does not understand tokens in the symbolic sense.

So `STARTTOK` and `ENDTOK` were learned simply as common character sequences.

That is why raw neural generation began emitting them literally.

This was handled in the next step by `nn_char_generate_markers.c`, which:

* hides `STARTTOK`
* turns `ENDTOK` into a line break
* lets output be read as verse-like units

## What is different from n-grams

### N-gram models

The earlier models stored exact observed sequences.

Example idea:

```text
the word of -> jehovah
```

They:

* preserve exact observed chains
* improve when context length grows
* become expensive and sparse as context grows
* drift because they only know local windows

### Neural character model

The neural model does not store exact phrases directly.

Instead it learns weights that make certain next characters more likely in certain contexts.

It can therefore:

* continue from unseen exact sequences
* invent plausible-looking word forms
* capture style without exact phrase lookup

However it still struggles with:

* spelling accuracy
* long-range coherence
* stable word boundaries

## How to use this stage

### Train

```bash
out/nn_char_model training/Darby.verses.txt learned/nn_char_darby.bin 5 8 96 0.05
```

### Generate plain

```bash
out/nn_char_generate learned/nn_char_darby.bin "And God " 300 random 12345
```

### Generate with temperature

```bash
out/nn_char_generate_temp learned/nn_char_darby.bin "Jesus said" 300 0.9 random 24823612
```

### Generate with marker cleanup

```bash
out/nn_char_generate_markers learned/nn_char_darby.bin "Thus say " 500 1.1 5 random 12345
```

## What this stage proved

This first neural stage showed clearly that:

* a very small neural net can already learn corpus style
* loss decreases in a healthy way under training
* character-level models can produce plausible scriptural fragments
* greedy decoding collapses into repetition
* random decoding with temperature is the useful mode
* marker handling matters for readability

## Why stop here for this stage

This is now a complete first neural step:

* train
* generate
* control temperature
* clean boundary markers

That is enough to make the contrast with n-grams very clear.

## Natural next steps

The next step should not be chosen blindly. Reasonable options now are:

* slightly longer context length
* slightly larger hidden layer
* token-level or word-level neural model
* cleaner handling of training markers in the training corpus itself
* documenting the neural targets in a separate `Makefile.neural`

The important thing is that the first neural step is now complete and can be referred back to later.


## Second neural stage: word-level model

The next step after the character model was to move to a word-level neural network.

The task changes from:

> given the last N characters, predict the next character

to:

> given the last N words, predict the next word

This increases complexity significantly:

* vocabulary becomes large (thousands of tokens instead of ~100 characters)
* model size grows quickly
* training time increases
* outputs become more structurally meaningful (words instead of fragments)

### Files added in this stage

#### `src/nn_word_model.c`

Trains a simple word-level neural network.

It:

* tokenises input text into words
* builds a vocabulary (fixed size cap)
* encodes context as word indices
* predicts the next word
* trains using backpropagation
* saves the model as a binary file

#### `src/nn_word_generate.c`

Generates text from a trained word model.

Supports:

* greedy decoding
* random decoding
* seeded randomness

#### `src/nn_word_generate_markers.c`

Handles presentation of special tokens.

Because the training corpus includes boundary markers, the model learns them as normal tokens.

This file:

* removes `starttok`
* removes `endtok`
* allows output to be presented as readable text

### Vocabulary size experiments

Two vocabulary sizes were tested:

#### 4K vocabulary

Command:

```bash
out/nn_word_model training/Darby.verses.txt learned/nn_word_darby.bin 3 4 64 0.05
```

Observed:

* training tokens: `841933`
* vocab size: `4096`
* epochs: `3`
* context length: `4`
* hidden size: `64`

Loss:

* epoch 1: `4.330761`
* epoch 2: `3.936000`
* epoch 3: `3.767138`

Runtime:

* about `24 minutes`

Behaviour:

* strong repetition of common structures
* frequent use of high-frequency words (`the`, `of`, etc.)
* noticeable fallback to generic phrase patterns

#### 8K vocabulary

Command:

```bash
out/nn_word_model training/Darby.verses.txt learned/nn_word_darby8k.bin 3 4 64 0.05
```

Observed:

* vocab size: `8192`

Loss:

* epoch 1: `4.508208`
* epoch 2: `4.099454`
* epoch 3: `3.924306`

Runtime:

* about `47 minutes`

Behaviour:

* broader word coverage
* less collapse into the most common phrase loops
* more varied and expressive output

Important note:

Loss values are not directly comparable between 4K and 8K models, because the prediction space is larger in the 8K case.

### Generation observations

With marker-aware generation:

Example:

```bash
out/nn_word_generate_markers learned/nn_word_darby8k.bin "and god" 40 3 random 12345
```

Observed outputs showed:

* recognisable structure at word level
* improved readability compared to character model
* still limited long-range coherence
* occasional unusual or rare word combinations

The model now produces full words instead of fragments, which is a major qualitative step forward.

### Model size observation

The neural model binary is significantly larger than the input text.

Example:

* training corpus: ~4.4MB
* 4K model: ~5.1MB
* 8K model: ~11MB

Reason:

* neural networks store dense weight matrices
* size scales with:

  * vocabulary size
  * hidden layer size
  * context length

This is fundamentally different from n-gram storage, which scales with observed sequences.

### Training data considerations

The model was trained using a single prepared corpus:

```text
training/Darby.verses.txt
```

Key points:

* one clean corpus is preferable at this stage
* multiple formats (JSON, CSV, YAML, SQL) are not additional information
* avoid duplicating the same text across formats
* introduce new texts deliberately and one at a time

### Neural Makefile

A dedicated build file was introduced:

```text
Makefile.neuralnet
```

This mirrors the earlier `Makefile.ngram` and provides:

* build targets for:

  * character model tools
  * word model tools
* clear separation of neural stage from n-gram stage

### What this stage proved

The transition from character to word model showed:

* vocabulary size has a strong effect on output quality
* training time scales significantly with vocabulary size
* word-level models improve readability immediately
* neural models store knowledge in weights, not explicit sequences

### Current position

At this point the neural path includes:

* character-level model
* word-level model
* marker-aware generation
* temperature-based sampling

This forms a complete second stage beyond n-grams.

Further expansion (larger corpora, multiple translations, token models) should be approached carefully to preserve clarity of learning outcomes.

