# Learning Model README

## Purpose

This directory builds a simple language-learning pipeline step by step, starting from visible and understandable rules rather than jumping straight into a neural network.

The work began with simple ordering and moved through:

* alphabet ordering
* word counting
* bigrams
* trigrams
* fourgrams
* deterministic generation
* weighted random generation
* verse boundary handling with `STARTTOK` and `ENDTOK`

The goal was to understand how a model can move from simple counting toward sentence-like generation, while keeping every step inspectable.

## Why this was done in stages

A neural network can come later, however starting with n-gram style models makes the mechanics plain.

At each step the model gains one more unit of context:

* unigram: which words exist and how often
* bigram: given one word, what tends to come next
* trigram: given two words, what tends to come next
* fourgram: given three words, what tends to come next

This shows the real tradeoff clearly:

* more context gives more coherent text
* more context costs more time and storage
* local phrasing improves before true long-range meaning does

## Source corpus

The working corpus was the Darby Bible.

The project used several forms of that corpus:

* `Darby.txt` for the early flat-text runs
* `Darby.fixed.txt` after repairing obvious broken joins such as missing spaces before `God`
* `Darby.verses.txt` for boundary-aware training, with one verse per line and markers:

  * `STARTTOK`
  * `ENDTOK`

Verse boundaries were chosen because the current models only remember a short context window. A verse is a better fit for short-memory models than a whole chapter.

## Main files

### 1. `alphabet.c`

A first toy step to show how simple ranking can be learned and then used for ordering.

Build:

```bash
gcc alphabet.c -o alphabet
```

Run:

```bash
./alphabet
```

What it demonstrates:

* a model can begin with no ordering
* repeated pairwise constraints can produce a ranking
* sorting is built on comparison

### 2. `word_model.c`

Reads text, tokenises words, counts them, and stores the learned vocabulary.

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 word_model.c -o word_model
```

Run on flat text:

```bash
./word_model Darby.txt learned_words.txt
```

Run on cleaned text:

```bash
./word_model Darby.fixed.txt learned_words_fixed.txt
```

Run on verse-bounded text:

```bash
./word_model Darby.verses.txt learned_words_verses.txt
```

What it demonstrates:

* the model can learn vocabulary frequency
* the corpus shape becomes visible through counts
* text cleanliness matters

### 3. `bigram_model.c`

Learns adjacent word pairs.

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 bigram_model.c -o bigram_model
```

Run:

```bash
./bigram_model Darby.fixed.txt learned_bigrams_fixed.txt
./bigram_model Darby.verses.txt learned_bigrams_verses.txt
```

What it demonstrates:

* given one word, what tends to follow
* local language structure begins to appear
* style and formulaic phrasing become visible

### 4. `query_bigrams.c`

Queries the bigram model to show what tends to follow a chosen word.

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 query_bigrams.c -o query_bigrams
```

Examples:

```bash
./query_bigrams learned_bigrams.txt jehovah
./query_bigrams learned_bigrams.txt shall
./query_bigrams learned_bigrams.txt thou
```

What it demonstrates:

* the model is learning real corpus transitions
* common continuations can be inspected directly

### 5. `trigram_model.c`

Learns three-word windows, meaning two words of context predicting the third.

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 trigram_model.c -o trigram_model
```

Run:

```bash
./trigram_model Darby.fixed.txt learned_trigrams_fixed.txt
./trigram_model Darby.verses.txt learned_trigrams_verses.txt
```

What it demonstrates:

* phrase structure becomes stronger than with bigrams
* generation begins to sound more verse-like
* loops still happen because memory is short

### 6. `query_trigrams.c`

Queries the trigram model for the most common continuation of a two-word prefix.

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 query_trigrams.c -o query_trigrams
```

Examples:

```bash
./query_trigrams learned_trigrams_fixed.txt children of
./query_trigrams learned_trigrams_fixed.txt came to
./query_trigrams learned_trigrams_fixed.txt jehovah said
./query_trigrams learned_trigrams_fixed.txt and thou
```

What it demonstrates:

* two-word context is much stronger than one-word context
* some phrase corridors become very obvious

### 7. `generate_trigram.c`

A deterministic trigram generator.

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 generate_trigram.c -o generate_trigram
```

Examples:

```bash
./generate_trigram learned_trigrams_fixed.txt jehovah said 20
./generate_trigram learned_trigrams_verses.txt starttok and 20
```

What it demonstrates:

* always choosing the strongest continuation causes loops
* local truth does not guarantee a globally faithful sentence
* `ENDTOK` is useful as a stop marker

### 8. `generate_trigram_random.c`

A weighted-random trigram generator.

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 generate_trigram_random.c -o generate_trigram_random
```

Examples:

```bash
./generate_trigram_random learned_trigrams_verses.txt jehovah said 20 12345
./generate_trigram_random learned_trigrams_verses.txt starttok in 20 12643
```

What it demonstrates:

* counts can be used as probabilities
* the seed controls reproducible randomness
* variation improves compared with greedy generation
* short-memory drift still remains

### 9. `fourgram_model.c`

Learns four-word windows, meaning three words of context predicting the fourth.

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 fourgram_model.c -o fourgram_model
```

Run:

```bash
rm -f learned_fourgrams_verses.txt
./fourgram_model Darby.verses.txt learned_fourgrams_verses.txt
```

What it demonstrates:

* one more word of memory improves phrase stability noticeably
* exact phrase corridors become much stronger
* training time rises sharply with a simple linear-search implementation

Important note:

The verse-based version was corrected so that each line is treated separately. This prevents false cross-verse patterns such as `endtok starttok ...` from dominating the learned model.

### 10. `query_fourgrams.c`

Queries the fourgram model for the most common continuation of a three-word prefix.

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 query_fourgrams.c -o query_fourgrams
```

Examples:

```bash
./query_fourgrams learned_fourgrams_verses.txt the children of
./query_fourgrams learned_fourgrams_verses.txt it came to
./query_fourgrams learned_fourgrams_verses.txt and he said
./query_fourgrams learned_fourgrams_verses.txt the word of
./query_fourgrams learned_fourgrams_verses.txt saith the lord
./query_fourgrams learned_fourgrams_verses.txt starttok and he
```

What it demonstrates:

* fourgram context is much more stable than trigram context
* some scriptural phrases become very sharp and clear

### 11. `generate_fourgram_random.c`

A weighted-random fourgram generator using three-word prefixes.

Build:

```bash
gcc -O2 -Wall -Wextra -std=c11 generate_fourgram_random.c -o generate_fourgram_random
```

Examples:

```bash
./generate_fourgram_random learned_fourgrams_verses.txt the children of 20 12345
./generate_fourgram_random learned_fourgrams_verses.txt it came to 20 12345
./generate_fourgram_random learned_fourgrams_verses.txt and he said 20 12345
./generate_fourgram_random learned_fourgrams_verses.txt the word of 20 12345
./generate_fourgram_random learned_fourgrams_verses.txt saith the lord 20 12345
./generate_fourgram_random learned_fourgrams_verses.txt and jesus said 100 298327
```

What it demonstrates:

* fourgram generation is noticeably more coherent than trigram generation
* local phrase structure can remain strong for longer spans
* the model still blends scriptural fragments rather than maintaining one full intention
* generation is slower because the whole fourgram file is rescanned at each step

## Corpus cleaning and repair

The corpus was not clean at the beginning. Missing spaces and punctuation issues produced tokens such as:

* `thygod`
* `thegod`
* `yourgod`
* `ourgod`

The JSON source was repaired with `jq` and then flattened again for training. The repaired text became `Darby.fixed.txt`, and later verse-bounded text became `Darby.verses.txt`.

This part of the work showed something important:

A model often fails first because the text is dirty, not because the model idea is wrong.

## What was learned

### 1. More context helps

The progression from bigram to trigram to fourgram clearly improved local coherence.

### 2. More context costs more

Training time rose sharply, especially for fourgrams.

### 3. Local truth is not global truth

Generated text can consist of real local fragments without being a true verse or a coherent whole sentence from the corpus.

### 4. Boundaries matter

Verse boundaries with `STARTTOK` and `ENDTOK` improved generation and prevented false cross-verse blending.

### 5. Randomness matters

Greedy generation loops badly. Weighted randomness gives more varied and often more natural output.

## Typical workflow

### Build everything

```bash
gcc alphabet.c -o alphabet
gcc -O2 -Wall -Wextra -std=c11 word_model.c -o word_model
gcc -O2 -Wall -Wextra -std=c11 bigram_model.c -o bigram_model
gcc -O2 -Wall -Wextra -std=c11 query_bigrams.c -o query_bigrams
gcc -O2 -Wall -Wextra -std=c11 trigram_model.c -o trigram_model
gcc -O2 -Wall -Wextra -std=c11 query_trigrams.c -o query_trigrams
gcc -O2 -Wall -Wextra -std=c11 generate_trigram.c -o generate_trigram
gcc -O2 -Wall -Wextra -std=c11 generate_trigram_random.c -o generate_trigram_random
gcc -O2 -Wall -Wextra -std=c11 fourgram_model.c -o fourgram_model
gcc -O2 -Wall -Wextra -std=c11 query_fourgrams.c -o query_fourgrams
gcc -O2 -Wall -Wextra -std=c11 generate_fourgram_random.c -o generate_fourgram_random
```

### Train verse-based models

```bash
rm -f learned_words_verses.txt learned_bigrams_verses.txt learned_trigrams_verses.txt learned_fourgrams_verses.txt

./word_model Darby.verses.txt learned_words_verses.txt
./bigram_model Darby.verses.txt learned_bigrams_verses.txt
./trigram_model Darby.verses.txt learned_trigrams_verses.txt
./fourgram_model Darby.verses.txt learned_fourgrams_verses.txt
```

### Query

```bash
./query_bigrams learned_bigrams_verses.txt baal
./query_trigrams learned_trigrams_verses.txt jehovah said
./query_fourgrams learned_fourgrams_verses.txt the word of
```

### Generate

```bash
./generate_trigram learned_trigrams_verses.txt starttok and 20
./generate_trigram_random learned_trigrams_verses.txt jehovah said 20 12345
./generate_fourgram_random learned_fourgrams_verses.txt and he said 20 12345
```

## Why stop here before a neural network

This project has now shown clearly what classical n-gram models can do and where they fail.

They can:

* learn style
* learn phrase corridors
* generate locally plausible text
* be inspected directly

They cannot:

* keep long-range meaning reliably
* generalise well beyond exact observed phrase chains
* understand topic or intention in the deeper sense

That makes this the right point to move on to a first neural model.

## Next step

The next stage should be a very small neural network, built with the same spirit:

* simple
* inspectable
* tested step by step
* small enough to understand fully

A character-level or very small token-level predictor would be the natural next build.

