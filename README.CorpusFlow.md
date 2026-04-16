# Corpus Flow README

## Purpose

The original `training/all_works.txt` approach mixed too many formats into one file:

* HTML
* XML
* CSV
* JSON
* markdown
* multilingual text
* navigation and metadata noise

That created a weak training corpus and made iteration slow.

The new flow separates corpus building from tokenization and training.

## Main idea

Use:

1. `python/0_prepare_literature_corpus.py`
2. `python/0_tokenize_all_works.py`
3. `python/1_train_gpt_all_works.py`

The corpus builder now emits sharded text files and can format structured data as chat-style supervision:

```text
STARTTOK User: What does Genesis 1:1 say? Assistant: In the beginning God created the heavens and the earth. ENDTOK
```

That is a better fit than plain `Question:` / `Answer:` labels for training a conversational next-token model.

## Recommended flows

### 1. Clean chat corpus

Best for:

* question answering
* factual response style
* lower-noise structured supervision

Run:

```bash
./prepare_chat_corpus.sh
./tokenize_corpus.sh training/corpus_chat learned/corpus_chat_tokenized
```

Default sources:

* `literature/english_dict.json.txt`
* selected English Bible CSV files

### 2. Mixed corpus

Best for:

* combining Q/A behaviour with general literary style
* broader language modeling without pulling in the whole raw library

Run:

```bash
./prepare_mixed_corpus.sh
./tokenize_corpus.sh training/corpus_mixed learned/corpus_mixed_tokenized
```

Default sources:

* the same structured chat sources as above
* Shakespeare HTML plays as cleaned plain-text records

### 3. Structured v2 continuation corpus

Best for:

* preserving the current mixed-model checkpoint line
* rebuilding the same material into cleaner source buckets
* continuing training with the existing vocabulary and token IDs

Run:

```bash
./prepare_structured_v2_corpus.sh
./tokenize_compatible_corpus.sh training/corpus_structured_v2 learned/corpus_structured_v2_compatible
./continue_structured_v2.sh
```

Default source buckets:

* `dictionary/`: term -> clean definition chat records
* `bible/`: one chosen English Bible CSV as reference -> verse chat records
* `fables/`: Aesop and selected Grimm / Perrault texts
* `stories/`: selected English prose stories
* `literature/`: selected literary style texts

### 4. Train or resume

Run:

```bash
./train_all_works.sh
```

The training wrapper will resume automatically from:

* `learned/gpt_all_works/checkpoint_latest.pt`
* then `learned/gpt_all_works/checkpoint.pt`

if either exists.

## Important notes

### The builder is still selective, not magical

The `literature/` tree contains a lot of unsuitable material:

* docs
* package files
* project metadata
* multilingual exports
* HTML navigation

The helper scripts intentionally target safer subsets first.

### Better Q/A needs structured sources

Proper `User:` / `Assistant:` pairs are strongest when the source already has a question-like key and a clean answer field.

Good examples:

* dictionary term -> definition
* Bible reference -> verse text

Weak examples:

* arbitrary prose forced into fake Q/A
* noisy HTML with no semantic structure

### Shards are accepted directly by the tokenizer

`python/0_tokenize_all_works.py` now accepts either:

* one input file
* a directory containing shard `.txt` files

So there is no need to concatenate them by hand first.

## Useful custom runs

### Small test run

```bash
./prepare_chat_corpus.sh --output training/corpus_chat_small --max-records 20000
./tokenize_corpus.sh training/corpus_chat_small learned/corpus_chat_small_tokenized --max-vocab-size 16384
```

### Custom subset

```bash
python/0_prepare_literature_corpus.py literature training/custom_corpus \
  --mode mixed \
  --qa-style chat \
  --include-glob 'english_dict.json.txt' \
  --include-glob 'bible_databases/formats/csv/Darby.csv' \
  --include-glob 'shakespeare/**/*.html' \
  --exclude-glob 'shakespeare/poetry/*'
```

### Compare current and structured-v2 lines

Run:

```bash
./eval_qa.sh
./eval_structured_v2.sh
./compare_eval_reports.sh \
  evaluation/reports/gpt_corpus_mixed_128k_checkpoint_latest.json \
  evaluation/reports/gpt_corpus_structured_v2_checkpoint_latest.json
```

## Files added for this flow

* `python/0_prepare_literature_corpus.py`
* `prepare_chat_corpus.sh`
* `prepare_mixed_corpus.sh`
* `prepare_structured_v2_corpus.sh`
* `tokenize_corpus.sh`
* `tokenize_compatible_corpus.sh`
* `train_all_works.sh`
* `finetune_chat_corpus.sh`
* `continue_structured_v2.sh`
* `eval_structured_v2.sh`
* `python/5_compare_eval_reports.py`

## Continue training on a new corpus

If you already have a trained checkpoint and want to continue on a better
dictionary/chat corpus, do not build a brand new vocabulary.

If you retokenize with a new vocabulary:

* token IDs change
* embedding rows change meaning
* output classes change meaning
* the checkpoint is no longer compatible

Use the existing vocabulary instead:

```bash
./prepare_chat_corpus.sh
./tokenize_compatible_corpus.sh training/corpus_chat learned/corpus_chat_compatible_tokenized
./finetune_chat_corpus.sh --epochs 1
```

That path keeps token IDs aligned with the current model so training can resume
from the existing checkpoint rather than restarting from scratch.

For the current mixed-model line, the compatible continuation flow is:

```bash
./prepare_structured_v2_corpus.sh
./tokenize_compatible_corpus.sh training/corpus_structured_v2 learned/corpus_structured_v2_compatible
./continue_structured_v2.sh
```
