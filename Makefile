# Learning Model Makefile
#
# Layout:
#   src/      - C source files
#   out/      - compiled binaries
#   learned/  - generated learned model text files
#   training/ - input corpus files
#
# Main teaching order:
#   1. alphabet
#   2. word_model
#   3. bigram_model + query_bigrams
#   4. trigram_model + query_trigrams + generators
#   5. fourgram_model + query_fourgrams + generator
#   6. json_word_model (corpus/json helper)

CC := gcc
CFLAGS := -O2 -Wall -Wextra -std=c11

SRC_DIR := src
OUT_DIR := out
LEARNED_DIR := learned
TRAINING_DIR := training

ALPHABET_SRC := $(SRC_DIR)/alphabet.c
WORD_MODEL_SRC := $(SRC_DIR)/word_model.c
BIGRAM_MODEL_SRC := $(SRC_DIR)/bigram_model.c
QUERY_BIGRAMS_SRC := $(SRC_DIR)/query_bigrams.c
TRIGRAM_MODEL_SRC := $(SRC_DIR)/trigram_model.c
QUERY_TRIGRAMS_SRC := $(SRC_DIR)/query_trigrams.c
GENERATE_TRIGRAM_SRC := $(SRC_DIR)/generate_trigram.c
GENERATE_TRIGRAM_RANDOM_SRC := $(SRC_DIR)/generate_trigram_random.c
FOURGRAM_MODEL_SRC := $(SRC_DIR)/fourgram_model.c
QUERY_FOURGRAMS_SRC := $(SRC_DIR)/query_fourgrams.c
GENERATE_FOURGRAM_RANDOM_SRC := $(SRC_DIR)/generate_fourgram_random.c
JSON_WORD_MODEL_SRC := $(SRC_DIR)/json_word_model.c

ALPHABET_BIN := $(OUT_DIR)/alphabet
WORD_MODEL_BIN := $(OUT_DIR)/word_model
BIGRAM_MODEL_BIN := $(OUT_DIR)/bigram_model
QUERY_BIGRAMS_BIN := $(OUT_DIR)/query_bigrams
TRIGRAM_MODEL_BIN := $(OUT_DIR)/trigram_model
QUERY_TRIGRAMS_BIN := $(OUT_DIR)/query_trigrams
GENERATE_TRIGRAM_BIN := $(OUT_DIR)/generate_trigram
GENERATE_TRIGRAM_RANDOM_BIN := $(OUT_DIR)/generate_trigram_random
FOURGRAM_MODEL_BIN := $(OUT_DIR)/fourgram_model
QUERY_FOURGRAMS_BIN := $(OUT_DIR)/query_fourgrams
GENERATE_FOURGRAM_RANDOM_BIN := $(OUT_DIR)/generate_fourgram_random
JSON_WORD_MODEL_BIN := $(OUT_DIR)/json_word_model

DARBY_TXT := $(TRAINING_DIR)/Darby.txt
DARBY_FIXED_TXT := $(TRAINING_DIR)/Darby.fixed.txt
DARBY_VERSES_TXT := $(TRAINING_DIR)/Darby.verses.txt

WORDS_TXT := $(LEARNED_DIR)/learned_words.txt
WORDS_FIXED_TXT := $(LEARNED_DIR)/learned_words_fixed.txt
WORDS_VERSES_TXT := $(LEARNED_DIR)/learned_words_verses.txt

BIGRAMS_TXT := $(LEARNED_DIR)/learned_bigrams.txt
BIGRAMS_FIXED_TXT := $(LEARNED_DIR)/learned_bigrams_fixed.txt
BIGRAMS_VERSES_TXT := $(LEARNED_DIR)/learned_bigrams_verses.txt

TRIGRAMS_FIXED_TXT := $(LEARNED_DIR)/learned_trigrams_fixed.txt
TRIGRAMS_VERSES_TXT := $(LEARNED_DIR)/learned_trigrams_verses.txt

FOURGRAMS_VERSES_TXT := $(LEARNED_DIR)/learned_fourgrams_verses.txt

.PHONY: help all dirs \
	stage1 stage2 stage3 stage4 stage5 stage6 \
	alphabet word-model bigram-model query-bigrams trigram-model query-trigrams \
	generate-trigram generate-trigram-random fourgram-model query-fourgrams \
	generate-fourgram-random json-word-model \
	train-flat train-fixed train-verses train-fourgrams \
	query-bigrams-demo query-trigrams-demo query-fourgrams-demo \
	generate-trigram-demo generate-trigram-random-demo generate-fourgram-random-demo \
	clean distclean

help:
	@echo "Learning Model Makefile"
	@echo
	@echo "Build stages in teaching order:"
	@echo "  make stage1   Build alphabet example"
	@echo "  make stage2   Build word model"
	@echo "  make stage3   Build bigram model and query tool"
	@echo "  make stage4   Build trigram model, query tool, and generators"
	@echo "  make stage5   Build fourgram model, query tool, and generator"
	@echo "  make stage6   Build JSON helper model"
	@echo
	@echo "Build everything:"
	@echo "  make all"
	@echo
	@echo "Training targets:"
	@echo "  make train-flat       Train unigram and bigram models on training/Darby.txt"
	@echo "  make train-fixed      Train unigram, bigram, trigram on training/Darby.fixed.txt"
	@echo "  make train-verses     Train unigram, bigram, trigram on training/Darby.verses.txt"
	@echo "  make train-fourgrams  Train fourgram model on training/Darby.verses.txt"
	@echo
	@echo "Query demos:"
	@echo "  make query-bigrams-demo"
	@echo "  make query-trigrams-demo"
	@echo "  make query-fourgrams-demo"
	@echo
	@echo "Generation demos:"
	@echo "  make generate-trigram-demo"
	@echo "  make generate-trigram-random-demo"
	@echo "  make generate-fourgram-random-demo"
	@echo
	@echo "Cleanup:"
	@echo "  make clean       Remove compiled binaries from out/"
	@echo "  make distclean   Remove compiled binaries and learned/*.txt"
	@echo
	@echo "Manual examples:"
	@echo "  $(OUT_DIR)/query_bigrams $(BIGRAMS_FIXED_TXT) jehovah"
	@echo "  $(OUT_DIR)/query_trigrams $(TRIGRAMS_VERSES_TXT) jehovah said"
	@echo "  $(OUT_DIR)/query_fourgrams $(FOURGRAMS_VERSES_TXT) the word of"
	@echo "  $(OUT_DIR)/generate_fourgram_random $(FOURGRAMS_VERSES_TXT) and jesus said 40 12345"

dirs:
	@mkdir -p $(OUT_DIR) $(LEARNED_DIR)

all: stage1 stage2 stage3 stage4 stage5 stage6

stage1: alphabet
stage2: word-model
stage3: bigram-model query-bigrams
stage4: trigram-model query-trigrams generate-trigram generate-trigram-random
stage5: fourgram-model query-fourgrams generate-fourgram-random
stage6: json-word-model

alphabet: $(ALPHABET_BIN)
word-model: $(WORD_MODEL_BIN)
bigram-model: $(BIGRAM_MODEL_BIN)
query-bigrams: $(QUERY_BIGRAMS_BIN)
trigram-model: $(TRIGRAM_MODEL_BIN)
query-trigrams: $(QUERY_TRIGRAMS_BIN)
generate-trigram: $(GENERATE_TRIGRAM_BIN)
generate-trigram-random: $(GENERATE_TRIGRAM_RANDOM_BIN)
fourgram-model: $(FOURGRAM_MODEL_BIN)
query-fourgrams: $(QUERY_FOURGRAMS_BIN)
generate-fourgram-random: $(GENERATE_FOURGRAM_RANDOM_BIN)
json-word-model: $(JSON_WORD_MODEL_BIN)

$(ALPHABET_BIN): $(ALPHABET_SRC) | dirs
	$(CC) $(ALPHABET_SRC) -o $(ALPHABET_BIN)

$(WORD_MODEL_BIN): $(WORD_MODEL_SRC) | dirs
	$(CC) $(CFLAGS) $(WORD_MODEL_SRC) -o $(WORD_MODEL_BIN)

$(BIGRAM_MODEL_BIN): $(BIGRAM_MODEL_SRC) | dirs
	$(CC) $(CFLAGS) $(BIGRAM_MODEL_SRC) -o $(BIGRAM_MODEL_BIN)

$(QUERY_BIGRAMS_BIN): $(QUERY_BIGRAMS_SRC) | dirs
	$(CC) $(CFLAGS) $(QUERY_BIGRAMS_SRC) -o $(QUERY_BIGRAMS_BIN)

$(TRIGRAM_MODEL_BIN): $(TRIGRAM_MODEL_SRC) | dirs
	$(CC) $(CFLAGS) $(TRIGRAM_MODEL_SRC) -o $(TRIGRAM_MODEL_BIN)

$(QUERY_TRIGRAMS_BIN): $(QUERY_TRIGRAMS_SRC) | dirs
	$(CC) $(CFLAGS) $(QUERY_TRIGRAMS_SRC) -o $(QUERY_TRIGRAMS_BIN)

$(GENERATE_TRIGRAM_BIN): $(GENERATE_TRIGRAM_SRC) | dirs
	$(CC) $(CFLAGS) $(GENERATE_TRIGRAM_SRC) -o $(GENERATE_TRIGRAM_BIN)

$(GENERATE_TRIGRAM_RANDOM_BIN): $(GENERATE_TRIGRAM_RANDOM_SRC) | dirs
	$(CC) $(CFLAGS) $(GENERATE_TRIGRAM_RANDOM_SRC) -o $(GENERATE_TRIGRAM_RANDOM_BIN)

$(FOURGRAM_MODEL_BIN): $(FOURGRAM_MODEL_SRC) | dirs
	$(CC) $(CFLAGS) $(FOURGRAM_MODEL_SRC) -o $(FOURGRAM_MODEL_BIN)

$(QUERY_FOURGRAMS_BIN): $(QUERY_FOURGRAMS_SRC) | dirs
	$(CC) $(CFLAGS) $(QUERY_FOURGRAMS_SRC) -o $(QUERY_FOURGRAMS_BIN)

$(GENERATE_FOURGRAM_RANDOM_BIN): $(GENERATE_FOURGRAM_RANDOM_SRC) | dirs
	$(CC) $(CFLAGS) $(GENERATE_FOURGRAM_RANDOM_SRC) -o $(GENERATE_FOURGRAM_RANDOM_BIN)

$(JSON_WORD_MODEL_BIN): $(JSON_WORD_MODEL_SRC) | dirs
	$(CC) $(CFLAGS) $(JSON_WORD_MODEL_SRC) -o $(JSON_WORD_MODEL_BIN)

train-flat: $(WORD_MODEL_BIN) $(BIGRAM_MODEL_BIN)
	$(WORD_MODEL_BIN) $(DARBY_TXT) $(WORDS_TXT)
	$(BIGRAM_MODEL_BIN) $(DARBY_TXT) $(BIGRAMS_TXT)

train-fixed: $(WORD_MODEL_BIN) $(BIGRAM_MODEL_BIN) $(TRIGRAM_MODEL_BIN)
	$(WORD_MODEL_BIN) $(DARBY_FIXED_TXT) $(WORDS_FIXED_TXT)
	$(BIGRAM_MODEL_BIN) $(DARBY_FIXED_TXT) $(BIGRAMS_FIXED_TXT)
	$(TRIGRAM_MODEL_BIN) $(DARBY_FIXED_TXT) $(TRIGRAMS_FIXED_TXT)

train-verses: $(WORD_MODEL_BIN) $(BIGRAM_MODEL_BIN) $(TRIGRAM_MODEL_BIN)
	$(WORD_MODEL_BIN) $(DARBY_VERSES_TXT) $(WORDS_VERSES_TXT)
	$(BIGRAM_MODEL_BIN) $(DARBY_VERSES_TXT) $(BIGRAMS_VERSES_TXT)
	$(TRIGRAM_MODEL_BIN) $(DARBY_VERSES_TXT) $(TRIGRAMS_VERSES_TXT)

train-fourgrams: $(FOURGRAM_MODEL_BIN)
	$(FOURGRAM_MODEL_BIN) $(DARBY_VERSES_TXT) $(FOURGRAMS_VERSES_TXT)

query-bigrams-demo: $(QUERY_BIGRAMS_BIN)
	$(QUERY_BIGRAMS_BIN) $(BIGRAMS_FIXED_TXT) jehovah
	@echo
	$(QUERY_BIGRAMS_BIN) $(BIGRAMS_FIXED_TXT) shall
	@echo
	$(QUERY_BIGRAMS_BIN) $(BIGRAMS_FIXED_TXT) thou

query-trigrams-demo: $(QUERY_TRIGRAMS_BIN)
	$(QUERY_TRIGRAMS_BIN) $(TRIGRAMS_VERSES_TXT) jehovah said
	@echo
	$(QUERY_TRIGRAMS_BIN) $(TRIGRAMS_VERSES_TXT) and thou
	@echo
	$(QUERY_TRIGRAMS_BIN) $(TRIGRAMS_VERSES_TXT) it came

query-fourgrams-demo: $(QUERY_FOURGRAMS_BIN)
	$(QUERY_FOURGRAMS_BIN) $(FOURGRAMS_VERSES_TXT) the children of
	@echo
	$(QUERY_FOURGRAMS_BIN) $(FOURGRAMS_VERSES_TXT) it came to
	@echo
	$(QUERY_FOURGRAMS_BIN) $(FOURGRAMS_VERSES_TXT) the word of

generate-trigram-demo: $(GENERATE_TRIGRAM_BIN)
	$(GENERATE_TRIGRAM_BIN) $(TRIGRAMS_VERSES_TXT) starttok and 20
	@echo
	$(GENERATE_TRIGRAM_BIN) $(TRIGRAMS_VERSES_TXT) jehovah said 20

generate-trigram-random-demo: $(GENERATE_TRIGRAM_RANDOM_BIN)
	$(GENERATE_TRIGRAM_RANDOM_BIN) $(TRIGRAMS_VERSES_TXT) jehovah said 20 12345
	@echo
	$(GENERATE_TRIGRAM_RANDOM_BIN) $(TRIGRAMS_VERSES_TXT) starttok in 20 12643

generate-fourgram-random-demo: $(GENERATE_FOURGRAM_RANDOM_BIN)
	$(GENERATE_FOURGRAM_RANDOM_BIN) $(FOURGRAMS_VERSES_TXT) and jesus said 40 12345
	@echo
	$(GENERATE_FOURGRAM_RANDOM_BIN) $(FOURGRAMS_VERSES_TXT) the word of 20 12345

clean:
	rm -f $(OUT_DIR)/*

distclean: clean
	rm -f $(LEARNED_DIR)/*.txt

