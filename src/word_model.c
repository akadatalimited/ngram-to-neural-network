#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>

#define MAX_WORD_LEN 128
#define INITIAL_CAPACITY 1024

struct WordEntry {
    char *word;
    unsigned long count;
};

struct Model {
    struct WordEntry *entries;
    size_t used;
    size_t capacity;
};

static char *xstrdup(const char *src)
{
    size_t len = strlen(src) + 1;
    char *copy = malloc(len);

    if (copy == NULL) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }

    memcpy(copy, src, len);
    return copy;
}

static void model_init(struct Model *model)
{
    model->entries = calloc(INITIAL_CAPACITY, sizeof(struct WordEntry));
    model->used = 0;
    model->capacity = INITIAL_CAPACITY;

    if (model->entries == NULL) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }
}

static void model_free(struct Model *model)
{
    size_t i;

    for (i = 0; i < model->used; i++) {
        free(model->entries[i].word);
    }

    free(model->entries);
    model->entries = NULL;
    model->used = 0;
    model->capacity = 0;
}

static void model_grow(struct Model *model)
{
    struct WordEntry *new_entries;
    size_t new_capacity = model->capacity * 2;

    new_entries = realloc(model->entries, new_capacity * sizeof(struct WordEntry));
    if (new_entries == NULL) {
        fprintf(stderr, "memory reallocation failed\n");
        exit(1);
    }

    memset(new_entries + model->capacity, 0,
           (new_capacity - model->capacity) * sizeof(struct WordEntry));

    model->entries = new_entries;
    model->capacity = new_capacity;
}

static ptrdiff_t model_find_word(const struct Model *model, const char *word)
{
    size_t i;

    for (i = 0; i < model->used; i++) {
        if (strcmp(model->entries[i].word, word) == 0) {
            return (ptrdiff_t)i;
        }
    }

    return -1;
}

static void model_add_word(struct Model *model, const char *word)
{
    ptrdiff_t index;

    if (word[0] == '\0') {
        return;
    }

    index = model_find_word(model, word);
    if (index >= 0) {
        model->entries[(size_t)index].count++;
        return;
    }

    if (model->used == model->capacity) {
        model_grow(model);
    }

    model->entries[model->used].word = xstrdup(word);
    model->entries[model->used].count = 1;
    model->used++;
}

static void model_save(const struct Model *model, const char *filename)
{
    FILE *fp;
    size_t i;

    fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("fopen");
        exit(1);
    }

    for (i = 0; i < model->used; i++) {
        fprintf(fp, "%s\t%lu\n",
                model->entries[i].word,
                model->entries[i].count);
    }

    fclose(fp);
}

static void model_load(struct Model *model, const char *filename)
{
    FILE *fp;
    char word[MAX_WORD_LEN];
    unsigned long count;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        return;
    }

    while (fscanf(fp, "%127s\t%lu", word, &count) == 2) {
        ptrdiff_t index = model_find_word(model, word);

        if (index >= 0) {
            model->entries[(size_t)index].count += count;
        } else {
            if (model->used == model->capacity) {
                model_grow(model);
            }

            model->entries[model->used].word = xstrdup(word);
            model->entries[model->used].count = count;
            model->used++;
        }
    }

    fclose(fp);
}

static void process_stream(FILE *fp, struct Model *model)
{
    int ch;
    char word[MAX_WORD_LEN];
    size_t pos = 0;

    while ((ch = fgetc(fp)) != EOF) {
        if (isalnum((unsigned char)ch)) {
            if (pos + 1 < MAX_WORD_LEN) {
                word[pos++] = (char)tolower((unsigned char)ch);
            }
        } else {
            if (pos > 0) {
                word[pos] = '\0';
                model_add_word(model, word);
                pos = 0;
            }
        }
    }

    if (pos > 0) {
        word[pos] = '\0';
        model_add_word(model, word);
    }
}

static int compare_count_desc(const void *pa, const void *pb)
{
    const struct WordEntry *a = (const struct WordEntry *)pa;
    const struct WordEntry *b = (const struct WordEntry *)pb;

    if (a->count < b->count) {
        return 1;
    }

    if (a->count > b->count) {
        return -1;
    }

    return strcmp(a->word, b->word);
}

static void model_print_top(struct Model *model, size_t top_n)
{
    size_t i;
    size_t limit;

    qsort(model->entries, model->used, sizeof(struct WordEntry), compare_count_desc);

    limit = (model->used < top_n) ? model->used : top_n;

    for (i = 0; i < limit; i++) {
        printf("%-20s %lu\n", model->entries[i].word, model->entries[i].count);
    }
}

int main(int argc, char *argv[])
{
    struct Model model;
    FILE *fp;

    if (argc < 2) {
        fprintf(stderr, "usage: %s input.txt [model.txt]\n", argv[0]);
        return 1;
    }

    model_init(&model);

    if (argc >= 3) {
        model_load(&model, argv[2]);
    }

    fp = fopen(argv[1], "r");
    if (fp == NULL) {
        perror("fopen");
        model_free(&model);
        return 1;
    }

    process_stream(fp, &model);
    fclose(fp);

    printf("unique words: %zu\n", model.used);
    printf("top words:\n");
    model_print_top(&model, 20);

    if (argc >= 3) {
        model_save(&model, argv[2]);
    }

    model_free(&model);
    return 0;
}