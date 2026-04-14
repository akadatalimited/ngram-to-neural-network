#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>

#define MAX_WORD_LEN 128
#define INITIAL_CAPACITY 4096

struct BigramEntry {
    char *first;
    char *second;
    unsigned long count;
};

struct BigramModel {
    struct BigramEntry *entries;
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

static void model_init(struct BigramModel *model)
{
    model->entries = calloc(INITIAL_CAPACITY, sizeof(struct BigramEntry));
    model->used = 0;
    model->capacity = INITIAL_CAPACITY;

    if (model->entries == NULL) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }
}

static void model_free(struct BigramModel *model)
{
    size_t i;

    for (i = 0; i < model->used; i++) {
        free(model->entries[i].first);
        free(model->entries[i].second);
    }

    free(model->entries);
    model->entries = NULL;
    model->used = 0;
    model->capacity = 0;
}

static void model_grow(struct BigramModel *model)
{
    struct BigramEntry *new_entries;
    size_t new_capacity = model->capacity * 2;

    new_entries = realloc(model->entries, new_capacity * sizeof(struct BigramEntry));
    if (new_entries == NULL) {
        fprintf(stderr, "memory reallocation failed\n");
        exit(1);
    }

    memset(new_entries + model->capacity, 0,
           (new_capacity - model->capacity) * sizeof(struct BigramEntry));

    model->entries = new_entries;
    model->capacity = new_capacity;
}

static ptrdiff_t model_find_bigram(const struct BigramModel *model,
                                   const char *first,
                                   const char *second)
{
    size_t i;

    for (i = 0; i < model->used; i++) {
        if (strcmp(model->entries[i].first, first) == 0 &&
            strcmp(model->entries[i].second, second) == 0) {
            return (ptrdiff_t)i;
        }
    }

    return -1;
}

static void model_add_bigram(struct BigramModel *model,
                             const char *first,
                             const char *second)
{
    ptrdiff_t index;

    if (first[0] == '\0' || second[0] == '\0') {
        return;
    }

    index = model_find_bigram(model, first, second);
    if (index >= 0) {
        model->entries[(size_t)index].count++;
        return;
    }

    if (model->used == model->capacity) {
        model_grow(model);
    }

    model->entries[model->used].first = xstrdup(first);
    model->entries[model->used].second = xstrdup(second);
    model->entries[model->used].count = 1;
    model->used++;
}

static void model_save(const struct BigramModel *model, const char *filename)
{
    FILE *fp;
    size_t i;

    fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("fopen");
        exit(1);
    }

    for (i = 0; i < model->used; i++) {
        fprintf(fp, "%s\t%s\t%lu\n",
                model->entries[i].first,
                model->entries[i].second,
                model->entries[i].count);
    }

    fclose(fp);
}

static void model_load(struct BigramModel *model, const char *filename)
{
    FILE *fp;
    char first[MAX_WORD_LEN];
    char second[MAX_WORD_LEN];
    unsigned long count;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        return;
    }

    while (fscanf(fp, "%127s\t%127s\t%lu", first, second, &count) == 3) {
        ptrdiff_t index = model_find_bigram(model, first, second);

        if (index >= 0) {
            model->entries[(size_t)index].count += count;
        } else {
            if (model->used == model->capacity) {
                model_grow(model);
            }

            model->entries[model->used].first = xstrdup(first);
            model->entries[model->used].second = xstrdup(second);
            model->entries[model->used].count = count;
            model->used++;
        }
    }

    fclose(fp);
}

static int read_next_word(FILE *fp, char *word, size_t word_size)
{
    int ch;
    size_t pos = 0;

    while ((ch = fgetc(fp)) != EOF) {
        if (isalnum((unsigned char)ch)) {
            if (pos + 1 < word_size) {
                word[pos++] = (char)tolower((unsigned char)ch);
            }
            break;
        }
    }

    if (ch == EOF) {
        return 0;
    }

    while ((ch = fgetc(fp)) != EOF) {
        if (isalnum((unsigned char)ch)) {
            if (pos + 1 < word_size) {
                word[pos++] = (char)tolower((unsigned char)ch);
            }
        } else {
            break;
        }
    }

    word[pos] = '\0';
    return 1;
}

static int compare_count_desc(const void *pa, const void *pb)
{
    const struct BigramEntry *a = (const struct BigramEntry *)pa;
    const struct BigramEntry *b = (const struct BigramEntry *)pb;

    if (a->count < b->count) {
        return 1;
    }

    if (a->count > b->count) {
        return -1;
    }

    if (strcmp(a->first, b->first) != 0) {
        return strcmp(a->first, b->first);
    }

    return strcmp(a->second, b->second);
}

static void model_print_top(struct BigramModel *model, size_t top_n)
{
    size_t i;
    size_t limit;

    qsort(model->entries, model->used, sizeof(struct BigramEntry), compare_count_desc);

    limit = (model->used < top_n) ? model->used : top_n;

    for (i = 0; i < limit; i++) {
        printf("%-20s %-20s %lu\n",
               model->entries[i].first,
               model->entries[i].second,
               model->entries[i].count);
    }
}

static void process_stream(FILE *fp, struct BigramModel *model)
{
    char prev[MAX_WORD_LEN];
    char curr[MAX_WORD_LEN];
    int have_prev = 0;

    while (read_next_word(fp, curr, sizeof(curr))) {
        if (have_prev) {
            model_add_bigram(model, prev, curr);
        }

        strncpy(prev, curr, sizeof(prev) - 1);
        prev[sizeof(prev) - 1] = '\0';
        have_prev = 1;
    }
}

int main(int argc, char *argv[])
{
    struct BigramModel model;
    FILE *fp;

    if (argc < 2) {
        fprintf(stderr, "usage: %s input.txt [bigrams.txt]\n", argv[0]);
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

    printf("unique bigrams: %zu\n", model.used);
    printf("top bigrams:\n");
    model_print_top(&model, 30);

    if (argc >= 3) {
        model_save(&model, argv[2]);
    }

    model_free(&model);
    return 0;
}