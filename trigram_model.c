#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>

#define MAX_WORD_LEN 128
#define INITIAL_CAPACITY 4096

struct TrigramEntry {
    char *first;
    char *second;
    char *third;
    unsigned long count;
};

struct TrigramModel {
    struct TrigramEntry *entries;
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

static void model_init(struct TrigramModel *model)
{
    model->entries = calloc(INITIAL_CAPACITY, sizeof(struct TrigramEntry));
    model->used = 0;
    model->capacity = INITIAL_CAPACITY;

    if (model->entries == NULL) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }
}

static void model_free(struct TrigramModel *model)
{
    size_t i;

    for (i = 0; i < model->used; i++) {
        free(model->entries[i].first);
        free(model->entries[i].second);
        free(model->entries[i].third);
    }

    free(model->entries);
    model->entries = NULL;
    model->used = 0;
    model->capacity = 0;
}

static void model_grow(struct TrigramModel *model)
{
    struct TrigramEntry *new_entries;
    size_t new_capacity = model->capacity * 2;

    new_entries = realloc(model->entries, new_capacity * sizeof(struct TrigramEntry));
    if (new_entries == NULL) {
        fprintf(stderr, "memory reallocation failed\n");
        exit(1);
    }

    memset(new_entries + model->capacity, 0,
           (new_capacity - model->capacity) * sizeof(struct TrigramEntry));

    model->entries = new_entries;
    model->capacity = new_capacity;
}

static ptrdiff_t model_find_trigram(const struct TrigramModel *model,
                                    const char *first,
                                    const char *second,
                                    const char *third)
{
    size_t i;

    for (i = 0; i < model->used; i++) {
        if (strcmp(model->entries[i].first, first) == 0 &&
            strcmp(model->entries[i].second, second) == 0 &&
            strcmp(model->entries[i].third, third) == 0) {
            return (ptrdiff_t)i;
        }
    }

    return -1;
}

static void model_add_trigram(struct TrigramModel *model,
                              const char *first,
                              const char *second,
                              const char *third)
{
    ptrdiff_t index;

    if (first[0] == '\0' || second[0] == '\0' || third[0] == '\0') {
        return;
    }

    index = model_find_trigram(model, first, second, third);
    if (index >= 0) {
        model->entries[(size_t)index].count++;
        return;
    }

    if (model->used == model->capacity) {
        model_grow(model);
    }

    model->entries[model->used].first = xstrdup(first);
    model->entries[model->used].second = xstrdup(second);
    model->entries[model->used].third = xstrdup(third);
    model->entries[model->used].count = 1;
    model->used++;
}

static void model_save(const struct TrigramModel *model, const char *filename)
{
    FILE *fp;
    size_t i;

    fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("fopen");
        exit(1);
    }

    for (i = 0; i < model->used; i++) {
        fprintf(fp, "%s\t%s\t%s\t%lu\n",
                model->entries[i].first,
                model->entries[i].second,
                model->entries[i].third,
                model->entries[i].count);
    }

    fclose(fp);
}

static void model_load(struct TrigramModel *model, const char *filename)
{
    FILE *fp;
    char first[MAX_WORD_LEN];
    char second[MAX_WORD_LEN];
    char third[MAX_WORD_LEN];
    unsigned long count;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        return;
    }

    while (fscanf(fp, "%127s\t%127s\t%127s\t%lu",
                  first, second, third, &count) == 4) {
        ptrdiff_t index = model_find_trigram(model, first, second, third);

        if (index >= 0) {
            model->entries[(size_t)index].count += count;
        } else {
            if (model->used == model->capacity) {
                model_grow(model);
            }

            model->entries[model->used].first = xstrdup(first);
            model->entries[model->used].second = xstrdup(second);
            model->entries[model->used].third = xstrdup(third);
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
    const struct TrigramEntry *a = (const struct TrigramEntry *)pa;
    const struct TrigramEntry *b = (const struct TrigramEntry *)pb;

    if (a->count < b->count) {
        return 1;
    }

    if (a->count > b->count) {
        return -1;
    }

    if (strcmp(a->first, b->first) != 0) {
        return strcmp(a->first, b->first);
    }

    if (strcmp(a->second, b->second) != 0) {
        return strcmp(a->second, b->second);
    }

    return strcmp(a->third, b->third);
}

static void model_print_top(struct TrigramModel *model, size_t top_n)
{
    size_t i;
    size_t limit;

    qsort(model->entries, model->used, sizeof(struct TrigramEntry), compare_count_desc);

    limit = (model->used < top_n) ? model->used : top_n;

    for (i = 0; i < limit; i++) {
        printf("%-16s %-16s %-16s %lu\n",
               model->entries[i].first,
               model->entries[i].second,
               model->entries[i].third,
               model->entries[i].count);
    }
}

static void process_stream(FILE *fp, struct TrigramModel *model)
{
    char w1[MAX_WORD_LEN];
    char w2[MAX_WORD_LEN];
    char w3[MAX_WORD_LEN];
    int have_w1 = 0;
    int have_w2 = 0;

    while (read_next_word(fp, w3, sizeof(w3))) {
        if (have_w1 && have_w2) {
            model_add_trigram(model, w1, w2, w3);
        }

        if (have_w2) {
            strncpy(w1, w2, sizeof(w1) - 1);
            w1[sizeof(w1) - 1] = '\0';
            have_w1 = 1;
        }

        strncpy(w2, w3, sizeof(w2) - 1);
        w2[sizeof(w2) - 1] = '\0';
        have_w2 = 1;
    }
}

int main(int argc, char *argv[])
{
    struct TrigramModel model;
    FILE *fp;

    if (argc < 2) {
        fprintf(stderr, "usage: %s input.txt [trigrams.txt]\n", argv[0]);
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

    printf("unique trigrams: %zu\n", model.used);
    printf("top trigrams:\n");
    model_print_top(&model, 30);

    if (argc >= 3) {
        model_save(&model, argv[2]);
    }

    model_free(&model);
    return 0;
}