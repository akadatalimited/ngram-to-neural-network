#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>

#define MAX_WORD_LEN 128
#define MAX_LINE_LEN 65536
#define INITIAL_CAPACITY 4096

struct FourgramEntry {
    char *first;
    char *second;
    char *third;
    char *fourth;
    unsigned long count;
};

struct FourgramModel {
    struct FourgramEntry *entries;
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

static void model_init(struct FourgramModel *model)
{
    model->entries = calloc(INITIAL_CAPACITY, sizeof(struct FourgramEntry));
    model->used = 0;
    model->capacity = INITIAL_CAPACITY;

    if (model->entries == NULL) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }
}

static void model_free(struct FourgramModel *model)
{
    size_t i;

    for (i = 0; i < model->used; i++) {
        free(model->entries[i].first);
        free(model->entries[i].second);
        free(model->entries[i].third);
        free(model->entries[i].fourth);
    }

    free(model->entries);
    model->entries = NULL;
    model->used = 0;
    model->capacity = 0;
}

static void model_grow(struct FourgramModel *model)
{
    struct FourgramEntry *new_entries;
    size_t new_capacity = model->capacity * 2;

    new_entries = realloc(model->entries, new_capacity * sizeof(struct FourgramEntry));
    if (new_entries == NULL) {
        fprintf(stderr, "memory reallocation failed\n");
        exit(1);
    }

    memset(new_entries + model->capacity, 0,
           (new_capacity - model->capacity) * sizeof(struct FourgramEntry));

    model->entries = new_entries;
    model->capacity = new_capacity;
}

static ptrdiff_t model_find_fourgram(const struct FourgramModel *model,
                                     const char *first,
                                     const char *second,
                                     const char *third,
                                     const char *fourth)
{
    size_t i;

    for (i = 0; i < model->used; i++) {
        if (strcmp(model->entries[i].first, first) == 0 &&
            strcmp(model->entries[i].second, second) == 0 &&
            strcmp(model->entries[i].third, third) == 0 &&
            strcmp(model->entries[i].fourth, fourth) == 0) {
            return (ptrdiff_t)i;
        }
    }

    return -1;
}

static void model_add_fourgram(struct FourgramModel *model,
                               const char *first,
                               const char *second,
                               const char *third,
                               const char *fourth)
{
    ptrdiff_t index;

    if (first[0] == '\0' || second[0] == '\0' ||
        third[0] == '\0' || fourth[0] == '\0') {
        return;
    }

    index = model_find_fourgram(model, first, second, third, fourth);
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
    model->entries[model->used].fourth = xstrdup(fourth);
    model->entries[model->used].count = 1;
    model->used++;
}

static void model_save(const struct FourgramModel *model, const char *filename)
{
    FILE *fp;
    size_t i;

    fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("fopen");
        exit(1);
    }

    for (i = 0; i < model->used; i++) {
        fprintf(fp, "%s\t%s\t%s\t%s\t%lu\n",
                model->entries[i].first,
                model->entries[i].second,
                model->entries[i].third,
                model->entries[i].fourth,
                model->entries[i].count);
    }

    fclose(fp);
}

static void model_load(struct FourgramModel *model, const char *filename)
{
    FILE *fp;
    char first[MAX_WORD_LEN];
    char second[MAX_WORD_LEN];
    char third[MAX_WORD_LEN];
    char fourth[MAX_WORD_LEN];
    unsigned long count;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        return;
    }

    while (fscanf(fp, "%127s\t%127s\t%127s\t%127s\t%lu",
                  first, second, third, fourth, &count) == 5) {
        ptrdiff_t index = model_find_fourgram(model, first, second, third, fourth);

        if (index >= 0) {
            model->entries[(size_t)index].count += count;
        } else {
            if (model->used == model->capacity) {
                model_grow(model);
            }

            model->entries[model->used].first = xstrdup(first);
            model->entries[model->used].second = xstrdup(second);
            model->entries[model->used].third = xstrdup(third);
            model->entries[model->used].fourth = xstrdup(fourth);
            model->entries[model->used].count = count;
            model->used++;
        }
    }

    fclose(fp);
}



static int compare_count_desc(const void *pa, const void *pb)
{
    const struct FourgramEntry *a = (const struct FourgramEntry *)pa;
    const struct FourgramEntry *b = (const struct FourgramEntry *)pb;

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

    if (strcmp(a->third, b->third) != 0) {
        return strcmp(a->third, b->third);
    }

    return strcmp(a->fourth, b->fourth);
}

static void model_print_top(struct FourgramModel *model, size_t top_n)
{
    size_t i;
    size_t limit;

    qsort(model->entries, model->used, sizeof(struct FourgramEntry), compare_count_desc);

    limit = (model->used < top_n) ? model->used : top_n;

    for (i = 0; i < limit; i++) {
        printf("%-14s %-14s %-14s %-14s %lu\n",
               model->entries[i].first,
               model->entries[i].second,
               model->entries[i].third,
               model->entries[i].fourth,
               model->entries[i].count);
    }
}

static void process_line(struct FourgramModel *model, const char *line)
{
    char w1[MAX_WORD_LEN];
    char w2[MAX_WORD_LEN];
    char w3[MAX_WORD_LEN];
    char w4[MAX_WORD_LEN];
    size_t pos = 0;
    int have_w1 = 0;
    int have_w2 = 0;
    int have_w3 = 0;
    size_t i;

    for (i = 0;; i++) {
        unsigned char ch = (unsigned char)line[i];

        if (isalnum(ch)) {
            if (pos + 1 < sizeof(w4)) {
                w4[pos++] = (char)tolower(ch);
            }
        } else {
            if (pos > 0) {
                w4[pos] = '\0';

                if (have_w1 && have_w2 && have_w3) {
                    model_add_fourgram(model, w1, w2, w3, w4);
                }

                if (have_w2 && have_w3) {
                    snprintf(w1, sizeof(w1), "%s", w2);
                    have_w1 = 1;
                }

                if (have_w3) {
                    snprintf(w2, sizeof(w2), "%s", w3);
                    have_w2 = 1;
                }

                snprintf(w3, sizeof(w3), "%s", w4);
                have_w3 = 1;

                pos = 0;
            }

            if (ch == '\0') {
                break;
            }
        }
    }
}

static void process_stream(FILE *fp, struct FourgramModel *model)
{
    char line[MAX_LINE_LEN];

    while (fgets(line, sizeof(line), fp) != NULL) {
        process_line(model, line);
    }
}

int main(int argc, char *argv[])
{
    struct FourgramModel model;
    FILE *fp;

    if (argc < 2) {
        fprintf(stderr, "usage: %s input.txt [fourgrams.txt]\n", argv[0]);
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

    printf("unique fourgrams: %zu\n", model.used);
    printf("top fourgrams:\n");
    model_print_top(&model, 30);

    if (argc >= 3) {
        model_save(&model, argv[2]);
    }

    model_free(&model);
    return 0;
}