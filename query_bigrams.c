#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WORD_LEN 128
#define INITIAL_CAPACITY 4096

struct BigramEntry {
    char first[MAX_WORD_LEN];
    char second[MAX_WORD_LEN];
    unsigned long count;
};

struct Result {
    char word[MAX_WORD_LEN];
    unsigned long count;
};

static int compare_desc(const void *pa, const void *pb)
{
    const struct Result *a = (const struct Result *)pa;
    const struct Result *b = (const struct Result *)pb;

    if (a->count < b->count) {
        return 1;
    }

    if (a->count > b->count) {
        return -1;
    }

    return strcmp(a->word, b->word);
}

int main(int argc, char *argv[])
{
    FILE *fp;
    char first[MAX_WORD_LEN];
    char second[MAX_WORD_LEN];
    unsigned long count;

    struct Result *results;
    size_t used = 0;
    size_t capacity = INITIAL_CAPACITY;

    if (argc < 3) {
        fprintf(stderr, "usage: %s bigrams.txt word\n", argv[0]);
        return 1;
    }

    results = malloc(capacity * sizeof(struct Result));
    if (results == NULL) {
        fprintf(stderr, "memory allocation failed\n");
        return 1;
    }

    fp = fopen(argv[1], "r");
    if (fp == NULL) {
        perror("fopen");
        free(results);
        return 1;
    }

    while (fscanf(fp, "%127s\t%127s\t%lu", first, second, &count) == 3) {
        if (strcmp(first, argv[2]) == 0) {
            if (used == capacity) {
                capacity *= 2;
                results = realloc(results, capacity * sizeof(struct Result));
                if (results == NULL) {
                    fprintf(stderr, "memory reallocation failed\n");
                    fclose(fp);
                    return 1;
                }
            }

            strncpy(results[used].word, second, MAX_WORD_LEN - 1);
            results[used].word[MAX_WORD_LEN - 1] = '\0';
            results[used].count = count;
            used++;
        }
    }

    fclose(fp);

    if (used == 0) {
        printf("no matches for '%s'\n", argv[2]);
        free(results);
        return 0;
    }

    qsort(results, used, sizeof(struct Result), compare_desc);

    printf("Top matches for '%s':\n\n", argv[2]);

    size_t limit = (used < 20) ? used : 20;
    for (size_t i = 0; i < limit; i++) {
        printf("%-20s %lu\n", results[i].word, results[i].count);
    }

    free(results);
    return 0;
}