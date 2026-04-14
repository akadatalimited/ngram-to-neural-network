#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_WORD_LEN 128
#define MAX_OUTPUT_WORDS 100
#define INITIAL_CAPACITY 256

struct Candidate {
    char word[MAX_WORD_LEN];
    unsigned long count;
};

static int load_candidates(const char *filename,
                           const char *first,
                           const char *second,
                           const char *third,
                           struct Candidate **candidates,
                           size_t *used,
                           size_t *capacity)
{
    FILE *fp;
    char f[MAX_WORD_LEN];
    char s[MAX_WORD_LEN];
    char t[MAX_WORD_LEN];
    char u[MAX_WORD_LEN];
    unsigned long count;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("fopen");
        return 0;
    }

    *used = 0;

    while (fscanf(fp, "%127s\t%127s\t%127s\t%127s\t%lu",
                  f, s, t, u, &count) == 5) {
        if (strcmp(f, first) == 0 &&
            strcmp(s, second) == 0 &&
            strcmp(t, third) == 0) {
            if (*used == *capacity) {
                struct Candidate *new_candidates;
                size_t new_capacity = (*capacity == 0) ? INITIAL_CAPACITY : (*capacity * 2);

                new_candidates = realloc(*candidates, new_capacity * sizeof(struct Candidate));
                if (new_candidates == NULL) {
                    fprintf(stderr, "memory reallocation failed\n");
                    fclose(fp);
                    return 0;
                }

                *candidates = new_candidates;
                *capacity = new_capacity;
            }

            snprintf((*candidates)[*used].word, MAX_WORD_LEN, "%s", u);
            (*candidates)[*used].count = count;
            (*used)++;
        }
    }

    fclose(fp);
    return 1;
}

static int choose_weighted_candidate(const struct Candidate *candidates,
                                     size_t used,
                                     char *chosen_word,
                                     size_t chosen_word_size,
                                     unsigned long *chosen_count)
{
    unsigned long long total = 0;
    unsigned long long r;
    unsigned long long acc = 0;
    size_t i;

    if (used == 0) {
        return 0;
    }

    for (i = 0; i < used; i++) {
        total += candidates[i].count;
    }

    if (total == 0) {
        return 0;
    }

    r = (unsigned long long)((double)rand() / ((double)RAND_MAX + 1.0) * total);

    for (i = 0; i < used; i++) {
        acc += candidates[i].count;
        if (r < acc) {
            snprintf(chosen_word, chosen_word_size, "%s", candidates[i].word);
            *chosen_count = candidates[i].count;
            return 1;
        }
    }

    snprintf(chosen_word, chosen_word_size, "%s", candidates[used - 1].word);
    *chosen_count = candidates[used - 1].count;
    return 1;
}

int main(int argc, char *argv[])
{
    const char *filename;
    char w1[MAX_WORD_LEN];
    char w2[MAX_WORD_LEN];
    char w3[MAX_WORD_LEN];
    char next[MAX_WORD_LEN];
    unsigned long chosen_count = 0;
    int max_words;
    int i;
    struct Candidate *candidates = NULL;
    size_t used = 0;
    size_t capacity = 0;
    unsigned int seed;
    int printed_any = 0;

    if (argc < 5 || argc > 7) {
        fprintf(stderr, "usage: %s fourgrams.txt word1 word2 word3 [max_words] [seed]\n", argv[0]);
        return 1;
    }

    filename = argv[1];
    snprintf(w1, sizeof(w1), "%s", argv[2]);
    snprintf(w2, sizeof(w2), "%s", argv[3]);
    snprintf(w3, sizeof(w3), "%s", argv[4]);

    max_words = 20;
    if (argc >= 6) {
        max_words = atoi(argv[5]);
        if (max_words <= 0 || max_words > MAX_OUTPUT_WORDS) {
            fprintf(stderr, "max_words must be between 1 and %d\n", MAX_OUTPUT_WORDS);
            return 1;
        }
    }

    if (argc == 7) {
        seed = (unsigned int)strtoul(argv[6], NULL, 10);
    } else {
        seed = (unsigned int)time(NULL);
    }

    srand(seed);

    if (strcmp(w1, "starttok") != 0 && strcmp(w1, "endtok") != 0) {
        printf("%s", w1);
        printed_any = 1;
    }

    if (strcmp(w2, "starttok") != 0 && strcmp(w2, "endtok") != 0) {
        if (printed_any) {
            printf(" ");
        }
        printf("%s", w2);
        printed_any = 1;
    }

    if (strcmp(w3, "starttok") != 0 && strcmp(w3, "endtok") != 0) {
        if (printed_any) {
            printf(" ");
        }
        printf("%s", w3);
        printed_any = 1;
    }

    for (i = 0; i < max_words; i++) {
        if (!load_candidates(filename, w1, w2, w3, &candidates, &used, &capacity)) {
            free(candidates);
            return 1;
        }

        if (used == 0) {
            break;
        }

        if (!choose_weighted_candidate(candidates, used, next, sizeof(next), &chosen_count)) {
            break;
        }

        if (strcmp(next, "endtok") == 0) {
            break;
        }

        if (strcmp(next, "starttok") == 0) {
            break;
        }

        if (printed_any) {
            printf(" ");
        }
        printf("%s", next);
        printed_any = 1;

        snprintf(w1, sizeof(w1), "%s", w2);
        snprintf(w2, sizeof(w2), "%s", w3);
        snprintf(w3, sizeof(w3), "%s", next);
    }

    printf("\n");

    free(candidates);
    return 0;
}