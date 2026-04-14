#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WORD_LEN 128
#define MAX_OUTPUT_WORDS 100

struct Candidate {
    char word[MAX_WORD_LEN];
    unsigned long count;
};

static int load_best_match(const char *filename,
                           const char *first,
                           const char *second,
                           char *best_word,
                           size_t best_word_size,
                           unsigned long *best_count)
{
    FILE *fp;
    char f[MAX_WORD_LEN];
    char s[MAX_WORD_LEN];
    char t[MAX_WORD_LEN];
    unsigned long count;
    int found = 0;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("fopen");
        return 0;
    }

    *best_count = 0;
    best_word[0] = '\0';

    while (fscanf(fp, "%127s\t%127s\t%127s\t%lu", f, s, t, &count) == 4) {
        if (strcmp(f, first) == 0 && strcmp(s, second) == 0) {
            if (!found || count > *best_count) {
                snprintf(best_word, best_word_size, "%s", t);
                *best_count = count;
                found = 1;
            }
        }
    }

    fclose(fp);
    return found;
}

int main(int argc, char *argv[])
{
    const char *filename;
    char w1[MAX_WORD_LEN];
    char w2[MAX_WORD_LEN];
    char next[MAX_WORD_LEN];
    unsigned long count;
    int max_words;
    int i;

    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s trigrams.txt word1 word2 [max_words]\n", argv[0]);
        return 1;
    }

    filename = argv[1];

    strncpy(w1, argv[2], sizeof(w1) - 1);
    w1[sizeof(w1) - 1] = '\0';

    strncpy(w2, argv[3], sizeof(w2) - 1);
    w2[sizeof(w2) - 1] = '\0';

    max_words = 20;
    if (argc == 5) {
        max_words = atoi(argv[4]);
        if (max_words <= 0 || max_words > MAX_OUTPUT_WORDS) {
            fprintf(stderr, "max_words must be between 1 and %d\n", MAX_OUTPUT_WORDS);
            return 1;
        }
    }

    printf("%s %s", w1, w2);

    for (i = 0; i < max_words; i++) {
        if (!load_best_match(filename, w1, w2, next, sizeof(next), &count)) {
            break;
        }

        if (strcmp(next, "endtok") == 0) {
            break;
        }

        printf(" %s", next);

        strncpy(w1, w2, sizeof(w1) - 1);
        w1[sizeof(w1) - 1] = '\0';

        strncpy(w2, next, sizeof(w2) - 1);
        w2[sizeof(w2) - 1] = '\0';
    }

    printf("\n");
    return 0;
}