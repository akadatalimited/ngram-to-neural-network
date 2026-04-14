#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define SYMBOLS 256
#define LR 0.1
#define EPOCHS 200

static double rank_score[SYMBOLS];

static void init_scores(void)
{
    int i;

    for (i = 0; i < SYMBOLS; i++) {
        rank_score[i] = 0.0;
    }
}

static void train_pair(unsigned char left, unsigned char right)
{
    /*
     * We want left < right.
     * If the model currently thinks otherwise,
     * push left down and right up.
     */
    if (rank_score[left] >= rank_score[right]) {
        rank_score[left] -= LR;
        rank_score[right] += LR;
    }
}

static void train_alphabet(void)
{
    const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    int epoch;
    size_t i;

    for (epoch = 0; epoch < EPOCHS; epoch++) {
        for (i = 0; i + 1 < strlen(alphabet); i++) {
            unsigned char a = (unsigned char)alphabet[i];
            unsigned char b = (unsigned char)alphabet[i + 1];

            train_pair(a, b);
        }
    }
}

static int compare_chars(unsigned char a, unsigned char b)
{
    double sa = rank_score[a];
    double sb = rank_score[b];

    if (sa < sb) {
        return -1;
    }

    if (sa > sb) {
        return 1;
    }

    return 0;
}

static int compare_words(const void *pa, const void *pb)
{
    const char *a = *(const char * const *)pa;
    const char *b = *(const char * const *)pb;
    size_t i = 0;

    while (a[i] != '\0' && b[i] != '\0') {
        unsigned char ca = (unsigned char)toupper((unsigned char)a[i]);
        unsigned char cb = (unsigned char)toupper((unsigned char)b[i]);
        int cmp = compare_chars(ca, cb);

        if (cmp != 0) {
            return cmp;
        }

        i++;
    }

    /*
     * If one word ends first, it comes first.
     * Example: APP comes before APPLE.
     */
    if (a[i] == '\0' && b[i] == '\0') {
        return 0;
    }

    if (a[i] == '\0') {
        return -1;
    }

    return 1;
}

static void print_scores(const char *symbols)
{
    size_t i;

    for (i = 0; i < strlen(symbols); i++) {
        unsigned char c = (unsigned char)symbols[i];

        printf("%c => %.2f\n", c, rank_score[c]);
    }
}

int main(void)
{
    char *words[] = {
        "banana",
        "apple",
        "grape",
        "apricot",
        "blueberry",
        "aardvark",
        "cherry",
        "app"
    };
    size_t count = sizeof(words) / sizeof(words[0]);
    size_t i;

    init_scores();
    train_alphabet();

    printf("Learned letter scores:\n");
    print_scores("ABCDEFGHIJKLMNOPQRSTUVWXYZ");

    qsort(words, count, sizeof(words[0]), compare_words);

    printf("\nSorted words:\n");

    for (i = 0; i < count; i++) {
        printf("%s\n", words[i]);
    }

    return 0;
}