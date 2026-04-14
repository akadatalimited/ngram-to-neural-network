#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>

#define MAX_WORD_LEN 128
#define INITIAL_CAPACITY 16384
#define MAX_TEXT_LEN 65536

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

static int is_title_word(const char *word)
{
    static const char *titles[] = {
        "God", "Jehovah", "Lord", "LORD", "Christ", "Spirit"
    };
    size_t i;

    for (i = 0; i < sizeof(titles) / sizeof(titles[0]); i++) {
        if (strcmp(word, titles[i]) == 0) {
            return 1;
        }
    }

    return 0;
}

static int split_title_suffix(const char *src, char *left, size_t left_size,
                              char *right, size_t right_size)
{
    static const char *titles[] = {
        "God", "Jehovah", "Lord", "LORD", "Christ", "Spirit"
    };
    size_t i;
    size_t src_len = strlen(src);

    for (i = 0; i < sizeof(titles) / sizeof(titles[0]); i++) {
        size_t tlen = strlen(titles[i]);

        if (src_len > tlen && strcmp(src + src_len - tlen, titles[i]) == 0) {
            size_t llen = src_len - tlen;

            if (llen == 0 || llen >= left_size || tlen >= right_size) {
                return 0;
            }

            memcpy(left, src, llen);
            left[llen] = '\0';
            memcpy(right, titles[i], tlen + 1);
            return 1;
        }
    }

    return 0;
}

static void emit_token(struct Model *model, const char *token)
{
    char lower[MAX_WORD_LEN];
    char left[MAX_WORD_LEN];
    char right[MAX_WORD_LEN];
    size_t i;
    size_t len = strlen(token);

    if (len == 0) {
        return;
    }

    if (split_title_suffix(token, left, sizeof(left), right, sizeof(right))) {
        emit_token(model, left);
        emit_token(model, right);
        return;
    }

    for (i = 0; i < len && i + 1 < sizeof(lower); i++) {
        lower[i] = (char)tolower((unsigned char)token[i]);
    }
    lower[i] = '\0';

    if (strcmp(lower, "s") == 0) {
        return;
    }

    model_add_word(model, lower);
}

static void process_text(struct Model *model, const char *text)
{
    char token[MAX_WORD_LEN];
    size_t pos = 0;
    size_t i;

    for (i = 0; text[i] != '\0'; i++) {
        unsigned char ch = (unsigned char)text[i];
        unsigned char next = (unsigned char)text[i + 1];

        if (isalnum(ch)) {
            if (pos + 1 < sizeof(token)) {
                token[pos++] = (char)ch;
            }

            if (pos > 0 &&
                islower(ch) &&
                isupper(next)) {
                token[pos] = '\0';
                emit_token(model, token);
                pos = 0;
            }
        } else if (ch == '\'' || ch == '-') {
            if (pos > 0 && isalnum(next)) {
                if (pos + 1 < sizeof(token)) {
                    token[pos++] = (char)ch;
                }
            } else {
                if (pos > 0) {
                    token[pos] = '\0';
                    emit_token(model, token);
                    pos = 0;
                }
            }
        } else {
            if (pos > 0) {
                token[pos] = '\0';
                emit_token(model, token);
                pos = 0;
            }
        }
    }

    if (pos > 0) {
        token[pos] = '\0';
        emit_token(model, token);
    }
}

static int read_json_string(FILE *fp, char *buf, size_t buf_size)
{
    int ch;
    size_t pos = 0;

    while ((ch = fgetc(fp)) != EOF) {
        if (ch == '"') {
            buf[pos] = '\0';
            return 1;
        }

        if (ch == '\\') {
            ch = fgetc(fp);
            if (ch == EOF) {
                return 0;
            }

            switch (ch) {
            case '"':
            case '\\':
            case '/':
                if (pos + 1 < buf_size) {
                    buf[pos++] = (char)ch;
                }
                break;
            case 'b':
                break;
            case 'f':
                break;
            case 'n':
            case 'r':
            case 't':
                if (pos + 1 < buf_size) {
                    buf[pos++] = ' ';
                }
                break;
            case 'u':
            {
                char hex[5];
                long codepoint;
                size_t j;

                for (j = 0; j < 4; j++) {
                    int h = fgetc(fp);
                    if (h == EOF) {
                        return 0;
                    }
                    hex[j] = (char)h;
                }
                hex[4] = '\0';
                codepoint = strtol(hex, NULL, 16);

                if (codepoint < 128 && pos + 1 < buf_size) {
                    buf[pos++] = (char)codepoint;
                } else if (pos + 1 < buf_size) {
                    buf[pos++] = ' ';
                }
                break;
            }
            default:
                if (pos + 1 < buf_size) {
                    buf[pos++] = ' ';
                }
                break;
            }
        } else {
            if (pos + 1 < buf_size) {
                buf[pos++] = (char)ch;
            }
        }
    }

    return 0;
}

static void process_json(FILE *fp, struct Model *model)
{
    int ch;
    char key[256];
    char text[MAX_TEXT_LEN];

    while ((ch = fgetc(fp)) != EOF) {
        if (ch != '"') {
            continue;
        }

        if (!read_json_string(fp, key, sizeof(key))) {
            break;
        }

        while ((ch = fgetc(fp)) != EOF && isspace((unsigned char)ch)) {
        }

        if (ch != ':') {
            continue;
        }

        while ((ch = fgetc(fp)) != EOF && isspace((unsigned char)ch)) {
        }

        if (strcmp(key, "text") == 0 && ch == '"') {
            if (!read_json_string(fp, text, sizeof(text))) {
                break;
            }
            process_text(model, text);
        }
    }
}

int main(int argc, char *argv[])
{
    struct Model model;
    FILE *fp;

    if (argc != 3) {
        fprintf(stderr, "usage: %s Darby.json learned_words_json.txt\n", argv[0]);
        return 1;
    }

    model_init(&model);

    fp = fopen(argv[1], "r");
    if (fp == NULL) {
        perror("fopen");
        model_free(&model);
        return 1;
    }

    process_json(fp, &model);
    fclose(fp);

    printf("unique words: %zu\n", model.used);
    printf("top words:\n");
    model_print_top(&model, 20);

    model_save(&model, argv[2]);
    model_free(&model);
    return 0;
}