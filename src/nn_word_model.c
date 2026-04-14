#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#define MAX_TEXT_LEN (16 * 1024 * 1024)
#define MAX_WORD_LEN 128
#define MAX_CONTEXT 16
#define MAX_VOCAB 8192 // started with 4096 and ran out of usable vocab
#define MODEL_MAGIC "NWM1"

struct TokenizedText {
    char **tokens;
    size_t count;
    size_t capacity;
};

struct VocabEntry {
    char *word;
    unsigned long count;
};

struct Vocab {
    struct VocabEntry *entries;
    size_t count;
    size_t capacity;

    int unk_index;
    int pad_index;
};

struct IndexedDataset {
    int *token_ids;
    size_t count;
};

struct Model {
    int context_len;
    int vocab_size;
    int input_size;
    int hidden_size;
    int output_size;

    float *w1;
    float *b1;
    float *w2;
    float *b2;
};

static void die(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static char *xstrdup(const char *src)
{
    size_t len = strlen(src) + 1;
    char *copy = malloc(len);

    if (copy == NULL) {
        die("malloc failed in xstrdup");
    }

    memcpy(copy, src, len);
    return copy;
}

static float frand_small(void)
{
    return ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
}

static unsigned char *read_file(const char *filename, size_t *out_len)
{
    FILE *fp;
    unsigned char *buf;
    long size;

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("fopen");
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        die("fseek end failed");
    }

    size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        die("ftell failed");
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        die("fseek start failed");
    }

    if ((size_t)size > MAX_TEXT_LEN) {
        fclose(fp);
        die("training file too large for this starter word model");
    }

    buf = malloc((size_t)size + 1);
    if (buf == NULL) {
        fclose(fp);
        die("malloc failed for training text");
    }

    if (fread(buf, 1, (size_t)size, fp) != (size_t)size) {
        free(buf);
        fclose(fp);
        die("fread failed");
    }

    fclose(fp);
    buf[size] = '\0';
    *out_len = (size_t)size;
    return buf;
}

static void tokenized_text_init(struct TokenizedText *tt)
{
    tt->tokens = NULL;
    tt->count = 0;
    tt->capacity = 0;
}

static void tokenized_text_add(struct TokenizedText *tt, const char *word)
{
    char **new_tokens;

    if (tt->count == tt->capacity) {
        size_t new_capacity = (tt->capacity == 0) ? 1024 : (tt->capacity * 2);

        new_tokens = realloc(tt->tokens, new_capacity * sizeof(char *));
        if (new_tokens == NULL) {
            die("realloc failed in tokenized_text_add");
        }

        tt->tokens = new_tokens;
        tt->capacity = new_capacity;
    }

    tt->tokens[tt->count++] = xstrdup(word);
}

static void tokenized_text_free(struct TokenizedText *tt)
{
    size_t i;

    for (i = 0; i < tt->count; i++) {
        free(tt->tokens[i]);
    }

    free(tt->tokens);
    tt->tokens = NULL;
    tt->count = 0;
    tt->capacity = 0;
}

static void tokenize_text(const unsigned char *text, struct TokenizedText *tt)
{
    char word[MAX_WORD_LEN];
    size_t pos = 0;
    size_t i;

    tokenized_text_init(tt);

    for (i = 0;; i++) {
        unsigned char ch = text[i];

        if (isalnum(ch)) {
            if (pos + 1 < sizeof(word)) {
                word[pos++] = (char)tolower(ch);
            }
        } else {
            if (pos > 0) {
                word[pos] = '\0';
                tokenized_text_add(tt, word);
                pos = 0;
            }

            if (ch == '\0') {
                break;
            }
        }
    }
}

static void vocab_init(struct Vocab *v)
{
    v->entries = NULL;
    v->count = 0;
    v->capacity = 0;
    v->unk_index = -1;
    v->pad_index = -1;
}

static int vocab_find(const struct Vocab *v, const char *word)
{
    size_t i;

    for (i = 0; i < v->count; i++) {
        if (strcmp(v->entries[i].word, word) == 0) {
            return (int)i;
        }
    }

    return -1;
}

static void vocab_add_or_increment(struct Vocab *v, const char *word)
{
    struct VocabEntry *new_entries;
    int idx = vocab_find(v, word);

    if (idx >= 0) {
        v->entries[idx].count++;
        return;
    }

    if (v->count == v->capacity) {
        size_t new_capacity = (v->capacity == 0) ? 1024 : (v->capacity * 2);

        new_entries = realloc(v->entries, new_capacity * sizeof(struct VocabEntry));
        if (new_entries == NULL) {
            die("realloc failed in vocab_add_or_increment");
        }

        v->entries = new_entries;
        v->capacity = new_capacity;
    }

    v->entries[v->count].word = xstrdup(word);
    v->entries[v->count].count = 1;
    v->count++;
}

static int compare_vocab_count_desc(const void *pa, const void *pb)
{
    const struct VocabEntry *a = (const struct VocabEntry *)pa;
    const struct VocabEntry *b = (const struct VocabEntry *)pb;

    if (a->count < b->count) {
        return 1;
    }

    if (a->count > b->count) {
        return -1;
    }

    return strcmp(a->word, b->word);
}

static void vocab_build_from_tokens(struct Vocab *v, const struct TokenizedText *tt)
{
    struct Vocab temp;
    size_t i;
    size_t keep_normal;

    vocab_init(&temp);

    for (i = 0; i < tt->count; i++) {
        vocab_add_or_increment(&temp, tt->tokens[i]);
    }

    qsort(temp.entries, temp.count, sizeof(struct VocabEntry), compare_vocab_count_desc);

    vocab_init(v);

    v->capacity = MAX_VOCAB;
    v->entries = calloc(v->capacity, sizeof(struct VocabEntry));
    if (v->entries == NULL) {
        die("calloc failed in vocab_build_from_tokens");
    }

    v->entries[0].word = xstrdup("<pad>");
    v->entries[0].count = 0;
    v->entries[1].word = xstrdup("<unk>");
    v->entries[1].count = 0;
    v->count = 2;

    keep_normal = 0;
    if (MAX_VOCAB > 2) {
        keep_normal = temp.count;
        if (keep_normal > (size_t)(MAX_VOCAB - 2)) {
            keep_normal = (size_t)(MAX_VOCAB - 2);
        }
    }

    for (i = 0; i < keep_normal; i++) {
        v->entries[v->count].word = xstrdup(temp.entries[i].word);
        v->entries[v->count].count = temp.entries[i].count;
        v->count++;
    }

    v->pad_index = 0;
    v->unk_index = 1;

    for (i = 0; i < temp.count; i++) {
        free(temp.entries[i].word);
    }
    free(temp.entries);
}

static void vocab_free(struct Vocab *v)
{
    size_t i;

    for (i = 0; i < v->count; i++) {
        free(v->entries[i].word);
    }

    free(v->entries);
    v->entries = NULL;
    v->count = 0;
    v->capacity = 0;
    v->unk_index = -1;
    v->pad_index = -1;
}

static void indexed_dataset_init(struct IndexedDataset *ds, size_t count)
{
    ds->token_ids = malloc(count * sizeof(int));
    if (ds->token_ids == NULL) {
        die("malloc failed for indexed dataset");
    }

    ds->count = count;
}

static void indexed_dataset_free(struct IndexedDataset *ds)
{
    free(ds->token_ids);
    ds->token_ids = NULL;
    ds->count = 0;
}

static void index_tokens(const struct TokenizedText *tt,
                         const struct Vocab *v,
                         struct IndexedDataset *ds)
{
    size_t i;

    indexed_dataset_init(ds, tt->count);

    for (i = 0; i < tt->count; i++) {
        int idx = vocab_find(v, tt->tokens[i]);

        if (idx < 0) {
            idx = v->unk_index;
        }

        ds->token_ids[i] = idx;
    }
}

static void model_init(struct Model *m, int context_len, int vocab_size, int hidden_size)
{
    int i;

    memset(m, 0, sizeof(*m));
    m->context_len = context_len;
    m->vocab_size = vocab_size;
    m->input_size = context_len * vocab_size;
    m->hidden_size = hidden_size;
    m->output_size = vocab_size;

    m->w1 = malloc((size_t)m->hidden_size * (size_t)m->input_size * sizeof(float));
    m->b1 = malloc((size_t)m->hidden_size * sizeof(float));
    m->w2 = malloc((size_t)m->output_size * (size_t)m->hidden_size * sizeof(float));
    m->b2 = malloc((size_t)m->output_size * sizeof(float));

    if (m->w1 == NULL || m->b1 == NULL || m->w2 == NULL || m->b2 == NULL) {
        die("malloc failed for model");
    }

    for (i = 0; i < m->hidden_size * m->input_size; i++) {
        m->w1[i] = frand_small();
    }

    for (i = 0; i < m->hidden_size; i++) {
        m->b1[i] = 0.0f;
    }

    for (i = 0; i < m->output_size * m->hidden_size; i++) {
        m->w2[i] = frand_small();
    }

    for (i = 0; i < m->output_size; i++) {
        m->b2[i] = 0.0f;
    }
}

static void model_free(struct Model *m)
{
    free(m->w1);
    free(m->b1);
    free(m->w2);
    free(m->b2);
    memset(m, 0, sizeof(*m));
}

static float sigmoidf(float x)
{
    if (x < -30.0f) {
        return 0.0f;
    }

    if (x > 30.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + expf(-x));
}

static void softmax(float *logits, int n)
{
    int i;
    float maxv = logits[0];
    float sum = 0.0f;

    for (i = 1; i < n; i++) {
        if (logits[i] > maxv) {
            maxv = logits[i];
        }
    }

    for (i = 0; i < n; i++) {
        logits[i] = expf(logits[i] - maxv);
        sum += logits[i];
    }

    if (sum == 0.0f) {
        return;
    }

    for (i = 0; i < n; i++) {
        logits[i] /= sum;
    }
}

static void build_input_vector(const struct IndexedDataset *ds,
                               size_t pos,
                               int context_len,
                               int vocab_size,
                               int pad_index,
                               float *x)
{
    int c, v;

    for (c = 0; c < context_len; c++) {
        for (v = 0; v < vocab_size; v++) {
            x[c * vocab_size + v] = 0.0f;
        }
    }

    for (c = 0; c < context_len; c++) {
        size_t src_pos = pos - (size_t)context_len + (size_t)c;
        int idx = pad_index;

        if (src_pos < ds->count) {
            idx = ds->token_ids[src_pos];
        }

        if (idx >= 0 && idx < vocab_size) {
            x[c * vocab_size + idx] = 1.0f;
        }
    }
}

static float train_one_example(struct Model *m,
                               const struct IndexedDataset *ds,
                               size_t pos,
                               int pad_index,
                               float learning_rate,
                               float *x,
                               float *hidden,
                               float *logits,
                               float *d_hidden)
{
    int i, j;
    int target_idx = ds->token_ids[pos];

    build_input_vector(ds, pos, m->context_len, m->vocab_size, pad_index, x);

    for (i = 0; i < m->hidden_size; i++) {
        float sum = m->b1[i];

        for (j = 0; j < m->input_size; j++) {
            sum += m->w1[i * m->input_size + j] * x[j];
        }

        hidden[i] = sigmoidf(sum);
    }

    for (i = 0; i < m->output_size; i++) {
        float sum = m->b2[i];

        for (j = 0; j < m->hidden_size; j++) {
            sum += m->w2[i * m->hidden_size + j] * hidden[j];
        }

        logits[i] = sum;
    }

    softmax(logits, m->output_size);

    {
        float prob = logits[target_idx];
        if (prob < 1e-8f) {
            prob = 1e-8f;
        }

        for (i = 0; i < m->hidden_size; i++) {
            d_hidden[i] = 0.0f;
        }

        for (i = 0; i < m->output_size; i++) {
            float grad = logits[i] - ((i == target_idx) ? 1.0f : 0.0f);

            for (j = 0; j < m->hidden_size; j++) {
                d_hidden[j] += grad * m->w2[i * m->hidden_size + j];
                m->w2[i * m->hidden_size + j] -= learning_rate * grad * hidden[j];
            }

            m->b2[i] -= learning_rate * grad;
        }

        for (i = 0; i < m->hidden_size; i++) {
            float grad_h = d_hidden[i] * hidden[i] * (1.0f - hidden[i]);

            for (j = 0; j < m->input_size; j++) {
                m->w1[i * m->input_size + j] -= learning_rate * grad_h * x[j];
            }

            m->b1[i] -= learning_rate * grad_h;
        }

        return -logf(prob);
    }
}

static void save_model(const char *filename, const struct Model *m, const struct Vocab *v)
{
    FILE *fp;
    int magic_len = 4;
    int i;

    fp = fopen(filename, "wb");
    if (fp == NULL) {
        perror("fopen");
        die("failed to save word model");
    }

    fwrite(MODEL_MAGIC, 1, (size_t)magic_len, fp);
    fwrite(&m->context_len, sizeof(int), 1, fp);
    fwrite(&m->vocab_size, sizeof(int), 1, fp);
    fwrite(&m->hidden_size, sizeof(int), 1, fp);

    for (i = 0; i < m->vocab_size; i++) {
        unsigned short len = (unsigned short)strlen(v->entries[i].word);

        fwrite(&len, sizeof(unsigned short), 1, fp);
        fwrite(v->entries[i].word, 1, len, fp);
    }

    fwrite(m->w1, sizeof(float), (size_t)m->hidden_size * (size_t)m->input_size, fp);
    fwrite(m->b1, sizeof(float), (size_t)m->hidden_size, fp);
    fwrite(m->w2, sizeof(float), (size_t)m->output_size * (size_t)m->hidden_size, fp);
    fwrite(m->b2, sizeof(float), (size_t)m->output_size, fp);

    fclose(fp);
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s training.txt model.bin [epochs] [context_len] [hidden_size] [learning_rate]\n",
            prog);
}

int main(int argc, char *argv[])
{
    unsigned char *text;
    size_t text_len;
    struct TokenizedText tt;
    struct Vocab vocab;
    struct IndexedDataset ds;
    struct Model m;
    int epochs = 3;
    int context_len = 4;
    int hidden_size = 64;
    float learning_rate = 0.05f;
    float *x;
    float *hidden;
    float *logits;
    float *d_hidden;
    int epoch;
    size_t pos;
    size_t example_count;

    if (argc < 3 || argc > 7) {
        usage(argv[0]);
        return 1;
    }

    if (argc >= 4) {
        epochs = atoi(argv[3]);
    }
    if (argc >= 5) {
        context_len = atoi(argv[4]);
    }
    if (argc >= 6) {
        hidden_size = atoi(argv[5]);
    }
    if (argc >= 7) {
        learning_rate = (float)atof(argv[6]);
    }

    if (context_len <= 0 || context_len > MAX_CONTEXT) {
        fprintf(stderr, "context_len must be between 1 and %d\n", MAX_CONTEXT);
        return 1;
    }

    if (hidden_size <= 0) {
        fprintf(stderr, "hidden_size must be > 0\n");
        return 1;
    }

    srand((unsigned int)time(NULL));

    text = read_file(argv[1], &text_len);
    if (text == NULL) {
        return 1;
    }

    tokenize_text(text, &tt);
    free(text);

    vocab_build_from_tokens(&vocab, &tt);
    index_tokens(&tt, &vocab, &ds);

    if (ds.count <= (size_t)context_len) {
        tokenized_text_free(&tt);
        vocab_free(&vocab);
        indexed_dataset_free(&ds);
        die("training text is too short");
    }

    model_init(&m, context_len, (int)vocab.count, hidden_size);

    x = malloc((size_t)m.input_size * sizeof(float));
    hidden = malloc((size_t)m.hidden_size * sizeof(float));
    logits = malloc((size_t)m.output_size * sizeof(float));
    d_hidden = malloc((size_t)m.hidden_size * sizeof(float));

    if (x == NULL || hidden == NULL || logits == NULL || d_hidden == NULL) {
        die("malloc failed for training buffers");
    }

    example_count = ds.count - (size_t)context_len;

    printf("training tokens: %zu\n", ds.count);
    printf("vocab size: %zu\n", vocab.count);
    printf("context len: %d words\n", context_len);
    printf("hidden size: %d\n", hidden_size);
    printf("epochs: %d\n", epochs);
    printf("learning rate: %.4f\n", learning_rate);

    for (epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        for (pos = (size_t)context_len; pos < ds.count; pos++) {
            total_loss += train_one_example(&m, &ds, pos, vocab.pad_index,
                                            learning_rate, x, hidden, logits, d_hidden);
        }

        printf("epoch %d/%d  avg_loss=%.6f\n",
               epoch + 1, epochs, total_loss / (float)example_count);
    }

    save_model(argv[2], &m, &vocab);

    free(x);
    free(hidden);
    free(logits);
    free(d_hidden);
    model_free(&m);
    indexed_dataset_free(&ds);
    vocab_free(&vocab);
    tokenized_text_free(&tt);

    return 0;
}
