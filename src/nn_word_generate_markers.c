#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#define MAX_WORD_LEN 128
#define MAX_CONTEXT 16
#define MAX_VOCAB 8192 // started with 4096 and ran out of words / vocab
#define MODEL_MAGIC "NWM1"

struct Model {
    int context_len;
    int vocab_size;
    int input_size;
    int hidden_size;
    int output_size;

    char **index_to_word;

    float *w1;
    float *b1;
    float *w2;
    float *b2;

    int pad_index;
    int unk_index;
};

struct TokenizedSeed {
    char **tokens;
    size_t count;
    size_t capacity;
};

static void die(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
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

static void tokenized_seed_init(struct TokenizedSeed *ts)
{
    ts->tokens = NULL;
    ts->count = 0;
    ts->capacity = 0;
}

static void tokenized_seed_add(struct TokenizedSeed *ts, const char *word)
{
    char **new_tokens;

    if (ts->count == ts->capacity) {
        size_t new_capacity = (ts->capacity == 0) ? 16 : (ts->capacity * 2);

        new_tokens = realloc(ts->tokens, new_capacity * sizeof(char *));
        if (new_tokens == NULL) {
            die("realloc failed in tokenized_seed_add");
        }

        ts->tokens = new_tokens;
        ts->capacity = new_capacity;
    }

    ts->tokens[ts->count++] = xstrdup(word);
}

static void tokenized_seed_free(struct TokenizedSeed *ts)
{
    size_t i;

    for (i = 0; i < ts->count; i++) {
        free(ts->tokens[i]);
    }

    free(ts->tokens);
    ts->tokens = NULL;
    ts->count = 0;
    ts->capacity = 0;
}

static void tokenize_seed(const char *text, struct TokenizedSeed *ts)
{
    char word[MAX_WORD_LEN];
    size_t pos = 0;
    size_t i;

    tokenized_seed_init(ts);

    for (i = 0;; i++) {
        unsigned char ch = (unsigned char)text[i];

        if (isalnum(ch)) {
            if (pos + 1 < sizeof(word)) {
                word[pos++] = (char)tolower(ch);
            }
        } else {
            if (pos > 0) {
                word[pos] = '\0';
                tokenized_seed_add(ts, word);
                pos = 0;
            }

            if (ch == '\0') {
                break;
            }
        }
    }
}

static void model_init_empty(struct Model *m)
{
    memset(m, 0, sizeof(*m));
    m->pad_index = -1;
    m->unk_index = -1;
}

static void model_free(struct Model *m)
{
    int i;

    if (m->index_to_word != NULL) {
        for (i = 0; i < m->vocab_size; i++) {
            free(m->index_to_word[i]);
        }
    }

    free(m->index_to_word);
    free(m->w1);
    free(m->b1);
    free(m->w2);
    free(m->b2);

    memset(m, 0, sizeof(*m));
}

static int find_word_index(const struct Model *m, const char *word)
{
    int i;

    for (i = 0; i < m->vocab_size; i++) {
        if (strcmp(m->index_to_word[i], word) == 0) {
            return i;
        }
    }

    return -1;
}

static void model_load(const char *filename, struct Model *m)
{
    FILE *fp;
    char magic[4];
    int i;

    model_init_empty(m);

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("fopen");
        die("failed to open word model file");
    }

    if (fread(magic, 1, 4, fp) != 4) {
        fclose(fp);
        die("failed to read model magic");
    }

    if (memcmp(magic, MODEL_MAGIC, 4) != 0) {
        fclose(fp);
        die("invalid word model file magic");
    }

    if (fread(&m->context_len, sizeof(int), 1, fp) != 1 ||
        fread(&m->vocab_size, sizeof(int), 1, fp) != 1 ||
        fread(&m->hidden_size, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        die("failed to read model header");
    }

    if (m->context_len <= 0 || m->context_len > MAX_CONTEXT) {
        fclose(fp);
        die("invalid context length in model");
    }

    if (m->vocab_size <= 0 || m->vocab_size > MAX_VOCAB) {
        fclose(fp);
        die("invalid vocab size in model");
    }

    if (m->hidden_size <= 0) {
        fclose(fp);
        die("invalid hidden size in model");
    }

    m->input_size = m->context_len * m->vocab_size;
    m->output_size = m->vocab_size;

    m->index_to_word = calloc((size_t)m->vocab_size, sizeof(char *));
    if (m->index_to_word == NULL) {
        fclose(fp);
        die("calloc failed for vocab");
    }

    for (i = 0; i < m->vocab_size; i++) {
        unsigned short len;
        char buf[MAX_WORD_LEN];

        if (fread(&len, sizeof(unsigned short), 1, fp) != 1) {
            fclose(fp);
            die("failed to read vocab word length");
        }

        if (len == 0 || len >= MAX_WORD_LEN) {
            fclose(fp);
            die("invalid vocab word length");
        }

        if (fread(buf, 1, len, fp) != len) {
            fclose(fp);
            die("failed to read vocab word");
        }

        buf[len] = '\0';
        m->index_to_word[i] = xstrdup(buf);
    }

    m->pad_index = find_word_index(m, "<pad>");
    m->unk_index = find_word_index(m, "<unk>");

    if (m->pad_index < 0 || m->unk_index < 0) {
        fclose(fp);
        die("special tokens not found in model");
    }

    m->w1 = malloc((size_t)m->hidden_size * (size_t)m->input_size * sizeof(float));
    m->b1 = malloc((size_t)m->hidden_size * sizeof(float));
    m->w2 = malloc((size_t)m->output_size * (size_t)m->hidden_size * sizeof(float));
    m->b2 = malloc((size_t)m->output_size * sizeof(float));

    if (m->w1 == NULL || m->b1 == NULL || m->w2 == NULL || m->b2 == NULL) {
        fclose(fp);
        die("malloc failed for model weights");
    }

    if (fread(m->w1, sizeof(float), (size_t)m->hidden_size * (size_t)m->input_size, fp) !=
        (size_t)m->hidden_size * (size_t)m->input_size) {
        fclose(fp);
        die("failed to read w1");
    }

    if (fread(m->b1, sizeof(float), (size_t)m->hidden_size, fp) != (size_t)m->hidden_size) {
        fclose(fp);
        die("failed to read b1");
    }

    if (fread(m->w2, sizeof(float), (size_t)m->output_size * (size_t)m->hidden_size, fp) !=
        (size_t)m->output_size * (size_t)m->hidden_size) {
        fclose(fp);
        die("failed to read w2");
    }

    if (fread(m->b2, sizeof(float), (size_t)m->output_size, fp) != (size_t)m->output_size) {
        fclose(fp);
        die("failed to read b2");
    }

    fclose(fp);
}

static void build_input_vector(const struct Model *m,
                               const int *context_ids,
                               float *x)
{
    int c, v;

    for (c = 0; c < m->context_len; c++) {
        for (v = 0; v < m->vocab_size; v++) {
            x[c * m->vocab_size + v] = 0.0f;
        }
    }

    for (c = 0; c < m->context_len; c++) {
        int idx = context_ids[c];

        if (idx >= 0 && idx < m->vocab_size) {
            x[c * m->vocab_size + idx] = 1.0f;
        }
    }
}

static void forward_pass(const struct Model *m,
                         const int *context_ids,
                         float *x,
                         float *hidden,
                         float *probs)
{
    int i, j;

    build_input_vector(m, context_ids, x);

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

        probs[i] = sum;
    }

    softmax(probs, m->output_size);
}

static int sample_index_greedy(const float *probs, int n)
{
    int i;
    int best = 0;
    float bestv = probs[0];

    for (i = 1; i < n; i++) {
        if (probs[i] > bestv) {
            bestv = probs[i];
            best = i;
        }
    }

    return best;
}

static int sample_index_random(const float *probs, int n)
{
    int i;
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float acc = 0.0f;

    for (i = 0; i < n; i++) {
        acc += probs[i];
        if (r < acc) {
            return i;
        }
    }

    return n - 1;
}

static void init_context_from_seed(const struct Model *m,
                                   const struct TokenizedSeed *seed,
                                   int *context_ids)
{
    int i;
    size_t start = 0;

    for (i = 0; i < m->context_len; i++) {
        context_ids[i] = m->pad_index;
    }

    if (seed->count > (size_t)m->context_len) {
        start = seed->count - (size_t)m->context_len;
    }

    for (i = 0; i < m->context_len && start + (size_t)i < seed->count; i++) {
        int idx = find_word_index(m, seed->tokens[start + (size_t)i]);
        if (idx < 0) {
            idx = m->unk_index;
        }
        context_ids[i] = idx;
    }
}

static void shift_context(int *context_ids, int context_len, int next_idx)
{
    int i;

    for (i = 0; i < context_len - 1; i++) {
        context_ids[i] = context_ids[i + 1];
    }

    context_ids[context_len - 1] = next_idx;
}

static int should_skip_word(const char *word)
{
    return strcmp(word, "<pad>") == 0 ||
           strcmp(word, "<unk>") == 0 ||
           strcmp(word, "starttok") == 0;
}

static int is_endtok(const char *word)
{
    return strcmp(word, "endtok") == 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s model.bin \"seed words\" output_len max_lines [mode] [seed]\n"
            "  mode: greedy or random (default: greedy)\n",
            prog);
}

int main(int argc, char *argv[])
{
    struct Model m;
    struct TokenizedSeed seed;
    const char *model_file;
    const char *seed_text;
    int output_len;
    int max_lines;
    const char *mode = "greedy";
    unsigned int rand_seed;
    int context_ids[MAX_CONTEXT];
    float *x;
    float *hidden;
    float *probs;
    int step;
    size_t i;
    int printed_any = 0;
    int line_count = 0;

    if (argc < 5 || argc > 7) {
        usage(argv[0]);
        return 1;
    }

    model_file = argv[1];
    seed_text = argv[2];
    output_len = atoi(argv[3]);
    max_lines = atoi(argv[4]);

    if (output_len <= 0) {
        fprintf(stderr, "output_len must be > 0\n");
        return 1;
    }

    if (max_lines <= 0) {
        fprintf(stderr, "max_lines must be > 0\n");
        return 1;
    }

    if (argc >= 6) {
        mode = argv[5];
        if (strcmp(mode, "greedy") != 0 && strcmp(mode, "random") != 0) {
            fprintf(stderr, "mode must be 'greedy' or 'random'\n");
            return 1;
        }
    }

    if (argc == 7) {
        rand_seed = (unsigned int)strtoul(argv[6], NULL, 10);
    } else {
        rand_seed = (unsigned int)time(NULL);
    }

    srand(rand_seed);

    model_load(model_file, &m);
    tokenize_seed(seed_text, &seed);

    x = malloc((size_t)m.input_size * sizeof(float));
    hidden = malloc((size_t)m.hidden_size * sizeof(float));
    probs = malloc((size_t)m.output_size * sizeof(float));

    if (x == NULL || hidden == NULL || probs == NULL) {
        tokenized_seed_free(&seed);
        model_free(&m);
        die("malloc failed for inference buffers");
    }

    init_context_from_seed(&m, &seed, context_ids);

    for (i = 0; i < seed.count; i++) {
        if (should_skip_word(seed.tokens[i]) || is_endtok(seed.tokens[i])) {
            continue;
        }

        if (printed_any) {
            putchar(' ');
        }
        fputs(seed.tokens[i], stdout);
        printed_any = 1;
    }

    for (step = 0; step < output_len; step++) {
        int next_idx;
        const char *next_word;

        forward_pass(&m, context_ids, x, hidden, probs);

        if (strcmp(mode, "random") == 0) {
            next_idx = sample_index_random(probs, m.output_size);
        } else {
            next_idx = sample_index_greedy(probs, m.output_size);
        }

        next_word = m.index_to_word[next_idx];
        shift_context(context_ids, m.context_len, next_idx);

        if (is_endtok(next_word)) {
            putchar('\n');
            printed_any = 0;
            line_count++;
            if (line_count >= max_lines) {
                break;
            }
            continue;
        }

        if (should_skip_word(next_word)) {
            continue;
        }

        if (printed_any) {
            putchar(' ');
        }
        fputs(next_word, stdout);
        printed_any = 1;
    }

    putchar('\n');

    free(x);
    free(hidden);
    free(probs);
    tokenized_seed_free(&seed);
    model_free(&m);

    return 0;
}
