#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_TEXT_LEN (8 * 1024 * 1024)
#define MAX_VOCAB 256
#define MAX_CONTEXT 32

struct Dataset {
    unsigned char *text;
    size_t text_len;
    int char_to_index[256];
    unsigned char index_to_char[MAX_VOCAB];
    int vocab_size;
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
        die("training file too large for this simple starter model");
    }

    buf = malloc((size_t)size);
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
    *out_len = (size_t)size;
    return buf;
}

static void dataset_init(struct Dataset *ds, const unsigned char *text, size_t text_len)
{
    int seen[256] = {0};
    int i;
    size_t p;

    memset(ds, 0, sizeof(*ds));
    ds->text = malloc(text_len);
    if (ds->text == NULL) {
        die("malloc failed for dataset text");
    }

    memcpy(ds->text, text, text_len);
    ds->text_len = text_len;

    for (i = 0; i < 256; i++) {
        ds->char_to_index[i] = -1;
    }

    for (p = 0; p < text_len; p++) {
        seen[text[p]] = 1;
    }

    ds->vocab_size = 0;
    for (i = 0; i < 256; i++) {
        if (seen[i]) {
            if (ds->vocab_size >= MAX_VOCAB) {
                die("vocab too large");
            }
            ds->char_to_index[i] = ds->vocab_size;
            ds->index_to_char[ds->vocab_size] = (unsigned char)i;
            ds->vocab_size++;
        }
    }
}

static void dataset_free(struct Dataset *ds)
{
    free(ds->text);
    ds->text = NULL;
    ds->text_len = 0;
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

static void build_input_vector(const struct Dataset *ds,
                               size_t pos,
                               int context_len,
                               float *x)
{
    int c, v;

    for (c = 0; c < context_len; c++) {
        for (v = 0; v < ds->vocab_size; v++) {
            x[c * ds->vocab_size + v] = 0.0f;
        }
    }

    for (c = 0; c < context_len; c++) {
        size_t src_pos = pos - (size_t)context_len + (size_t)c;
        unsigned char ch = ds->text[src_pos];
        int idx = ds->char_to_index[ch];

        if (idx >= 0) {
            x[c * ds->vocab_size + idx] = 1.0f;
        }
    }
}

static float train_one_example(struct Model *m,
                               const struct Dataset *ds,
                               size_t pos,
                               float learning_rate,
                               float *x,
                               float *hidden,
                               float *logits,
                               float *d_hidden)
{
    int i, j;
    unsigned char target_ch;
    int target_idx;

    build_input_vector(ds, pos, m->context_len, x);

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

    target_ch = ds->text[pos];
    target_idx = ds->char_to_index[target_ch];
    if (target_idx < 0) {
        return 0.0f;
    }

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

static void save_model(const char *filename, const struct Model *m, const struct Dataset *ds)
{
    FILE *fp;
    int i;

    fp = fopen(filename, "wb");
    if (fp == NULL) {
        perror("fopen");
        die("failed to save model");
    }

    fwrite("NCM1", 1, 4, fp);
    fwrite(&m->context_len, sizeof(int), 1, fp);
    fwrite(&m->vocab_size, sizeof(int), 1, fp);
    fwrite(&m->hidden_size, sizeof(int), 1, fp);
    fwrite(ds->index_to_char, 1, (size_t)ds->vocab_size, fp);

    fwrite(m->w1, sizeof(float), (size_t)m->hidden_size * (size_t)m->input_size, fp);
    fwrite(m->b1, sizeof(float), (size_t)m->hidden_size, fp);
    fwrite(m->w2, sizeof(float), (size_t)m->output_size * (size_t)m->hidden_size, fp);
    fwrite(m->b2, sizeof(float), (size_t)m->output_size, fp);

    fclose(fp);

    for (i = 0; i < ds->vocab_size; i++) {
        (void)i;
    }
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s training.txt model.bin [epochs] [context_len] [hidden_size] [learning_rate]\n",
            prog);
}

int main(int argc, char *argv[])
{
    struct Dataset ds;
    struct Model m;
    unsigned char *text;
    size_t text_len;
    int epochs = 3;
    int context_len = 8;
    int hidden_size = 64;
    float learning_rate = 0.05f;
    int epoch;
    size_t pos;
    size_t example_count;
    float *x;
    float *hidden;
    float *logits;
    float *d_hidden;

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

    dataset_init(&ds, text, text_len);
    free(text);

    if (ds.text_len <= (size_t)context_len) {
        dataset_free(&ds);
        die("training text is too short");
    }

    model_init(&m, context_len, ds.vocab_size, hidden_size);

    x = malloc((size_t)m.input_size * sizeof(float));
    hidden = malloc((size_t)m.hidden_size * sizeof(float));
    logits = malloc((size_t)m.output_size * sizeof(float));
    d_hidden = malloc((size_t)m.hidden_size * sizeof(float));

    if (x == NULL || hidden == NULL || logits == NULL || d_hidden == NULL) {
        die("malloc failed for training buffers");
    }

    example_count = ds.text_len - (size_t)context_len;

    printf("training chars: %zu\n", ds.text_len);
    printf("vocab size: %d\n", ds.vocab_size);
    printf("context len: %d\n", context_len);
    printf("hidden size: %d\n", hidden_size);
    printf("epochs: %d\n", epochs);
    printf("learning rate: %.4f\n", learning_rate);

    for (epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        for (pos = (size_t)context_len; pos < ds.text_len; pos++) {
            total_loss += train_one_example(&m, &ds, pos, learning_rate,
                                            x, hidden, logits, d_hidden);
        }

        printf("epoch %d/%d  avg_loss=%.6f\n",
               epoch + 1, epochs, total_loss / (float)example_count);
    }

    save_model(argv[2], &m, &ds);

    free(x);
    free(hidden);
    free(logits);
    free(d_hidden);
    model_free(&m);
    dataset_free(&ds);

    return 0;
}
