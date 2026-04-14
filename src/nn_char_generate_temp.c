#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_VOCAB 256
#define MAX_CONTEXT 32
#define MODEL_MAGIC "NCM1"

struct Model {
    int context_len;
    int vocab_size;
    int input_size;
    int hidden_size;
    int output_size;

    int char_to_index[256];
    unsigned char index_to_char[MAX_VOCAB];

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

static void softmax_with_temperature(float *logits, int n, float temperature)
{
    int i;
    float maxv;
    float sum = 0.0f;

    if (temperature <= 0.0f) {
        temperature = 1.0f;
    }

    for (i = 0; i < n; i++) {
        logits[i] /= temperature;
    }

    maxv = logits[0];
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

static void model_init_empty(struct Model *m)
{
    int i;

    memset(m, 0, sizeof(*m));

    for (i = 0; i < 256; i++) {
        m->char_to_index[i] = -1;
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

static void model_load(const char *filename, struct Model *m)
{
    FILE *fp;
    char magic[4];
    int i;

    model_init_empty(m);

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("fopen");
        die("failed to open model file");
    }

    if (fread(magic, 1, 4, fp) != 4) {
        fclose(fp);
        die("failed to read model magic");
    }

    if (memcmp(magic, MODEL_MAGIC, 4) != 0) {
        fclose(fp);
        die("invalid model file magic");
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

    if (fread(m->index_to_char, 1, (size_t)m->vocab_size, fp) != (size_t)m->vocab_size) {
        fclose(fp);
        die("failed to read vocab mapping");
    }

    for (i = 0; i < 256; i++) {
        m->char_to_index[i] = -1;
    }

    for (i = 0; i < m->vocab_size; i++) {
        m->char_to_index[m->index_to_char[i]] = i;
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
                               const unsigned char *context,
                               float *x)
{
    int c, v;

    for (c = 0; c < m->context_len; c++) {
        for (v = 0; v < m->vocab_size; v++) {
            x[c * m->vocab_size + v] = 0.0f;
        }
    }

    for (c = 0; c < m->context_len; c++) {
        int idx = m->char_to_index[context[c]];

        if (idx >= 0) {
            x[c * m->vocab_size + idx] = 1.0f;
        }
    }
}

static void forward_pass(const struct Model *m,
                         const unsigned char *context,
                         float *x,
                         float *hidden,
                         float *probs,
                         float temperature)
{
    int i, j;

    build_input_vector(m, context, x);

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

    softmax_with_temperature(probs, m->output_size, temperature);
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

static void shift_context(unsigned char *context, int context_len, unsigned char next_ch)
{
    int i;

    for (i = 0; i < context_len - 1; i++) {
        context[i] = context[i + 1];
    }

    context[context_len - 1] = next_ch;
}

static void init_context_from_seed(const struct Model *m,
                                   const char *seed,
                                   unsigned char *context)
{
    size_t seed_len = strlen(seed);
    int i;

    for (i = 0; i < m->context_len; i++) {
        context[i] = ' ';
    }

    if (seed_len >= (size_t)m->context_len) {
        memcpy(context, seed + seed_len - (size_t)m->context_len, (size_t)m->context_len);
    } else {
        memcpy(context + (m->context_len - (int)seed_len), seed, seed_len);
    }
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s model.bin seed_text output_len temperature [mode] [seed]\n"
            "  mode: greedy or random (default: random)\n",
            prog);
}

int main(int argc, char *argv[])
{
    struct Model m;
    const char *model_file;
    const char *seed_text;
    int output_len;
    float temperature;
    const char *mode = "random";
    unsigned int rand_seed;
    unsigned char context[MAX_CONTEXT];
    float *x;
    float *hidden;
    float *probs;
    int step;

    if (argc < 5 || argc > 7) {
        usage(argv[0]);
        return 1;
    }

    model_file = argv[1];
    seed_text = argv[2];
    output_len = atoi(argv[3]);
    temperature = (float)atof(argv[4]);

    if (output_len <= 0) {
        fprintf(stderr, "output_len must be > 0\n");
        return 1;
    }

    if (temperature <= 0.0f) {
        fprintf(stderr, "temperature must be > 0\n");
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

    x = malloc((size_t)m.input_size * sizeof(float));
    hidden = malloc((size_t)m.hidden_size * sizeof(float));
    probs = malloc((size_t)m.output_size * sizeof(float));

    if (x == NULL || hidden == NULL || probs == NULL) {
        model_free(&m);
        die("malloc failed for inference buffers");
    }

    init_context_from_seed(&m, seed_text, context);

    printf("%s", seed_text);

    for (step = 0; step < output_len; step++) {
        int next_idx;
        unsigned char next_ch;

        forward_pass(&m, context, x, hidden, probs, temperature);

        if (strcmp(mode, "greedy") == 0) {
            next_idx = sample_index_greedy(probs, m.output_size);
        } else {
            next_idx = sample_index_random(probs, m.output_size);
        }

        next_ch = m.index_to_char[next_idx];
        putchar((int)next_ch);
        shift_context(context, m.context_len, next_ch);
    }

    putchar('\n');

    free(x);
    free(hidden);
    free(probs);
    model_free(&m);

    return 0;
}
