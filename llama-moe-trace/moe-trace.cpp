// moe-trace: capture MoE router expert selections ("ffn_moe_topk-<layer>")
// for every token via the backend-scheduler eval callback, with the model
// weights untouched. Output is a raw binary stream of records:
//     int32 layer, int32 n_tokens, int32 k, then n_tokens*k int32 expert ids
// (one record per MoE layer per decoded chunk), parsed downstream into the
// moe-routing-lab trace format.
//
// Usage:
//   MOE_TRACE_OUT=trace.bin llama-moe-trace -m model.gguf -f corpus.txt \
//       -c 512 --no-warmup [--max-trace-tokens N]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct moe_trace_state {
    FILE * out = nullptr;
    size_t records = 0;
};

static bool moe_trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool match = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (ask) {
        return match;
    }
    if (!match) {
        return true;
    }
    auto * st = (moe_trace_state *) user_data;

    const int32_t layer    = atoi(t->name + 13);
    const int32_t k        = (int32_t) t->ne[0];
    const int32_t n_tokens = (int32_t) t->ne[1];

    // ffn_moe_topk is a non-contiguous view (top-8 slice of the 128-wide
    // argsort result): a flat copy would read whole argsort rows instead of
    // the selected experts, so copy row by row honoring the stride
    std::vector<int32_t> ids((size_t) k * n_tokens);
    for (int32_t i = 0; i < n_tokens; i++) {
        ggml_backend_tensor_get(t, ids.data() + (size_t) i * k,
                                (size_t) i * t->nb[1], k * sizeof(int32_t));
    }

    fwrite(&layer,    sizeof(int32_t), 1, st->out);
    fwrite(&n_tokens, sizeof(int32_t), 1, st->out);
    fwrite(&k,        sizeof(int32_t), 1, st->out);
    fwrite(ids.data(), sizeof(int32_t), ids.size(), st->out);
    st->records++;
    return true;
}

int main(int argc, char ** argv) {
    moe_trace_state st;

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * out_path = getenv("MOE_TRACE_OUT");
    if (!out_path) {
        out_path = "moe_trace.bin";
    }
    st.out = fopen(out_path, "wb");
    if (!st.out) {
        LOG_ERR("cannot open %s for writing\n", out_path);
        return 1;
    }

    long max_trace_tokens = 0;   // 0 = whole corpus
    if (const char * s = getenv("MOE_TRACE_MAX_TOKENS")) {
        max_trace_tokens = atol(s);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = moe_trace_cb;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init model/context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens =
        common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (max_trace_tokens > 0 && (long) tokens.size() > max_trace_tokens) {
        tokens.resize(max_trace_tokens);
    }
    LOG_INF("tracing %zu tokens in chunks of %d -> %s\n",
            tokens.size(), params.n_batch, out_path);

    const int chunk = params.n_batch;
    // logits requested for every position: prevents llama.cpp from pruning
    // the last layer's FFN to the final token, which would truncate its trace
    llama_batch batch = llama_batch_init(chunk, 0, 1);
    for (size_t start = 0; start < tokens.size(); start += chunk) {
        const int n = (int) std::min((size_t) chunk, tokens.size() - start);
        if (n < 8) {
            break;
        }
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i]    = tokens[start + i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = true;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch)) {
            LOG_ERR("decode failed at offset %zu\n", start);
            return 1;
        }
        LOG_INF("  %zu / %zu tokens\n",
                std::min(start + (size_t) n, tokens.size()), tokens.size());
    }

    llama_batch_free(batch);
    fclose(st.out);
    LOG_INF("wrote %zu records to %s\n", st.records, out_path);

    llama_backend_free();
    return 0;
}
