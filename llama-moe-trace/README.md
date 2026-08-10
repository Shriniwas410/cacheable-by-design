# llama-moe-trace

A ~120-line addition to [llama.cpp](https://github.com/ggml-org/llama.cpp) that records the
MoE router's top-*k* expert selection (`ffn_moe_topk`) for every layer and token, via the
backend eval-callback — **with the model weights completely untouched**. It emits a compact
binary stream of `(layer, n_tokens, k, ids…)` records, converted into the analysis format by
`../routing-lab/06_convert_gguf_trace.py`.

## Build

1. Clone llama.cpp and drop `moe-trace.cpp` into `examples/eval-callback/`.
2. Add a target to `examples/eval-callback/CMakeLists.txt`:

   ```cmake
   set(TARGET llama-moe-trace)
   add_executable(${TARGET} moe-trace.cpp)
   install(TARGETS ${TARGET} RUNTIME)
   target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
   target_compile_features(${TARGET} PRIVATE cxx_std_17)
   ```
3. Configure and build:

   ```bash
   cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target llama-moe-trace -j
   ```

## Run

```bash
MOE_TRACE_OUT=trace_code.bin MOE_TRACE_MAX_TOKENS=8000 \
  ./build/bin/llama-moe-trace -m model.gguf -f corpus.txt -c 512 -b 512 -t 16 -ngl 0
python ../routing-lab/06_convert_gguf_trace.py --bin trace_code.bin \
  --out traces/code.npz --model-id <name> --num-experts 128
```

## Implementation note (important)

`ffn_moe_topk` is a **non-contiguous view** — the top-*k* slice of a wide argsort tensor. A
flat `ggml_backend_tensor_get` reads whole argsort rows (i.e. garbage). The tell-tale is every
expert id appearing *exactly* k times with sub-chance reuse. `moe-trace.cpp` copies **row by
row honoring the stride** (`i * t->nb[1]`). See the paper's corrections log.

Tested against llama.cpp built from source in August 2026 (Qwen3-30B-A3B, Qwen3-235B-A22B GGUF).
