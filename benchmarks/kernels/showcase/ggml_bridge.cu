// Benchmark-only bridge to an unmodified llama.cpp GGML CUDA build.
// Private context access is used solely to obtain the backend's timing stream.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"
#include <exception>
#include <stdexcept>
#include <cstring>

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

extern "C" int showcase_ggml(int m, int n, int k, int type,
        const void * weights, const float * activation, float * output,
        int iterations, int repetitions, double * samples, char * error) {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    try {
        // Disable nested GGML capture: the benchmark captures the full operation
        // (activation quantization, multiplication and any fixup) repeatedly.
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
        ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        if (!ctx) throw std::runtime_error("ggml_init failed");
        backend = ggml_backend_cuda_init(0);
        if (!backend) throw std::runtime_error("ggml_backend_cuda_init failed");
        auto w = ggml_new_tensor_2d(ctx, (ggml_type)type, k, n);
        auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
        auto out = ggml_mul_mat(ctx, w, x);
        auto gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("ggml allocation failed");
        ggml_backend_tensor_set(w, weights, 0, ggml_nbytes(w));
        ggml_backend_tensor_set(x, activation, 0, ggml_nbytes(x));
        for (int i = 0; i < 5; ++i) {
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("ggml graph failed");
        }
        ggml_backend_tensor_get(out, output, 0, ggml_nbytes(out));
        auto cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
        auto stream = cuda_ctx->stream();
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < iterations; ++i) {
            if (ggml_backend_graph_compute_async(backend, gf) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("ggml capture failed");
        }
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int i = 0; i < 3; ++i) check(cudaGraphLaunch(exec, stream));
        check(cudaStreamSynchronize(stream));
        check(cudaEventCreate(&begin));
        check(cudaEventCreate(&end));
        for (int i = 0; i < repetitions; ++i) {
            check(cudaEventRecord(begin, stream));
            check(cudaGraphLaunch(exec, stream));
            check(cudaEventRecord(end, stream));
            check(cudaEventSynchronize(end));
            float ms;
            check(cudaEventElapsedTime(&ms, begin, end));
            samples[i] = ms * 1000.0 / iterations;
        }
        check(cudaEventDestroy(begin));
        check(cudaEventDestroy(end));
        check(cudaGraphExecDestroy(exec));
        check(cudaGraphDestroy(graph));
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 0;
    } catch (const std::exception & e) {
        std::strncpy(error, e.what(), 2047);
        return 1;
    }
}
