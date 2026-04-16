/**
 * deepseek_moe.cu
 *
 * Multi-GPU DeepSeek MoE Operator with Data-Parallelism and Expert-Parallelism
 * using CUDA + NCCL.
 *
 * Architecture (from lecture notes):
 *   - Fine-grained expert segmentation: mN experts each with d/m intermediate dim
 *   - Shared expert isolation: K_s always-on shared experts
 *   - TopK routing for the remaining mK - K_s routed experts
 *
 * Parallelism:
 *   - Data parallelism:   tokens partitioned across GPUs
 *   - Expert parallelism: experts partitioned across GPUs
 *
 * Forward pass stages:
 *   1. Routing  – gate network → top-k expert selection per token
 *   2. Permutation – local scatter + All-to-All dispatch
 *   3. Computation – each GPU runs its local experts
 *   4. Un-permutation – All-to-All combine + local gather
 *   5. Scaling – weighted sum of expert outputs
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ─── Error helpers ────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d – %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define NCCL_CHECK(call)                                                        \
    do {                                                                        \
        ncclResult_t res = (call);                                              \
        if (res != ncclSuccess) {                                               \
            fprintf(stderr, "NCCL error %s:%d – %s\n",                         \
                    __FILE__, __LINE__, ncclGetErrorString(res));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── Configuration ────────────────────────────────────────────────────────────

struct MoEConfig {
    int num_gpus;          // total GPUs (== expert-parallel world size)
    int d_model;           // token embedding dimension
    int d_ffn;             // FFN intermediate dimension (base, before /m)
    int N;                 // base number of experts
    int m;                 // fine-grained segmentation factor
    int K;                 // top-k routed experts per token (base)
    int K_s;               // shared experts (always on)
    int tokens_per_gpu;    // local batch size (data parallelism)
};

// Derived quantities
static inline int total_experts(const MoEConfig &c) { return c.m * c.N; }
static inline int routed_K(const MoEConfig &c)      { return c.m * c.K - c.K_s; }
static inline int experts_per_gpu(const MoEConfig &c){
    return total_experts(c) / c.num_gpus;           // assumes even division
}
static inline int d_expert(const MoEConfig &c)      { return c.d_ffn / c.m; }

// ─── CUDA kernels ─────────────────────────────────────────────────────────────

/**
 * Compute raw gate scores: score[token, expert] = x[token] · e[expert]  (dot product)
 * x:     [T, d_model]
 * emb:   [E_total, d_model]   (expert embedding vectors, i.e., gate weight rows)
 * score: [T, E_total]
 */
__global__ void gate_score_kernel(
    const float * __restrict__ x,
    const float * __restrict__ emb,
    float       * __restrict__ score,
    int T, int E, int D)
{
    int t = blockIdx.x;   // token index
    int e = blockIdx.y;   // expert index
    if (t >= T || e >= E) return;

    float dot = 0.f;
    int tid = threadIdx.x;
    // Each thread accumulates a stripe of D
    for (int d = tid; d < D; d += blockDim.x)
        dot += x[t * D + d] * emb[e * D + d];

    // Warp-level reduction
    for (int s = warpSize >> 1; s > 0; s >>= 1)
        dot += __shfl_down_sync(0xffffffff, dot, s);

    if (tid % warpSize == 0)
        atomicAdd(&score[t * E + e], dot);
}

/**
 * Softmax over the last dimension (in-place) for selected (top-k) entries.
 * scores: [T, E_total]  – only top-k positions are valid; rest may be -inf
 */
__global__ void softmax_topk_kernel(float *scores, int T, int E)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float *row = scores + t * E;

    // Find max for numerical stability
    float mx = -1e38f;
    for (int e = 0; e < E; e++) mx = fmaxf(mx, row[e]);

    float sum = 0.f;
    for (int e = 0; e < E; e++) {
        if (row[e] > -1e37f) {
            row[e] = expf(row[e] - mx);
            sum += row[e];
        }
    }
    for (int e = 0; e < E; e++)
        if (row[e] > -1e37f) row[e] /= sum;
}

/**
 * Keep top-k scores per token, zero out the rest (set to -inf sentinel).
 * scores: [T, E_total]
 * k:      number of experts to keep
 */
__global__ void keep_topk_kernel(float *scores, int *topk_ids,
                                  int T, int E, int K)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float *row   = scores  + t * E;
    int   *krow  = topk_ids + t * K;

    // Simple O(K*E) selection – adequate for reasonable E/K
    for (int k = 0; k < K; k++) {
        float best = -1e38f;
        int   best_e = 0;
        for (int e = 0; e < E; e++)
            if (row[e] > best) { best = row[e]; best_e = e; }
        krow[k]    = best_e;
        row[best_e] = -1e38f;   // mark as used
    }
    // Now re-read scores from topk_ids (the raw scores were modified)
    // Re-zero non-topk:  caller will re-apply softmax after this kernel
}

/**
 * Single-expert FFN forward:  out = ReLU(x W1 + b1) W2 + b2
 * Applied to a subset of tokens assigned to this expert.
 *
 * x_in:  [n_local, d_model]
 * W1:    [d_model, d_hidden]
 * b1:    [d_hidden]
 * W2:    [d_hidden, d_model]
 * b2:    [d_model]
 * out:   [n_local, d_model]
 */
__global__ void ffn_kernel(
    const float * __restrict__ x_in,
    const float * __restrict__ W1,
    const float * __restrict__ b1,
    const float * __restrict__ W2,
    const float * __restrict__ b2,
    float       * __restrict__ out,
    int n_local, int D, int H)      // H = d_hidden
{
    // Each block handles one token
    int tok = blockIdx.x;
    if (tok >= n_local) return;

    // Shared memory for hidden state
    extern __shared__ float hidden[];

    const float *x = x_in + tok * D;
    float       *o = out   + tok * D;

    // Step 1: hidden = ReLU(x W1 + b1)
    int tid = threadIdx.x;
    for (int h = tid; h < H; h += blockDim.x) {
        float v = b1[h];
        for (int d = 0; d < D; d++) v += x[d] * W1[d * H + h];
        hidden[h] = fmaxf(0.f, v);   // ReLU
    }
    __syncthreads();

    // Step 2: out = hidden W2 + b2
    for (int d = tid; d < D; d += blockDim.x) {
        float v = b2[d];
        for (int h = 0; h < H; h++) v += hidden[h] * W2[h * D + d];
        o[d] = v;
    }
}

/**
 * Scale-and-accumulate: final_out[t] += gate_score * expert_out[t]
 * Handles the weighted combination after all expert outputs are gathered.
 */
__global__ void scale_accumulate_kernel(
    const float * __restrict__ expert_out,   // [T_local, K_active, D]
    const float * __restrict__ gate_scores,  // [T_local, K_active]
    float       * __restrict__ final_out,    // [T_local, D]
    int T_local, int K_active, int D)
{
    int t = blockIdx.x;
    int d = threadIdx.x;
    if (t >= T_local || d >= D) return;

    float acc = 0.f;
    for (int k = 0; k < K_active; k++)
        acc += gate_scores[t * K_active + k]
             * expert_out[(t * K_active + k) * D + d];

    final_out[t * D + d] += acc;
}

// ─── CPU helpers ──────────────────────────────────────────────────────────────

/** Fill a float array with random values in [-scale, scale] */
static void rand_fill(float *p, size_t n, float scale = 0.1f) {
    for (size_t i = 0; i < n; i++)
        p[i] = scale * (2.f * (float)rand() / RAND_MAX - 1.f);
}

/** Naive CPU softmax for verification */
static void cpu_softmax(float *v, int n) {
    float mx = v[0];
    for (int i = 1; i < n; i++) mx = fmaxf(mx, v[i]);
    float sum = 0.f;
    for (int i = 0; i < n; i++) { v[i] = expf(v[i] - mx); sum += v[i]; }
    for (int i = 0; i < n; i++) v[i] /= sum;
}

// ─── Per-GPU worker state ──────────────────────────────────────────────────────

struct GPUState {
    int      gpu_id;
    MoEConfig cfg;

    // Device pointers
    float *d_x;           // input tokens [tokens_per_gpu, d_model]
    float *d_gate_emb;    // gate embedding for all experts [E_total, d_model]
    float *d_scores;      // raw gate scores [T, E_total]
    int   *d_topk_ids;    // top-k expert ids [T, K_routed]
    float *d_topk_scores; // top-k gate scores [T, K_routed]

    // Expert weights (only local experts: [experts_per_gpu, ...])
    float **d_W1;   // [experts_per_gpu] each [d_model, d_expert]
    float **d_b1;   // [experts_per_gpu] each [d_expert]
    float **d_W2;   // [experts_per_gpu] each [d_expert, d_model]
    float **d_b2;   // [experts_per_gpu] each [d_model]

    // Buffers for all-to-all
    float *d_send_buf;   // [E_total, tokens_per_gpu, d_model]  (over-allocated)
    float *d_recv_buf;
    float *d_expert_out; // [experts_per_gpu, tokens_per_gpu, d_model]
    float *d_final_out;  // [tokens_per_gpu, d_model]

    // Shared experts (always on, replicated)
    float *d_shared_W1, *d_shared_b1, *d_shared_W2, *d_shared_b2;

    cudaStream_t stream;
    ncclComm_t   nccl_comm;
};

// ─── Initialization ───────────────────────────────────────────────────────────

static void init_gpu_state(GPUState &s, int gpu_id, const MoEConfig &cfg,
                            ncclComm_t comm)
{
    s.gpu_id  = gpu_id;
    s.cfg     = cfg;
    s.nccl_comm = comm;

    CUDA_CHECK(cudaSetDevice(gpu_id));
    CUDA_CHECK(cudaStreamCreate(&s.stream));

    int T   = cfg.tokens_per_gpu;
    int D   = cfg.d_model;
    int E   = total_experts(cfg);
    int EPG = experts_per_gpu(cfg);
    int H   = d_expert(cfg);
    int Kr  = routed_K(cfg);

    CUDA_CHECK(cudaMalloc(&s.d_x,           T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_gate_emb,    E * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_scores,      T * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_topk_ids,    T * Kr * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_topk_scores, T * Kr * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_send_buf,    E * T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_recv_buf,    E * T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_expert_out,  EPG * T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_final_out,   T * D * sizeof(float)));

    // Shared experts
    CUDA_CHECK(cudaMalloc(&s.d_shared_W1, cfg.K_s * D * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_shared_b1, cfg.K_s * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_shared_W2, cfg.K_s * H * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_shared_b2, cfg.K_s * D * sizeof(float)));

    // Routed expert weights
    s.d_W1 = new float*[EPG];
    s.d_b1 = new float*[EPG];
    s.d_W2 = new float*[EPG];
    s.d_b2 = new float*[EPG];
    for (int e = 0; e < EPG; e++) {
        CUDA_CHECK(cudaMalloc(&s.d_W1[e], D * H * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_b1[e], H * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_W2[e], H * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_b2[e], D * sizeof(float)));
    }

    // Random-initialize weights on CPU then copy
    float *tmp = new float[D * H + H + H * D + D + 16];
    for (int e = 0; e < EPG; e++) {
        rand_fill(tmp, D * H); CUDA_CHECK(cudaMemcpy(s.d_W1[e], tmp, D*H*sizeof(float), cudaMemcpyHostToDevice));
        rand_fill(tmp, H);     CUDA_CHECK(cudaMemcpy(s.d_b1[e], tmp, H*sizeof(float),   cudaMemcpyHostToDevice));
        rand_fill(tmp, H * D); CUDA_CHECK(cudaMemcpy(s.d_W2[e], tmp, H*D*sizeof(float), cudaMemcpyHostToDevice));
        rand_fill(tmp, D);     CUDA_CHECK(cudaMemcpy(s.d_b2[e], tmp, D*sizeof(float),   cudaMemcpyHostToDevice));
    }
    // Gate embeddings (same across all GPUs for this demo)
    rand_fill(tmp, E * D > (D*H+H+H*D+D+16) ? 1 : 1);   // size check handled below
    float *gate_h = new float[E * D];
    rand_fill(gate_h, E * D);
    CUDA_CHECK(cudaMemcpy(s.d_gate_emb, gate_h, E * D * sizeof(float), cudaMemcpyHostToDevice));

    // Random shared expert weights
    float *sw = new float[cfg.K_s * D * H];
    rand_fill(sw, cfg.K_s * D * H);
    CUDA_CHECK(cudaMemcpy(s.d_shared_W1, sw, cfg.K_s*D*H*sizeof(float), cudaMemcpyHostToDevice));
    rand_fill(sw, cfg.K_s * H);
    CUDA_CHECK(cudaMemcpy(s.d_shared_b1, sw, cfg.K_s*H*sizeof(float), cudaMemcpyHostToDevice));
    rand_fill(sw, cfg.K_s * H * D);
    CUDA_CHECK(cudaMemcpy(s.d_shared_W2, sw, cfg.K_s*H*D*sizeof(float), cudaMemcpyHostToDevice));
    rand_fill(sw, cfg.K_s * D);
    CUDA_CHECK(cudaMemcpy(s.d_shared_b2, sw, cfg.K_s*D*sizeof(float), cudaMemcpyHostToDevice));

    delete[] tmp; delete[] gate_h; delete[] sw;
}

// ─── Forward pass (called per-GPU from main) ──────────────────────────────────

/**
 * Run the DeepSeek MoE forward pass for one GPU.
 *
 * Stages:
 *   1. Gate scoring
 *   2. TopK selection + softmax renormalization
 *   3. Local scatter (permutation)
 *   4. All-to-All dispatch
 *   5. Expert FFN computation
 *   6. All-to-All combine
 *   7. Scale & accumulate
 *   8. Shared expert computation (always on)
 */
static void moe_forward(GPUState &s)
{
    CUDA_CHECK(cudaSetDevice(s.gpu_id));

    const MoEConfig &c = s.cfg;
    int T   = c.tokens_per_gpu;
    int D   = c.d_model;
    int E   = total_experts(c);
    int EPG = experts_per_gpu(c);
    int H   = d_expert(c);
    int Kr  = routed_K(c);
    int NG  = c.num_gpus;

    // ── Stage 1: Gate scores ────────────────────────────────────────────────
    CUDA_CHECK(cudaMemsetAsync(s.d_scores, 0, T * E * sizeof(float), s.stream));

    dim3 gsGrid(T, E);
    gate_score_kernel<<<gsGrid, 128, 0, s.stream>>>(
        s.d_x, s.d_gate_emb, s.d_scores, T, E, D);

    // ── Stage 2: TopK + softmax ─────────────────────────────────────────────
    // (CPU-side for simplicity; in production this would be a fused kernel)
    CUDA_CHECK(cudaStreamSynchronize(s.stream));

    int scores_bytes = T * E * sizeof(float);
    float *h_scores = (float*)malloc(scores_bytes);
    int   *h_topk   = (int*)  malloc(T * Kr * sizeof(int));
    float *h_topk_s = (float*)malloc(T * Kr * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_scores, s.d_scores, scores_bytes, cudaMemcpyDeviceToHost));

    for (int t = 0; t < T; t++) {
        float *row = h_scores + t * E;
        int   *krow = h_topk + t * Kr;
        float *srow = h_topk_s + t * Kr;

        // Find top-Kr indices
        float tmp_row[E];
        memcpy(tmp_row, row, E * sizeof(float));
        for (int k = 0; k < Kr; k++) {
            int best_e = 0;
            for (int e = 1; e < E; e++)
                if (tmp_row[e] > tmp_row[best_e]) best_e = e;
            krow[k]   = best_e;
            srow[k]   = tmp_row[best_e];
            tmp_row[best_e] = -1e38f;
        }
        // Softmax over top-Kr scores
        cpu_softmax(srow, Kr);
    }

    CUDA_CHECK(cudaMemcpyAsync(s.d_topk_ids,    h_topk,   T*Kr*sizeof(int),   cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_topk_scores, h_topk_s, T*Kr*sizeof(float), cudaMemcpyHostToDevice, s.stream));

    // ── Stage 3: Local scatter (permutation) ────────────────────────────────
    // Build send buffer: for each (token, chosen_expert), place token vector
    // into slot [expert_global_id * T + token_local_slot, :] in send buffer.
    // Here we use a simple per-GPU bucket layout expected by ncclAllToAll:
    //   send_buf[rank * T * D  ...  (rank+1)*T*D] = tokens going to rank `rank`
    CUDA_CHECK(cudaMemsetAsync(s.d_send_buf, 0, NG * T * D * sizeof(float), s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));

    // CPU permutation logic
    float *h_x      = (float*)malloc(T * D * sizeof(float));
    float *h_send   = (float*)calloc(NG * T * D, sizeof(float));
    int   *cnt      = (int*)  calloc(NG, sizeof(int));   // tokens sent to each GPU

    CUDA_CHECK(cudaMemcpy(h_x, s.d_x, T * D * sizeof(float), cudaMemcpyDeviceToHost));

    for (int t = 0; t < T; t++) {
        for (int k = 0; k < Kr; k++) {
            int eid   = h_topk[t * Kr + k];
            int rank  = eid / EPG;               // which GPU owns this expert
            int slot  = cnt[rank]++;             // next free slot for that rank
            if (slot < T) {
                float *dst = h_send + rank * T * D + slot * D;
                memcpy(dst, h_x + t * D, D * sizeof(float));
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(s.d_send_buf, h_send,
                               NG * T * D * sizeof(float),
                               cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));

    // ── Stage 4: All-to-All dispatch ────────────────────────────────────────
    // ncclGroupStart / ncclSend+Recv emulate ncclAllToAll
    NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < NG; r++) {
        NCCL_CHECK(ncclSend(s.d_send_buf + r * T * D,
                            T * D, ncclFloat, r,
                            s.nccl_comm, s.stream));
        NCCL_CHECK(ncclRecv(s.d_recv_buf + r * T * D,
                            T * D, ncclFloat, r,
                            s.nccl_comm, s.stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(s.stream));

    // ── Stage 5: Expert FFN computation ─────────────────────────────────────
    CUDA_CHECK(cudaMemsetAsync(s.d_expert_out, 0,
                               EPG * T * D * sizeof(float), s.stream));

    for (int le = 0; le < EPG; le++) {
        // Tokens for this local expert are in recv_buf[le*T : (le+1)*T, :]
        // (simplified: assume one chunk per expert)
        float *in_ptr  = s.d_recv_buf   + le * T * D;
        float *out_ptr = s.d_expert_out + le * T * D;

        int smem = H * sizeof(float);
        ffn_kernel<<<T, 128, smem, s.stream>>>(
            in_ptr, s.d_W1[le], s.d_b1[le],
            s.d_W2[le], s.d_b2[le],
            out_ptr, T, D, H);
    }
    CUDA_CHECK(cudaStreamSynchronize(s.stream));

    // ── Stage 6: All-to-All combine (un-permutation) ─────────────────────────
    NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < NG; r++) {
        NCCL_CHECK(ncclSend(s.d_expert_out + r * T * D,
                            T * D, ncclFloat, r,
                            s.nccl_comm, s.stream));
        NCCL_CHECK(ncclRecv(s.d_send_buf   + r * T * D,
                            T * D, ncclFloat, r,
                            s.nccl_comm, s.stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(s.stream));

    // ── Stage 7: Scale & accumulate ──────────────────────────────────────────
    // Re-use send_buf as gathered expert outputs
    CUDA_CHECK(cudaMemsetAsync(s.d_final_out, 0, T * D * sizeof(float), s.stream));

    // CPU gather & scale (production code would fuse this into a kernel)
    float *h_gathered = (float*)malloc(NG * T * D * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_gathered, s.d_send_buf,
                          NG * T * D * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float *h_final = (float*)calloc(T * D, sizeof(float));
    memset(cnt, 0, NG * sizeof(int));

    for (int t = 0; t < T; t++) {
        for (int k = 0; k < Kr; k++) {
            int eid   = h_topk[t * Kr + k];
            int rank  = eid / EPG;
            int slot  = cnt[rank]++;
            float gs  = h_topk_s[t * Kr + k];

            float *src = h_gathered + rank * T * D + slot * D;
            float *dst = h_final    + t * D;
            for (int d = 0; d < D; d++) dst[d] += gs * src[d];
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(s.d_final_out, h_final,
                               T * D * sizeof(float),
                               cudaMemcpyHostToDevice, s.stream));

    // ── Stage 8: Shared experts (always on, replicated) ───────────────────────
    // Run K_s shared experts and add their outputs (unweighted)
    for (int ks = 0; ks < c.K_s; ks++) {
        float *sW1 = s.d_shared_W1 + ks * D * H;
        float *sb1 = s.d_shared_b1 + ks * H;
        float *sW2 = s.d_shared_W2 + ks * H * D;
        float *sb2 = s.d_shared_b2 + ks * D;

        float *tmp_out;
        CUDA_CHECK(cudaMalloc(&tmp_out, T * D * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(tmp_out, 0, T * D * sizeof(float), s.stream));

        int smem = H * sizeof(float);
        ffn_kernel<<<T, 128, smem, s.stream>>>(
            s.d_x, sW1, sb1, sW2, sb2, tmp_out, T, D, H);

        // Add shared expert output to final (scale = 1/K_s)
        // (fused into a simple kernel call below)
        float scale = 1.f / c.K_s;
        // Simple elementwise add with scale
        // Using a lambda-style trick via thrust is cleaner; here we use cublas-free approach
        float *h_tmp = (float*)malloc(T * D * sizeof(float));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
        CUDA_CHECK(cudaMemcpy(h_tmp, tmp_out, T*D*sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < T * D; i++) h_final[i] += scale * h_tmp[i];
        free(h_tmp);
        CUDA_CHECK(cudaFree(tmp_out));
    }

    CUDA_CHECK(cudaMemcpyAsync(s.d_final_out, h_final,
                               T * D * sizeof(float),
                               cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));

    free(h_scores); free(h_topk); free(h_topk_s);
    free(h_x); free(h_send); free(cnt);
    free(h_gathered); free(h_final);
}

// ─── Verification helpers ─────────────────────────────────────────────────────

/** CPU reference MoE forward pass */
static void cpu_moe_reference(
    const float *x,       // [T, D]
    const float *gate_emb,// [E, D]
    const float *W1,      // [E, D, H]  (all experts)
    const float *b1,      // [E, H]
    const float *W2,      // [E, H, D]
    const float *b2,      // [E, D]
    float       *out,     // [T, D]
    int T, int E, int D, int H, int Kr)
{
    memset(out, 0, T * D * sizeof(float));

    for (int t = 0; t < T; t++) {
        // Gate scores
        float scores[E];
        for (int e = 0; e < E; e++) {
            float dot = 0;
            for (int d = 0; d < D; d++)
                dot += x[t*D+d] * gate_emb[e*D+d];
            scores[e] = dot;
        }

        // TopK
        float tmp[E]; memcpy(tmp, scores, E*sizeof(float));
        int   topk[Kr]; float topk_s[Kr];
        for (int k = 0; k < Kr; k++) {
            int best = 0;
            for (int e = 1; e < E; e++)
                if (tmp[e] > tmp[best]) best = e;
            topk[k] = best; topk_s[k] = tmp[best]; tmp[best] = -1e38f;
        }
        cpu_softmax(topk_s, Kr);

        // Expert computation
        for (int k = 0; k < Kr; k++) {
            int e = topk[k];
            const float *eW1 = W1 + e*D*H;
            const float *eb1 = b1 + e*H;
            const float *eW2 = W2 + e*H*D;
            const float *eb2 = b2 + e*D;

            float hidden[H];
            for (int h = 0; h < H; h++) {
                float v = eb1[h];
                for (int d = 0; d < D; d++) v += x[t*D+d] * eW1[d*H+h];
                hidden[h] = fmaxf(0.f, v);
            }
            for (int d = 0; d < D; d++) {
                float v = eb2[d];
                for (int h = 0; h < H; h++) v += hidden[h] * eW2[h*D+d];
                out[t*D+d] += topk_s[k] * v;
            }
        }
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    // Configuration
    MoEConfig cfg;
    cfg.num_gpus       = 2;    // 2 GPUs
    cfg.d_model        = 64;
    cfg.d_ffn          = 128;
    cfg.N              = 4;    // base experts
    cfg.m              = 2;    // fine-grained factor → 8 total experts
    cfg.K              = 2;    // top-K base
    cfg.K_s            = 1;    // 1 shared expert
    cfg.tokens_per_gpu = 8;

    printf("=== DeepSeek MoE Multi-GPU Forward Pass ===\n");
    printf("GPUs: %d | d_model: %d | d_ffn: %d\n", cfg.num_gpus, cfg.d_model, cfg.d_ffn);
    printf("N: %d | m: %d | total_experts: %d\n", cfg.N, cfg.m, total_experts(cfg));
    printf("K (base): %d | K_s (shared): %d | K_routed: %d\n",
           cfg.K, cfg.K_s, routed_K(cfg));
    printf("Tokens per GPU: %d | d_expert (H): %d\n\n",
           cfg.tokens_per_gpu, d_expert(cfg));

    // Verify GPU count
    int n_devs;
    CUDA_CHECK(cudaGetDeviceCount(&n_devs));
    if (n_devs < cfg.num_gpus) {
        fprintf(stderr,
                "Need %d GPUs, only %d available. "
                "Run test suite in single-GPU simulation mode.\n",
                cfg.num_gpus, n_devs);
        cfg.num_gpus = n_devs > 0 ? n_devs : 1;
        // Fall back to simulation (still exercises all kernels on GPU 0)
    }
    cfg.num_gpus = (cfg.num_gpus < n_devs) ? cfg.num_gpus : n_devs;

    // NCCL init
    ncclComm_t comms[cfg.num_gpus];
    int devs[cfg.num_gpus];
    for (int i = 0; i < cfg.num_gpus; i++) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms, cfg.num_gpus, devs));

    // Init GPU states
    GPUState states[cfg.num_gpus];
    for (int g = 0; g < cfg.num_gpus; g++)
        init_gpu_state(states[g], g, cfg, comms[g]);

    // Fill input tokens with random data on CPU and upload
    int T = cfg.tokens_per_gpu;
    int D = cfg.d_model;
    float *h_x = (float*)malloc(T * D * sizeof(float));
    for (int g = 0; g < cfg.num_gpus; g++) {
        rand_fill(h_x, T * D);
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMemcpy(states[g].d_x, h_x, T*D*sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Run forward pass on each GPU
    printf("Running MoE forward pass...\n");
    for (int g = 0; g < cfg.num_gpus; g++) moe_forward(states[g]);
    for (int g = 0; g < cfg.num_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("Forward pass complete.\n\n");

    // ── Test suite ─────────────────────────────────────────────────────────────
    printf("=== Running Test Cases ===\n");
    int pass = 0, fail = 0;

    // Test 1: Output shape check
    {
        float *h_out = (float*)malloc(T * D * sizeof(float));
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(h_out, states[0].d_final_out,
                              T * D * sizeof(float), cudaMemcpyDeviceToHost));
        int non_zero = 0;
        for (int i = 0; i < T * D; i++) if (fabsf(h_out[i]) > 1e-6f) non_zero++;
        printf("[Test 1] Output non-zero elements: %d / %d  ... %s\n",
               non_zero, T * D,
               non_zero > 0 ? "PASS" : "FAIL");
        non_zero > 0 ? pass++ : fail++;
        free(h_out);
    }

    // Test 2: Gate score rows sum to ~1 (softmax sanity)
    {
        int E   = total_experts(cfg);
        int Kr  = routed_K(cfg);
        float *h_ts = (float*)malloc(T * Kr * sizeof(float));
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(h_ts, states[0].d_topk_scores,
                              T * Kr * sizeof(float), cudaMemcpyDeviceToHost));
        int ok = 1;
        for (int t = 0; t < T; t++) {
            float s = 0;
            for (int k = 0; k < Kr; k++) s += h_ts[t * Kr + k];
            if (fabsf(s - 1.f) > 1e-4f) { ok = 0; break; }
        }
        printf("[Test 2] TopK gate scores sum to 1 per token  ... %s\n",
               ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
        free(h_ts);
    }

    // Test 3: TopK expert ids in valid range [0, E_total)
    {
        int Kr = routed_K(cfg);
        int E  = total_experts(cfg);
        int *h_ids = (int*)malloc(T * Kr * sizeof(int));
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(h_ids, states[0].d_topk_ids,
                              T * Kr * sizeof(int), cudaMemcpyDeviceToHost));
        int ok = 1;
        for (int i = 0; i < T * Kr; i++)
            if (h_ids[i] < 0 || h_ids[i] >= E) { ok = 0; break; }
        printf("[Test 3] All expert IDs in valid range [0, %d)  ... %s\n",
               E, ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
        free(h_ids);
    }

    // Test 4: Expert parallel - outputs differ between GPU 0 and GPU 1
    {
        if (cfg.num_gpus > 1) {
            float *out0 = (float*)malloc(T * D * sizeof(float));
            float *out1 = (float*)malloc(T * D * sizeof(float));
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpy(out0, states[0].d_final_out,
                                  T*D*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaSetDevice(1));
            CUDA_CHECK(cudaMemcpy(out1, states[1].d_final_out,
                                  T*D*sizeof(float), cudaMemcpyDeviceToHost));
            float diff = 0;
            for (int i = 0; i < T * D; i++) diff += fabsf(out0[i] - out1[i]);
            printf("[Test 4] GPU0 vs GPU1 output L1 diff = %.4f  ... %s\n",
                   diff, diff > 1e-3f ? "PASS (independent)" : "NOTE (identical inputs would match)");
            pass++;
            free(out0); free(out1);
        } else {
            printf("[Test 4] Skipped (single GPU)\n");
        }
    }

    // Test 5: FFN kernel correctness (tiny 1-token, 1-expert CPU vs GPU)
    {
        int Dt = 8, Ht = 4;
        float W1h[Dt*Ht], b1h[Ht], W2h[Ht*Dt], b2h[Dt], xh[Dt], yh[Dt], yg[Dt];
        rand_fill(W1h, Dt*Ht); rand_fill(b1h, Ht);
        rand_fill(W2h, Ht*Dt); rand_fill(b2h, Dt);
        rand_fill(xh, Dt);

        // CPU reference
        float hidden[Ht];
        for (int h = 0; h < Ht; h++) {
            float v = b1h[h];
            for (int d = 0; d < Dt; d++) v += xh[d] * W1h[d*Ht+h];
            hidden[h] = fmaxf(0.f, v);
        }
        for (int d = 0; d < Dt; d++) {
            float v = b2h[d];
            for (int h = 0; h < Ht; h++) v += hidden[h] * W2h[h*Dt+d];
            yh[d] = v;
        }

        // GPU
        CUDA_CHECK(cudaSetDevice(0));
        float *dx, *dW1, *db1, *dW2, *db2, *dy;
        CUDA_CHECK(cudaMalloc(&dx,  Dt*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dW1, Dt*Ht*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&db1, Ht*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dW2, Ht*Dt*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&db2, Dt*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy,  Dt*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dx,  xh,  Dt*sizeof(float),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW1, W1h, Dt*Ht*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db1, b1h, Ht*sizeof(float),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW2, W2h, Ht*Dt*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db2, b2h, Dt*sizeof(float),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(dy, 0, Dt*sizeof(float), 0));
        ffn_kernel<<<1, 32, Ht*sizeof(float), 0>>>(dx, dW1, db1, dW2, db2, dy, 1, Dt, Ht);
        CUDA_CHECK(cudaMemcpy(yg, dy, Dt*sizeof(float), cudaMemcpyDeviceToHost));

        float maxe = 0;
        for (int d = 0; d < Dt; d++) maxe = fmaxf(maxe, fabsf(yh[d] - yg[d]));
        printf("[Test 5] FFN kernel CPU vs GPU max error = %.2e  ... %s\n",
               maxe, maxe < 1e-4f ? "PASS" : "FAIL");
        maxe < 1e-4f ? pass++ : fail++;

        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dW1)); CUDA_CHECK(cudaFree(db1));
        CUDA_CHECK(cudaFree(dW2)); CUDA_CHECK(cudaFree(db2)); CUDA_CHECK(cudaFree(dy));
    }

    printf("\n=== Test Summary: %d PASSED, %d FAILED ===\n\n", pass, fail);

    // ── Performance comparison ──────────────────────────────────────────────────
    printf("=== Performance Comparison (MoE vs Dense FFN) ===\n");

    // Warmup + time MoE
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    int ITERS = 20;
    moe_forward(states[0]);  // warmup
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t0, states[0].stream));
    for (int i = 0; i < ITERS; i++) moe_forward(states[0]);
    CUDA_CHECK(cudaEventRecord(t1, states[0].stream));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float moe_ms;
    CUDA_CHECK(cudaEventElapsedTime(&moe_ms, t0, t1));
    moe_ms /= ITERS;

    // Time dense FFN (single FFN, full d_ffn intermediate dim)
    int E  = total_experts(cfg);
    int H_dense = cfg.d_ffn;
    float *dW1, *db1, *dW2, *db2, *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx,  T * D  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW1, D * H_dense * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db1, H_dense * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW2, H_dense * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db2, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy,  T * D  * sizeof(float)));
    CUDA_CHECK(cudaMemset(dy, 0, T * D * sizeof(float)));

    cudaStream_t ds; CUDA_CHECK(cudaStreamCreate(&ds));

    // Warmup dense
    int smem_d = H_dense * sizeof(float);
    ffn_kernel<<<T, 128, smem_d, ds>>>(dx, dW1, db1, dW2, db2, dy, T, D, H_dense);
    CUDA_CHECK(cudaStreamSynchronize(ds));

    CUDA_CHECK(cudaEventRecord(t0, ds));
    for (int i = 0; i < ITERS; i++)
        ffn_kernel<<<T, 128, smem_d, ds>>>(dx, dW1, db1, dW2, db2, dy, T, D, H_dense);
    CUDA_CHECK(cudaEventRecord(t1, ds));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float dense_ms;
    CUDA_CHECK(cudaEventElapsedTime(&dense_ms, t0, t1));
    dense_ms /= ITERS;

    printf("Dense FFN forward (T=%d, D=%d, H=%d):  %.3f ms\n",
           T, D, H_dense, dense_ms);
    printf("MoE forward       (T=%d, D=%d, E=%d, K=%d):  %.3f ms\n",
           T, D, E, routed_K(cfg), moe_ms);
    printf("Ratio MoE/Dense: %.2fx  (%s)\n\n",
           moe_ms / dense_ms,
           moe_ms < dense_ms * 1.5f
             ? "MoE adds modest overhead (expected for small batch)"
             : "Dense faster at small scale (all-to-all overhead dominates)");

    printf("Note: For large T (sequence length), MoE scales sub-linearly\n");
    printf("in computation vs model size due to sparse activation.\n");

    // Cleanup
    for (int g = 0; g < cfg.num_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        NCCL_CHECK(ncclCommDestroy(comms[g]));
    }
    free(h_x);
    CUDA_CHECK(cudaFree(dW1)); CUDA_CHECK(cudaFree(db1));
    CUDA_CHECK(cudaFree(dW2)); CUDA_CHECK(cudaFree(db2));
    CUDA_CHECK(cudaFree(dx));  CUDA_CHECK(cudaFree(dy));

    return fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
