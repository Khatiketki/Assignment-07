"""
deepseek_moe_numpy_tests.py

Pure NumPy test suite for the DeepSeek MoE operator.
No GPU or PyTorch required — verifies all mathematical properties
described in the Week 8 lecture notes.

Run with:   python3 deepseek_moe_numpy_tests.py
"""

import math
import time
import numpy as np

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Core math helpers
# ─────────────────────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def ffn_forward(x, W1, b1, W2, b2):
    """Two-layer FFN: ReLU(x W1 + b1) W2 + b2"""
    h = relu(x @ W1 + b1)
    return h @ W2 + b2


# ─────────────────────────────────────────────────────────────────────────────
# TopK Router  (TopKRouter from lecture notes)
# ─────────────────────────────────────────────────────────────────────────────

def top_k_router(x, W_gate, K):
    """
    TopKRouter(x_i) = Softmax(KeepTopK(x_i W_gate))

    x:       [T, d_model]
    W_gate:  [d_model, E_total]
    K:       number of experts to select

    Returns:
      gate_scores: [T, K]   (softmax-normalised)
      topk_ids:    [T, K]   (expert indices)
    """
    T, D = x.shape
    E = W_gate.shape[1]

    logits = x @ W_gate                          # [T, E]

    # KeepTopK: sort and pick top-K indices
    topk_ids = np.argsort(-logits, axis=-1)[:, :K]   # [T, K]

    # Gather top-K logit values
    rows = np.arange(T)[:, None]
    topk_vals = logits[rows, topk_ids]               # [T, K]

    # Softmax over the K selected scores
    gate_scores = softmax(topk_vals, axis=-1)         # [T, K]

    return gate_scores, topk_ids


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek MoE forward (NumPy reference)
# ─────────────────────────────────────────────────────────────────────────────

class DeepSeekMoENumpy:
    """
    DeepSeek MoE layer (NumPy reference implementation).

    Parameters
    ----------
    d_model : int
    d_ffn   : int   - base FFN intermediate dim; each expert uses d_ffn // m
    N       : int   - base number of experts
    m       : int   - fine-grained segmentation factor  → mN total experts
    K       : int   - top-K base  → mK − K_s routed experts
    K_s     : int   - number of always-on shared experts
    """

    def __init__(self, d_model, d_ffn, N, m, K, K_s, rng=None):
        rng = rng or np.random.default_rng(0)
        self.d_model  = d_model
        self.d_ffn    = d_ffn
        self.N        = N
        self.m        = m
        self.K        = K
        self.K_s      = K_s

        self.E_total  = m * N
        self.d_hidden = d_ffn // m
        self.K_routed = m * K - K_s

        D, H, E = d_model, self.d_hidden, self.E_total
        scale = 0.1

        def rand(*shape): return rng.standard_normal(shape) * scale

        # Gate weight  [D, E]
        self.W_gate = rand(D, E)

        # Shared expert weights (K_s experts)
        self.shared = [
            (rand(D, H), rand(H), rand(H, D), rand(D))
            for _ in range(K_s)
        ]

        # Routed expert weights (E_total experts)
        self.experts = [
            (rand(D, H), rand(H), rand(H, D), rand(D))
            for _ in range(E)
        ]

    def forward(self, u):
        """
        u:  [T, d_model]
        returns: [T, d_model]

        Implements:
            h_t = Σ_shared FFN_i(u_t) / K_s
                + Σ_routed g_{i,t} FFN_i(u_t)
                + u_t       (residual)
        """
        T, D = u.shape

        # ── Shared experts (always on) ────────────────────────────────────
        shared_out = np.zeros_like(u)
        for (W1, b1, W2, b2) in self.shared:
            shared_out += ffn_forward(u, W1, b1, W2, b2)
        if self.K_s > 0:
            shared_out /= self.K_s

        # ── Routing ───────────────────────────────────────────────────────
        routed_out = np.zeros_like(u)
        if self.K_routed == 0:
            return shared_out + routed_out + u

        gate_scores, topk_ids = top_k_router(u, self.W_gate, self.K_routed)

        # ── Routed expert computation ─────────────────────────────────────
        for t in range(T):
            for k in range(self.K_routed):
                e  = topk_ids[t, k]
                gs = gate_scores[t, k]
                W1, b1, W2, b2 = self.experts[e]
                routed_out[t] += gs * ffn_forward(u[t:t+1], W1, b1, W2, b2)[0]

        # ── Residual ──────────────────────────────────────────────────────
        return shared_out + routed_out + u


# ─────────────────────────────────────────────────────────────────────────────
# Expert-parallel forward (simulates all-to-all across 'num_ranks' workers)
# ─────────────────────────────────────────────────────────────────────────────

def expert_parallel_forward(moe, u, num_ranks):
    """
    Simulate the 5-stage expert-parallel forward pass:
      1. Routing
      2. Permutation  (local scatter + all-to-all dispatch)
      3. Computation  (each rank processes its assigned experts)
      4. Un-permutation (all-to-all combine + local gather)
      5. Scaling
    """
    T, D = u.shape
    EPR  = moe.E_total // num_ranks    # experts per rank

    # ── Shared experts ────────────────────────────────────────────────────
    shared_out = np.zeros_like(u)
    for (W1, b1, W2, b2) in moe.shared:
        shared_out += ffn_forward(u, W1, b1, W2, b2)
    if moe.K_s > 0:
        shared_out /= moe.K_s

    # ── Stage 1: Routing ─────────────────────────────────────────────────
    gate_scores, topk_ids = top_k_router(u, moe.W_gate, moe.K_routed)

    # ── Stage 2: Permutation (build per-rank send buffers) ────────────────
    # send_bufs[rank] = list of (token_vec, gate_score, original_token_idx, local_expert_idx)
    send_bufs = [[] for _ in range(num_ranks)]
    for t in range(T):
        for k in range(moe.K_routed):
            e    = topk_ids[t, k]
            rank = e // EPR
            send_bufs[rank].append((u[t].copy(), gate_scores[t, k], t, e % EPR))

    # ── Stage 3: Computation (each rank processes its local experts) ──────
    # (In real NCCL: after all-to-all dispatch, each rank has recv_buf)
    recv_results = [[] for _ in range(num_ranks)]
    for rank in range(num_ranks):
        for (tok_vec, gs, src_t, local_eid) in send_bufs[rank]:
            global_eid = rank * EPR + local_eid
            W1, b1, W2, b2 = moe.experts[global_eid]
            out = ffn_forward(tok_vec[None], W1, b1, W2, b2)[0]
            recv_results[rank].append((out, gs, src_t))

    # ── Stage 4+5: Un-permutation + Scaling ──────────────────────────────
    routed_out = np.zeros_like(u)
    for rank in range(num_ranks):
        for (out, gs, src_t) in recv_results[rank]:
            routed_out[src_t] += gs * out

    return shared_out + routed_out + u


# ─────────────────────────────────────────────────────────────────────────────
# Test suite
# ─────────────────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if cond: PASS += 1
    else:    FAIL += 1


def test_softmax_router():
    print("\n=== Test Group 1: Softmax / TopK Router ===")

    T, D, E, K = 8, 16, 6, 2
    x      = np.random.randn(T, D)
    W_gate = np.random.randn(D, E) * 0.1

    gs, ids = top_k_router(x, W_gate, K)

    check("Output shapes correct",
          gs.shape == (T, K) and ids.shape == (T, K))
    check("Gate scores sum to 1 per token",
          np.allclose(gs.sum(-1), 1.0, atol=1e-6),
          f"max deviation={np.abs(gs.sum(-1)-1).max():.2e}")
    check("Gate scores non-negative", (gs >= 0).all())
    check("Expert IDs in valid range",
          ((ids >= 0) & (ids < E)).all())
    check("TopK IDs unique per token",
          all(len(set(ids[t])) == K for t in range(T)))
    check("K=1 is SwitchRouter",
          top_k_router(x, W_gate, 1)[0].shape == (T, 1))
    # SoftmaxRouter: K=E gives all-expert routing
    gs_all, _ = top_k_router(x, W_gate, E)
    check("K=E (SoftmaxRouter): scores sum to 1",
          np.allclose(gs_all.sum(-1), 1.0, atol=1e-6))


def test_ffn():
    print("\n=== Test Group 2: FFN Expert ===")

    T, D, H = 4, 8, 16
    W1 = np.random.randn(D, H) * 0.1
    b1 = np.zeros(H)
    W2 = np.random.randn(H, D) * 0.1
    b2 = np.zeros(D)
    x  = np.random.randn(T, D)

    y = ffn_forward(x, W1, b1, W2, b2)
    check("FFN output shape", y.shape == (T, D))
    check("FFN output finite", np.isfinite(y).all())

    # ReLU kills negatives in hidden layer
    hidden = relu(x @ W1 + b1)
    check("ReLU hidden >= 0", (hidden >= 0).all())

    # Zero input → output = b2 (since ReLU(b1) @ W2 + b2 when x=0)
    y0 = ffn_forward(np.zeros((1, D)), W1, b1, W2, b2)[0]
    expected = relu(b1) @ W2 + b2
    check("Zero input gives correct output",
          np.allclose(y0, expected, atol=1e-7))


def test_deepseek_moe_architecture():
    print("\n=== Test Group 3: DeepSeek MoE Architecture ===")

    d_model, d_ffn, N, m, K, K_s = 32, 64, 4, 2, 2, 1

    moe = DeepSeekMoENumpy(d_model, d_ffn, N, m, K, K_s)

    check(f"Total experts = mN = {m}*{N}",
          moe.E_total == m * N, str(moe.E_total))
    check(f"d_hidden = d_ffn/m = {d_ffn}/{m}",
          moe.d_hidden == d_ffn // m, str(moe.d_hidden))
    check(f"K_routed = mK - K_s = {m}*{K} - {K_s}",
          moe.K_routed == m * K - K_s, str(moe.K_routed))
    check(f"Num shared experts = K_s = {K_s}",
          len(moe.shared) == K_s)
    check(f"Num routed experts = E_total = {m*N}",
          len(moe.experts) == m * N)

    T  = 12
    u  = np.random.randn(T, d_model)
    y  = moe.forward(u)
    check("Forward output shape", y.shape == (T, d_model))
    check("Output finite",        np.isfinite(y).all())
    check("Output != input (transformation happened)",
          not np.allclose(y, u))


def test_routing_properties():
    print("\n=== Test Group 4: Routing Properties ===")

    d_model, d_ffn, N, m, K, K_s = 64, 128, 4, 2, 2, 1
    moe = DeepSeekMoENumpy(d_model, d_ffn, N, m, K, K_s)

    T = 16
    u = np.random.randn(T, d_model)

    gs, ids = top_k_router(u, moe.W_gate, moe.K_routed)

    check("Exactly K_routed experts per token",
          ids.shape[1] == moe.K_routed)
    check("Gate scores in (0, 1)",
          (gs > 0).all() and (gs <= 1 + 1e-7).all())
    check("No duplicate expert per token",
          all(len(set(ids[t])) == moe.K_routed for t in range(T)))

    # With fine-grained segmentation: K_routed > K_base  (more diverse routing)
    check("K_routed > K (fine-grained gives more diversity)",
          moe.K_routed > K,
          f"{moe.K_routed} > {K}")


def test_expert_parallelism():
    print("\n=== Test Group 5: Expert Parallelism ===")

    d_model, d_ffn, N, m, K, K_s = 32, 64, 4, 2, 2, 1
    moe = DeepSeekMoENumpy(d_model, d_ffn, N, m, K, K_s)

    T = 8
    u = np.random.randn(T, d_model)

    y_seq = moe.forward(u)
    y_ep2 = expert_parallel_forward(moe, u, num_ranks=2)

    diff = np.abs(y_seq - y_ep2).max()
    check("EP(2 ranks) matches sequential",
          diff < 1e-10, f"max diff={diff:.2e}")

    y_ep4 = expert_parallel_forward(moe, u, num_ranks=4)
    diff4 = np.abs(y_seq - y_ep4).max()
    check("EP(4 ranks) matches sequential",
          diff4 < 1e-10, f"max diff={diff4:.2e}")


def test_moe_vs_dense():
    print("\n=== Test Group 6: MoE Sparsity vs Dense ===")

    d_model, d_ffn, N, m, K, K_s = 32, 128, 8, 2, 2, 1
    moe = DeepSeekMoENumpy(d_model, d_ffn, N, m, K, K_s)

    # Count active FLOPs per token for MoE
    active = moe.K_routed + moe.K_s   # total experts evaluated
    total  = moe.E_total
    pct    = 100.0 * active / total
    check(f"Only {active}/{total} = {pct:.0f}% of experts active per token",
          active < total, f"K_routed={moe.K_routed}, K_s={moe.K_s}")

    # Parameter count ratio: MoE has total_experts * expert_size params
    moe_params   = (moe.E_total + moe.K_s) * (
                       d_model * moe.d_hidden + moe.d_hidden +
                       moe.d_hidden * d_model + d_model) \
                 + d_model * moe.E_total          # gate
    # Dense FFN with same intermediate dim
    dense_params = (d_model * d_ffn + d_ffn +
                    d_ffn * d_model + d_model)
    ratio = moe_params / dense_params
    check(f"MoE has more params than dense (ratio={ratio:.1f}x)",
          moe_params > dense_params)


def test_edge_cases():
    print("\n=== Test Group 7: Edge Cases ===")

    # Single token
    moe = DeepSeekMoENumpy(16, 32, 2, 2, 1, 1)
    y = moe.forward(np.random.randn(1, 16))
    check("Single token", y.shape == (1, 16) and np.isfinite(y).all())

    # Large batch
    y = moe.forward(np.random.randn(512, 16))
    check("Large batch T=512", y.shape == (512, 16) and np.isfinite(y).all())

    # K=1 (SwitchRouter)
    moe_k1 = DeepSeekMoENumpy(16, 32, 4, 1, 1, 1)
    check("K=1 SwitchRouter K_routed correct", moe_k1.K_routed == 0)
    # When K_routed=0, output = shared_out + u
    x = np.random.randn(4, 16)
    y = moe_k1.forward(x)
    check("K=1 output finite", np.isfinite(y).all())

    # All-zeros input
    moe2 = DeepSeekMoENumpy(16, 32, 2, 2, 1, 1)
    y0 = moe2.forward(np.zeros((4, 16)))
    check("Zero input: output finite", np.isfinite(y0).all())


# ─────────────────────────────────────────────────────────────────────────────
# Performance micro-benchmark (pure NumPy, shows sub-linear scaling)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark():
    print("\n=== Performance: MoE vs Dense (NumPy, shows scaling trend) ===\n")

    d, Hd, N, m, K, Ks = 128, 256, 8, 4, 2, 1
    moe = DeepSeekMoENumpy(d, Hd, N, m, K, Ks)

    # Dense FFN  (same intermediate dim)
    W1d = np.random.randn(d, Hd) * 0.1
    b1d = np.zeros(Hd)
    W2d = np.random.randn(Hd, d) * 0.1
    b2d = np.zeros(d)

    RUNS = 3
    print(f"{'T':>6}  {'Dense (ms)':>12}  {'MoE (ms)':>12}  "
          f"{'Active%':>8}  {'Params ratio':>13}")
    print("-" * 60)

    for T in [8, 32, 128, 512]:
        x = np.random.randn(T, d)

        t0 = time.perf_counter()
        for _ in range(RUNS): ffn_forward(x, W1d, b1d, W2d, b2d)
        dense_ms = (time.perf_counter() - t0) / RUNS * 1000

        t0 = time.perf_counter()
        for _ in range(RUNS): moe.forward(x)
        moe_ms = (time.perf_counter() - t0) / RUNS * 1000

        active_pct = 100.0 * (moe.K_routed + moe.K_s) / moe.E_total
        param_ratio = (moe.E_total * d * moe.d_hidden * 2) / (d * Hd * 2)
        print(f"{T:>6}  {dense_ms:>12.2f}  {moe_ms:>12.2f}  "
              f"{active_pct:>7.0f}%  {param_ratio:>12.1f}x")

    print()
    print(f"MoE total experts: {moe.E_total}, "
          f"active per token: {moe.K_routed + moe.K_s} "
          f"({100*(moe.K_routed+moe.K_s)/moe.E_total:.0f}%)")
    print("MoE has more parameters but the SAME compute per token as a")
    print("dense model using only the active expert fraction.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_softmax_router()
    test_ffn()
    test_deepseek_moe_architecture()
    test_routing_properties()
    test_expert_parallelism()
    test_moe_vs_dense()
    test_edge_cases()

    print(f"\n{'='*55}")
    print(f"  TOTAL: {PASS} PASSED  |  {FAIL} FAILED")
    print(f"{'='*55}")

    benchmark()

    import sys
    sys.exit(0 if FAIL == 0 else 1)
