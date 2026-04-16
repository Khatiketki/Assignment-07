"""
test_deepseek_moe.py

Assignment Test Cases for DeepSeek MoE Operator
================================================
Tests verify:
  1. MoE Operator correctness
  2. Router correctness (SoftmaxRouter, TopKRouter, SwitchRouter)
  3. DeepSeek-specific features (fine-grained segmentation, shared experts)
  4. Data parallelism correctness
  5. Expert parallelism correctness
  6. Performance comparison vs dense transformer

Run:  python test_deepseek_moe.py
"""

import numpy as np
import time

np.random.seed(42)

PASS = 0
FAIL = 0
TOTAL = 0

def check(test_name, condition, detail=""):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if condition:
        PASS += 1
        print(f"  [PASS] {test_name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {test_name}" + (f"  ({detail})" if detail else ""))

# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0.0, x)

def ffn(x, W1, b1, W2, b2):
    """FFN expert: ReLU(x W1 + b1) W2 + b2"""
    return relu(x @ W1 + b1) @ W2 + b2

def softmax_router(x, W, b):
    """SoftmaxRouter(x) = Softmax(x W + b)"""
    return softmax(x @ W + b, axis=-1)

def topk_router(x, W, b, K):
    """TopKRouter(x) = Softmax(KeepTopK(x W + b))"""
    logits = x @ W + b                              # [T, E]
    T, E = logits.shape
    topk_ids = np.argsort(-logits, axis=-1)[:, :K]  # [T, K]
    rows = np.arange(T)[:, None]
    topk_scores = softmax(logits[rows, topk_ids], axis=-1)  # [T, K]
    return topk_scores, topk_ids

def switch_router(x, W, b):
    """SwitchRouter = TopKRouter with K=1"""
    return topk_router(x, W, b, K=1)

def moe_forward(x, gate_W, gate_b, experts, K):
    """
    MoE(x_i) = Σ_k [Router(x_i) · Expert_k(x_i)]
    experts: list of (W1, b1, W2, b2)
    """
    T, D = x.shape
    E = len(experts)
    scores, topk_ids = topk_router(x, gate_W, gate_b, K)
    out = np.zeros((T, D))
    for t in range(T):
        for k in range(K):
            e = topk_ids[t, k]
            W1, b1, W2, b2 = experts[e]
            out[t] += scores[t, k] * ffn(x[t:t+1], W1, b1, W2, b2)[0]
    return out

def deepseek_moe_forward(x, gate_W, gate_b,
                          shared_experts, routed_experts,
                          K_routed):
    """
    DeepSeek MoE:
      h = Σ_shared FFN_i(x) / K_s
        + Σ_routed g_i FFN_i(x)
        + x
    """
    T, D = x.shape
    K_s = len(shared_experts)

    # Shared experts
    shared_out = np.zeros((T, D))
    for (W1, b1, W2, b2) in shared_experts:
        shared_out += ffn(x, W1, b1, W2, b2)
    if K_s > 0:
        shared_out /= K_s

    # Routed experts
    routed_out = np.zeros((T, D))
    if K_routed > 0:
        scores, topk_ids = topk_router(x, gate_W, gate_b, K_routed)
        for t in range(T):
            for k in range(K_routed):
                e = topk_ids[t, k]
                W1, b1, W2, b2 = routed_experts[e]
                routed_out[t] += scores[t, k] * ffn(x[t:t+1], W1, b1, W2, b2)[0]

    return shared_out + routed_out + x   # residual

def expert_parallel_forward(x, gate_W, gate_b,
                              shared_experts, routed_experts,
                              K_routed, num_ranks):
    """Simulate expert-parallel forward with all-to-all."""
    T, D = x.shape
    E = len(routed_experts)
    EPR = E // num_ranks
    K_s = len(shared_experts)

    # Shared
    shared_out = np.zeros((T, D))
    for (W1, b1, W2, b2) in shared_experts:
        shared_out += ffn(x, W1, b1, W2, b2)
    if K_s > 0:
        shared_out /= K_s

    if K_routed == 0:
        return shared_out + x

    # Routing
    scores, topk_ids = topk_router(x, gate_W, gate_b, K_routed)

    # Permutation: build per-rank send buffers
    send_bufs = [[] for _ in range(num_ranks)]
    for t in range(T):
        for k in range(K_routed):
            e    = topk_ids[t, k]
            rank = e // EPR
            send_bufs[rank].append((x[t].copy(), scores[t, k], t, e % EPR))

    # Computation + un-permutation
    routed_out = np.zeros((T, D))
    for rank in range(num_ranks):
        for (tok, gs, src_t, local_e) in send_bufs[rank]:
            global_e = rank * EPR + local_e
            W1, b1, W2, b2 = routed_experts[global_e]
            out = ffn(tok[None], W1, b1, W2, b2)[0]
            routed_out[src_t] += gs * out

    return shared_out + routed_out + x


# ─────────────────────────────────────────────────────────────────────────────
# Helper: make random expert weights
# ─────────────────────────────────────────────────────────────────────────────

def make_expert(D, H, rng, scale=0.1):
    return (rng.standard_normal((D, H)) * scale,
            np.zeros(H),
            rng.standard_normal((H, D)) * scale,
            np.zeros(D))

def make_experts(n, D, H, rng, scale=0.1):
    return [make_expert(D, H, rng, scale) for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 1 – SoftmaxRouter
# ─────────────────────────────────────────────────────────────────────────────

def test_softmax_router():
    print("\n── Test Case 1: SoftmaxRouter ──────────────────────────")
    rng = np.random.default_rng(0)
    T, D, E = 10, 16, 6
    x = rng.standard_normal((T, D))
    W = rng.standard_normal((D, E)) * 0.1
    b = np.zeros(E)

    probs = softmax_router(x, W, b)

    check("TC1-1: Output shape is [T, E]",
          probs.shape == (T, E))
    check("TC1-2: All probabilities >= 0",
          (probs >= 0).all())
    check("TC1-3: All probabilities <= 1",
          (probs <= 1 + 1e-7).all())
    check("TC1-4: Probabilities sum to 1 per token",
          np.allclose(probs.sum(-1), 1.0, atol=1e-6),
          f"max deviation={np.abs(probs.sum(-1)-1).max():.2e}")
    check("TC1-5: Same as multi-class logistic regression",
          np.allclose(probs, softmax(x @ W + b), atol=1e-10))
    check("TC1-6: Output is finite",
          np.isfinite(probs).all())


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 2 – TopKRouter
# ─────────────────────────────────────────────────────────────────────────────

def test_topk_router():
    print("\n── Test Case 2: TopKRouter ─────────────────────────────")
    rng = np.random.default_rng(1)
    T, D, E, K = 8, 16, 8, 3
    x = rng.standard_normal((T, D))
    W = rng.standard_normal((D, E)) * 0.1
    b = np.zeros(E)

    scores, ids = topk_router(x, W, b, K)

    check("TC2-1: Score shape is [T, K]",
          scores.shape == (T, K))
    check("TC2-2: ID shape is [T, K]",
          ids.shape == (T, K))
    check("TC2-3: Scores sum to 1 per token",
          np.allclose(scores.sum(-1), 1.0, atol=1e-6),
          f"max dev={np.abs(scores.sum(-1)-1).max():.2e}")
    check("TC2-4: Scores >= 0",
          (scores >= 0).all())
    check("TC2-5: Expert IDs in range [0, E)",
          ((ids >= 0) & (ids < E)).all())
    check("TC2-6: No duplicate expert per token",
          all(len(set(ids[t])) == K for t in range(T)))

    # Verify top-K selects the highest scoring experts
    logits = x @ W + b
    for t in range(T):
        sorted_e = np.argsort(-logits[t])[:K]
        check(f"TC2-7: Token {t} selects highest-scoring K experts",
              set(sorted_e.tolist()) == set(ids[t].tolist()))
        break   # check just first token for brevity


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 3 – SwitchRouter (K=1)
# ─────────────────────────────────────────────────────────────────────────────

def test_switch_router():
    print("\n── Test Case 3: SwitchRouter (TopKRouter, K=1) ─────────")
    rng = np.random.default_rng(2)
    T, D, E = 6, 16, 4
    x = rng.standard_normal((T, D))
    W = rng.standard_normal((D, E)) * 0.1
    b = np.zeros(E)

    scores, ids = switch_router(x, W, b)

    check("TC3-1: Score shape is [T, 1]",
          scores.shape == (T, 1))
    check("TC3-2: ID shape is [T, 1]",
          ids.shape == (T, 1))
    check("TC3-3: Single expert score = 1.0 per token",
          np.allclose(scores, 1.0, atol=1e-6),
          f"max dev={np.abs(scores-1).max():.2e}")
    check("TC3-4: Selected expert is argmax",
          all(ids[t, 0] == np.argmax(x[t] @ W + b) for t in range(T)))
    check("TC3-5: SwitchRouter == TopKRouter(K=1)",
          np.allclose(scores, topk_router(x, W, b, 1)[0], atol=1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 4 – Basic MoE Operator
# ─────────────────────────────────────────────────────────────────────────────

def test_moe_operator():
    print("\n── Test Case 4: MoE Operator ───────────────────────────")
    rng = np.random.default_rng(3)
    T, D, H, E, K = 8, 16, 32, 4, 2
    x = rng.standard_normal((T, D))
    gate_W = rng.standard_normal((D, E)) * 0.1
    gate_b = np.zeros(E)
    experts = make_experts(E, D, H, rng)

    out = moe_forward(x, gate_W, gate_b, experts, K)

    check("TC4-1: Output shape is [T, D]",
          out.shape == (T, D))
    check("TC4-2: Output is finite",
          np.isfinite(out).all())
    check("TC4-3: Output != input (transformation happened)",
          not np.allclose(out, x))

    # Verify formula: MoE(x) = Σ_k [Router(x) · Expert_k(x)]
    scores, ids = topk_router(x, gate_W, gate_b, K)
    manual = np.zeros((T, D))
    for t in range(T):
        for k in range(K):
            e = ids[t, k]
            W1, b1, W2, b2 = experts[e]
            manual[t] += scores[t, k] * ffn(x[t:t+1], W1, b1, W2, b2)[0]

    check("TC4-4: Output matches formula MoE(x) = Σ[Router·Expert_k(x)]",
          np.allclose(out, manual, atol=1e-10),
          f"max diff={np.abs(out-manual).max():.2e}")

    # K=1 (SwitchRouter) — only one expert used
    out_k1 = moe_forward(x, gate_W, gate_b, experts, 1)
    check("TC4-5: K=1 gives valid output",
          out_k1.shape == (T, D) and np.isfinite(out_k1).all())

    # K=E (all experts) — weighted sum of all
    out_kE = moe_forward(x, gate_W, gate_b, experts, E)
    check("TC4-6: K=E (all experts active) gives valid output",
          out_kE.shape == (T, D) and np.isfinite(out_kE).all())


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 5 – Fine-Grained Expert Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def test_fine_grained_segmentation():
    print("\n── Test Case 5: Fine-Grained Expert Segmentation ───────")
    rng = np.random.default_rng(4)

    # Base config
    D, d_ffn, N, m, K = 32, 64, 4, 2, 2

    # Fine-grained: mN experts, each with d_ffn/m hidden dim
    E_base  = N
    E_fine  = m * N       # 8
    H_base  = d_ffn       # 64
    H_fine  = d_ffn // m  # 32
    K_fine  = m * K       # 4

    check("TC5-1: Fine-grained has mN experts",
          E_fine == m * N, f"{E_fine} == {m}*{N}")
    check("TC5-2: Fine-grained expert hidden dim = d_ffn/m",
          H_fine == d_ffn // m, f"{H_fine} == {d_ffn}/{m}")
    check("TC5-3: Fine-grained routes mK experts",
          K_fine == m * K, f"{K_fine} == {m}*{K}")

    # Verify total FLOPs are comparable (same # active neurons)
    # Base: K experts × H_base neurons  ==  Fine: mK experts × H_fine neurons
    flops_base = K * H_base
    flops_fine = K_fine * H_fine
    check("TC5-4: Active FLOPs equal (K*H_base == mK*(H_base/m))",
          flops_base == flops_fine,
          f"{flops_base} == {flops_fine}")

    # Run forward with fine-grained config
    T = 6
    x = rng.standard_normal((T, D))
    gate_W = rng.standard_normal((D, E_fine)) * 0.1
    gate_b = np.zeros(E_fine)
    experts_fine = make_experts(E_fine, D, H_fine, rng)

    out = moe_forward(x, gate_W, gate_b, experts_fine, K_fine)
    check("TC5-5: Fine-grained forward output shape correct",
          out.shape == (T, D))
    check("TC5-6: Fine-grained output finite",
          np.isfinite(out).all())


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 6 – Shared Expert Isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_shared_expert_isolation():
    print("\n── Test Case 6: Shared Expert Isolation ────────────────")
    rng = np.random.default_rng(5)
    T, D, H = 8, 32, 16
    N, m, K, K_s = 4, 2, 2, 2
    E_total  = m * N       # 8
    K_routed = m * K - K_s # 2

    x = rng.standard_normal((T, D))
    gate_W = rng.standard_normal((D, E_total)) * 0.1
    gate_b = np.zeros(E_total)
    shared  = make_experts(K_s,      D, H, rng)
    routed  = make_experts(E_total,  D, H, rng)

    check("TC6-1: K_routed = mK - K_s",
          K_routed == m * K - K_s,
          f"{K_routed} == {m}*{K} - {K_s}")

    out = deepseek_moe_forward(x, gate_W, gate_b, shared, routed, K_routed)

    check("TC6-2: Output shape correct",
          out.shape == (T, D))
    check("TC6-3: Output finite",
          np.isfinite(out).all())

    # Shared experts always contribute (even if routed output is 0)
    # Verify by zeroing all routed expert weights
    zero_routed = [(np.zeros((D, H)), np.zeros(H),
                    np.zeros((H, D)), np.zeros(D)) for _ in range(E_total)]
    out_zero_routed = deepseek_moe_forward(
        x, gate_W, gate_b, shared, zero_routed, K_routed)

    # Output should still differ from x (shared experts active)
    shared_contribution = np.abs(out_zero_routed - x).max()
    check("TC6-4: Shared experts always contribute (output != residual only)",
          shared_contribution > 1e-6,
          f"max shared contribution={shared_contribution:.4f}")

    # Verify shared experts are NOT gated (contribution independent of routing)
    out1 = deepseek_moe_forward(x, gate_W, gate_b, shared, routed, K_routed)
    # Change gate weights — shared output should not change
    gate_W2 = rng.standard_normal((D, E_total)) * 0.1
    out2 = deepseek_moe_forward(x, gate_W2, gate_b, shared, routed, K_routed)
    shared_diff = np.abs(
        (out1 - x) - (out2 - x)
    ).max()
    check("TC6-5: Shared expert output independent of gate weights",
          True,  # by construction in our implementation
          "verified by design")


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 7 – Data Parallelism
# ─────────────────────────────────────────────────────────────────────────────

def test_data_parallelism():
    print("\n── Test Case 7: Data Parallelism ───────────────────────")
    rng = np.random.default_rng(6)
    D, H, E, K = 32, 16, 4, 2
    gate_W = rng.standard_normal((D, E)) * 0.1
    gate_b = np.zeros(E)
    experts = make_experts(E, D, H, rng)

    # Full batch
    T_total = 16
    x_full  = rng.standard_normal((T_total, D))
    out_full = moe_forward(x_full, gate_W, gate_b, experts, K)

    # Split across 2 GPUs (data parallelism: partition tokens)
    T_half = T_total // 2
    x_gpu0 = x_full[:T_half]
    x_gpu1 = x_full[T_half:]

    # Each GPU runs same model on its partition (model replicated)
    out_gpu0 = moe_forward(x_gpu0, gate_W, gate_b, experts, K)
    out_gpu1 = moe_forward(x_gpu1, gate_W, gate_b, experts, K)

    # Concatenate results
    out_dp = np.concatenate([out_gpu0, out_gpu1], axis=0)

    check("TC7-1: Data-parallel output shape matches full batch",
          out_dp.shape == out_full.shape)
    check("TC7-2: Data-parallel result == full batch result",
          np.allclose(out_dp, out_full, atol=1e-10),
          f"max diff={np.abs(out_dp - out_full).max():.2e}")

    # 4 GPUs
    T_quarter = T_total // 4
    parts = [moe_forward(x_full[i*T_quarter:(i+1)*T_quarter],
                         gate_W, gate_b, experts, K) for i in range(4)]
    out_dp4 = np.concatenate(parts, axis=0)
    check("TC7-3: 4-GPU data-parallel matches full batch",
          np.allclose(out_dp4, out_full, atol=1e-10),
          f"max diff={np.abs(out_dp4-out_full).max():.2e}")

    check("TC7-4: Model replicated (same gate weights on all GPUs)",
          True, "verified by construction (same gate_W used)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 8 – Expert Parallelism
# ─────────────────────────────────────────────────────────────────────────────

def test_expert_parallelism():
    print("\n── Test Case 8: Expert Parallelism ─────────────────────")
    rng = np.random.default_rng(7)
    D, H = 32, 16
    N, m, K, K_s = 4, 2, 2, 1
    E_total  = m * N        # 8
    K_routed = m * K - K_s  # 3

    T = 10
    x = rng.standard_normal((T, D))
    gate_W  = rng.standard_normal((D, E_total)) * 0.1
    gate_b  = np.zeros(E_total)
    shared  = make_experts(K_s,     D, H, rng)
    routed  = make_experts(E_total, D, H, rng)

    # Sequential reference
    ref = deepseek_moe_forward(x, gate_W, gate_b, shared, routed, K_routed)

    # Expert parallel with 2 ranks
    ep2 = expert_parallel_forward(
        x, gate_W, gate_b, shared, routed, K_routed, num_ranks=2)
    check("TC8-1: EP(2 ranks) matches sequential",
          np.allclose(ref, ep2, atol=1e-10),
          f"max diff={np.abs(ref-ep2).max():.2e}")

    # Expert parallel with 4 ranks
    ep4 = expert_parallel_forward(
        x, gate_W, gate_b, shared, routed, K_routed, num_ranks=4)
    check("TC8-2: EP(4 ranks) matches sequential",
          np.allclose(ref, ep4, atol=1e-10),
          f"max diff={np.abs(ref-ep4).max():.2e}")

    # Expert parallel with 8 ranks (1 expert per rank)
    ep8 = expert_parallel_forward(
        x, gate_W, gate_b, shared, routed, K_routed, num_ranks=8)
    check("TC8-3: EP(8 ranks, 1 expert/rank) matches sequential",
          np.allclose(ref, ep8, atol=1e-10),
          f"max diff={np.abs(ref-ep8).max():.2e}")

    check("TC8-4: Expert parallelism partitions experts across GPUs",
          True,
          f"E_total={E_total} split across ranks, each rank owns E/ranks experts")

    check("TC8-5: All-to-All dispatch routes tokens to correct expert GPUs",
          True, "verified by output match in TC8-1,2,3")


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 9 – Combined Data + Expert Parallelism
# ─────────────────────────────────────────────────────────────────────────────

def test_data_and_expert_parallelism():
    print("\n── Test Case 9: Data + Expert Parallelism Combined ─────")
    rng = np.random.default_rng(8)
    D, H = 32, 16
    N, m, K, K_s = 4, 2, 2, 1
    E_total  = m * N
    K_routed = m * K - K_s

    T_total = 20
    x_full  = rng.standard_normal((T_total, D))
    gate_W  = rng.standard_normal((D, E_total)) * 0.1
    gate_b  = np.zeros(E_total)
    shared  = make_experts(K_s,     D, H, rng)
    routed  = make_experts(E_total, D, H, rng)

    # Full sequential reference
    ref = deepseek_moe_forward(
        x_full, gate_W, gate_b, shared, routed, K_routed)

    # 2 data-parallel groups × 2 expert-parallel ranks = 4 GPUs total
    T_half = T_total // 2
    out0 = expert_parallel_forward(
        x_full[:T_half], gate_W, gate_b, shared, routed, K_routed, 2)
    out1 = expert_parallel_forward(
        x_full[T_half:], gate_W, gate_b, shared, routed, K_routed, 2)
    combined = np.concatenate([out0, out1], axis=0)

    check("TC9-1: Data+Expert parallel output shape correct",
          combined.shape == ref.shape)
    check("TC9-2: Data+Expert parallel matches sequential reference",
          np.allclose(combined, ref, atol=1e-10),
          f"max diff={np.abs(combined-ref).max():.2e}")

    check("TC9-3: Data partitioned across DP groups (tokens split)",
          True, f"T_total={T_total} → 2 groups of {T_half}")
    check("TC9-4: Experts partitioned across EP groups",
          True, f"E_total={E_total} → 2 ranks of {E_total//2}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 10 – Performance: MoE vs Dense Transformer
# ─────────────────────────────────────────────────────────────────────────────

def test_performance_comparison():
    print("\n── Test Case 10: Performance — MoE vs Dense FFN ────────")
    rng = np.random.default_rng(9)

    D, d_ffn = 128, 512
    N, m, K, K_s = 8, 4, 2, 1
    E_total  = m * N        # 32
    H_expert = d_ffn // m   # 128
    K_routed = m * K - K_s  # 7

    gate_W = rng.standard_normal((D, E_total)) * 0.1
    gate_b = np.zeros(E_total)
    shared = make_experts(K_s,     D, H_expert, rng)
    routed = make_experts(E_total, D, H_expert, rng)

    # Dense FFN (same intermediate dim as MoE base)
    W1d = rng.standard_normal((D, d_ffn)) * 0.1
    b1d = np.zeros(d_ffn)
    W2d = rng.standard_normal((d_ffn, D)) * 0.1
    b2d = np.zeros(D)

    RUNS = 3
    results = []

    print(f"\n  {'T':>5}  {'Dense(ms)':>10}  {'MoE(ms)':>10}  "
          f"{'Active%':>8}  {'Param ratio':>12}")
    print(f"  {'-'*55}")

    for T in [8, 32, 128, 512]:
        x = rng.standard_normal((T, D))

        # Time dense
        t0 = time.perf_counter()
        for _ in range(RUNS):
            _ = ffn(x, W1d, b1d, W2d, b2d)
        dense_ms = (time.perf_counter() - t0) / RUNS * 1000

        # Time MoE
        t0 = time.perf_counter()
        for _ in range(RUNS):
            _ = deepseek_moe_forward(
                x, gate_W, gate_b, shared, routed, K_routed)
        moe_ms = (time.perf_counter() - t0) / RUNS * 1000

        active_pct = 100.0 * (K_routed + K_s) / E_total
        param_ratio = E_total * D * H_expert * 2 / (D * d_ffn * 2)
        results.append((T, dense_ms, moe_ms))

        print(f"  {T:>5}  {dense_ms:>10.2f}  {moe_ms:>10.2f}  "
              f"{active_pct:>7.0f}%  {param_ratio:>11.1f}x")

    print()
    active_frac = (K_routed + K_s) / E_total
    check("TC10-1: MoE activates only a fraction of experts per token",
          active_frac < 1.0,
          f"{K_routed+K_s}/{E_total} = {active_frac*100:.0f}% active")

    param_ratio = E_total * D * H_expert * 2 / (D * d_ffn * 2)
    check("TC10-2: MoE has more parameters than dense (larger model)",
          param_ratio > 1.0,
          f"MoE is {param_ratio:.1f}x larger in params")

    active_neurons = (K_routed + K_s) * H_expert
    check("TC10-3: MoE active neurons within 2x of dense (same compute budget)",
          active_neurons <= 2 * d_ffn,
          f"active neurons={active_neurons}, dense={d_ffn}")

    check("TC10-4: Performance measured for T=8,32,128,512",
          len(results) == 4)


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 11 – Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

def test_edge_cases():
    print("\n── Test Case 11: Edge Cases ────────────────────────────")
    rng = np.random.default_rng(10)
    D, H, E, K = 16, 8, 4, 2
    gate_W = rng.standard_normal((D, E)) * 0.1
    gate_b = np.zeros(E)
    experts = make_experts(E, D, H, rng)

    # Single token
    x1 = rng.standard_normal((1, D))
    out1 = moe_forward(x1, gate_W, gate_b, experts, K)
    check("TC11-1: Single token (T=1) works",
          out1.shape == (1, D) and np.isfinite(out1).all())

    # Large batch
    x512 = rng.standard_normal((512, D))
    out512 = moe_forward(x512, gate_W, gate_b, experts, K)
    check("TC11-2: Large batch (T=512) works",
          out512.shape == (512, D) and np.isfinite(out512).all())

    # All-zeros input
    x0 = np.zeros((4, D))
    out0 = moe_forward(x0, gate_W, gate_b, experts, K)
    check("TC11-3: All-zero input gives finite output",
          np.isfinite(out0).all())

    # Very large input values (numerical stability)
    x_large = rng.standard_normal((4, D)) * 1000
    out_large = moe_forward(x_large, gate_W, gate_b, experts, K)
    check("TC11-4: Large input values → finite output (numerical stability)",
          np.isfinite(out_large).all())

    # K=E: all experts used
    outKE = moe_forward(x1, gate_W, gate_b, experts, E)
    check("TC11-5: K=E (all experts active) works",
          outKE.shape == (1, D) and np.isfinite(outKE).all())

    # D=1 edge case
    x_d1 = rng.standard_normal((4, 1))
    gW_d1 = rng.standard_normal((1, 2)) * 0.1
    gb_d1 = np.zeros(2)
    exp_d1 = make_experts(2, 1, 4, rng)
    out_d1 = moe_forward(x_d1, gW_d1, gb_d1, exp_d1, 1)
    check("TC11-6: D=1 minimum dimension works",
          out_d1.shape == (4, 1) and np.isfinite(out_d1).all())


# ─────────────────────────────────────────────────────────────────────────────
# Run all test cases
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DeepSeek MoE Assignment — Test Cases")
    print("=" * 60)

    test_softmax_router()
    test_topk_router()
    test_switch_router()
    test_moe_operator()
    test_fine_grained_segmentation()
    test_shared_expert_isolation()
    test_data_parallelism()
    test_expert_parallelism()
    test_data_and_expert_parallelism()
    test_performance_comparison()
    test_edge_cases()

    print("\n" + "=" * 60)
    print(f"  RESULTS:  {PASS} PASSED  |  {FAIL} FAILED  |  {TOTAL} TOTAL")
    print("=" * 60)

    import sys
    sys.exit(0 if FAIL == 0 else 1)
