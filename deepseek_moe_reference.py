"""
deepseek_moe_reference.py

Pure PyTorch reference implementation of the DeepSeek MoE operator.
Serves two purposes:
  1.  Ground-truth for validating the CUDA implementation.
  2.  Self-contained demo that runs on CPU or single GPU.

Architecture (from Week 8 lecture notes):
  - Fine-grained expert segmentation:
      mN experts, each with d_ffn/m intermediate dim
  - Shared expert isolation:
      K_s experts always activated
  - TopKRouter with K_routed = mK - K_s routed experts
  - Data + Expert parallelism simulated via tensor operations
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Routing helpers
# ─────────────────────────────────────────────────────────────────────────────

def top_k_router(x: torch.Tensor,
                 gate_weight: torch.Tensor,
                 K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TopKRouter: Softmax(KeepTopK(x W_gate))

    Args:
        x:            [T, d_model]
        gate_weight:  [d_model, E_total]   (learnable)
        K:            number of experts to route to

    Returns:
        topk_scores:  [T, K]   (softmax-normalised, sum=1 per token)
        topk_ids:     [T, K]   (expert indices)
    """
    # [T, E_total]  raw logits
    logits = x @ gate_weight          # (T, d) × (d, E) → (T, E)

    # KeepTopK: zero out (set to -inf) all but top-K
    topk_vals, topk_ids = torch.topk(logits, K, dim=-1)   # [T, K]

    # Softmax only over the top-K scores
    topk_scores = F.softmax(topk_vals, dim=-1)             # [T, K]

    return topk_scores, topk_ids


# ─────────────────────────────────────────────────────────────────────────────
# Single FFN expert
# ─────────────────────────────────────────────────────────────────────────────

class ExpertFFN(nn.Module):
    """One expert: two-layer FFN with ReLU.  d_hidden = d_ffn / m."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek MoE operator
# ─────────────────────────────────────────────────────────────────────────────

class DeepSeekMoE(nn.Module):
    """
    DeepSeek MoE layer with:
      - Fine-grained expert segmentation (m × N total experts)
      - Shared expert isolation (K_s always-on experts)
      - TopK routing for mK − K_s routed experts

    Formula (lecture notes):
        h_t = Σ_{i=1}^{K_s} FFN_i(u_t)
            + Σ_{i=K_s+1}^{mN} g_{i,t} FFN_i(u_t)
            + u_t

    where g_{i,t} = s_{i,t} if s_{i,t} ∈ TopK({s_{j,t}}, mK−K_s) else 0
    and   s_{i,t} = Softmax_i(u_t^T e_i)
    """

    def __init__(self,
                 d_model: int,
                 d_ffn:   int,
                 N:       int,
                 m:       int,
                 K:       int,
                 K_s:     int):
        super().__init__()
        self.d_model  = d_model
        self.d_ffn    = d_ffn
        self.N        = N
        self.m        = m
        self.K        = K
        self.K_s      = K_s

        E_total  = m * N
        d_hidden = d_ffn // m
        K_routed = m * K - K_s

        self.E_total  = E_total
        self.d_hidden = d_hidden
        self.K_routed = K_routed

        # Shared experts (always on, no gating)
        self.shared_experts = nn.ModuleList(
            [ExpertFFN(d_model, d_hidden) for _ in range(K_s)])

        # Routed experts
        self.routed_experts = nn.ModuleList(
            [ExpertFFN(d_model, d_hidden) for _ in range(E_total)])

        # Gate / router weight  [d_model, E_total]
        self.gate = nn.Linear(d_model, E_total, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u:  [T, d_model]   (token hidden states after self-attention)
        returns: [T, d_model]
        """
        T = u.shape[0]

        # ── Stage 1: Shared experts (always on) ────────────────────────────
        shared_out = torch.zeros_like(u)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(u)
        # Average shared expert contributions
        if self.K_s > 0:
            shared_out = shared_out / self.K_s

        # ── Stage 2: Routing ────────────────────────────────────────────────
        gate_scores, topk_ids = top_k_router(u, self.gate.weight.T, self.K_routed)
        # gate_scores: [T, K_routed],  topk_ids: [T, K_routed]

        # ── Stage 3: Routed expert computation ──────────────────────────────
        routed_out = torch.zeros_like(u)

        for k in range(self.K_routed):
            # Which expert does each token use at position k?
            eids = topk_ids[:, k]       # [T]
            gs   = gate_scores[:, k]    # [T]

            # Group tokens by expert (permutation)
            for e in range(self.E_total):
                mask = (eids == e)      # [T] bool
                if not mask.any():
                    continue
                tokens_e = u[mask]                           # [n_e, D]
                expert_out = self.routed_experts[e](tokens_e)  # [n_e, D]
                routed_out[mask] += gs[mask].unsqueeze(-1) * expert_out

        # ── Stage 4: Residual (skip connection modelled as add u) ──────────
        return shared_out + routed_out + u


# ─────────────────────────────────────────────────────────────────────────────
# Data + Expert Parallelism simulation
# ─────────────────────────────────────────────────────────────────────────────

class ExpertParallelMoE(nn.Module):
    """
    Simulates expert-parallelism by partitioning experts across 'num_ep_ranks'
    virtual "devices" (here just Python objects; real version uses NCCL all-to-all).

    Demonstrates the 5-stage forward pass described in the lecture:
      Routing → Permutation → Computation → Un-permutation → Scaling
    """

    def __init__(self, moe: DeepSeekMoE, num_ep_ranks: int):
        super().__init__()
        self.moe          = moe
        self.num_ep_ranks = num_ep_ranks
        assert moe.E_total % num_ep_ranks == 0, \
            "E_total must be divisible by num_ep_ranks"
        self.experts_per_rank = moe.E_total // num_ep_ranks

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        T  = u.shape[0]
        D  = u.shape[1]
        Kr = self.moe.K_routed
        E  = self.moe.E_total
        EPR = self.experts_per_rank
        NG  = self.num_ep_ranks

        # ── Shared experts ──────────────────────────────────────────────────
        shared_out = torch.zeros_like(u)
        for exp in self.moe.shared_experts:
            shared_out += exp(u)
        if self.moe.K_s > 0:
            shared_out /= self.moe.K_s

        # ── Stage 1: Routing ────────────────────────────────────────────────
        gate_scores, topk_ids = top_k_router(u, self.moe.gate.weight.T, Kr)

        # ── Stage 2: Permutation ─────────────────────────────────────────────
        # Build send buffers: send_buf[rank] = list of (token, gate_score) tuples
        # for experts on that rank.
        send_bufs   = [[] for _ in range(NG)]   # list of (token_vec, score, src_t, k_pos)
        for t in range(T):
            for k in range(Kr):
                e    = topk_ids[t, k].item()
                rank = e // EPR
                send_bufs[rank].append((u[t], gate_scores[t, k], t, e % EPR))

        # All-to-All dispatch (simulated: just use the local Python lists)
        # In CUDA+NCCL: ncclAllToAll / grouped ncclSend+Recv

        # ── Stage 3: Computation ─────────────────────────────────────────────
        recv_bufs = [[] for _ in range(NG)]   # outputs indexed by original token

        for rank in range(NG):
            for (tok_vec, gs, src_t, local_eid) in send_bufs[rank]:
                expert = self.moe.routed_experts[rank * EPR + local_eid]
                out    = expert(tok_vec.unsqueeze(0)).squeeze(0)
                recv_bufs[rank].append((out, gs, src_t))

        # ── Stage 4: Un-permutation + Stage 5: Scaling ──────────────────────
        routed_out = torch.zeros(T, D, device=u.device, dtype=u.dtype)
        for rank in range(NG):
            for (out, gs, src_t) in recv_bufs[rank]:
                routed_out[src_t] += gs * out

        return shared_out + routed_out + u


# ─────────────────────────────────────────────────────────────────────────────
# Test suite
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on: {device}\n")

    torch.manual_seed(42)

    # Config matching CUDA implementation
    d_model = 64
    d_ffn   = 128
    N       = 4
    m       = 2
    K       = 2
    K_s     = 1
    T       = 16     # total tokens (data-parallel: 8 per GPU)
    num_ep  = 2      # expert-parallel ranks

    moe = DeepSeekMoE(d_model, d_ffn, N, m, K, K_s).to(device)
    ep_moe = ExpertParallelMoE(moe, num_ep)

    x = torch.randn(T, d_model, device=device)

    pass_count = fail_count = 0

    def check(name, cond):
        nonlocal pass_count, fail_count
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}")
        if cond: pass_count += 1
        else:    fail_count += 1

    # ── Test 1: Output shape ─────────────────────────────────────────────────
    print("=== Test Group 1: Output Shape ===")
    with torch.no_grad():
        y = moe(x)
    check("Output shape matches input shape", y.shape == x.shape)
    check("Output is finite", torch.isfinite(y).all().item())

    # ── Test 2: Gate probabilities ───────────────────────────────────────────
    print("\n=== Test Group 2: Gate Probabilities ===")
    gate_scores, topk_ids = top_k_router(x, moe.gate.weight.T, moe.K_routed)
    check("Gate scores sum to 1 per token",
          torch.allclose(gate_scores.sum(-1), torch.ones(T, device=device), atol=1e-5))
    check("Gate scores non-negative", (gate_scores >= 0).all().item())
    check("Expert IDs in valid range [0, E_total)",
          ((topk_ids >= 0) & (topk_ids < moe.E_total)).all().item())
    check("TopK IDs are unique per token",
          all(len(set(topk_ids[t].tolist())) == moe.K_routed for t in range(T)))

    # ── Test 3: Expert parallelism consistency ───────────────────────────────
    print("\n=== Test Group 3: Expert Parallelism Consistency ===")
    with torch.no_grad():
        y_seq = moe(x)
        y_ep  = ep_moe(x)
    diff = (y_seq - y_ep).abs().max().item()
    check(f"EP output matches sequential (max diff={diff:.2e})", diff < 1e-4)

    # ── Test 4: Shared vs routed experts ────────────────────────────────────
    print("\n=== Test Group 4: DeepSeek-Specific Features ===")
    check(f"Correct number of total experts ({moe.E_total})",
          moe.E_total == m * N)
    check(f"Correct K_routed ({moe.K_routed} = m*K - K_s = {m}*{K} - {K_s})",
          moe.K_routed == m * K - K_s)
    check(f"Expert hidden dim ({moe.d_hidden}) = d_ffn/m ({d_ffn//m})",
          moe.d_hidden == d_ffn // m)

    # ── Test 5: Residual connection ──────────────────────────────────────────
    print("\n=== Test Group 5: Residual / Gradient Flow ===")
    x_req = x.clone().requires_grad_(True)
    y_grad = moe(x_req)
    loss = y_grad.sum()
    loss.backward()
    check("Gradients flow back through MoE", x_req.grad is not None)
    check("Gradients are finite",
          x_req.grad is not None and torch.isfinite(x_req.grad).all().item())

    # ── Test 6: Batch size invariance ────────────────────────────────────────
    print("\n=== Test Group 6: Batch Size Invariance ===")
    for Tb in [1, 4, 32]:
        xb = torch.randn(Tb, d_model, device=device)
        with torch.no_grad():
            yb = moe(xb)
        check(f"T={Tb}: output shape [{Tb}, {d_model}]", yb.shape == (Tb, d_model))

    print(f"\n{'='*50}")
    print(f"Results: {pass_count} PASSED  |  {fail_count} FAILED")
    print(f"{'='*50}\n")
    return fail_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Performance benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Performance Benchmark: MoE vs Dense FFN ===\n")

    configs = [
        # (T,    d_model, d_ffn,  N,  m,  K,  K_s)
        (32,    512,     2048,   8,  4,  2,  1),
        (128,   512,     2048,   8,  4,  2,  1),
        (512,   512,     2048,   8,  4,  2,  1),
        (2048,  512,     2048,   8,  4,  2,  1),
    ]

    for (T, D, Hd, N, m, K, Ks) in configs:
        # Dense FFN (same parameter budget as MoE)
        dense = nn.Sequential(nn.Linear(D, Hd), nn.ReLU(), nn.Linear(Hd, D)).to(device)
        moe   = DeepSeekMoE(D, Hd, N, m, K, Ks).to(device)

        x = torch.randn(T, D, device=device)

        WARMUP, RUNS = 5, 20

        def time_fn(fn, x):
            with torch.no_grad():
                for _ in range(WARMUP): fn(x)
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(RUNS): fn(x)
                if device.type == "cuda": torch.cuda.synchronize()
                return (time.perf_counter() - t0) / RUNS * 1000  # ms

        dense_ms = time_fn(dense, x)
        moe_ms   = time_fn(moe,   x)

        E_active  = K * m - Ks
        pct_active = 100.0 * E_active / (m * N)
        print(f"T={T:5d}  D={D}  E={m*N} (K={E_active} active, {pct_active:.0f}%)")
        print(f"  Dense FFN:  {dense_ms:.3f} ms")
        print(f"  DeepSeek MoE:  {moe_ms:.3f} ms  (ratio: {moe_ms/dense_ms:.2f}x)")
        print()

    print("Note: MoE overhead at small T is dominated by routing logic.")
    print("At large T, MoE's sparse activation means parameter count can scale")
    print("while compute grows sub-linearly (matching the lecture notes).\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ok = run_tests()
    benchmark()
    exit(0 if ok else 1)
