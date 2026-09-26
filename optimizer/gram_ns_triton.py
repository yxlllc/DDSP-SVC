"""
Triton implementation of the Gram Newton-Schulz iteration used by Muon.

Selected by ``Muon.step`` when the optimizer is built with
``use_fused_kernels=True`` (the reference cuBLAS chain stays in
``muon.gram_newton_schulz``). Same numerics: fp16 storage with fp32
accumulation, identical control flow. Changes versus the cuBLAS chain:

1. fused Frobenius-normalize + fp16 cast (one pass instead of ~3),
2. symmetric Gram kernel: only the lower triangle of tiles is computed and
   mirrored at store time — ~half the FLOPs of ``bmm(X, X.mT)``,
3. one generic batched GEMM with a fused linear epilogue
   ``OUT = alpha * (A @ B) + beta * C`` used for the Z / Q / RZ / R updates
   and the polynomial apply; an optional second store folds ``Q = Z + a*I``
   in, removing the clone + diagonal-add kernels.

Buffers ping-pong instead of being re-allocated/cloned inside the loop.

Autotuning: block configs are NOT hardcoded per GPU — both GEMM kernels are
``@triton.autotune``d and pick the fastest config on whatever GPU runs them.
Both tune with ``cache_results=True``: Triton persists every tuning key's
per-config timings to its own on-disk cache (``~/.triton/cache/<hash>/
<kernel>.autotune.json``; the hash covers the triton build, GPU-arch backend,
kernel source, tuning key, and the full config pool, so editing any of those
re-tunes automatically). The full benchmark sweep therefore runs only on a
cold cache — ``warmup_gram_ns``, called by train_reflow.py at startup under
``train.use_fused_kernels`` with the model's real param shape groups, just
loads the recorded timings + compiled binaries on later process starts
(minutes -> seconds). To force a fresh tune, e.g. after a driver update,
delete the ``*.autotune.json`` files under the triton cache dir.
Steady-state launches bypass the Autotuner wrapper: its per-call Python
overhead (~tens of µs building the key tuple) would dominate the small
launch-bound groups, so after a key is tuned we relaunch the underlying
JITFunction directly with the winning config (_BEST_* caches below — that is
a per-call overhead optimization, orthogonal to the on-disk result cache).
"""

import torch
import triton
import triton.language as tl

_A, _B, _C = 3.4445, -4.7750, 2.0315  # Newton-Schulz quintic coefficients

# Cross-GPU config pools. Autotune benchmarks every config on first use of
# each key and prunes configs that don't fit the GPU (shared memory, etc.),
# so shipping large H100-class tiles alongside consumer-class ones is safe.
_GEMM_CONFIGS = [
    triton.Config({'BM': 64,  'BN': 64,  'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64,  'BN': 64,  'BK': 32}, num_warps=4, num_stages=4),
    triton.Config({'BM': 64,  'BN': 64,  'BK': 64}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64,  'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 64,  'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 64,  'BK': 32}, num_warps=4, num_stages=4),
    triton.Config({'BM': 128, 'BN': 64,  'BK': 64}, num_warps=4, num_stages=2),
    triton.Config({'BM': 64,  'BN': 256, 'BK': 64}, num_warps=4, num_stages=2),
    triton.Config({'BM': 64,  'BN': 256, 'BK': 64}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=4),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_warps=8, num_stages=3),
    triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=3),
]
# NOTE: _gram_kernel tiles must stay SQUARE (BM == BN) — the triangular
# tiling assumes a tile is either entirely below or above the diagonal;
# rectangular tiles would straddle it and silently drop R entries.
_GRAM_CONFIGS = [
    triton.Config({'BM': 64,  'BN': 64,  'BK': 32}, num_warps=4, num_stages=2),
    triton.Config({'BM': 64,  'BN': 64,  'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64,  'BN': 64,  'BK': 64}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64,  'BN': 64,  'BK': 64}, num_warps=4, num_stages=4),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=4),
    triton.Config({'BM': 256, 'BN': 256, 'BK': 32}, num_warps=8, num_stages=3),
]

# Keep the per-key autotune benchmark short: config gaps here are >=10%, and
# with cache_results=True the sweep only runs on a cold triton disk cache.
_AUTOTUNE_KW = dict(warmup=5, rep=20)


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@triton.jit
def _normalize_cast_kernel(X, NORM, Y, numel, BLOCK: tl.constexpr):
    """Y[b] = X[b] / max(||X[b]||_F, 1e-12), cast to fp16. X contiguous (B, numel)."""
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(X + b * numel + offs, mask=mask, other=0.0)
    n = tl.maximum(tl.load(NORM + b), 1e-12)
    tl.store(Y + b * numel + offs, tl.math.div_rn(x, n).to(tl.float16), mask=mask)


@triton.autotune(configs=_GRAM_CONFIGS, key=['NB', 'K', 'L', 'M_LT_N'],
                 cache_results=True, **_AUTOTUNE_KW)
@triton.jit
def _gram_kernel(X, R, NB, K, L,
                 M_LT_N: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """R[b] = V @ V^T with V = X[b] (K, L) if M_LT_N else X[b]^T (L, K case:
    V[i, l] = X[b][l, i]). Only tiles tj <= ti are computed; the symmetric
    counterpart is written with a transposed store. fp16 in, fp32 acc.
    NB (batch count) is only an autotune key."""
    b = tl.program_id(0)
    ti = tl.program_id(1)
    tj = tl.program_id(2)
    if tj > ti:
        return
    rm = ti * BM + tl.arange(0, BM)
    rn = tj * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, L, BK):
        ks = k0 + rk
        if M_LT_N:  # X is (K, L), R = X @ X^T
            a = tl.load(X + b * K * L + rm[:, None] * L + ks[None, :],
                        mask=(rm[:, None] < K) & (ks[None, :] < L), other=0.0)
            bt = tl.load(X + b * K * L + rn[:, None] * L + ks[None, :],
                         mask=(rn[:, None] < K) & (ks[None, :] < L), other=0.0)
            acc = tl.dot(a, tl.trans(bt), acc)
        else:       # X is (L, K), R = X^T @ X
            a = tl.load(X + b * L * K + ks[:, None] * K + rm[None, :],
                        mask=(ks[:, None] < L) & (rm[None, :] < K), other=0.0)
            bt = tl.load(X + b * L * K + ks[:, None] * K + rn[None, :],
                         mask=(ks[:, None] < L) & (rn[None, :] < K), other=0.0)
            acc = tl.dot(tl.trans(a), bt, acc)
    out = acc.to(tl.float16)
    rbase = R + b * K * K
    mi = rm[:, None] < K
    mj = rn[None, :] < K
    tl.store(rbase + rm[:, None] * K + rn[None, :], out, mask=mi & mj)
    # mirrored store: target tile is (BN, BM), build its mask accordingly
    # (skip the diagonal tile: identical addresses & values)
    m2 = (rn[:, None] < K) & (rm[None, :] < K) & (ti != tj)
    tl.store(rbase + rn[:, None] * K + rm[None, :], tl.trans(out), mask=m2)


@triton.autotune(configs=_GEMM_CONFIGS, key=['NB', 'P', 'Q', 'K', 'HAS_BETA', 'WRITE_Q2'],
                 cache_results=True, **_AUTOTUNE_KW)
@triton.jit
def _gemm_epilogue_kernel(A, Bm, C, OUT, Q2,
                          NB, P, Q, K,
                          alpha, beta, diag_a,
                          HAS_BETA: tl.constexpr, WRITE_Q2: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Batched: OUT = alpha * (A @ Bm) + beta * C.
    WRITE_Q2 additionally stores  Q2 = OUT + diag_a * I  (used to fold
    Q = Z + a*I into the Z kernel). All batch tensors contiguous.
    NB (batch count) is only an autotune key."""
    b = tl.program_id(0)
    pi = tl.program_id(1)
    pj = tl.program_id(2)
    rm = pi * BM + tl.arange(0, BM)
    rn = pj * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + rk
        a = tl.load(A + b * P * K + rm[:, None] * K + ks[None, :],
                    mask=(rm[:, None] < P) & (ks[None, :] < K), other=0.0)
        bt = tl.load(Bm + b * K * Q + ks[:, None] * Q + rn[None, :],
                     mask=(ks[:, None] < K) & (rn[None, :] < Q), other=0.0)
        acc = tl.dot(a, bt, acc)
    outm = (rm[:, None] < P) & (rn[None, :] < Q)
    if HAS_BETA:
        c = tl.load(C + b * P * Q + rm[:, None] * Q + rn[None, :],
                    mask=outm, other=0.0).to(tl.float32)
        acc = alpha * acc + beta * c
    else:
        acc = alpha * acc
    out = acc.to(tl.float16)
    tl.store(OUT + b * P * Q + rm[:, None] * Q + rn[None, :], out, mask=outm)
    if WRITE_Q2:
        d = tl.where(rm[:, None] == rn[None, :], diag_a, 0.0)
        tl.store(Q2 + b * P * Q + rm[:, None] * Q + rn[None, :],
                 (out.to(tl.float32) + d).to(tl.float16), mask=outm)


# --------------------------------------------------------------------------
# host-side launch helpers
# --------------------------------------------------------------------------
# key -> (BM, BN, BK, num_warps, num_stages), filled from best_config after
# the first (autotuned) call per shape key — on warm disk caches that call
# replays recorded timings, no benchmarks run; afterwards the raw
# JITFunction is launched directly, skipping the Autotuner wrapper's
# per-call Python overhead.
_BEST_GRAM = {}
_BEST_GEP = {}


def _normalize_cast(X32):
    B, M, N = X32.shape
    norms = torch.linalg.vector_norm(X32, dim=(1, 2))  # (B,) fp32
    Y = torch.empty((B, M, N), device=X32.device, dtype=torch.float16)
    numel = M * N
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK), B)
    _normalize_cast_kernel[grid](X32, norms, Y, numel, BLOCK=BLOCK)
    return Y


def _gram(X, R, K, L, m_lt_n):
    B = X.shape[0]
    key = (B, K, L, bool(m_lt_n))
    cfg = _BEST_GRAM.get(key)
    if cfg is not None:
        BM, BN, BK, w, s = cfg
        grid = (B, triton.cdiv(K, BM), triton.cdiv(K, BN))
        _gram_kernel.fn[grid](X, R, B, K, L, M_LT_N=m_lt_n,
                              BM=BM, BN=BN, BK=BK, num_warps=w, num_stages=s)
        return
    grid = lambda META: (B, triton.cdiv(K, META['BM']), triton.cdiv(K, META['BN']))
    _gram_kernel[grid](X, R, B, K, L, M_LT_N=m_lt_n)  # autotunes, then runs
    best = getattr(_gram_kernel, 'best_config', None)
    if best is not None:
        _BEST_GRAM[key] = (best.kwargs['BM'], best.kwargs['BN'], best.kwargs['BK'],
                           best.num_warps, best.num_stages)


def _gep(out, a, b, c, alpha, beta, diag_a=0.0, q2=None):
    Bn, P, K = a.shape
    Q = b.shape[2]
    key = (Bn, P, Q, K, c is not None, q2 is not None)
    cfg = _BEST_GEP.get(key)
    if cfg is not None:
        BM, BN, BK, w, s = cfg
        grid = (Bn, triton.cdiv(P, BM), triton.cdiv(Q, BN))
        _gemm_epilogue_kernel.fn[grid](
            a, b, c if c is not None else out, out,
            q2 if q2 is not None else out,
            Bn, P, Q, K, alpha, beta, diag_a,
            HAS_BETA=c is not None, WRITE_Q2=q2 is not None,
            BM=BM, BN=BN, BK=BK, num_warps=w, num_stages=s)
        return
    grid = lambda META: (Bn, triton.cdiv(P, META['BM']), triton.cdiv(Q, META['BN']))
    _gemm_epilogue_kernel[grid](
        a, b, c if c is not None else out, out,
        q2 if q2 is not None else out,
        Bn, P, Q, K, alpha, beta, diag_a,
        HAS_BETA=c is not None, WRITE_Q2=q2 is not None)  # autotunes, then runs
    best = getattr(_gemm_epilogue_kernel, 'best_config', None)
    if best is not None:
        _BEST_GEP[key] = (best.kwargs['BM'], best.kwargs['BN'], best.kwargs['BK'],
                          best.num_warps, best.num_stages)


def _apply_q(X, Q, m_lt_n):
    """X <- Q @ X (m_lt_n, X is (B, K, L)) or X <- X @ Q (X is (B, L, K))."""
    out = torch.empty_like(X)
    if m_lt_n:
        _gep(out, Q, X, None, 1.0, 0.0)
    else:
        _gep(out, X, Q, None, 1.0, 0.0)
    return out


# --------------------------------------------------------------------------
# public entry
# --------------------------------------------------------------------------
def gram_ns_triton(G: torch.Tensor, steps: int, reset_iterations) -> torch.Tensor:
    """Triton drop-in for muon.gram_newton_schulz. Same numerics (fp16 GEMMs
    with fp32 accumulate), same control flow, so step count / resets behave
    identically."""
    assert G.ndim == 3
    B, M, N = G.shape
    dev = G.device
    resets = reset_iterations if reset_iterations is not None else []

    X32 = G.to(dtype=torch.float32).contiguous()
    X = _normalize_cast(X32)                      # fp16, unit Frobenius norm

    K = min(M, N)
    L = max(M, N)
    m_lt_n = M < N

    if M != N:
        R0 = torch.empty((B, K, K), device=dev, dtype=torch.float16)
        R1 = torch.empty_like(R0)
        Z = torch.empty_like(R0)
        RZ = torch.empty_like(R0)
        Q0 = torch.empty_like(R0)
        Q1 = torch.empty_like(R0)

        _gram(X, R0, K, L, m_lt_n)
        R = R0
        Q = None
        for i in range(steps):
            if i in resets and i != 0:
                X = _apply_q(X, Q, m_lt_n)
                _gram(X, R, K, L, m_lt_n)  # R buffer is fully overwritten
                Q = None
            if i != 0 and i not in resets:
                # Z = b*R + c*(R@R) ; Q = a*Q + Q@Z
                _gep(Z, R, R, R, _C, _B)
                Qn = Q1 if Q is Q0 else Q0
                _gep(Qn, Q, Z, Q, 1.0, _A)
                Q = Qn
            else:
                # Z = b*R + c*(R@R) ; Q = Z + a*I  (fused second store).
                # On this branch Q is always None (initial state / post-reset).
                _gep(Z, R, R, R, _C, _B, diag_a=_A, q2=Q0)
                Q = Q0
            if i < steps - 1 and (i + 1) not in resets:
                # RZ = a*R + R@Z ; R = a*RZ + Z@RZ
                _gep(RZ, R, Z, R, 1.0, _A)
                Rn = R1 if R is R0 else R0
                _gep(Rn, Z, RZ, RZ, 1.0, _A)
                R = Rn
        X = _apply_q(X, Q, m_lt_n)
    else:
        # square case: A = X@X^T (symmetric kernel), B = b*A + c*(A@A),
        #              X = a*X + B@X   — same 3-GEMM chain, A at half FLOPs.
        Amat = torch.empty((B, K, K), device=dev, dtype=torch.float16)
        Bm = torch.empty_like(Amat)
        Xn = torch.empty_like(X)
        for _ in range(steps):
            _gram(X, Amat, K, K, True)
            _gep(Bm, Amat, Amat, Amat, _C, _B)
            _gep(Xn, Bm, X, X, 1.0, _A)
            X, Xn = Xn, X
        # X may now alias the 'Xn' scratch; that's fine — it is the result.

    return X


# --------------------------------------------------------------------------
# startup warmup (trigger autotune outside of any training step)
# --------------------------------------------------------------------------
def warmup_gram_ns(shape_groups, device, steps=5, reset_iterations=(2,)):
    """Pre-compile + autotune every kernel/shape key the training run needs.

    `shape_groups` is an iterable of (B, M, N) tuples — B = param count per
    shape group, exactly what gram_ns_triton will see at optimizer time; get
    it from ``muon.gram_ns_shape_groups(get_params_for_muon(model))``.

    Call this once per training process, after the model is on the GPU,
    before training. With ``cache_results=True`` on both kernels the
    benchmark sweep only runs on a cold triton disk cache; on warm starts
    this just loads the recorded timings + compiled binaries (~seconds).
    Returns the number of warmed shape groups.
    """
    n = 0
    for Bsz, M, N in shape_groups:
        G = torch.randn(Bsz, M, N, device=device)
        with torch.no_grad():
            gram_ns_triton(G, steps, list(reset_iterations))
        del G
        n += 1
    torch.cuda.synchronize(device)
    return n
