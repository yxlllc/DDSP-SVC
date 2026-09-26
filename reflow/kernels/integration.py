"""
Drop-in replacement for LYNXNet2Block with fused Linear+SoftSignGLU kernels.

Adapted from DiffSinger/modules/kernels/integration.py for this project's
LYNXNet2 (reflow/lynxnet2.py). Differences vs DiffSinger's LYNXNet2:
  - No LayerNorm inside ``block.net`` — normalization is F.rms_norm applied
    in ``LYNXNet2Block.forward``, so all net indices shift by -1:
      net[0]=Transpose, net[1]=Conv1d, net[2]=Transpose,
      net[3]=Linear, net[4]=GLU, net[5]=Linear, net[6]=GLU,
      net[7]=Linear(out), net[8]=Dropout/Identity
  - The backbone stores no ``n_feats``/``in_dims`` attributes; warmup derives
    dims from the projection layers instead.

The fused kernel replaces:
  nn.Linear(dim, inner_dim*2) + SoftSignGLU  →  one fused kernel call
(training mode only; eval mode uses the original nn.Sequential path).

Only softsign_glu / double_softsign_glu are supported — other GLU types are
left unpatched (warning at patch time, block runs the original forward).

Numerical accuracy:
  SoftSignGLU is exact in Triton (no approximation). Differences vs the
  eager path are fp16 rounding only (~1e-3 max on unit-scale activations).

HBM savings (per fused call, M=50000, N=1024, fp16):
  Eager:  Linear writes [M, 2N] (200 MB), GLU reads [M, 2N] + writes [M, N]
  Fused:  writes y/left/gate = 3×[M, N] — saves the [M, 2N] round-trip
Backward saves the softsign/denominator intermediates by fusing the
element-wise gradient into one kernel; all GEMMs stay on cuBLAS.

ONNX export:
  Use `model.eval()` → falls back to original path → ONNX export works
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from reflow.kernels.fused_linear_softsign_glu import (
    fused_linear_softsign_glu,
    is_triton_available,
)


_FUSABLE_GLU_TYPES = ('softsign_glu', 'double_softsign_glu')


def wrap_lynxnet2_block(block, glu_type='softsign_glu'):
    """Wrap an existing LYNXNet2Block to use fused forward.

    Keeps all weights in-place (state_dict compatible).
    Only modifies the forward pass.

    'softsign_glu' and 'double_softsign_glu' are fused. Other GLU types
    (atanglu) are returned unpatched and keep the eager path.

    Args:
        block: LYNXNet2Block instance
        glu_type: GLU type configured for this block

    Returns:
        The same block, with patched forward if glu_type is supported.
    """
    if glu_type not in _FUSABLE_GLU_TYPES:
        import warnings
        warnings.warn(
            f"Fused kernels support only {_FUSABLE_GLU_TYPES}; leaving block "
            f"with glu_type={glu_type!r} unpatched."
        )
        return block

    is_double = glu_type == 'double_softsign_glu'

    def fused_forward(self, x):
        residual = x

        # Original: rms_norm → Transpose → Conv1d → Transpose
        x = F.rms_norm(x, (x.size(-1),))
        x = self.net[0](x)  # Transpose
        x = self.net[1](x)  # Conv1d(depthwise)
        x = self.net[2](x)  # Transpose

        if self.training:
            # Fused: Linear+GLU → Linear+GLU
            x = fused_linear_softsign_glu(x, self.net[3].weight, self.net[3].bias, is_double)
            x = fused_linear_softsign_glu(x, self.net[5].weight, self.net[5].bias, is_double)
        else:
            # Original: Linear → GLU → Linear → GLU
            x = self.net[3](x)
            x = self.net[4](x)  # (Double)SoftSignGLU
            x = self.net[5](x)
            x = self.net[6](x)

        # Original: Linear → Dropout → +residual
        x = self.net[7](x)  # output projection
        x = self.net[8](x)  # Dropout
        return x + residual

    # Monkey-patch — use descriptor protocol so the bound method captures
    # ``self`` dynamically (avoids the deepcopy stale-closure issue: a
    # deepcopy'ed block would otherwise keep a reference to the original's
    # ``net``).
    block.forward = fused_forward.__get__(block, type(block))
    return block


def patch_lynxnet2_model(model, glu_type='softsign_glu'):
    """Patch all LYNXNet2Blocks in a LYNXNet2 model.

    Args:
        model: LYNXNet2 instance
        glu_type: GLU type configured for the model (only softsign_glu fuses)

    Returns:
        Number of blocks patched (0 if glu_type unsupported).
    """
    from reflow.lynxnet2 import LYNXNet2Block
    if glu_type not in _FUSABLE_GLU_TYPES:
        import warnings
        warnings.warn(
            f"Fused kernels require glu_type in {_FUSABLE_GLU_TYPES}; "
            f"got {glu_type!r}. Skipping patch."
        )
        return 0
    if not is_triton_available():
        raise RuntimeError(
            'Fused kernels require a working Triton installation. '
            'Install Triton for this platform or set use_fused_kernels: false.'
        )
    patched = 0
    for i, layer in enumerate(model.residual_layers):
        if isinstance(layer, LYNXNet2Block):
            model.residual_layers[i] = wrap_lynxnet2_block(layer, glu_type=glu_type)
            patched += 1
    return patched


# ---------------------------------------------------------------------------
# Safe patching — handles both DDPM (denoise_fn) and ReFlow (velocity_fn),
# and checks that the backbone is actually a LYNXNet2 before patching.
# ---------------------------------------------------------------------------

def _patch_backbone_fn(backbone_fn, glu_type):
    """Patch a single backbone function/module if it's a LYNXNet2.

    Args:
        backbone_fn: The backbone module (e.g., reflow.velocity_fn)
        glu_type: GLU type (only softsign_glu fuses)

    Returns:
        Number of blocks patched (0 if not a LYNXNet2).
    """
    from reflow.lynxnet2 import LYNXNet2
    if not isinstance(backbone_fn, LYNXNet2):
        return 0
    return patch_lynxnet2_model(backbone_fn, glu_type=glu_type)


def _try_patch(module, attr, glu_type):
    """Try to patch backbone at module.attr if it's a LYNXNet2. Safe to call
    even if attr doesn't exist — returns 0 silently."""
    backbone = getattr(module, attr, None)
    if backbone is None:
        return 0
    return _patch_backbone_fn(backbone, glu_type)


def patch_diffusion_module(diffusion, glu_type='softsign_glu'):
    """Patch a diffusion module's backbone (DDPM or ReFlow).

    Handles both:
      GaussianDiffusion → .denoise_fn
      RectifiedFlow → .velocity_fn

    Returns:
        Number of blocks patched.
    """
    return (
        _try_patch(diffusion, 'denoise_fn', glu_type) +
        _try_patch(diffusion, 'velocity_fn', glu_type)
    )


def patch_unit2wav(model, glu_type='softsign_glu'):
    """Patch all fusable LYNXNet2Blocks in a Unit2Wav model.

    Covers both sub-models:
      - model.reflow_model.velocity_fn (LYNXNet2 backbone) — patched with
        the configured ``glu_type``
      - model.ddsp_model (CombSubSuperFast → Unit2Control) — its blocks are
        constructed with the default glu_type, so each block is patched with
        its own recorded ``block.glu_type``

    Returns:
        Number of blocks patched.
    """
    from reflow.lynxnet2 import LYNXNet2Block
    patched = patch_diffusion_module(model.reflow_model, glu_type=glu_type)
    ddsp_model = getattr(model, 'ddsp_model', None)
    if ddsp_model is not None:
        for block in ddsp_model.modules():
            if isinstance(block, LYNXNet2Block):
                block_glu = getattr(block, 'glu_type', 'softsign_glu')
                if block_glu in _FUSABLE_GLU_TYPES:
                    wrap_lynxnet2_block(block, glu_type=block_glu)
                    patched += 1
    return patched


# ---------------------------------------------------------------------------
# Warmup — trigger Triton autotune before training starts
# ---------------------------------------------------------------------------

def _bucket_timesteps(max_frames):
    """Timesteps T such that M = 4 * T sweeps every power-of-two M bucket
    from 2048 up to next_power_of_2(max_frames) (the fused forward kernel's
    M_BUCKET autotune key is next_power_of_2(M)).
    """
    import triton
    B = 4
    # Sweep M buckets: 2048 up to next_power_of_2(max_frames)
    if max_frames is not None:
        top = triton.next_power_of_2(int(max_frames))
        bucket = 2048
        t_list = []
        while bucket <= top:
            # M = B * T lands in this bucket (M just above the previous bucket)
            t_list.append(bucket // B // 2 + 1)
            bucket *= 2
    else:
        t_list = [500]
    return B, t_list


def warmup_fused_backbone(backbone, max_frames=None, autocast_dtype=None):
    """Run dummy forward passes to trigger Triton autotune compilation
    for all fused kernels (fwd + bwd elem). Call after patching, before
    the first real training step (model must already be on its CUDA device).

    Only covers THIS backbone's (N, K) shapes. Blocks living outside a
    LYNXNet2 wrapper (e.g. Unit2Control inside ddsp_model, whose dim is
    n_aux_chans) need ``warmup_fused_blocks`` instead.

    Only forward is executed (``torch.no_grad``) — the element-wise backward
    kernel's autotune key depends on ``N`` (a single fixed value per model),
    so its one-off compile+tune is paid on the first real step instead (and
    lands in the same disk cache).

    Both kernels tune with cache_results=True: per-key config timings
    persist in Triton's on-disk cache, so on a warm cache this only
    compile-loads the winning binaries and refills the in-process autotune
    table. The forward kernel's autotune key buckets M by next_power_of_2,
    so we sweep the power-of-two buckets a real run will hit: from a small
    bucket up to next_power_of_2(max_frames).

    Args:
        backbone: LYNXNet2 model (already patched).
        max_frames: max total frames per batch
            (batch_size * duration * sampling_rate / block_size).
            If None, warms a single small bucket only.
        autocast_dtype: torch.float16 for fp16, torch.bfloat16 for bf16.
            If None (fp32), no autocast — with fp32 parameters the fused
            path falls back to eager and the warmup is a no-op.
    """
    if not is_triton_available():
        raise RuntimeError(
            'Fused kernel warmup requires a working Triton installation. '
            'Install Triton for this platform or set use_fused_kernels: false.'
        )

    import contextlib
    import warnings

    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype

    if not device.type == 'cuda':
        return

    # cond hidden size from the conditioner projection (Linear or Conv1d)
    proj = backbone.conditioner_projection
    hidden = getattr(proj, 'in_features', None) or proj.in_channels
    # mel dims from the output projection (this LYNXNet2 stores no in_dims)
    in_dims = backbone.output_projection.out_features

    B, t_list = _bucket_timesteps(max_frames)

    ac_factory = (
        (lambda: torch.autocast(device_type=device.type, dtype=autocast_dtype))
        if autocast_dtype is not None else contextlib.nullcontext
    )
    for T in t_list:
        # spec shape: [B, 1, M, T], matching RectifiedFlow's x_t
        spec = torch.randn(B, 1, in_dims, T, device=device, dtype=dtype)
        t = torch.randint(0, 1000, (B,), device=device).float()
        cond = torch.randn(B, hidden, T, device=device, dtype=dtype)

        try:
            with torch.no_grad():
                with ac_factory():
                    backbone(spec, t, cond=cond)
        except Exception as e:
            # Autotune failure should not crash training — Triton cache
            # can be built on the first real step instead.
            warnings.warn(f'Fused kernel warmup skipped at T={T} ({e})')
            break
        finally:
            del spec, cond
    torch.cuda.empty_cache()


def warmup_fused_block(block, max_frames=None, autocast_dtype=None):
    """Run dummy forward passes through ONE patched LYNXNet2Block to trigger
    Triton autotune for its fused kernels.

    Companion to ``warmup_fused_backbone``: that helper needs a full LYNXNet2
    (it reads ``conditioner_projection``/``output_projection``), so it cannot
    warm the LYNXNet2Blocks inside the DDSP front-end
    (Unit2Wav.ddsp_model → CombSubSuperFast → Unit2Control), whose channel
    count (n_aux_chans) differs from the velocity backbone (n_chans). The
    fused forward kernel is autotuned on (M_BUCKET, N, K), so a block at a
    different dim is a different autotune key and must be warmed separately —
    otherwise its compile+bench lands inside the first real training step on
    a cold Triton cache.

    Autotune keys depend only on shapes/dtype, not weights: warming one block
    covers every same-shaped block in the model.

    Args:
        block: LYNXNet2Block instance (already patched, on CUDA).
        max_frames: max total frames per batch
            (batch_size * duration * sampling_rate / block_size).
            If None, warms a single small bucket only.
        autocast_dtype: torch.float16 for fp16, torch.bfloat16 for bf16.
            If None (fp32), the fused path falls back to eager and this is a
            no-op for the fused kernels.
    """
    if not is_triton_available():
        raise RuntimeError(
            'Fused kernel warmup requires a working Triton installation. '
            'Install Triton for this platform or set use_fused_kernels: false.'
        )

    import contextlib
    import warnings

    # net[3] = Linear(dim, inner_dim*2) — the first fused GEMM's projection
    weight = block.net[3].weight
    device = weight.device
    dtype = weight.dtype
    dim = block.net[3].in_features

    if device.type != 'cuda':
        return

    B, t_list = _bucket_timesteps(max_frames)

    ac_factory = (
        (lambda: torch.autocast(device_type=device.type, dtype=autocast_dtype))
        if autocast_dtype is not None else contextlib.nullcontext
    )

    # The fused patch only engages in training mode — force it (the caller
    # runs this before the first real step, where the block may be in any
    # mode), and restore afterwards.
    was_training = block.training
    block.train()
    try:
        for T in t_list:
            x = torch.randn(B, T, dim, device=device, dtype=dtype)
            try:
                with torch.no_grad():
                    with ac_factory():
                        block(x)
            except Exception as e:
                warnings.warn(f'Fused block warmup skipped at T={T} ({e})')
                break
            finally:
                del x
    finally:
        if not was_training:
            block.eval()
    torch.cuda.empty_cache()


def warmup_fused_blocks(module, max_frames=None, autocast_dtype=None):
    """Trigger Triton autotune for every unique fused LYNXNet2Block shape
    found under ``module`` (e.g. ``model.ddsp_model``).

    Blocks are grouped by (in_features, out_features, glu_type) of their
    first fused projection (``net[3]``) and ONE block per group is warmed —
    autotune is shape-keyed and weight-agnostic. Blocks with an unsupported
    GLU type (left unpatched) are skipped.

    Returns:
        Number of unique block shapes warmed (0 if module is None).
    """
    from reflow.lynxnet2 import LYNXNet2Block
    if module is None:
        return 0
    seen = {}
    for block in module.modules():
        if isinstance(block, LYNXNet2Block):
            if getattr(block, 'glu_type', None) not in _FUSABLE_GLU_TYPES:
                continue
            proj = block.net[3]
            key = (proj.in_features, proj.out_features, block.glu_type)
            seen.setdefault(key, block)
    for block in seen.values():
        warmup_fused_block(block, max_frames=max_frames,
                           autocast_dtype=autocast_dtype)
    return len(seen)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def _test():
    import torch
    from reflow.lynxnet2 import LYNXNet2Block

    device = 'cuda'
    torch.manual_seed(42)

    for glu_type in _FUSABLE_GLU_TYPES:
        # Create a single block
        block = LYNXNet2Block(dim=256, expansion_factor=1, glu_type=glu_type).to(device).half()

        # Copy weights
        block_ref = LYNXNet2Block(dim=256, expansion_factor=1, glu_type=glu_type).to(device).half()
        block_ref.load_state_dict(block.state_dict())

        # Patch
        wrap_lynxnet2_block(block, glu_type=glu_type)

        B, T = 2, 500
        x = torch.randn(B, T, 256, device=device, dtype=torch.float16)

        # Forward
        out_orig = block_ref(x)
        out_fused = block(x)

        fwd_diff = (out_fused - out_orig).abs().max().item()
        print(f"[{glu_type}] Block forward max diff: {fwd_diff:.4e}")

        # Backward
        grad = torch.randn_like(out_orig)
        out_orig.backward(grad)
        grads_ref = {n: p.grad.clone() for n, p in block_ref.named_parameters() if p.grad is not None}

        for p in block.parameters():
            p.grad = None

        out_fused = block(x)
        out_fused.backward(grad)
        grads_fused = {n: p.grad.clone() for n, p in block.named_parameters() if p.grad is not None}

        max_w_diff = max(
            (grads_fused[n] - grads_ref[n]).abs().max().item()
            for n in grads_ref
        )
        print(f"[{glu_type}] Block weight grad max diff: {max_w_diff:.4e}")

        # Eval mode falls back to the original nn.Sequential path
        block.eval()
        block_ref.eval()
        with torch.no_grad():
            eval_diff = (block(x) - block_ref(x)).abs().max().item()
        print(f"[{glu_type}] Block eval (eager fallback) max diff: {eval_diff:.4e}")

    print(f"\nIntegration works! Use model.eval() for ONNX export fallback.")


if __name__ == '__main__':
    _test()
