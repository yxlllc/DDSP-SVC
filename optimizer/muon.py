import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, Embedding
from typing import List
from collections import Counter
from .chained_optimizer import ChainedOptimizer, OptimizerSpec

from .gram_ns_triton import gram_ns_triton


def gram_newton_schulz(G: Tensor, steps: int, reset_iterations: List[int]) -> Tensor:
    """
    Gram Newton-Schulz iteration to compute the orthogonalization of G.
    Mathematically identical to standard Newton-Schulz but computes iterating
    on the smaller NxN Gram matrix to save up to 50% FLOPs.
    """
    assert G.ndim == 3

    X = G.to(dtype=torch.float32)
    X = F.normalize(X, p=2.0, dim=(-2, -1))
    X = X.to(dtype=torch.float16)
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    if X.size(-2) != X.size(-1):
        R = torch.bmm(X, X.mT) if X.size(-2) < X.size(-1) else torch.bmm(X.mT, X)
        Q = None
        for i in range(steps):
            if i in reset_iterations and i != 0:
                X = torch.bmm(Q, X) if X.size(-2) < X.size(-1) else torch.bmm(X, Q)
                R = torch.bmm(X, X.mT) if X.size(-2) < X.size(-1) else torch.bmm(X.mT, X)
                Q = None
            Z = torch.baddbmm(R, R, R, beta=b, alpha=c)
            if i != 0 and i not in reset_iterations:
                Q = torch.baddbmm(Q, Q, Z, beta=a, alpha=1.0)
            else:
                Q = Z.clone()
                Q.diagonal(dim1=-2, dim2=-1).add_(a)
            if i < steps - 1 and (i + 1) not in reset_iterations:
                RZ = torch.baddbmm(R, R, Z, beta=a, alpha=1.0)
                R = torch.baddbmm(RZ, Z, RZ, beta=a, alpha=1.0)
        X = torch.bmm(Q, X) if X.size(-2) < X.size(-1) else torch.bmm(X, Q)
    else:
        for _ in range(steps):
            A = torch.bmm(X, X.mT)
            B = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)

    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
        use_fused_kernels: Run the Gram Newton-Schulz orthogonalization with the fused
            Triton kernels from gram_ns_triton.py (same numerics: fp16 storage, fp32
            accumulate) instead of the cuBLAS reference chain, for every CUDA shape
            group (non-CUDA gradients stay on the reference). When enabled,
            gram_ns_triton.warmup_gram_ns should run once at startup
            (train_reflow.py does this under train.use_fused_kernels)
            so Triton autotune never fires inside a training step.
    """

    def __init__(self, params, lr=5e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5, reset_iterations=[2], use_fused_kernels=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, reset_iterations=reset_iterations, use_fused_kernels=use_fused_kernels)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            shape_groups = {}
            for p in filter(lambda p: p.grad is not None, group["params"]):
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                key = (p.shape, p.device, p.dtype)
                if key not in shape_groups:
                    shape_groups[key] = {"params": [], "grads": [], "buffers": []}
                shape_groups[key]["params"].append(p)
                shape_groups[key]["grads"].append(g)
                shape_groups[key]["buffers"].append(state["momentum_buffer"])
            for key in shape_groups:
                group_data = shape_groups[key]
                p, g, buf, m = group_data["params"], group_data["grads"], group_data["buffers"], group["momentum"]
                torch._foreach_lerp_(buf, g, 1-m)
                if group["nesterov"]:
                    torch._foreach_lerp_(g, buf, m)
                    g = torch.stack(g)
                else:
                    g = torch.stack(buf)
                original_shape = g.shape
                if g.ndim >= 4:  # for the case of conv filters
                    g = g.view(g.size(0), g.size(1), -1)
                use_triton_ns = group["use_fused_kernels"] and g.is_cuda
                if use_triton_ns:
                    g = gram_ns_triton(g, steps=group["ns_steps"], reset_iterations=group["reset_iterations"])
                else:
                    g = gram_newton_schulz(g, steps=group["ns_steps"], reset_iterations=group["reset_iterations"])
                if group["weight_decay"] > 0:
                    torch._foreach_mul_(p, 1 - group["lr"] * group["weight_decay"])
                torch._foreach_add_(p, g.view(original_shape).unbind(0), alpha=-group["lr"] * max(g[0].size()) ** 0.5)


def get_params_for_muon(model) -> List[Parameter]:
    """
    Filter parameters of a module into two groups: those that can be optimized by Muon,
    and those that should be optimized by a standard optimizer.
    Args:
        module: The module to filter parameters for.
    Returns:
        A list of parameters that should be optimized with muon.
    """
    muon_params = []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if name == 'weight_g':
                continue
            if not isinstance(module, nn.Embedding) and param.ndim >= 2:
                muon_params.append(param)
    return muon_params


def gram_ns_shape_groups(params) -> List[tuple]:
    """
    Collapse Muon-eligible parameters into the (B, M, N) shapes that
    gram_newton_schulz will actually see at optimizer-step time, mirroring the
    grouping of Muon.step: B = number of params sharing the shape, conv filters
    (out, in, ...) flattened to (out, in*...). Used by gram_ns_triton.warmup_gram_ns
    to pre-tune the Triton kernels for exactly these shapes at startup.
    """
    groups = Counter()
    for p in params:
        if p.ndim >= 3:  # conv filter: (out, in, ...) -> (out, in*...)
            m, n = p.shape[0], p.numel() // p.shape[0]
        else:
            m, n = p.shape
        groups[(m, n)] += 1
    return [(count, m, n) for (m, n), count in sorted(groups.items())]


class Muon_AdamW(ChainedOptimizer):
    def __init__(self, model, lr=0.0005, weight_decay=0.0, muon_args={}, adamw_args={}, verbose=False):
        muon_params_id_set = set(id(p) for p in get_params_for_muon(model))
        spec_muon = OptimizerSpec(Muon, muon_args, lambda param: id(param) in muon_params_id_set)
        spec_adamw = OptimizerSpec(torch.optim.AdamW, adamw_args, None)
        specs = [spec_muon, spec_adamw]
        callback = None
        if verbose:
            callback = lambda p, spec_idx: print(
            f"Adding param {p.shape} to optimizer{spec_idx} {str(specs[spec_idx].class_type)}"
        )
        super().__init__(model.parameters(), specs, lr=lr, weight_decay=weight_decay, optimizer_selection_callback=callback)