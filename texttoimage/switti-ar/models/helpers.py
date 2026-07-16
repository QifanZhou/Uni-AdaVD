from __future__ import annotations

import torch
from torch import nn as nn
from torch.nn import functional as F


def _prefix_cumsum_via_matmul(x: torch.Tensor) -> torch.Tensor:
    """
    Deterministic prefix-sum alternative for CUDA where torch.cumsum may be disallowed under
    `torch.use_deterministic_algorithms(True)`.

    x: [..., V] float tensor
    returns: same shape, inclusive prefix sums along last dim.

    Note: this is O(V^2) and only intended for small V (Switti VQ vocab is 4096). It is still
    expensive but avoids calling the CUDA cumsum kernel.
    """
    V = x.shape[-1]
    # Lower-triangular (inclusive) matrix for prefix sums.
    # Use float32 for numerical stability.
    tri = torch.tril(torch.ones((V, V), device=x.device, dtype=torch.float32))
    x_f = x.float()
    # (..., V) @ (V, V) -> (..., V)
    return torch.matmul(x_f, tri)


def sample_with_top_k_top_p_(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    rng=None,
    num_samples=1,
) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(
            top_k, largest=True, sorted=False, dim=-1
        )[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    # Nucleus sampling (top-p). We implement a deterministic alternative to CUDA cumsum when strict
    # determinism is enabled. The fallback is slower (O(V^2)) but keeps runs reproducible.
    if 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        probs = sorted_logits.softmax(dim=-1)
        if torch.are_deterministic_algorithms_enabled() and probs.is_cuda:
            cdf = _prefix_cumsum_via_matmul(probs)
        else:
            cdf = probs.cumsum(dim=-1)
        sorted_idx_to_remove = cdf <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(
            sorted_idx_to_remove.scatter(
                sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove
            ),
            -torch.inf,
        )
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(
        logits_BlV.softmax(dim=-1).view(-1, V),
        num_samples=num_samples,
        replacement=replacement,
        generator=rng,
    ).view(B, l, num_samples)


def gumbel_softmax_with_rng(
    logits: torch.Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_(generator=rng)
        .log()
    )
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):  # taken from timm
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"(drop_prob=...)"
