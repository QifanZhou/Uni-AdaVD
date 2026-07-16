from __future__ import annotations

import math
import warnings

import os
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.functional import scaled_dot_product_attention  # q, k, v: BHLc

from models.helpers import DropPath
from models.rope import apply_rotary_emb

try:
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError:
    fused_mlp_func = None

# this file only provides the blocks used in Switti transformer
__all__ = ["FFN", "SwiGLUFFN", "RMSNorm", "AdaLNSelfCrossAttn", "AdaLNBeforeHead"]


try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ImportError:
    warnings.warn("Cannot import apex RMSNorm, switch to vanilla implementation")

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            """
            Initialize the RMSNorm normalization layer.

            Args:
                dim (int): The dimension of the input tensor.
                eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

            Attributes:
                eps (float): A small value added to the denominator for numerical stability.
                weight (nn.Parameter): Learnable scaling parameter.

            """
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def _norm(self, x):
            """
            Apply the RMSNorm normalization to the input tensor.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The normalized tensor.

            """
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            """
            Forward pass through the RMSNorm layer.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor after applying RMSNorm.

            """
            output = self._norm(x.float()).type_as(x)
            return output * self.weight


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        fused_if_available=True,
    ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(
                self.fused_mlp_func(
                    x=x,
                    weight1=self.fc1.weight,
                    weight2=self.fc2.weight,
                    bias1=self.fc1.bias,
                    bias2=self.fc2.bias,
                    activation="gelu_approx",
                    save_pre_act=self.training,
                    return_residual=False,
                    checkpoint_lvl=0,
                    heuristic=0,
                    process_group=None,
                )
            )
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))

    def extra_repr(self) -> str:
        return f"fused_mlp_func={self.fused_mlp_func is not None}"


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_mult: float = 8 / 3,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            ff_mult (float, optional): Custom multiplier for hidden dimension. Defaults to 4.
        """
        super().__init__()
        hidden_dim = int(dim * ff_mult)

        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.fused_mlp_func = None
        self._init()

    def _init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # @torch.compile
    def _forward_silu_gating(self, x_gate: torch.Tensor, x_up: torch.Tensor):
        return F.silu(x_gate) * x_up

    def forward(self, x: torch.Tensor):
        return self.down_proj(
            self._forward_silu_gating(self.gate_proj(x), self.up_proj(x))
        )

    def extra_repr(self) -> str:
        return f"fused_mlp_func={self.fused_mlp_func is not None}"


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        context_dim: int = 2048,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert attn_drop == 0.0

        self.num_heads, self.head_dim = (
            num_heads,
            embed_dim // num_heads,
        )
        self.qk_norm = qk_norm
        self.scale = 1 / math.sqrt(self.head_dim)

        self.q_norm = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.to_kv = nn.Linear(context_dim, embed_dim * 2, bias=True)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = (
            nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        )
        self.attn_drop = attn_drop

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

        # AdaVD (value-space orthogonal decomposition) config (disabled by default).
        # This is designed to mirror SD1-4 style cross-attn intervention:
        # - target context is preprocessed as last_subject and repeated to length 77
        # - only value `v` is modified (retain/erase), key `k` untouched
        self.adavd_mode: str = "original"  # original|retain|erase
        self.adavd_sigmoid_setting: tuple[float, float, float] | None = None  # (a,b,c)
        self.adavd_target_context: torch.Tensor | None = None  # [B, L_ctx, context_dim]
        # Multi-concept target contexts: list of [B, L_ctx, context_dim]. When set and mode is retain/erase,
        # span-subspace erasure is applied using all concepts jointly (SD1-4 multi-concept style).
        self.adavd_target_contexts: list[torch.Tensor] | None = None
        self.adavd_target_concept_names: list[str] | None = None
        self.adavd_record_target: bool = True
        self.adavd_cfg_active: bool = True  # True when batch is [cond, uncond] for CFG
        self.adavd_apply_to_uncond: bool = False  # SD1-4 aligned default: only modify text/cond branch
        self._adavd_cached_target_v: torch.Tensor | None = None  # [B, H, L_ctx, head_dim]
        self._adavd_cached_target_key: tuple | None = None
        self._adavd_cached_target_vs: list[torch.Tensor] | None = None  # list of [B,H,L,hd]
        self._adavd_cached_target_vs_key: tuple | None = None
        self.adavd_block_idx: int | None = None

        # Debug: cosine similarity stats (printed from forward, limited).
        self.adavd_debug_cos: bool = False
        self.adavd_debug_print_limit: int = 1
        self._adavd_debug_printed: int = 0
        self.adavd_debug_cos_log_path: str | None = None
        self.adavd_debug_cos_dump_tokens: bool = False
        self.adavd_debug_cos_dump_tokens_per_concept: bool = False

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    def set_adavd(
        self,
        mode: str,
        target_context: torch.Tensor | None,
        target_contexts: list[torch.Tensor] | None = None,
        target_concept_names: list[str] | None = None,
        sigmoid_setting: tuple[float, float, float] | None = None,
        record_target: bool = True,
        cfg_active: bool = True,
        apply_to_uncond: bool = False,
    ) -> None:
        mode = (mode or "original").strip().lower()
        if mode not in {"original", "retain", "erase"}:
            raise ValueError(f"Unsupported AdaVD mode: {mode}")
        self.adavd_mode = mode
        self.adavd_sigmoid_setting = sigmoid_setting
        self.adavd_record_target = bool(record_target)
        self.adavd_cfg_active = bool(cfg_active)
        self.adavd_target_context = target_context
        self.adavd_target_contexts = target_contexts
        self.adavd_target_concept_names = target_concept_names
        self.adavd_apply_to_uncond = bool(apply_to_uncond)
        self._adavd_cached_target_v = None
        self._adavd_cached_target_key = None
        self._adavd_cached_target_vs = None
        self._adavd_cached_target_vs_key = None

    def set_adavd_cfg_active(self, cfg_active: bool) -> None:
        self.adavd_cfg_active = bool(cfg_active)

    def set_adavd_debug(
        self,
        enable: bool,
        print_limit: int = 1,
        cos_log_path: str | None = None,
        dump_tokens: bool = False,
        dump_tokens_per_concept: bool = False,
    ) -> None:
        self.adavd_debug_cos = bool(enable)
        self.adavd_debug_print_limit = int(print_limit)
        self._adavd_debug_printed = 0
        self.adavd_debug_cos_log_path = cos_log_path
        self.adavd_debug_cos_dump_tokens = bool(dump_tokens)
        self.adavd_debug_cos_dump_tokens_per_concept = bool(dump_tokens_per_concept)

    def set_adavd_target_context(self, target_context: torch.Tensor | None) -> None:
        self.adavd_target_context = target_context
        self.adavd_target_contexts = None
        self._adavd_cached_target_v = None
        self._adavd_cached_target_key = None
        self._adavd_cached_target_vs = None
        self._adavd_cached_target_vs_key = None

    def set_adavd_target_contexts(self, target_contexts: list[torch.Tensor] | None) -> None:
        self.adavd_target_contexts = target_contexts
        # Keep single-context field in sync for safety.
        self.adavd_target_context = None
        self.adavd_target_concept_names = None
        self._adavd_cached_target_v = None
        self._adavd_cached_target_key = None
        self._adavd_cached_target_vs = None
        self._adavd_cached_target_vs_key = None

    def clear_adavd(self) -> None:
        self.set_adavd(mode="original", target_context=None, sigmoid_setting=None)

    @staticmethod
    def _adavd_sigmoid(x: torch.Tensor, setting: tuple[float, float, float] | None) -> torch.Tensor:
        if setting is None:
            return x
        a, b, c = setting
        return c / (1.0 + torch.exp(-a * (x - b)))

    def _adavd_get_target_v(self, target_context: torch.Tensor, context_L: int) -> torch.Tensor:
        # Cache per layer (depends on to_kv weights). Must match current batch size.
        B = target_context.shape[0]
        key = (B, context_L, target_context.device, target_context.dtype)
        if self._adavd_cached_target_v is not None and self._adavd_cached_target_key == key:
            return self._adavd_cached_target_v

        kv_t = self.to_kv(target_context).view(B, context_L, 2, -1)
        k_t, v_t = kv_t.permute(2, 0, 1, 3).unbind(dim=0)
        if self.qk_norm:
            k_t = self.k_norm(k_t)
        # Keep identical reshape path with normal context.
        _ = k_t.view(B, context_L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_t = v_t.view(B, context_L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.adavd_record_target:
            self._adavd_cached_target_v = v_t
            self._adavd_cached_target_key = key
        return v_t

    def _adavd_get_target_vs(self, target_contexts: list[torch.Tensor], context_L: int) -> list[torch.Tensor]:
        # Cache per layer (depends on to_kv weights). Must match current batch size.
        if len(target_contexts) == 0:
            raise ValueError("target_contexts is empty")
        B = target_contexts[0].shape[0]
        for tc in target_contexts:
            if tc.shape[0] != B:
                raise ValueError("All target_contexts must have the same batch size")
        key = (B, context_L, len(target_contexts), target_contexts[0].device, target_contexts[0].dtype)
        if self._adavd_cached_target_vs is not None and self._adavd_cached_target_vs_key == key:
            return self._adavd_cached_target_vs

        out: list[torch.Tensor] = []
        for tc in target_contexts:
            out.append(self._adavd_get_target_v(tc, context_L))

        if self.adavd_record_target:
            self._adavd_cached_target_vs = out
            self._adavd_cached_target_vs_key = key
        return out

    @staticmethod
    def _adavd_percentiles_approx(x: torch.Tensor) -> tuple[float, float, float]:
        # x is 1D float tensor on GPU. Use sort since it's small (<= B*L).
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            return (float("nan"), float("nan"), float("nan"))
        xs, _ = torch.sort(x)
        n = xs.numel()
        p10 = xs[int(0.10 * (n - 1))].item()
        p50 = xs[int(0.50 * (n - 1))].item()
        p90 = xs[int(0.90 * (n - 1))].item()
        return (p10, p50, p90)

    def _adavd_maybe_log_cos(self, cos_raw: torch.Tensor, cos_gated: torch.Tensor) -> None:
        if (not self.adavd_debug_cos) and (not self.adavd_debug_cos_dump_tokens):
            return
        if self._adavd_debug_printed >= max(self.adavd_debug_print_limit, 0):
            return

        # cos_*: [B, L]
        with torch.no_grad():
            # Full 77-token dump (raw cos-sim), written to a log file if provided.
            # Intended for checking sigmoid-b: compare raw cos distribution vs threshold.
            if self.adavd_debug_cos_dump_tokens and self.adavd_debug_cos_log_path is not None:
                v = cos_raw[0].detach().float().cpu().tolist()
                # Mark max token (excluding token0 by default since we don't modify token0 anyway).
                if len(v) > 1:
                    sub = v[1:]
                    max_rel = int(max(range(len(sub)), key=lambda i: sub[i]))
                    max_idx = max_rel + 1
                else:
                    max_idx = 0
                max_val = v[max_idx] if len(v) > 0 else float("nan")
                blk = self.adavd_block_idx
                blk_s = "?" if blk is None else str(blk)
                line = (
                    f"block={blk_s} mode={self.adavd_mode} "
                    f"sigmoid(a,b,c)={self.adavd_sigmoid_setting} "
                    f"max_idx={max_idx} max_val={max_val:.6f} "
                    f"cos_raw={v}\n"
                )
                try:
                    os.makedirs(os.path.dirname(self.adavd_debug_cos_log_path) or ".", exist_ok=True)
                    with open(self.adavd_debug_cos_log_path, "a", encoding="utf-8") as f:
                        f.write(line)
                except Exception as e:
                    print(f"[AdaVD cos] failed to write log to {self.adavd_debug_cos_log_path}: {e}")

            if not self.adavd_debug_cos:
                self._adavd_debug_printed += 1
                return

            # Skip token 0 (SOT) for readability; that's also what we keep untouched.
            if cos_raw.shape[1] > 1:
                raw = cos_raw[:, 1:].reshape(-1).float()
                gated = cos_gated[:, 1:].reshape(-1).float()
            else:
                raw = cos_raw.reshape(-1).float()
                gated = cos_gated.reshape(-1).float()

            raw_p10, raw_p50, raw_p90 = self._adavd_percentiles_approx(raw)
            gated_p10, gated_p50, gated_p90 = self._adavd_percentiles_approx(gated)

            blk = self.adavd_block_idx
            blk_s = "?" if blk is None else str(blk)
            print(
                f"[AdaVD cos] block={blk_s} "
                f"raw(mean={raw.mean().item():.4f}, min={raw.min().item():.4f}, p10={raw_p10:.4f}, "
                f"p50={raw_p50:.4f}, p90={raw_p90:.4f}, max={raw.max().item():.4f}) "
                f"gated(mean={gated.mean().item():.4f}, min={gated.min().item():.4f}, p10={gated_p10:.4f}, "
                f"p50={gated_p50:.4f}, p90={gated_p90:.4f}, max={gated.max().item():.4f}) "
                f"sigmoid(a,b,c)={self.adavd_sigmoid_setting}"
            )
            self._adavd_debug_printed += 1

    def _adavd_maybe_log_cos_span(self, cos_raw: torch.Tensor) -> None:
        # cos_raw: [B, L, n]
        if not self.adavd_debug_cos_dump_tokens_per_concept:
            return
        if self.adavd_debug_cos_log_path is None:
            return
        if self._adavd_debug_printed >= max(self.adavd_debug_print_limit, 0):
            return
        with torch.no_grad():
            v = cos_raw[0].detach().float().cpu()  # [L,n]
            L, n = v.shape
            names = self.adavd_target_concept_names or [f"concept_{i}" for i in range(n)]
            if len(names) != n:
                names = [f"concept_{i}" for i in range(n)]

            blk = self.adavd_block_idx
            blk_s = "?" if blk is None else str(blk)
            lines = []
            for j in range(n):
                vv = v[:, j].tolist()  # len L
                if L > 1:
                    sub = vv[1:]
                    max_rel = int(max(range(len(sub)), key=lambda i: sub[i]))
                    max_idx = max_rel + 1
                else:
                    max_idx = 0
                max_val = vv[max_idx] if len(vv) > 0 else float("nan")
                lines.append(
                    f"block={blk_s} mode={self.adavd_mode} concept={names[j]} "
                    f"sigmoid(a,b,c)={self.adavd_sigmoid_setting} "
                    f"max_idx={max_idx} max_val={max_val:.6f} "
                    f"cos_raw={vv}\n"
                )

            try:
                os.makedirs(os.path.dirname(self.adavd_debug_cos_log_path) or ".", exist_ok=True)
                with open(self.adavd_debug_cos_log_path, "a", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line)
            except Exception as e:
                print(f"[AdaVD cos] failed to write span log to {self.adavd_debug_cos_log_path}: {e}")
            self._adavd_debug_printed += 1

    def _adavd_decompose_value(self, v: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
        # v, target_v: [B, H, L, hd]
        input_dtype = v.dtype
        B, H, L, hd = v.shape
        v_flat = v.permute(0, 2, 1, 3).reshape(B, L, H * hd).float()
        t_flat = target_v.permute(0, 2, 1, 3).reshape(B, L, H * hd).float()

        dot1 = (t_flat * v_flat).sum(dim=-1)
        dot2 = torch.clamp((t_flat * t_flat).sum(dim=-1), min=1e-6)
        cos_raw = F.cosine_similarity(t_flat, v_flat, dim=-1)
        cos_gated = self._adavd_sigmoid(cos_raw, self.adavd_sigmoid_setting)
        self._adavd_maybe_log_cos(cos_raw, cos_gated)

        w = torch.nan_to_num(cos_gated * (dot1 / dot2), nan=0.0, posinf=0.0, neginf=0.0)
        if L > 0:
            w[:, 0] = 0.0

        erase = w.unsqueeze(-1) * t_flat
        retain = v_flat - erase
        out_flat = erase if self.adavd_mode == "erase" else retain
        return out_flat.view(B, L, H, hd).permute(0, 2, 1, 3).to(dtype=input_dtype)

    def _adavd_decompose_value_span(self, v: torch.Tensor, target_vs: list[torch.Tensor]) -> torch.Tensor:
        # v: [B,H,L,hd], target_vs: list of [B,H,L,hd], length = n_concepts
        input_dtype = v.dtype
        B, H, L, hd = v.shape
        n = len(target_vs)
        if n == 0:
            return v

        v_flat = v.permute(0, 2, 1, 3).reshape(B, L, H * hd).float()  # [B,L,D]
        t_stack = torch.stack(
            [tv.permute(0, 2, 1, 3).reshape(B, L, H * hd).float() for tv in target_vs],
            dim=2,
        )  # [B,L,n,D]

        # Cosine-sim gating per concept, per token (SD1-4 aligned).
        # cos_raw: [B,L,n]
        v_exp = v_flat.unsqueeze(2).expand(-1, -1, n, -1)
        cos_raw = F.cosine_similarity(t_stack, v_exp, dim=-1)
        cos_gated = self._adavd_sigmoid(cos_raw, self.adavd_sigmoid_setting)
        self._adavd_maybe_log_cos_span(cos_raw)

        # SD1-4 multi-concept path uses a Gram-Schmidt style projection_matrix + ortho_basis per token.
        # We implement an equivalent per-(B,L) Gram-Schmidt on the (gated) target vectors.
        t_g = cos_gated.unsqueeze(-1) * t_stack  # [B,L,n,D]

        # Flatten (B,L) so we can batch small-matrix ops.
        BL = B * L
        T = t_g.reshape(BL, n, -1)  # [BL,n,D]
        V = v_flat.reshape(BL, -1)  # [BL,D]

        # Gram-Schmidt orthogonalization (not normalized) on T along the concept axis.
        # Q[:,k,:] are orthogonal basis vectors spanning span(T).
        Q = torch.zeros_like(T)
        eps = 1e-6
        for k in range(n):
            vk = T[:, k, :]
            uk = vk
            for j in range(k):
                qj = Q[:, j, :]
                denom = (qj * qj).sum(dim=-1, keepdim=True).clamp_min(eps)
                proj = ((uk * qj).sum(dim=-1, keepdim=True) / denom) * qj
                uk = uk - proj
            Q[:, k, :] = uk

        # Compute coefficients using orthogonal basis (SD1-4's weight ~ dot(Q, V)/dot(Q,Q)).
        dot1 = (Q * V[:, None, :]).sum(dim=-1)  # [BL,n]
        dot2 = (Q * Q).sum(dim=-1).clamp_min(eps)  # [BL,n]
        weight = torch.nan_to_num(dot1 / dot2, nan=0.0, posinf=0.0, neginf=0.0)  # [BL,n]

        # Reconstruct projection (sum_k weight_k * Q_k).
        proj = (weight[:, :, None] * Q).sum(dim=1)  # [BL,D]
        proj = proj.reshape(B, L, -1)  # [B,L,D]

        if L > 0:
            proj[:, 0, :] = 0.0

        out_flat = proj if self.adavd_mode == "erase" else (v_flat - proj)
        return out_flat.view(B, L, H, hd).permute(0, 2, 1, 3).to(dtype=input_dtype)

    def forward(self, x, context, context_attn_bias=None, freqs_cis=None):
        B, L, C = x.shape
        context_B, context_L, context_C = context.shape
        assert B == context_B

        q = self.to_q(x).view(B, L, -1)  # BLD , self.num_heads, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)

        q = q.view(B, L, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # BHLc

        if self.cached_k is None:
            # not using caches or first scale inference
            kv = self.to_kv(context).view(B, context_L, 2, -1)  # qkv: BL3D
            k, v = kv.permute(2, 0, 1, 3).unbind(dim=0)  # q or k or v: BLHD

            if self.qk_norm:
                k = self.k_norm(k)

            k = k.view(B, context_L, self.num_heads, self.head_dim)
            k = k.permute(0, 2, 1, 3)  # BHLc

            v = v.view(B, context_L, self.num_heads, self.head_dim)
            v = v.permute(0, 2, 1, 3)  # BHLc

            if self.caching:
                self.cached_k = k
                self.cached_v = v
        else:
            k = self.cached_k
            v = self.cached_v

        if self.adavd_mode in {"retain", "erase"}:
            # Do not mutate cached_v in-place.
            v_work = v.clone()
            if self.adavd_target_contexts is not None:
                target_vs = self._adavd_get_target_vs(self.adavd_target_contexts, context_L)
                if self.adavd_cfg_active and B % 2 == 0 and (not self.adavd_apply_to_uncond):
                    cond_bs = B // 2
                    v_work[:cond_bs] = self._adavd_decompose_value_span(
                        v_work[:cond_bs], [tv[:cond_bs] for tv in target_vs]
                    )
                else:
                    v_work = self._adavd_decompose_value_span(v_work, target_vs)
                v = v_work
            else:
                if self.adavd_target_context is None:
                    raise ValueError("AdaVD enabled but target_context/target_contexts is None.")
                target_v = self._adavd_get_target_v(self.adavd_target_context, context_L)
                if self.adavd_cfg_active and B % 2 == 0 and (not self.adavd_apply_to_uncond):
                    cond_bs = B // 2
                    v_work[:cond_bs] = self._adavd_decompose_value(v_work[:cond_bs], target_v[:cond_bs])
                else:
                    v_work = self._adavd_decompose_value(v_work, target_v)
                v = v_work

        if context_attn_bias is not None:
            context_attn_bias = rearrange(context_attn_bias, "b j -> b 1 1 j")

        dropout_p = self.attn_drop if self.training else 0.0
        out = (
            scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                attn_mask=context_attn_bias,
                dropout_p=dropout_p,
            )
            .transpose(1, 2)
            .reshape(B, L, C)
        )

        return self.proj_drop(self.proj(out))


class SelfAttention(nn.Module):
    def __init__(
        self,
        block_idx: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = (
            block_idx,
            num_heads,
            embed_dim // num_heads,
        )
        self.qk_norm = qk_norm
        self.scale = 1 / math.sqrt(self.head_dim)

        self.q_norm = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)

        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = (
            nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        )
        self.attn_drop = attn_drop

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias, freqs_cis: torch.Tensor = None):
        B, L, C = x.shape

        qkv = self.to_qkv(x).view(B, L, 3, -1)
        q, k, v = qkv.permute(2, 0, 1, 3).unbind(dim=0)  # q or k or v: BLD

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.view(B, L, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # BHLc
        k = k.view(B, L, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # BHLc
        v = v.view(B, L, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # BHLc
        dim_cat = 2

        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis=freqs_cis)
            k = apply_rotary_emb(k, freqs_cis=freqs_cis)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0
        out = (
            scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                attn_mask=attn_bias,
                dropout_p=dropout_p,
            )
            .transpose(1, 2)
            .reshape(B, L, C)
        )

        return self.proj_drop(self.proj(out))

    def extra_repr(self) -> str:
        return f"attn_l2_norm={self.qk_norm}"


class AdaLNSelfCrossAttn(nn.Module):
    def __init__(
        self,
        block_idx,
        last_drop_p,
        embed_dim,
        cond_dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        qk_norm=False,
        context_dim=None,
        use_swiglu_ffn=False,
        norm_eps=1e-6,
        use_crop_cond=False,
    ):
        super().__init__()
        assert attn_drop == 0.0
        assert qk_norm

        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn = SelfAttention(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
        )

        if context_dim:
            self.cross_attn = CrossAttention(
                embed_dim=embed_dim,
                context_dim=context_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=drop,
                qk_norm=qk_norm,
            )
        else:
            self.cross_attn = None

        if use_swiglu_ffn:
            self.ffn = SwiGLUFFN(dim=embed_dim)
        else:
            self.ffn = FFN(
                in_features=embed_dim,
                hidden_features=round(embed_dim * mlp_ratio),
                drop=drop,
            )

        self.self_attention_norm1 = RMSNorm(embed_dim, eps=norm_eps)
        self.self_attention_norm2 = RMSNorm(embed_dim, eps=norm_eps)
        self.cross_attention_norm1 = RMSNorm(embed_dim, eps=norm_eps)
        self.cross_attention_norm2 = RMSNorm(embed_dim, eps=norm_eps)

        self.ffn_norm1 = RMSNorm(embed_dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(embed_dim, eps=norm_eps)

        self.attention_y_norm = RMSNorm(context_dim, eps=norm_eps)

        # AdaLN
        lin = nn.Linear(cond_dim, 6 * embed_dim)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)

        self.fused_add_norm_fn = None
        
        self.use_crop_cond = use_crop_cond
        if use_crop_cond:
            self.crop_cond_scales = nn.Parameter(torch.zeros(1, cond_dim))

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(
        self,
        x,
        cond_BD,
        attn_bias,
        crop_cond=None,
        context=None,
        context_attn_bias=None,
        freqs_cis=None,
    ):  # C: embed_dim, D: cond_dim
        
        if self.use_crop_cond:
            assert crop_cond is not None
            cond_BD = cond_BD + self.crop_cond_scales * crop_cond
            
        gamma1, gamma2, scale1, scale2, shift1, shift2 = (
            self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        )
        x = x + self.self_attention_norm2(
            self.attn(
                self.self_attention_norm1(x).mul(scale1.add(1)).add(shift1),
                attn_bias=attn_bias,
                freqs_cis=freqs_cis,
            )
        ).mul(gamma1)
        if context is not None:
            x = x + self.cross_attention_norm2(
                self.cross_attn(
                    self.cross_attention_norm1(x),
                    self.attention_y_norm(context),
                    context_attn_bias=context_attn_bias,
                    freqs_cis=freqs_cis,
                )
            )
        x = x + self.ffn_norm2(
            self.ffn(self.ffn_norm1(x).mul(scale2.add(1)).add(shift2))
        ).mul(gamma2)
        return x


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):  # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2 * C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
