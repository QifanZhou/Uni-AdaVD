import os
import re
import random
import argparse
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
if _alloc_conf and "expandable_segments" in _alloc_conf:
    _norm = _alloc_conf.replace(";", ",")
    _parts = [p.strip() for p in _norm.split(",") if p.strip()]
    _parts = [p for p in _parts if not p.split(":", 1)[0].strip().startswith("expandable_segments")]
    if _parts:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(_parts)
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

print("Importing dependencies...", flush=True)
import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from diffusers.models.attention_processor import FluxAttnProcessor2_0

from template import template_dict
print("Dependencies imported.", flush=True)

INSTANCE_FIXED_SIGMOID_B_VALUES_19 = [
    0.61, 0.43, 0.58, 0.49, 0.46, 0.41, 0.42, 0.42, 0.42, 0.42,
    0.42, 0.35, 0.35, 0.42, 0.42, 0.43, 0.42, 0.43, 0.34,
]

CELEBRITY_FIXED_SIGMOID_B_VALUES_19 = [
    0.65, 0.53, 0.61, 0.55, 0.55, 0.58, 0.4, 0.55, 0.5, 0.53,
    0.5, 0.37, 0.4, 0.42, 0.36, 0.4, 0.4, 0.4, 0.4,
]

STYLE_FIXED_SIGMOID_B_VALUES_19 = [
    0.41, 0.43, 0.55, 0.45, 0.53, 0.52, 0.46, 0.5, 0.53, 0.56,
    0.55, 0.6, 0.49, 0.5, 0.39, 0.5, 0.5, 0.45, 0.5,
]

NUDITY_FIXED_SIGMOID_B_VALUES_19 = [
    0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
    0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
]
def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_cast_tensor(tensor, dtype):
    if not torch.is_floating_point(tensor):
        return tensor.to(dtype=dtype)

    work = tensor.to(dtype=torch.float32)
    finite_mask = torch.isfinite(work)
    if not finite_mask.all():
        work = torch.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)

    finfo = torch.finfo(dtype)
    work = work.clamp(min=finfo.min, max=finfo.max)
    return work.to(dtype=dtype)


class JointAttnProcessor:
    _timestep_counter = {}
    _cached_sigmoid_b = {}
    _use_cached_sigmoid_b = False

    @classmethod
    def reset_phase_cache(cls):
        cls._timestep_counter = {}
        cls._cached_sigmoid_b = {}
        cls._use_cached_sigmoid_b = False

    @classmethod
    def enable_sigmoid_b_cache(cls):
        cls._use_cached_sigmoid_b = True

    @classmethod
    def disable_sigmoid_b_cache(cls):
        cls._use_cached_sigmoid_b = False
        cls._cached_sigmoid_b = {}

    @classmethod
    def reset_timestep_counter(cls):
        cls._timestep_counter = {}

    def __init__(
        self,
        sigmoid_setting=None,
        layer_index=None,
        layer_sigmoid_b_values=None,
        phase='early',
        num_targets=1,
        **kwargs,
    ):
        self.sigmoid_setting = sigmoid_setting
        self.layer_index = layer_index
        self.layer_sigmoid_b_values = layer_sigmoid_b_values
        self.phase = phase
        self.num_targets = max(1, int(num_targets)) if num_targets is not None else 1

    def sigmoid(self, x, setting):
        a, b, c = setting
        return c / (1 + torch.exp(-a * (x - b)))

    def _phase_info(self, current_timestep):
        if current_timestep < 10:
            return 'early'
        if current_timestep < 20:
            return 'mid'
        return 'late'

    def _normalize_phase(self, phase):
        if not phase:
            return ''
        alias = {
            'eraly': 'early',
        }
        phase = str(phase).strip().lower()
        return alias.get(phase, phase)

    def _phase_in_range(self, phase, current_timestep):
        phase = (phase or '').strip().lower()
        if not phase:
            return False
        if '-' in phase:
            parts = [p.strip() for p in phase.split('-', 1)]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start = int(parts[0])
                end = int(parts[1])
                return start <= current_timestep <= end
        return False

    def _should_erase_in_phase(self, current_phase):
        phase = self._normalize_phase(self.phase)
        current_phase = self._normalize_phase(current_phase)
        if phase == 'all':
            return True
        if '+' in phase:
            for item in phase.split('+'):
                item = self._normalize_phase(item)
                if self._phase_in_range(item, current_timestep=JointAttnProcessor._timestep_counter.get(self.layer_index, 0)):
                    return True
                if current_phase == item:
                    return True
            return False
        if self._phase_in_range(phase, current_timestep=JointAttnProcessor._timestep_counter.get(self.layer_index, 0)):
            return True
        return current_phase == phase

    def _normalized_abl_type(self):
        return 'text'

    def _need_decomp(self, target_name):
        return target_name == 'value'

    def _select_dynamic_sigmoid_b_reference(self, cos_sim_raw):
        length = int(cos_sim_raw.shape[0])
        if length <= 0:
            return 0.0, 1
        reference_rank = 2 if length >= 2 else 1
        ref_values, _ = torch.topk(cos_sim_raw, k=reference_rank)
        return float(ref_values[reference_rank - 1].item()), reference_rank

    def record_ortho_decomp(self, target_value, current_value, sot_idx=None):
        erase_record, retain_record = [], []

        if self.layer_index not in JointAttnProcessor._timestep_counter:
            JointAttnProcessor._timestep_counter[self.layer_index] = 0
        current_timestep = JointAttnProcessor._timestep_counter[self.layer_index]
        current_phase = self._phase_info(current_timestep)
        should_erase = self._should_erase_in_phase(current_phase)

        for tar_record, pro_record in zip(target_value, current_value):
            input_dtype = tar_record.dtype
            L = tar_record.shape[1]
            num_targets = int(tar_record.shape[0])

            tar_float = tar_record.to(dtype=torch.float32)
            pro_float = pro_record.to(dtype=torch.float32)

            mask = torch.ones((L,), device=tar_record.device, dtype=torch.float32)
            if sot_idx is not None and 0 <= sot_idx < mask.shape[0]:
                mask[sot_idx] = 0

            if num_targets == 1:
                tar_flat = tar_float.permute(1, 0, 2).reshape(L, -1)
                pro_flat = pro_float.permute(1, 0, 2).reshape(L, -1)
                dot1 = (tar_flat * pro_flat).sum(-1)
                dot2 = torch.clamp((tar_flat * tar_flat).sum(-1), min=1e-6)
                cos_sim_raw = torch.cosine_similarity(tar_flat, pro_flat, dim=-1)
            else:
                tar_tokens = tar_float.permute(1, 0, 2)
                pro_tokens = pro_float.permute(1, 0, 2)[:, 0, :]
                cos_sim_all = torch.cosine_similarity(
                    tar_tokens,
                    pro_tokens.unsqueeze(1),
                    dim=-1,
                )
                cos_sim_raw = cos_sim_all.max(dim=1).values
            token_indices = torch.arange(L, device=cos_sim_raw.device)
            token_values = cos_sim_raw

            erase_mask = torch.zeros_like(cos_sim_raw)
            coeffs = []
            dynamic_sigmoid_b = None
            use_fixed_coeff = self.sigmoid_setting[0] == 0

            if should_erase:
                if use_fixed_coeff:
                    for idx in token_indices:
                        erase_mask[idx] = 0.0
                        coeffs.append(0.0)
                else:
                    reference_cosim, _ = self._select_dynamic_sigmoid_b_reference(cos_sim_raw)
                    cache_key = (self.layer_index, current_timestep)
                    if self.layer_sigmoid_b_values and self.layer_index is not None and self.layer_index < len(self.layer_sigmoid_b_values):
                        dynamic_sigmoid_b = self.layer_sigmoid_b_values[self.layer_index]
                    elif JointAttnProcessor._use_cached_sigmoid_b and cache_key in JointAttnProcessor._cached_sigmoid_b:
                        dynamic_sigmoid_b = JointAttnProcessor._cached_sigmoid_b[cache_key]
                    else:
                        dynamic_sigmoid_b = reference_cosim
                        JointAttnProcessor._cached_sigmoid_b[cache_key] = dynamic_sigmoid_b

                    dynamic_setting = (self.sigmoid_setting[0], dynamic_sigmoid_b, self.sigmoid_setting[2])
                    for idx, val in zip(token_indices, token_values):
                        sigmoid_val = self.sigmoid(val, dynamic_setting)
                        erase_mask[idx] = sigmoid_val
                        coeffs.append(float(sigmoid_val.item()))
            else:
                coeffs = [0.0] * L

            if num_targets == 1:
                coeff = erase_mask * mask * (dot1 / dot2)
                era_record = coeff.unsqueeze(0).unsqueeze(-1) * tar_float
            else:
                tar_tokens = tar_float.permute(1, 0, 2)
                pro_tokens = pro_float.permute(1, 0, 2)[:, 0, :]
                gram = torch.matmul(tar_tokens, tar_tokens.transpose(-1, -2))
                eye = torch.eye(num_targets, device=gram.device, dtype=gram.dtype).unsqueeze(0)
                gram = gram + 1e-6 * eye
                rhs = torch.matmul(tar_tokens, pro_tokens.unsqueeze(-1))
                coef = torch.linalg.solve(gram, rhs)
                proj = torch.matmul(tar_tokens.transpose(-1, -2), coef).squeeze(-1)
                coeff = erase_mask * mask
                proj = proj * coeff.unsqueeze(-1)
                era_record = proj.unsqueeze(0)
            ret_record = pro_float - era_record

            erase_record.append(
                safe_cast_tensor(
                    era_record,
                    input_dtype,
                )
            )
            retain_record.append(
                safe_cast_tensor(
                    ret_record,
                    input_dtype,
                )
            )

        JointAttnProcessor._timestep_counter[self.layer_index] += 1

        retain_record = torch.stack(retain_record, dim=0)
        erase_record = torch.stack(erase_record, dim=0)
        return erase_record, retain_record

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            sample_bs, sample_c, sample_h, sample_w = hidden_states.shape
            hidden_states = hidden_states.view(sample_bs, sample_c, sample_h * sample_w).transpose(1, 2)

        context_input_ndim = None
        if encoder_hidden_states is not None:
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                ctx_bs, ctx_c, ctx_h, ctx_w = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(ctx_bs, ctx_c, ctx_h * ctx_w).transpose(1, 2)

        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]
        query_proj = attn.to_q(hidden_states)
        image_key_proj = attn.to_k(hidden_states)
        image_value_proj = attn.to_v(hidden_states)

        inner_dim = image_key_proj.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)

        if encoder_hidden_states is not None:
            txt_len = encoder_hidden_states.shape[1]
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            current_abl_type = self._normalized_abl_type()
            group = 1 + self.num_targets
            if batch_size % group == 0:
                bs_pair = batch_size // group
                prompt_idx = slice(0, bs_pair)
                target_start = bs_pair
                target_end = bs_pair * group
                sot_idx = 0 if current_abl_type in ('all', 'text') else None

                def _reshape_targets(tensor):
                    target_block = tensor[target_start:target_end]
                    if target_block.numel() == 0:
                        return None
                    target_block = target_block.view(self.num_targets, bs_pair, *tensor.shape[1:])
                    return target_block.permute(1, 0, 2, 3).contiguous()

                if self._need_decomp('key'):
                    if current_abl_type == 'all':
                        combined_key = torch.cat([encoder_key, image_key_proj], dim=1)
                        prompt_key = combined_key[prompt_idx].unsqueeze(1)
                        target_key = _reshape_targets(combined_key)
                        if target_key is not None:
                            _, retain_record = self.record_ortho_decomp(target_key, prompt_key, sot_idx=sot_idx)
                            combined_key[prompt_idx] = retain_record[:, 0]
                            encoder_key = combined_key[:, :txt_len, :]
                            image_key_proj = combined_key[:, txt_len:, :]
                    elif current_abl_type == 'text':
                        prompt_key = encoder_key[prompt_idx].unsqueeze(1)
                        target_key = _reshape_targets(encoder_key)
                        if target_key is not None:
                            _, retain_record = self.record_ortho_decomp(target_key, prompt_key, sot_idx=sot_idx)
                            encoder_key[prompt_idx] = retain_record[:, 0]
                    elif current_abl_type == 'image':
                        prompt_key = image_key_proj[prompt_idx].unsqueeze(1)
                        target_key = _reshape_targets(image_key_proj)
                        if target_key is not None:
                            _, retain_record = self.record_ortho_decomp(target_key, prompt_key, sot_idx=sot_idx)
                            image_key_proj[prompt_idx] = retain_record[:, 0]

                if self._need_decomp('value'):
                    if current_abl_type == 'all':
                        combined_value = torch.cat([encoder_value, image_value_proj], dim=1)
                        prompt_value = combined_value[prompt_idx].unsqueeze(1)
                        target_value = _reshape_targets(combined_value)
                        if target_value is not None:
                            _, retain_record = self.record_ortho_decomp(target_value, prompt_value, sot_idx=sot_idx)
                            combined_value[prompt_idx] = retain_record[:, 0]
                            encoder_value = combined_value[:, :txt_len, :]
                            image_value_proj = combined_value[:, txt_len:, :]
                    elif current_abl_type == 'text':
                        prompt_value = encoder_value[prompt_idx].unsqueeze(1)
                        target_value = _reshape_targets(encoder_value)
                        if target_value is not None:
                            _, retain_record = self.record_ortho_decomp(target_value, prompt_value, sot_idx=sot_idx)
                            encoder_value[prompt_idx] = retain_record[:, 0]
                    elif current_abl_type == 'image':
                        prompt_value = image_value_proj[prompt_idx].unsqueeze(1)
                        target_value = _reshape_targets(image_value_proj)
                        if target_value is not None:
                            _, retain_record = self.record_ortho_decomp(target_value, prompt_value, sot_idx=sot_idx)
                            image_value_proj[prompt_idx] = retain_record[:, 0]

            encoder_query = encoder_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_key = encoder_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_value = encoder_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            image_key = image_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            image_value = image_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_k is not None:
                image_key = attn.norm_k(image_key)
            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, image_key], dim=2)
            value = torch.cat([encoder_value, image_value], dim=2)

            if image_rotary_emb is not None:
                query, key = apply_rope(query, key, image_rotary_emb)

            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = safe_cast_tensor(
                hidden_states,
                query.dtype,
            )

            encoder_hidden_states, hidden_states = (
                hidden_states[:, :txt_len],
                hidden_states[:, txt_len:],
            )

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = safe_cast_tensor(
                hidden_states,
                query.dtype,
            )
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            encoder_hidden_states = safe_cast_tensor(
                encoder_hidden_states,
                query.dtype,
            )

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(sample_bs, sample_c, sample_h, sample_w)
            if context_input_ndim == 4:
                encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(ctx_bs, ctx_c, ctx_h, ctx_w)

            return hidden_states, encoder_hidden_states
        image_key = image_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        image_value = image_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_k is not None:
            image_key = attn.norm_k(image_key)
        if image_rotary_emb is not None:
            query, image_key = apply_rope(query, image_key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, image_key, image_value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = safe_cast_tensor(
            hidden_states,
            query.dtype,
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = safe_cast_tensor(
            hidden_states,
            query.dtype,
        )
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(sample_bs, sample_c, sample_h, sample_w)

        return hidden_states
class SingleAttnProcessor:
    _timestep_counter = {}
    _cached_sigmoid_b = {}
    _use_cached_sigmoid_b = False

    @classmethod
    def reset_phase_cache(cls):
        cls._timestep_counter = {}
        cls._cached_sigmoid_b = {}
        cls._use_cached_sigmoid_b = False

    @classmethod
    def enable_sigmoid_b_cache(cls):
        cls._use_cached_sigmoid_b = True

    @classmethod
    def disable_sigmoid_b_cache(cls):
        cls._use_cached_sigmoid_b = False
        cls._cached_sigmoid_b = {}

    @classmethod
    def reset_timestep_counter(cls):
        cls._timestep_counter = {}

    def __init__(
        self,
        sigmoid_setting=None,
        layer_index=None,
        layer_sigmoid_b_values=None,
        phase='early',
        num_targets=1,
        **kwargs,
    ):
        self.sigmoid_setting = sigmoid_setting
        self.layer_index = layer_index
        self.layer_sigmoid_b_values = layer_sigmoid_b_values
        self.phase = phase
        self.num_targets = max(1, int(num_targets)) if num_targets is not None else 1

    def sigmoid(self, x, setting):
        a, b, c = setting
        return c / (1 + torch.exp(-a * (x - b)))

    def _phase_info(self, current_timestep):
        if current_timestep < 10:
            return 'early'
        if current_timestep < 20:
            return 'mid'
        return 'late'

    def _normalize_phase(self, phase):
        if not phase:
            return ''
        alias = {
            'eraly': 'early',
        }
        phase = str(phase).strip().lower()
        return alias.get(phase, phase)

    def _phase_in_range(self, phase, current_timestep):
        phase = (phase or '').strip().lower()
        if not phase:
            return False
        if '-' in phase:
            parts = [p.strip() for p in phase.split('-', 1)]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start = int(parts[0])
                end = int(parts[1])
                return start <= current_timestep <= end
        return False

    def _should_erase_in_phase(self, current_phase):
        phase = self._normalize_phase(self.phase)
        current_phase = self._normalize_phase(current_phase)
        if phase == 'all':
            return True
        if '+' in phase:
            for item in phase.split('+'):
                item = self._normalize_phase(item)
                if self._phase_in_range(item, current_timestep=SingleAttnProcessor._timestep_counter.get(self.layer_index, 0)):
                    return True
                if current_phase == item:
                    return True
            return False
        if self._phase_in_range(phase, current_timestep=SingleAttnProcessor._timestep_counter.get(self.layer_index, 0)):
            return True
        return current_phase == phase

    def _normalized_abl_type(self):
        return 'text'

    def _need_decomp(self, target_name):
        return target_name == 'value'

    def _select_dynamic_sigmoid_b_reference(self, cos_sim_raw):
        length = int(cos_sim_raw.shape[0])
        if length <= 0:
            return 0.0, 1
        reference_rank = 2 if length >= 2 else 1
        ref_values, _ = torch.topk(cos_sim_raw, k=reference_rank)
        return float(ref_values[reference_rank - 1].item()), reference_rank

    def record_ortho_decomp(self, target_value, current_value, sot_idx=None):
        erase_record, retain_record = [], []

        if self.layer_index not in SingleAttnProcessor._timestep_counter:
            SingleAttnProcessor._timestep_counter[self.layer_index] = 0
        current_timestep = SingleAttnProcessor._timestep_counter[self.layer_index]
        current_phase = self._phase_info(current_timestep)
        should_erase = self._should_erase_in_phase(current_phase)

        for tar_record, pro_record in zip(target_value, current_value):
            input_dtype = tar_record.dtype
            L = tar_record.shape[1]
            num_targets = int(tar_record.shape[0])

            tar_float = tar_record.to(dtype=torch.float32)
            pro_float = pro_record.to(dtype=torch.float32)

            mask = torch.ones((L,), device=tar_record.device, dtype=torch.float32)
            if sot_idx is not None and 0 <= sot_idx < mask.shape[0]:
                mask[sot_idx] = 0

            if num_targets == 1:
                tar_flat = tar_float.permute(1, 0, 2).reshape(L, -1)
                pro_flat = pro_float.permute(1, 0, 2).reshape(L, -1)
                dot1 = (tar_flat * pro_flat).sum(-1)
                dot2 = torch.clamp((tar_flat * tar_flat).sum(-1), min=1e-6)
                cos_sim_raw = torch.cosine_similarity(tar_flat, pro_flat, dim=-1)
            else:
                tar_tokens = tar_float.permute(1, 0, 2)
                pro_tokens = pro_float.permute(1, 0, 2)[:, 0, :]
                cos_sim_all = torch.cosine_similarity(
                    tar_tokens,
                    pro_tokens.unsqueeze(1),
                    dim=-1,
                )
                cos_sim_raw = cos_sim_all.max(dim=1).values
            token_indices = torch.arange(L, device=cos_sim_raw.device)
            token_values = cos_sim_raw

            erase_mask = torch.zeros_like(cos_sim_raw)
            coeffs = []
            dynamic_sigmoid_b = None
            use_fixed_coeff = self.sigmoid_setting[0] == 0

            if should_erase:
                if use_fixed_coeff:
                    for idx in token_indices:
                        erase_mask[idx] = 0.0
                        coeffs.append(0.0)
                else:
                    reference_cosim, _ = self._select_dynamic_sigmoid_b_reference(cos_sim_raw)
                    cache_key = (self.layer_index, current_timestep)
                    if self.layer_sigmoid_b_values and self.layer_index is not None and self.layer_index < len(self.layer_sigmoid_b_values):
                        dynamic_sigmoid_b = self.layer_sigmoid_b_values[self.layer_index]
                    elif SingleAttnProcessor._use_cached_sigmoid_b and cache_key in SingleAttnProcessor._cached_sigmoid_b:
                        dynamic_sigmoid_b = SingleAttnProcessor._cached_sigmoid_b[cache_key]
                    else:
                        dynamic_sigmoid_b = reference_cosim
                        SingleAttnProcessor._cached_sigmoid_b[cache_key] = dynamic_sigmoid_b

                    dynamic_setting = (self.sigmoid_setting[0], dynamic_sigmoid_b, self.sigmoid_setting[2])
                    for idx, val in zip(token_indices, token_values):
                        sigmoid_val = self.sigmoid(val, dynamic_setting)
                        erase_mask[idx] = sigmoid_val
                        coeffs.append(float(sigmoid_val.item()))
            else:
                coeffs = [0.0] * L

            if num_targets == 1:
                coeff = erase_mask * mask * (dot1 / dot2)
                era_record = coeff.unsqueeze(0).unsqueeze(-1) * tar_float
            else:
                tar_tokens = tar_float.permute(1, 0, 2)
                pro_tokens = pro_float.permute(1, 0, 2)[:, 0, :]
                gram = torch.matmul(tar_tokens, tar_tokens.transpose(-1, -2))
                eye = torch.eye(num_targets, device=gram.device, dtype=gram.dtype).unsqueeze(0)
                gram = gram + 1e-6 * eye
                rhs = torch.matmul(tar_tokens, pro_tokens.unsqueeze(-1))
                coef = torch.linalg.solve(gram, rhs)
                proj = torch.matmul(tar_tokens.transpose(-1, -2), coef).squeeze(-1)
                coeff = erase_mask * mask
                proj = proj * coeff.unsqueeze(-1)
                era_record = proj.unsqueeze(0)
            ret_record = pro_float - era_record

            erase_record.append(
                safe_cast_tensor(
                    era_record,
                    input_dtype,
                )
            )
            retain_record.append(
                safe_cast_tensor(
                    ret_record,
                    input_dtype,
                )
            )

        SingleAttnProcessor._timestep_counter[self.layer_index] += 1

        retain_record = torch.stack(retain_record, dim=0)
        erase_record = torch.stack(erase_record, dim=0)
        return erase_record, retain_record

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        current_abl_type = self._normalized_abl_type()
        group = 1 + self.num_targets
        if batch_size % group == 0 and current_abl_type in ('all', 'image'):
            bs_pair = batch_size // group
            prompt_idx = slice(0, bs_pair)
            target_start = bs_pair
            target_end = bs_pair * group

            def _reshape_targets(tensor):
                target_block = tensor[target_start:target_end]
                if target_block.numel() == 0:
                    return None
                target_block = target_block.view(self.num_targets, bs_pair, *tensor.shape[1:])
                return target_block.permute(1, 0, 2, 3).contiguous()

            if self._need_decomp('key'):
                prompt_key = key[prompt_idx].unsqueeze(1)
                target_key = _reshape_targets(key)
                if target_key is not None:
                    _, retain_record = self.record_ortho_decomp(target_key, prompt_key, sot_idx=None)
                    key[prompt_idx] = retain_record[:, 0]

            if self._need_decomp('value'):
                prompt_value = value[prompt_idx].unsqueeze(1)
                target_value = _reshape_targets(value)
                if target_value is not None:
                    _, retain_record = self.record_ortho_decomp(target_value, prompt_value, sot_idx=None)
                    value[prompt_idx] = retain_record[:, 0]

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query, key = apply_rope(query, key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = safe_cast_tensor(
            hidden_states,
            query.dtype,
        )

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        return hidden_states
def set_attenprocessor(
    pipe,
    joint_layers=None,
    sigmoid_setting=None,
    layer_sigmoid_b_values=None,
    phase='early',
    num_targets=1,
):
    if joint_layers is None:
        joint_layers = list(range(19))

    for idx, block in enumerate(pipe.transformer.transformer_blocks):
        if idx in joint_layers:
            block.attn.set_processor(
                JointAttnProcessor(
                    sigmoid_setting=sigmoid_setting,
                    layer_index=idx,
                    layer_sigmoid_b_values=layer_sigmoid_b_values,
                    phase=phase,
                    num_targets=num_targets,
                )
            )

    return pipe

class FluxAdaVD:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.args.save_root, exist_ok=True)

        self.pipe_dtype = self._resolve_pipe_dtype()
        print(f"Loading FLUX model from: {self.args.sd_ckpt}", flush=True)
        print(f"Using pipeline dtype: {self.args.torch_dtype} -> {self.pipe_dtype}", flush=True)
        self.pipe = FluxPipeline.from_pretrained(
            self.args.sd_ckpt,
            torch_dtype=self.pipe_dtype,
        )
        print("Moving pipeline to device...", flush=True)
        self.pipe.to(self.device)
        print("Model loaded.", flush=True)

    def _resolve_pipe_dtype(self):
        raw = (getattr(self.args, 'torch_dtype', 'auto') or 'auto').strip().lower()
        mapping = {
            'fp16': torch.float16,
            'float16': torch.float16,
            'half': torch.float16,
            'bf16': torch.bfloat16,
            'bfloat16': torch.bfloat16,
            'fp32': torch.float32,
            'float32': torch.float32,
            'float': torch.float32,
        }
        if raw != 'auto':
            if raw not in mapping:
                raise ValueError(f"Unsupported --torch_dtype: {self.args.torch_dtype}")
            return mapping[raw]

        if self.device.type == 'cuda':
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
            return torch.float16

        return torch.float32

    def _stable_seed(self, key):
        digest = hashlib.md5(key.encode()).hexdigest()[:8]
        return int(digest, 16)

    def _seed_from_value(self, value):
        if value is None or pd.isna(value):
            return None
        try:
            seed_int = int(float(value))
        except Exception:
            return None
        return seed_int & 0x7FFFFFFFFFFFFFFF

    def _csv_seed_from_row(self, row):
        for col in (
            'seed',
            'Seed',
            'sd_seed',
            'SD_Seed',
            'evaluation_seed',
            'Evaluation_Seed',
            'eval_seed',
            'Eval_Seed',
        ):
            if col in row.index:
                seed_val = self._seed_from_value(row[col])
                if seed_val is not None:
                    return seed_val
        return None

    def _parse_layer_sigmoid_b_values(self):
        target_list = self._parse_target_concepts(self.args.target_concept)
        target = (target_list[0] if target_list else '').strip().lower()
        if target in ('nudity', 'nsfw'):
            return NUDITY_FIXED_SIGMOID_B_VALUES_19.copy()

        erase_type = (self.args.erase_type or '').strip().lower()
        if erase_type == 'celebrity':
            return CELEBRITY_FIXED_SIGMOID_B_VALUES_19.copy()
        if erase_type == 'instance':
            return INSTANCE_FIXED_SIGMOID_B_VALUES_19.copy()
        if erase_type in ('nudity', 'nsfw'):
            return NUDITY_FIXED_SIGMOID_B_VALUES_19.copy()
        if erase_type == 'style':
            return STYLE_FIXED_SIGMOID_B_VALUES_19.copy()

        return None

    def _sigmoid_setting(self):
        return (self.args.sigmoid_a, 0.0, self.args.sigmoid_c)

    def _normalize_adavd_mode(self, mode=None):
        return 'retain'

    def _sigmoid_setting_for_mode(self, mode=None):
        return self._sigmoid_setting()

    def _layer_sigmoid_b_values_for_mode(self, mode=None):
        return self._parse_layer_sigmoid_b_values()

    def _parse_target_concepts(self, raw=None):
        if raw is None:
            raw = self.args.target_concept
        raw = raw or ''
        concepts = [c.strip() for c in str(raw).split(',') if c.strip()]
        return concepts if concepts else ['']

    def _process_target_t5_embeddings(self, target_concepts, target_embeds):
        tokenizer = getattr(self.pipe, 'tokenizer_2', None)
        seq_len = target_embeds.shape[1]
        processed = []

        for idx, concept in enumerate(target_concepts):
            valid_len = seq_len
            eos_positions = None
            if tokenizer is not None:
                tokens = tokenizer(
                    concept,
                    padding='max_length',
                    max_length=seq_len,
                    truncation=True,
                    return_tensors='pt',
                )
                input_ids = tokens.input_ids[0]
                eos_token_id = tokenizer.eos_token_id
                if eos_token_id is not None:
                    eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) > 0:
                        valid_len = max(1, min(seq_len, int(eos_positions[0].item())))
                elif hasattr(tokens, 'attention_mask'):
                    valid_len = max(1, min(seq_len, int(tokens.attention_mask[0].sum().item())))

            current = target_embeds[idx]
            valid_tokens = current[:valid_len]
            vector = valid_tokens.mean(dim=0, keepdim=True)
            processed.append(vector.repeat(seq_len, 1))

        return torch.stack(processed, dim=0).to(device=target_embeds.device, dtype=target_embeds.dtype)

    def setup_attention_processors(self, mode='retain'):
        joint_layers = list(range(19))
        num_targets = len(self._parse_target_concepts(self.args.target_concept))

        set_attenprocessor(
            self.pipe,
            joint_layers=joint_layers,
            sigmoid_setting=self._sigmoid_setting_for_mode(mode),
            layer_sigmoid_b_values=self._layer_sigmoid_b_values_for_mode(mode),
            phase=self.args.phase,
            num_targets=num_targets,
        )

    def restore_original_attention_processors(self):
        for block in self.pipe.transformer.transformer_blocks:
            block.attn.set_processor(FluxAttnProcessor2_0())

    def _build_latents(self, batch_size, seed):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        latents, _ = self.pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            self.args.height,
            self.args.width,
            self.pipe.transformer.dtype,
            self.device,
            gen,
            None,
        )
        return latents

    def _pipe_kwargs_base(self, latents, guidance_scale=None, max_sequence_length=None):
        if guidance_scale is None:
            guidance_scale = self.args.guidance_scale
        if max_sequence_length is None:
            max_sequence_length = self.args.max_sequence_length
        return {
            'height': self.args.height,
            'width': self.args.width,
            'num_inference_steps': self.args.num_inference_steps,
            'guidance_scale': guidance_scale,
            'max_sequence_length': max_sequence_length,
            'latents': latents,
        }

    def _clear_external_erasure_states(self):
        return None

    @torch.no_grad()
    def generate_original(self, prompt, latents, guidance_scale=None):
        self._clear_external_erasure_states()
        prompt_list = [prompt] * self.args.batch_size
        kwargs = self._pipe_kwargs_base(latents, guidance_scale=guidance_scale)
        images = self.pipe(
            prompt=prompt_list,
            prompt_2=prompt_list,
            **kwargs,
        ).images
        return images[: self.args.batch_size]

    @torch.no_grad()
    def generate_retain(self, prompt, target_concept, latents, guidance_scale=None, adavd_mode='retain'):
        self._clear_external_erasure_states()
        adavd_mode = self._normalize_adavd_mode(adavd_mode)
        target_concepts = self._parse_target_concepts(target_concept)
        self.setup_attention_processors(mode=adavd_mode)

        prompt_list = [prompt] * self.args.batch_size
        target_list = []
        for concept in target_concepts:
            target_list.extend([concept] * self.args.batch_size)

        prompt_embeds, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt_list,
            prompt_2=prompt_list,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=self.args.max_sequence_length,
        )
        target_embeds, pooled_target_embeds, _ = self.pipe.encode_prompt(
            prompt=target_list,
            prompt_2=target_list,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=self.args.max_sequence_length,
        )
        target_embeds = self._process_target_t5_embeddings(target_list, target_embeds)

        combined_prompt_embeds = torch.cat([prompt_embeds, target_embeds], dim=0)
        combined_pooled = torch.cat([pooled_prompt_embeds, pooled_target_embeds], dim=0)
        repeat_count = 1 + max(1, len(target_concepts))
        latents_pair = torch.cat([latents] * repeat_count, dim=0)

        kwargs = self._pipe_kwargs_base(latents_pair, guidance_scale=guidance_scale)
        all_images = self.pipe(
            prompt_embeds=combined_prompt_embeds,
            pooled_prompt_embeds=combined_pooled,
            **kwargs,
        ).images

        return all_images[: self.args.batch_size]

    @torch.no_grad()
    def _save_images(self, save_dir, mode, prompt, sample_idx, images):
        mode_dir = os.path.join(save_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        base_name = re.sub(r'[^\w\s]', '', prompt).replace(' ', '_')[:120]
        for img_idx, img in enumerate(images):
            suffix = f"_{img_idx}" if img_idx > 0 else ""
            filename = f"{base_name}_{sample_idx}{suffix}.png"
            img.save(os.path.join(mode_dir, filename))

    def _normalized_modes(self):
        modes = [m.strip() for m in self.args.mode.split(',') if m.strip()]
        normalized = []
        for m in modes:
            if m in ('retain', 'erase'):
                normalized.append('retain')
            elif m == 'original':
                normalized.append('original')
        normalized = list(dict.fromkeys(normalized))
        if not normalized:
            raise ValueError("Unsupported mode. Use a comma-separated subset of: original, retain, erase.")
        return normalized

    def _csv_outputs_exist(self, csv_idx, modes):
        for mode in modes:
            for img_idx in range(self.args.batch_size):
                save_path = os.path.join(
                    self.args.save_root,
                    f"{mode}_prompt_{csv_idx:05d}_img_{img_idx:02d}.png",
                )
                if not os.path.exists(save_path):
                    return False
        return True

    def _template_outputs_exist(self, concept_dir, mode, prompt, sample_idx):
        mode_dir = os.path.join(concept_dir, mode)
        base_name = re.sub(r'[^\w\s]', '', prompt).replace(' ', '_')[:120]
        for img_idx in range(self.args.batch_size):
            suffix = f"_{img_idx}" if img_idx > 0 else ""
            filename = f"{base_name}_{sample_idx}{suffix}.png"
            if not os.path.exists(os.path.join(mode_dir, filename)):
                return False
        return True

    def run_template_mode(self):
        concept_list = [item.strip() for item in self.args.contents.split(',') if item.strip()]
        if not concept_list:
            return

        if self.args.erase_type not in template_dict:
            raise ValueError(f"Unknown erase_type: {self.args.erase_type}")

        target_concept_full = (self.args.target_concept or '').strip()
        target_concept_first = target_concept_full.split(',')[0].strip() if target_concept_full else concept_list[0]
        if target_concept_first in concept_list:
            ordered_concepts = [target_concept_first] + [c for c in concept_list if c != target_concept_first]
        else:
            ordered_concepts = concept_list

        prompt_templates = template_dict[self.args.erase_type]
        os.makedirs(self.args.save_root, exist_ok=True)

        modes = self._normalized_modes()

        for sample_idx in range(self.args.num_samples):
            for prompt_idx, prompt_template in enumerate(prompt_templates):
                JointAttnProcessor.reset_phase_cache()
                SingleAttnProcessor.reset_phase_cache()

                for concept_idx, concept in enumerate(ordered_concepts):
                    prompt = prompt_template.format(concept)
                    concept_dir = os.path.join(self.args.save_root, concept)
                    os.makedirs(concept_dir, exist_ok=True)

                    done_markers = {
                        mode: os.path.join(concept_dir, f"{mode}_prompt_{prompt_idx:02d}_sample_{sample_idx:02d}.done")
                        for mode in modes
                    }
                    pending_modes = [mode for mode in modes if not os.path.exists(done_markers[mode])]
                    if self.args.skip_if_exists and pending_modes:
                        still_pending = []
                        for mode in pending_modes:
                            if self._template_outputs_exist(concept_dir, mode, prompt, sample_idx):
                                with open(done_markers[mode], 'w') as f:
                                    f.write('done')
                            else:
                                still_pending.append(mode)
                        pending_modes = still_pending
                    if not pending_modes:
                        continue
                    JointAttnProcessor.reset_timestep_counter()
                    SingleAttnProcessor.reset_timestep_counter()

                    if concept_idx == 0:
                        JointAttnProcessor.disable_sigmoid_b_cache()
                        SingleAttnProcessor.disable_sigmoid_b_cache()
                    else:
                        JointAttnProcessor.enable_sigmoid_b_cache()
                        SingleAttnProcessor.enable_sigmoid_b_cache()

                    seed_key = f"{concept}|{prompt}|{sample_idx}|{self.args.seed}"
                    base_seed = self._stable_seed(seed_key)
                    base_latents = self._build_latents(self.args.batch_size, base_seed)

                    decoded_imgs = {}
                    for mode in pending_modes:
                        if mode == 'original':
                            decoded_imgs[mode] = self.generate_original(prompt, base_latents.clone())
                        elif mode == 'retain':
                            decoded_imgs[mode] = self.generate_retain(
                                prompt,
                                target_concept_full or target_concept_first,
                                base_latents.clone(),
                                adavd_mode=mode,
                            )

                    for mode in pending_modes:
                        if mode in decoded_imgs:
                            self._save_images(concept_dir, mode, prompt, sample_idx, decoded_imgs[mode])
                            with open(done_markers[mode], 'w') as f:
                                f.write('done')

                    print(f"Done: concept={concept} prompt_idx={prompt_idx} sample_idx={sample_idx} modes={','.join(pending_modes)}")

    def run_csv_mode(self):
        df = pd.read_csv(self.args.prompt_file, encoding='utf-8')
        if 'prompt' not in df.columns:
            available_columns = ', '.join(map(str, df.columns.tolist()))
            raise ValueError(
                f"Prompt column 'prompt' not found in CSV '{self.args.prompt_file}'. "
                f"Available columns: {available_columns}"
            )
        selected_rows = df.iloc[self.args.prompt_start:self.args.prompt_end]

        os.makedirs(self.args.save_root, exist_ok=True)
        modes = self._normalized_modes()
        target_concept_full = (self.args.target_concept or '').strip() or 'nsfw'

        for local_idx, (_, row) in enumerate(tqdm(selected_rows.iterrows(), total=len(selected_rows), desc='Processing CSV prompts')):
            csv_idx = self.args.prompt_start + local_idx
            if self.args.skip_if_exists and self._csv_outputs_exist(csv_idx, modes):
                continue
            prompt = row['prompt']
            if prompt is None or pd.isna(prompt):
                continue
            prompt = str(prompt).strip()
            if not prompt:
                continue

            JointAttnProcessor.reset_phase_cache()
            SingleAttnProcessor.reset_phase_cache()
            JointAttnProcessor.disable_sigmoid_b_cache()
            SingleAttnProcessor.disable_sigmoid_b_cache()

            row_seed = self._csv_seed_from_row(row)
            if row_seed is None:
                seed_key = f"csv|{csv_idx}|{self.args.seed}"
                base_seed = self._stable_seed(seed_key)
            else:
                base_seed = row_seed
            base_latents = self._build_latents(self.args.batch_size, base_seed)

            decoded_imgs = {}
            for mode in modes:
                if mode == 'original':
                    decoded_imgs[mode] = self.generate_original(prompt, base_latents.clone())
                elif mode == 'retain':
                    decoded_imgs[mode] = self.generate_retain(
                        prompt,
                        target_concept_full,
                        base_latents.clone(),
                        adavd_mode=mode,
                    )

            for mode in modes:
                if mode not in decoded_imgs:
                    continue
                for img_idx, img in enumerate(decoded_imgs[mode]):
                    save_path = os.path.join(
                        self.args.save_root,
                        f"{mode}_prompt_{csv_idx:05d}_img_{img_idx:02d}.png",
                    )
                    img.save(save_path)

    def run_single_prompt(self):
        os.makedirs(self.args.save_root, exist_ok=True)
        modes = self._normalized_modes()
        target_concept_full = (self.args.target_concept or '').strip() or 'nsfw'

        JointAttnProcessor.reset_phase_cache()
        SingleAttnProcessor.reset_phase_cache()
        JointAttnProcessor.disable_sigmoid_b_cache()
        SingleAttnProcessor.disable_sigmoid_b_cache()

        seed_key = f"single|{self.args.prompt}|{self.args.seed}"
        base_seed = self._stable_seed(seed_key)
        base_latents = self._build_latents(self.args.batch_size, base_seed)

        decoded_imgs = {}
        for mode in modes:
            if mode == 'original':
                decoded_imgs[mode] = self.generate_original(self.args.prompt, base_latents.clone())
            elif mode == 'retain':
                decoded_imgs[mode] = self.generate_retain(
                    self.args.prompt,
                    target_concept_full,
                    base_latents.clone(),
                    adavd_mode=mode,
                )

        for mode, images in decoded_imgs.items():
            self._save_images(self.args.save_root, mode, self.args.prompt, 0, images)

    def run(self):
        if self.args.contents:
            self.run_template_mode()
        elif self.args.prompt_file:
            self.run_csv_mode()
        else:
            self.run_single_prompt()


def main():
    parser = argparse.ArgumentParser(description='Flux AdaVD')

    parser.add_argument('--sd_ckpt', type=str, default='black-forest-labs/FLUX.1-dev', help='Local checkpoint path or Hugging Face model ID for FLUX.')
    parser.add_argument('--torch_dtype', type=str, default='auto', choices=['auto', 'bfloat16', 'bf16', 'float16', 'fp16', 'float32', 'fp32'], help='Weight dtype for loading the FLUX pipeline.')
    parser.add_argument('--target_concept', type=str, default='nsfw', help='Target concept(s) to suppress. Use commas for multi-concept erasure.')
    parser.add_argument('--save_root', type=str, default='./output_flux', help='Root directory for saved outputs.')
    parser.add_argument('--prompt', type=str, default='a photo of a person', help='Single prompt used when --contents and --prompt_file are not provided.')

    parser.add_argument('--mode', type=str, default='original', help='original,retain,erase')
    parser.add_argument('--contents', type=str, default='', help='Comma-separated concepts to fill template prompts.')
    parser.add_argument('--erase_type', type=str, default='instance', choices=['instance', 'style', 'celebrity', 'nsfw', 'nudity'], help='Erasure setting. Use instance/style/celebrity for explicit concepts and nsfw/nudity for implicit concepts.')

    parser.add_argument('--prompt_file', type=str, default='', help='CSV prompt file for benchmark mode.')
    parser.add_argument('--prompt_start', type=int, default=0, help='Start index in the prompt CSV.')
    parser.add_argument('--prompt_end', type=int, default=10, help='End index in the prompt CSV (exclusive).')
    parser.add_argument('--skip_if_exists', action='store_true', help='Skip prompts whose output files already exist.')

    parser.add_argument('--sigmoid_a', type=float, default=100.0, help='Sigmoid steepness used for AdaVD value scaling.')
    parser.add_argument('--sigmoid_c', type=float, default=5.0, help='Sigmoid upper bound used for AdaVD value scaling.')

    parser.add_argument('--phase', type=str, default='early', help='Which denoising phase(s) to intervene in: early, mid, late, all, or phase combinations.')

    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate per prompt in template mode.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of prompts or images processed together.')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed.')

    parser.add_argument('--height', type=int, default=1024, help='Output image height.')
    parser.add_argument('--width', type=int, default=1024, help='Output image width.')
    parser.add_argument('--num_inference_steps', type=int, default=30, help='Number of denoising steps.')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Classifier-free guidance scale.')
    parser.add_argument('--max_sequence_length', type=int, default=512, help='Maximum text sequence length passed to the FLUX text encoder.')

    args = parser.parse_args()

    seed_everything(args.seed)

    adavd = FluxAdaVD(args)
    adavd.run()


if __name__ == '__main__':
    main()
