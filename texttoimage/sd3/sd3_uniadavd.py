import os, sys
import re
import random
import argparse
import hashlib
import numpy as np
import pandas as pd
import csv
from PIL import Image
from tqdm import tqdm
import torch
from diffusers import StableDiffusion3Pipeline
import torch.nn.functional as F
from torch import nn
from template import template_dict


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

INSTANCE_SIGMOID_B_VALUES = (
    0.83, 0.95, 0.86, 0.83, 0.71, 0.65, 0.66, 0.53,
    0.51, 0.47, 0.57, 0.49, 0.49, 0.57, 0.71, 0.71,
    0.50, 0.61, 0.63, 0.56, 0.52, 0.53, 0.71, 0.61,
)

STYLE_SIGMOID_B_VALUES = (
    0.73, 0.92, 0.76, 0.80, 0.70, 0.64, 0.60, 0.54,
    0.51, 0.48, 0.51, 0.55, 0.47, 0.59, 0.56, 0.64,
    0.55, 0.61, 0.63, 0.56, 0.52, 0.46, 0.58, 0.52,
)

CELEBRITY_SIGMOID_B_VALUES = (
    0.80, 0.95, 0.82, 0.82, 0.72, 0.70, 0.65, 0.60,
    0.62, 0.62, 0.64, 0.65, 0.56, 0.62, 0.74, 0.70,
    0.62, 0.67, 0.70, 0.62, 0.55, 0.60, 0.68, 0.56,
)

NSFW_NUDITY_CSV_SIGMOID_B_VALUES = (0.55,) * 24
IMPLICIT_TOPK_CONCEPTS = {"nudity"}


def _validate_sigmoid_b_values(values, preset_name):
    if len(values) != 24:
        raise ValueError(
            f"Expected 24 layer-wise sigmoid_b values for {preset_name}, got {len(values)}"
        )
    return values


def is_nsfw_nudity_csv_case(args, use_csv):
    normalized_erase_type = (args.erase_type or '').strip().lower()
    normalized_target_concept = (args.target_concept or '').split(',')[0].strip().lower()
    return use_csv and normalized_erase_type == 'nsfw' and normalized_target_concept == 'nudity'


def get_effective_sigmoid_b(args, use_csv):
    normalized_erase_type = (args.erase_type or '').strip().lower()
    if is_nsfw_nudity_csv_case(args, use_csv):
        return _validate_sigmoid_b_values(NSFW_NUDITY_CSV_SIGMOID_B_VALUES, 'nsfw CSV nudity')
    if normalized_erase_type == 'instance':
        return _validate_sigmoid_b_values(INSTANCE_SIGMOID_B_VALUES, 'instance')
    if normalized_erase_type == 'style':
        return _validate_sigmoid_b_values(STYLE_SIGMOID_B_VALUES, 'style')
    if normalized_erase_type == 'celebrity':
        return _validate_sigmoid_b_values(CELEBRITY_SIGMOID_B_VALUES, 'celebrity')
    return args.sigmoid_b


def get_effective_top_k(args):
    normalized_erase_type = (args.erase_type or "").strip().lower()
    target_concepts = {
        concept.strip().lower()
        for concept in (args.target_concept or "").split(",")
        if concept.strip()
    }
    if normalized_erase_type == "nsfw" or target_concepts.intersection(IMPLICIT_TOPK_CONCEPTS):
        return 50
    return 1


def resolve_layer_sigmoid_b(sigmoid_b_setting, layer_num):
    if isinstance(sigmoid_b_setting, (list, tuple)):
        if layer_num >= len(sigmoid_b_setting):
            raise ValueError(
                f"Layer {layer_num} exceeds sigmoid_b preset length {len(sigmoid_b_setting)}"
            )
        return float(sigmoid_b_setting[layer_num])
    return float(sigmoid_b_setting)


def get_sigmoid_b_tag(sigmoid_b_setting, args, use_csv):
    if isinstance(sigmoid_b_setting, (list, tuple)):
        if is_nsfw_nudity_csv_case(args, use_csv):
            return "nsfw24x0.55csv"
        normalized_erase_type = (args.erase_type or '').strip().lower()
        if normalized_erase_type in {"instance", "style", "celebrity"}:
            return f"{normalized_erase_type}24preset"
        return "24preset"
    return str(sigmoid_b_setting)


def describe_sigmoid_b_setting(sigmoid_b_setting, args, use_csv):
    if isinstance(sigmoid_b_setting, (list, tuple)):
        return f"layer-wise sigmoid_b preset ({get_sigmoid_b_tag(sigmoid_b_setting, args, use_csv)})"
    return f"static sigmoid_b = {sigmoid_b_setting}"


def get_target_concept_names():
    concept_names = []
    argv = sys.argv
    for idx, arg in enumerate(argv):
        if arg == '--target_concept' and idx + 1 < len(argv):
            concept_names = [item.strip() for item in argv[idx + 1].split(',') if item.strip()]
            break
        if arg.startswith('--target_concept='):
            concept_names = [item.strip() for item in arg.split('=', 1)[1].split(',') if item.strip()]
            break
    return concept_names


def _should_decompose_layer(processor, layer_num):
    return True


def _current_phase(current_timestep):
    if current_timestep < 10:
        return 'early'
    if current_timestep < 20:
        return 'mid'
    return 'late'


def _should_erase_in_phase(processor, current_phase):
    if processor.phase == 'all':
        return True
    if '+' in processor.phase:
        return current_phase in processor.phase.split('+')
    return processor.phase == current_phase


def _get_multiconcept_token_coefficients(processor, cos_sim_raw, layer_num, current_timestep, current_phase):
    should_erase = _should_erase_in_phase(processor, current_phase)
    if not should_erase:
        return torch.zeros_like(cos_sim_raw)

    sigmoid_a = processor.sigmoid_setting[0]
    sigmoid_c = processor.sigmoid_setting[2]
    if sigmoid_a == 0:
        raw_coeff = torch.full_like(cos_sim_raw, sigmoid_c)
    else:
        layer_sigmoid_b = resolve_layer_sigmoid_b(processor.sigmoid_setting[1], layer_num)
        raw_coeff = processor.sigmoid(cos_sim_raw, (sigmoid_a, layer_sigmoid_b, sigmoid_c))

    if cos_sim_raw.ndim > 1:
        coeff = torch.zeros_like(raw_coeff)
        top_concept_indices = torch.argmax(cos_sim_raw, dim=-1)
        token_indices = torch.arange(cos_sim_raw.shape[0], device=cos_sim_raw.device)
        coeff[token_indices, top_concept_indices] = raw_coeff[token_indices, top_concept_indices]
    else:
        coeff = raw_coeff

    top_k = int(getattr(processor, "top_k", 0) or 0)
    if top_k <= 0 or coeff.shape[0] == 0:
        return coeff

    token_scores = cos_sim_raw.max(dim=-1).values if cos_sim_raw.ndim > 1 else cos_sim_raw
    top_k = min(top_k, token_scores.shape[0])
    if top_k >= token_scores.shape[0]:
        return coeff

    _, top_indices = torch.topk(token_scores, k=top_k)
    masked_coeff = torch.zeros_like(coeff)
    masked_coeff[top_indices] = coeff[top_indices]
    return masked_coeff

class VisualAttentionProcess(nn.Module):
    def __init__(self, module_name=None, atten_type='original', record=False,
    record_type=None, sigmoid_setting=None, token_sim=False, decomp_timestep=0, target_concept_encodings=None,
    phase='all', top_k=1, **kwargs):
        super().__init__()
        self.module_name = module_name
        self.atten_type = atten_type
        self.record = record
        self.record_type = record_type
        self.sigmoid_setting = sigmoid_setting
        self.token_sim = token_sim
        self.decomp_timestep = decomp_timestep
        self.target_concept_encodings = target_concept_encodings
        self.phase = phase
        self.top_k = top_k

    def __call__(self, attn, hidden_states, encoder_hidden_states, *args, **kwargs):
        attn._modules.pop("processor")
        attn.processor = JointAttnProcessor(
            module_name=self.module_name,
            atten_type=self.atten_type,
            record=self.record,
            record_type=self.record_type,
            sigmoid_setting=self.sigmoid_setting,
            token_sim=self.token_sim,
            decomp_timestep=self.decomp_timestep,
            target_concept_encodings=self.target_concept_encodings,
            phase=self.phase,
            top_k=self.top_k,
        )
        return attn.processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)


class JointAttnProcessor:
    _timestep_counter = {}

    @classmethod
    def reset_phase_cache(cls):
        cls._timestep_counter = {}

    def __init__(
        self, module_name=None, atten_type='original', record=False,
        record_type=None, sigmoid_setting=None, token_sim=False,
        decomp_timestep=0, target_concept_encodings=None, phase='all', top_k=1
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0.")

        self.module_name = module_name
        self.atten_type = atten_type
        self.record = record
        self.record_type = record_type.strip().split(',') if record_type is not None else []
        self.records = {key: {} for key in self.record_type} if record_type is not None else {}
        self.sigmoid_setting = sigmoid_setting
        self.token_sim = token_sim
        self.decomp_timestep = decomp_timestep
        self.target_concept_encodings = target_concept_encodings
        self.phase = phase
        self.top_k = top_k
        self.multi_concept_mode = (
            target_concept_encodings is not None and target_concept_encodings.shape[0] > 1
        )

    def sigmoid(self, x, setting):
        a, b, c = setting
        return c / (1 + torch.exp(-a * (x - b)))

    def flatten_target_concepts(self, target_value):
        return target_value.permute(2, 0, 1, 3).reshape(target_value.shape[2], target_value.shape[0], -1)

    def solve_order_invariant_projection(self, target_flat, prompt_flat):
        concept_gram = torch.bmm(target_flat, target_flat.transpose(1, 2))
        eye = torch.eye(
            concept_gram.shape[-1], dtype=concept_gram.dtype, device=concept_gram.device
        ).unsqueeze(0)
        concept_gram = concept_gram + 1e-6 * eye
        concept_gram_pinv = torch.linalg.pinv(concept_gram)
        prompt_concept_dot = torch.bmm(prompt_flat.unsqueeze(1), target_flat.transpose(1, 2)).squeeze(1)
        concept_weights = torch.bmm(prompt_concept_dot.unsqueeze(1), concept_gram_pinv).squeeze(1)
        return concept_weights

    def record_ortho_decomp(self, target_value, current_value, sot_idx=None):
        layer_num = int(self.module_name.split('.')[1])
        should_decompose = _should_decompose_layer(self, layer_num)

        if self.multi_concept_mode:
            if not should_decompose:
                return current_value, current_value

            erase_record, retain_record = [], []
            if self.token_sim:
                if self.module_name not in JointAttnProcessor._timestep_counter:
                    JointAttnProcessor._timestep_counter[self.module_name] = 0
                current_timestep = JointAttnProcessor._timestep_counter[self.module_name]
                current_phase = _current_phase(current_timestep)
            else:
                current_timestep = None
                current_phase = None

            concept_count = target_value.shape[0]
            if concept_count == 1:
                target_record = target_value[0]
                target_flat = target_record.permute(1, 0, 2).reshape(target_record.shape[1], -1).to(dtype=torch.float32)

                for pro_record in current_value:
                    input_dtype = pro_record.dtype
                    pro_flat = pro_record.permute(1, 0, 2).reshape(pro_record.shape[1], -1).to(dtype=torch.float32)
                    dot1 = (target_flat * pro_flat).sum(-1)
                    dot2 = torch.clamp((target_flat * target_flat).sum(-1), min=1e-6)
                    coeff = dot1 / dot2

                    if sot_idx is not None and sot_idx < coeff.shape[0]:
                        coeff[sot_idx] = 0

                    if self.token_sim:
                        cos_sim_raw = torch.cosine_similarity(target_flat, pro_flat, dim=-1)
                        coeff = coeff * _get_multiconcept_token_coefficients(
                            self, cos_sim_raw, layer_num, current_timestep, current_phase
                        )

                    era_record = coeff.unsqueeze(0).unsqueeze(-1) * target_record
                    ret_record = pro_record - era_record
                    erase_record.append(era_record.to(dtype=input_dtype))
                    retain_record.append(ret_record.to(dtype=input_dtype))
            else:
                target_flat = self.flatten_target_concepts(target_value.to(dtype=torch.float32))

                for pro_record in current_value:
                    input_dtype = pro_record.dtype
                    pro_flat = pro_record.permute(1, 0, 2).reshape(pro_record.shape[1], -1).to(dtype=torch.float32)
                    concept_weights = self.solve_order_invariant_projection(target_flat, pro_flat)

                    if sot_idx is not None and sot_idx < concept_weights.shape[0]:
                        concept_weights[sot_idx] = 0

                    if self.token_sim:
                        expanded_prompt = pro_flat.unsqueeze(1).expand(-1, concept_count, -1)
                        cos_sim_raw = torch.cosine_similarity(target_flat, expanded_prompt, dim=-1)
                        concept_coeff = _get_multiconcept_token_coefficients(
                            self, cos_sim_raw, layer_num, current_timestep, current_phase
                        )
                    else:
                        concept_coeff = torch.ones(
                            target_flat.shape[:2], device=target_flat.device, dtype=target_flat.dtype
                        )

                    weighted_concept_weights = concept_weights * concept_coeff
                    era_flat = torch.bmm(weighted_concept_weights.unsqueeze(1), target_flat).squeeze(1)
                    era_record = era_flat.view(target_flat.shape[0], 2, -1).permute(1, 0, 2)
                    ret_record = pro_record - era_record

                    erase_record.append(era_record.to(dtype=input_dtype))
                    retain_record.append(ret_record.to(dtype=input_dtype))

            if self.token_sim:
                JointAttnProcessor._timestep_counter[self.module_name] += 1

            retain_record = torch.stack(retain_record, dim=0)
            erase_record = torch.stack(erase_record, dim=0)
            return erase_record, retain_record

        if should_decompose:
            erase_record, retain_record = [], []
            for tar_record, pro_record in zip(target_value, current_value):
                input_dtype = tar_record.dtype
                L = tar_record.shape[1]

                tar_flat = tar_record.permute(1, 0, 2).reshape(L, -1).to(dtype=torch.float32)
                pro_flat = pro_record.permute(1, 0, 2).reshape(L, -1).to(dtype=torch.float32)

                dot1 = (tar_flat * pro_flat).sum(-1)
                dot2 = torch.clamp((tar_flat * tar_flat).sum(-1), min=1e-6)

                mask = torch.ones_like(dot1)
                if sot_idx is not None:
                    if sot_idx < mask.shape[0]:
                        mask[sot_idx] = 0

                if self.token_sim:
                    cos_sim_raw = torch.cosine_similarity(tar_flat, pro_flat, dim=-1)

                    if self.module_name not in JointAttnProcessor._timestep_counter:
                        JointAttnProcessor._timestep_counter[self.module_name] = 0

                    current_timestep = JointAttnProcessor._timestep_counter[self.module_name]
                    current_phase = _current_phase(current_timestep)
                    should_erase_in_phase = _should_erase_in_phase(self, current_phase)
                    erase_mask = torch.zeros_like(cos_sim_raw)

                    if should_erase_in_phase:
                        sigmoid_a = self.sigmoid_setting[0]
                        sigmoid_c = self.sigmoid_setting[2]
                        use_fixed_coeff = (sigmoid_a == 0)
                        if use_fixed_coeff:
                            erase_mask.fill_(sigmoid_c)
                        else:
                            layer_sigmoid_b = resolve_layer_sigmoid_b(self.sigmoid_setting[1], layer_num)
                            erase_mask = self.sigmoid(cos_sim_raw, (self.sigmoid_setting[0], layer_sigmoid_b, self.sigmoid_setting[2]))

                        top_k = min(int(self.top_k or 0), erase_mask.shape[0])
                        if 0 < top_k < erase_mask.shape[0]:
                            _, top_indices = torch.topk(cos_sim_raw, k=top_k)
                            topk_mask = torch.zeros_like(erase_mask)
                            topk_mask[top_indices] = 1
                            erase_mask = erase_mask * topk_mask

                    JointAttnProcessor._timestep_counter[self.module_name] += 1
                    cos_sim = erase_mask
                else:
                    cos_sim = torch.ones_like(dot1)

                coeff = cos_sim * mask * (dot1 / dot2)

                era_record = coeff.unsqueeze(0).unsqueeze(-1) * tar_record
                ret_record = pro_record - era_record

                erase_record.append(era_record.to(dtype=input_dtype))
                retain_record.append(ret_record.to(dtype=input_dtype))

            retain_record = torch.stack(retain_record, dim=0)
            erase_record = torch.stack(erase_record, dim=0)
            return erase_record, retain_record
        else:
            return current_value, current_value

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        *args,
        **kwargs
    ):

        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj   = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key   = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        sot_idx = 0

        if (
            self.multi_concept_mode
            and not self.record
            and self.atten_type in ['erase', 'retain']
            and hasattr(self, 'target_concept_encodings')
            and batch_size % 2 == 0
        ):
            target_count = int(self.target_concept_encodings.shape[0])
            batch_per_condition = batch_size // 2
            prompt_count = batch_per_condition - target_count

            if prompt_count > 0:
                uncond_prompt_idx = slice(0, prompt_count)
                uncond_target_idx = slice(prompt_count, batch_per_condition)
                cond_prompt_idx = slice(batch_per_condition, batch_per_condition + prompt_count)
                cond_target_idx = slice(batch_per_condition + prompt_count, batch_size)

                def split_prompt_target(record_tensor):
                    neg_prompt = record_tensor[uncond_prompt_idx]
                    pos_prompt = record_tensor[cond_prompt_idx]
                    neg_target = record_tensor[uncond_target_idx]
                    pos_target = record_tensor[cond_target_idx]
                    prompt_record = torch.stack([neg_prompt, pos_prompt], dim=1)
                    target_record = torch.stack([neg_target, pos_target], dim=1)
                    return prompt_record, target_record

                text_value = encoder_hidden_states_value_proj
                prompt_text, target_text = split_prompt_target(text_value)
                erase_record, retain_record = self.record_ortho_decomp(target_text, prompt_text, sot_idx=sot_idx)
                processed_prompt = retain_record if self.atten_type == 'retain' else erase_record
                text_value[uncond_prompt_idx] = processed_prompt[:, 0]
                text_value[cond_prompt_idx] = processed_prompt[:, 1]
                value[:, -text_value.shape[1]:, :] = text_value

        elif not self.record and self.atten_type in ['erase', 'retain'] and hasattr(self, 'target_concept_encodings') and batch_size % 4 == 0:
            bs_orig = batch_size // 4
            uncond_prompt_idx = slice(0, bs_orig)
            uncond_target_idx = slice(bs_orig, 2 * bs_orig)
            cond_prompt_idx = slice(2 * bs_orig, 3 * bs_orig)
            cond_target_idx = slice(3 * bs_orig, 4 * bs_orig)

            text_value = encoder_hidden_states_value_proj
            neg_prompt = text_value[uncond_prompt_idx]
            pos_prompt = text_value[cond_prompt_idx]
            neg_target = text_value[uncond_target_idx]
            pos_target = text_value[cond_target_idx]

            prompt_text = torch.stack([neg_prompt, pos_prompt], dim=1)
            target_text = torch.stack([neg_target, pos_target], dim=1)

            erase_record, retain_record = self.record_ortho_decomp(target_text, prompt_text, sot_idx=sot_idx)
            processed_prompt = retain_record if self.atten_type == 'retain' else erase_record

            text_value[uncond_prompt_idx] = processed_prompt[:, 0]
            text_value[cond_prompt_idx] = processed_prompt[:, 1]
            value[:, -text_value.shape[1]:, :] = text_value

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


def set_attenprocessor(
    transformer,
    atten_type='original',
    record=False,
    record_type=None,
    sigmoid_setting=None,
    token_sim=False,
    decomp_timestep=0,
    target_concept_encodings=None,
    phase='all',
    top_k=1,
):
    for name, m in transformer.named_modules():
        if name.endswith('attn'):
            m.set_processor(VisualAttentionProcess(
                module_name=name,
                atten_type=atten_type,
                record=record,
                record_type=record_type,
                sigmoid_setting=sigmoid_setting,
                token_sim=token_sim,
                decomp_timestep=decomp_timestep,
                target_concept_encodings=target_concept_encodings,
                phase=phase,
                top_k=top_k,
            ))
    return transformer


@torch.no_grad()
def main():
    def get_csv_param(row, column_names, cast_fn, default_value):
        if isinstance(column_names, str):
            column_names = [column_names]

        value = None
        for column_name in column_names:
            if column_name in row:
                value = row[column_name]
                if not pd.isna(value):
                    break
        else:
            return default_value

        try:
            if cast_fn is int:
                return int(float(value))
            return cast_fn(value)
        except (TypeError, ValueError):
            return default_value

    def stable_uint32_seed(*parts):
        hasher = hashlib.sha256()
        for part in parts:
            hasher.update(str(part).encode("utf-8"))
            hasher.update(b"\0")
        return int.from_bytes(hasher.digest()[:8], "big") % (2**32)

    def build_latents(sample_seed, dtype):
        latent_device = 'cuda'
        generator = torch.Generator(device=latent_device)
        generator.manual_seed(int(sample_seed))
        return torch.randn(bs, 16, 128, 128, generator=generator, device=latent_device, dtype=dtype)

    def configure_memory_optimizations(runtime_pipe):
        runtime_pipe.enable_attention_slicing()
        if hasattr(runtime_pipe, "vae") and runtime_pipe.vae is not None:
            if hasattr(runtime_pipe.vae, "enable_slicing"):
                runtime_pipe.vae.enable_slicing()
            if hasattr(runtime_pipe.vae, "enable_tiling"):
                runtime_pipe.vae.enable_tiling()

    def enable_runtime_cpu_offload(runtime_pipe):
        if getattr(args, "disable_cpu_offload", False):
            print("[INFO] CPU offload disabled; keeping pipeline on GPU.", flush=True)
            try:
                runtime_pipe.to('cuda')
            except Exception as exc:
                print(f"[WARN] Failed to move pipeline to CUDA: {exc}", flush=True)
            return
        try:
            if hasattr(runtime_pipe, "enable_sequential_cpu_offload"):
                runtime_pipe.enable_sequential_cpu_offload()
            elif hasattr(runtime_pipe, "enable_model_cpu_offload"):
                runtime_pipe.enable_model_cpu_offload()
        except Exception as exc:
            print(f"[WARN] CPU offload not enabled: {exc}")

    def _temporary_offload_for_vae_decode(runtime_pipe):
        offload_names = ['transformer', 'text_encoder', 'text_encoder_2', 'text_encoder_3']
        moved_components = []

        for name in offload_names:
            module = getattr(runtime_pipe, name, None)
            if module is None:
                continue
            try:
                module.to('cpu')
                moved_components.append((name, module))
            except Exception:
                pass

        torch.cuda.empty_cache()
        return moved_components

    def _restore_offloaded_components(runtime_pipe, moved_components):
        for name, module in moved_components:
            try:
                module.to('cuda')
            except Exception:
                print(f"[WARN] Failed to move {name} back to CUDA after VAE decode fallback.")
        if moved_components:
            torch.cuda.empty_cache()

    def run_pipe_and_decode(runtime_pipe, output_type='pil', decode_batch_size=1, **call_kwargs):
        latent_output = runtime_pipe(output_type='latent', **call_kwargs).images
        if output_type == 'latent':
            return latent_output

        latents = (latent_output / runtime_pipe.vae.config.scaling_factor) + runtime_pipe.vae.config.shift_factor
        decoded_images = []
        moved_components = []

        try:
            for start_idx in range(0, latents.shape[0], decode_batch_size):
                latent_chunk = latents[start_idx:start_idx + decode_batch_size]
                try:
                    decoded_chunk = runtime_pipe.vae.decode(latent_chunk, return_dict=False)[0]
                except torch.cuda.OutOfMemoryError:
                    print("[WARN] CUDA OOM during VAE decode. Offloading non-VAE modules to CPU and retrying once.")
                    del latent_chunk
                    torch.cuda.empty_cache()
                    moved_components = _temporary_offload_for_vae_decode(runtime_pipe)
                    latent_chunk = latents[start_idx:start_idx + decode_batch_size]
                    decoded_chunk = runtime_pipe.vae.decode(latent_chunk, return_dict=False)[0]

                decoded_images.extend(runtime_pipe.image_processor.postprocess(decoded_chunk, output_type=output_type))
                del latent_chunk, decoded_chunk
                torch.cuda.empty_cache()
        finally:
            _restore_offloaded_components(runtime_pipe, moved_components)

        del latent_output, latents
        torch.cuda.empty_cache()
        return decoded_images

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, default='', help='Root directory for saved outputs.')
    parser.add_argument('--sd_ckpt', type=str, default='stabilityai/stable-diffusion-3-medium-diffusers', help='Local checkpoint path or Hugging Face model ID for Stable Diffusion 3.')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed. Template-mode sample seeds are derived from this seed.')

    parser.add_argument('--mode', type=str, default='original', help='original, erase, retain')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Fallback classifier-free guidance scale. CSV mode uses per-row guidance when available.')
    parser.add_argument('--total_timesteps', type=int, default=30, help='Number of denoising steps.')
    parser.add_argument('--decomp_timestep', type=int, default=0, help='The decomp calculation will keep until this hyper-parameter')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate per prompt.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--erase_type', type=str, default='', help='instance, nsfw, style, celebrity')
    parser.add_argument('--target_concept', type=str, default='', help='Target concept(s) to suppress. Use commas for multi-concept erasure.')
    parser.add_argument('--contents', type=str, default='', help='Comma-separated content concepts used to fill templates in template mode.')

    parser.add_argument('--prompt_file', type=str, default='', help='Path to CSV file (I2P/COCO)')
    parser.add_argument('--prompt_start', type=int, default=0, help='Start index for CSV')
    parser.add_argument('--prompt_end', type=int, default=10, help='End index for CSV')

    parser.add_argument('--token_processing', type=str, default='last_subject', choices=['last_subject', 'last_subject_eot_mean'], help='Strategy for constructing the target token representation.')

    parser.add_argument('--token_sim', default=False, action='store_true', help='Record token-wise cosine similarity statistics during value decomposition.')
    parser.add_argument('--sigmoid_a', type=float, default=100.0, help='Sigmoid steepness (0=fixed coeff mode, >0=sigmoid mode, default: 100)')
    parser.add_argument('--sigmoid_b', type=float, default=0.43, help='Static sigmoid center b')
    parser.add_argument('--sigmoid_c', type=float, default=2, help='Sigmoid max value')
    parser.add_argument('--phase', type=str, default='early', help='Which phase(s) to apply erasure (early/mid/late/all/early+mid/mid+late/early+late, default: early)')
    parser.add_argument('--disable_cpu_offload', action='store_true',
                        help='Keep SD3 pipeline on GPU instead of using CPU offload')

    args = parser.parse_args()
    effective_top_k = get_effective_top_k(args)
    record_type = 'values'
    bs = args.batch_size
    mode_list = args.mode.replace(' ', '').split(',')

    use_csv = args.prompt_file and os.path.exists(args.prompt_file)

    if use_csv:
        print(f"Loading prompts from CSV: {args.prompt_file}")
        df = pd.read_csv(args.prompt_file, encoding='iso-8859-1', quoting=csv.QUOTE_ALL,
                         engine='python', on_bad_lines='skip')
        df = df[df['prompt'].notna()].reset_index(drop=True)
        df = df[df['prompt'].astype(str).str.strip() != ''].reset_index(drop=True)

        if args.prompt_end > len(df):
            args.prompt_end = len(df)
        selected_prompts_df = df.iloc[args.prompt_start:args.prompt_end]
        print(f"Processing {len(selected_prompts_df)} prompts from CSV (index {args.prompt_start} to {args.prompt_end})")
        concept_list = None
        prompt_list = None
    else:
        concept_list_tmp = [item.strip() for item in args.contents.split(',') if item.strip()]
        concept_list = []
        if mode_list == ['retain']:
            SAMPLE_NUM = {'instance': 800, 'style': 300, 'celebrity': 250}
            for concept in concept_list_tmp:
                check_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept, 'retain')
                os.makedirs(check_path, exist_ok=True)
                if len(os.listdir(check_path)) != SAMPLE_NUM[args.erase_type]:
                    concept_list.append(concept)
        else:
            concept_list = concept_list_tmp
        if len(concept_list) == 0:
            sys.exit()
        selected_prompts_df = None

    print(f"\n{'='*70}")
    print("Loading model with method: adavd")
    print(f"{'='*70}\n")

    pipe = StableDiffusion3Pipeline.from_pretrained(args.sd_ckpt, torch_dtype=torch.float16)
    configure_memory_optimizations(pipe)
    print("Using AdaVD training-free method (no LoRA weights needed)")
    print(f"{'='*70}\n")
    enable_runtime_cpu_offload(pipe)

    def cal_targ_embedding(target_concepts, pipe, token_processing='last_subject'):
        target_embeds, negative_target_embeds, pooled_target_embeds, negative_pooled_target_embeds = pipe.encode_prompt(
            prompt=target_concepts,
            prompt_2=target_concepts,
            prompt_3=target_concepts,
            negative_prompt='',
            negative_prompt_2='',
            negative_prompt_3=''
        )

        clip_targets_full = target_embeds[:, :77]
        t5_targets_full = target_embeds[:, 77:]

        clip_targets_processed = []
        t5_targets_processed = []

        for idx, concept in enumerate(target_concepts):
            clip_tokens = pipe.tokenizer(concept, padding="max_length", max_length=77, return_tensors="pt")
            clip_input_ids = clip_tokens.input_ids[0]
            eos_token_id = pipe.tokenizer.eos_token_id
            clip_eot_idxs = (clip_input_ids == eos_token_id).nonzero(as_tuple=True)[0]
            clip_eot_idx = clip_eot_idxs[0].item() if len(clip_eot_idxs) > 0 else 76

            curr_clip_emb = clip_targets_full[idx]

            if token_processing == 'last_subject':
                target_idx = max(0, clip_eot_idx - 1)
                vector = curr_clip_emb[target_idx:target_idx+1, :]
            elif token_processing == 'last_subject_eot_mean':
                subj_idx = max(0, clip_eot_idx - 1)
                subj_vec = curr_clip_emb[subj_idx:subj_idx+1, :]
                if len(clip_eot_idxs) > 0:
                    eot_vecs = curr_clip_emb[clip_eot_idxs, :]
                    combined = torch.cat([subj_vec, eot_vecs], dim=0)
                    vector = combined.mean(dim=0, keepdim=True)
                else:
                    vector = subj_vec
            else:
                target_idx = max(0, clip_eot_idx - 1)
                vector = curr_clip_emb[target_idx:target_idx+1, :]

            clip_expanded = vector.repeat(77, 1)
            clip_targets_processed.append(clip_expanded)

            t5_tokens = pipe.tokenizer_3(concept, padding="max_length", max_length=256, return_tensors="pt")
            t5_input_ids = t5_tokens.input_ids[0]
            t5_eos_token_id = pipe.tokenizer_3.eos_token_id
            t5_eot_idxs = (t5_input_ids == t5_eos_token_id).nonzero(as_tuple=True)[0]
            t5_eot_idx = t5_eot_idxs[0].item() if len(t5_eot_idxs) > 0 else 255

            curr_t5_emb = t5_targets_full[idx]

            if token_processing == 'last_subject':
                vector_t5 = curr_t5_emb[:t5_eot_idx].mean(dim=0, keepdim=True)
            elif token_processing == 'last_subject_eot_mean':
                vector_t5 = curr_t5_emb[:t5_eot_idx].mean(dim=0, keepdim=True)
            else:
                vector_t5 = curr_t5_emb[:t5_eot_idx].mean(dim=0, keepdim=True)

            t5_expanded = vector_t5.repeat(256, 1)
            t5_targets_processed.append(t5_expanded)

        clip_targets_final = torch.stack(clip_targets_processed, dim=0)
        t5_targets_final = torch.stack(t5_targets_processed, dim=0)
        target_embeds = torch.concat([clip_targets_final, t5_targets_final], dim=1)

        return target_embeds, negative_target_embeds, pooled_target_embeds, negative_pooled_target_embeds

    target_concepts = [item.strip() for item in args.target_concept.split(',') if item.strip()]
    target_embeds, negative_target_embeds, pooled_target_embeds, negative_pooled_target_embeds = cal_targ_embedding(
        target_concepts, pipe, token_processing=args.token_processing
    )

    seed_everything(args.seed, True)

    if not use_csv:
        prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]
    else:
        prompt_list = None

    effective_sigmoid_b = get_effective_sigmoid_b(args, use_csv)
    if isinstance(effective_sigmoid_b, (list, tuple)):
        print(f"Using {describe_sigmoid_b_setting(effective_sigmoid_b, args, use_csv)}")
    elif effective_sigmoid_b != args.sigmoid_b:
        print(
            f"Overriding sigmoid_b from {args.sigmoid_b} to {effective_sigmoid_b} "
            f"for CSV nudity/nsfw mode"
        )
    sigmoid_a_values = [args.sigmoid_a]
    sigmoid_b_values = [effective_sigmoid_b]
    sigmoid_c_values = [args.sigmoid_c]

    fixed_latents = {}
    latent_seeds = {}
    latent_dtype = target_embeds.dtype

    if use_csv:
        for idx, row in selected_prompts_df.iterrows():
            csv_seed = get_csv_param(row, ['sd_seed', 'seed', 'evaluation_seed'], int, args.seed)
            for i in range(int(args.num_samples // bs)):
                latent_key = f"csv_{idx}_{i}"
                sample_seed = (csv_seed + i) % (2**32)
                latent_seeds[latent_key] = sample_seed
                fixed_latents[latent_key] = build_latents(sample_seed, latent_dtype)
    else:
        prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]
        for concept, prompts in zip(concept_list, prompt_list):
            for prompt in prompts:
                for i in range(int(args.num_samples // bs)):
                    latent_key = f"{concept}_{prompt}_{i}"
                    sample_seed = stable_uint32_seed(args.seed, args.erase_type, concept, prompt, i)
                    latent_seeds[latent_key] = sample_seed
                    fixed_latents[latent_key] = build_latents(sample_seed, latent_dtype)

    def get_save_path(prompt, concept, sigmoid_a, sigmoid_b, sigmoid_c):
        sigb_tag = get_sigmoid_b_tag(sigmoid_b, args, use_csv)
        second_level = f"text_value_alllayers_sigA{sigmoid_a}_sigB{sigb_tag}_sigC{sigmoid_c}_phase{args.phase}"
        if use_csv:
            return os.path.join(args.save_root, second_level)
        return os.path.join(args.save_root, second_level, concept)

    def get_save_filename(prompt, prompt_idx, idx_img, sample_idx):
        if use_csv:
            prompt_text = re.sub(r'[^\w\s]', '', prompt).replace(' ', '_')[:40]
            return f"{prompt_idx}_{prompt_text}_{int(idx_img + bs * sample_idx)}.png"
        prompt_stub = re.sub(r'[^\w\s]', '', prompt).replace(' ', '_')
        return f"{prompt_stub}_{int(idx_img + bs * sample_idx)}.png"

    def sample_outputs_exist(save_path, prompt, prompt_idx, sample_idx):
        expected_paths = []
        for idx_img in range(bs):
            save_filename = get_save_filename(prompt, prompt_idx, idx_img, sample_idx)
            for mode in mode_list:
                expected_paths.append(os.path.join(save_path, mode, save_filename))
            if len(mode_list) > 1:
                expected_paths.append(os.path.join(save_path, 'combine', save_filename))
        return all(os.path.exists(path) for path in expected_paths)

    def combine_images_horizontally(images):
        widths, heights = zip(*(img.size for img in images))
        new_img = Image.new('RGB', (sum(widths), max(heights)))
        for i_img, img in enumerate(images):
            new_img.paste(img, (sum(widths[:i_img]), 0))
        return new_img

    def generate_mode_images(prompt, latent, guidance_scale, sigmoid_a, sigmoid_b, sigmoid_c):
        decoded_imgs = {}
        cached_conditioned = None

        for mode in mode_list:
            if mode == 'original':
                pipe.transformer = set_attenprocessor(
                    pipe.transformer,
                    atten_type='original',
                    record=False,
                    record_type=record_type,
                    sigmoid_setting=(sigmoid_a, sigmoid_b, sigmoid_c),
                    token_sim=args.token_sim,
                    decomp_timestep=args.decomp_timestep,
                    target_concept_encodings=None,
                    phase=args.phase,
                    top_k=effective_top_k,
                )
                decoded_imgs['original'] = run_pipe_and_decode(
                    pipe,
                    prompt=[prompt] * bs,
                    num_images_per_prompt=1,
                    latents=latent,
                    guidance_scale=guidance_scale,
                    num_inference_steps=args.total_timesteps
                )
                continue

            if cached_conditioned is None:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                    prompt=[prompt] * bs,
                    prompt_2=[prompt] * bs,
                    prompt_3=[prompt] * bs,
                    negative_prompt=[""] * bs,
                    negative_prompt_2=[""] * bs,
                    negative_prompt_3=[""] * bs
                )

                tgt = target_embeds
                neg_tgt = negative_target_embeds
                pooled_tgt = pooled_target_embeds
                neg_pooled_tgt = negative_pooled_target_embeds

                if target_embeds.shape[0] > 1:
                    tgt_expanded = tgt
                    neg_tgt_expanded = neg_tgt
                    pooled_tgt_expanded = pooled_tgt
                    neg_pooled_tgt_expanded = neg_pooled_tgt
                elif tgt.shape[0] != bs:
                    tgt_expanded = tgt.repeat((bs, 1, 1))[:bs]
                    neg_tgt_expanded = neg_tgt.repeat((bs, 1, 1))[:bs]
                    pooled_tgt_expanded = pooled_tgt.repeat((bs, 1))[:bs]
                    neg_pooled_tgt_expanded = neg_pooled_tgt.repeat((bs, 1))[:bs]
                else:
                    tgt_expanded = tgt
                    neg_tgt_expanded = neg_tgt
                    pooled_tgt_expanded = pooled_tgt
                    neg_pooled_tgt_expanded = neg_pooled_tgt

                combined_prompt_embeds = torch.cat([prompt_embeds, tgt_expanded], dim=0)
                combined_negative_prompt_embeds = torch.cat([negative_prompt_embeds, neg_tgt_expanded], dim=0)
                combined_pooled = torch.cat([pooled_prompt_embeds, pooled_tgt_expanded], dim=0)
                combined_negative_pooled = torch.cat([negative_pooled_prompt_embeds, neg_pooled_tgt_expanded], dim=0)

                if target_embeds.shape[0] > 1:
                    target_latents = latent.repeat(tgt_expanded.shape[0], 1, 1, 1)
                    latent_bs = torch.cat([latent, target_latents], dim=0)
                else:
                    latent_bs = latent.repeat(2, 1, 1, 1)

                cached_conditioned = (
                    combined_prompt_embeds,
                    combined_negative_prompt_embeds,
                    combined_pooled,
                    combined_negative_pooled,
                    latent_bs,
                )

            combined_prompt_embeds, combined_negative_prompt_embeds, combined_pooled, combined_negative_pooled, latent_bs = cached_conditioned
            JointAttnProcessor.reset_phase_cache()
            pipe.transformer = set_attenprocessor(
                pipe.transformer,
                atten_type=('retain' if mode == 'retain' else 'erase'),
                record=False,
                record_type=record_type,
                sigmoid_setting=(sigmoid_a, sigmoid_b, sigmoid_c),
                token_sim=args.token_sim,
                decomp_timestep=args.decomp_timestep,
                target_concept_encodings=target_embeds,
                phase=args.phase,
                top_k=effective_top_k,
            )
            all_imgs = run_pipe_and_decode(
                pipe,
                prompt_embeds=combined_prompt_embeds,
                negative_prompt_embeds=combined_negative_prompt_embeds,
                pooled_prompt_embeds=combined_pooled,
                negative_pooled_prompt_embeds=combined_negative_pooled,
                latents=latent_bs,
                num_images_per_prompt=1,
                guidance_scale=guidance_scale,
                num_inference_steps=args.total_timesteps
            )
            decoded_imgs[mode] = all_imgs[:bs]

        return decoded_imgs

    for sigmoid_a in sigmoid_a_values:
        for sigmoid_b in sigmoid_b_values:
            for sigmoid_c in sigmoid_c_values:
                if use_csv:
                    for idx, row in selected_prompts_df.iterrows():
                        prompt = row['prompt']
                        csv_guidance_scale = get_csv_param(row, ['sd_guidance_scale', 'guidance_scale'], float, args.guidance_scale)
                        for i in range(int(args.num_samples // bs)):
                            save_path = get_save_path(prompt, None, sigmoid_a, sigmoid_b, sigmoid_c)
                            if sample_outputs_exist(save_path, prompt, idx, i):
                                print(f"Skipping existing sample: {get_save_filename(prompt, idx, 0, i)}")
                                continue

                            latent_key = f"csv_{idx}_{i}"
                            latent = fixed_latents[latent_key]
                            decoded_imgs = generate_mode_images(prompt, latent, csv_guidance_scale, sigmoid_a, sigmoid_b, sigmoid_c)

                            for mode in mode_list:
                                os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                            if len(mode_list) > 1:
                                os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)

                            for idx_img in range(len(decoded_imgs[mode_list[0]])):
                                save_filename = get_save_filename(prompt, idx, idx_img, i)
                                images_to_combine = []
                                for mode in mode_list:
                                    decoded_imgs[mode][idx_img].save(os.path.join(save_path, mode, save_filename))
                                    images_to_combine.append(decoded_imgs[mode][idx_img])
                                if len(mode_list) > 1:
                                    img_combined = combine_images_horizontally(images_to_combine)
                                    img_combined.save(os.path.join(save_path, 'combine', save_filename))

                            del decoded_imgs
                            torch.cuda.empty_cache()
                else:
                    target_concept = args.target_concept.split(',')[0].strip()
                    ordered_concept_list = []
                    ordered_prompt_list = []

                    for idx, concept in enumerate(concept_list):
                        if concept == target_concept:
                            ordered_concept_list.insert(0, concept)
                            ordered_prompt_list.insert(0, prompt_list[idx])
                        else:
                            ordered_concept_list.append(concept)
                            ordered_prompt_list.append(prompt_list[idx])

                    for i in range(int(args.num_samples // bs)):
                        num_prompts = len(ordered_prompt_list[0]) if ordered_prompt_list else 0
                        for prompt_idx in range(num_prompts):
                            for concept, prompts in zip(ordered_concept_list, ordered_prompt_list):
                                prompt = prompts[prompt_idx]
                                save_path = get_save_path(prompt, concept, sigmoid_a, sigmoid_b, sigmoid_c)
                                if sample_outputs_exist(save_path, prompt, prompt_idx, i):
                                    print(f"Skipping existing sample: {concept}/{get_save_filename(prompt, prompt_idx, 0, i)}")
                                    continue

                                latent_key = f"{concept}_{prompt}_{i}"
                                latent = fixed_latents[latent_key]
                                decoded_imgs = generate_mode_images(prompt, latent, args.guidance_scale, sigmoid_a, sigmoid_b, sigmoid_c)

                                for mode in mode_list:
                                    os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                                if len(mode_list) > 1:
                                    os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)

                                for idx_img in range(len(decoded_imgs[mode_list[0]])):
                                    save_filename = get_save_filename(prompt, prompt_idx, idx_img, i)
                                    images_to_combine = []
                                    for mode in mode_list:
                                        decoded_imgs[mode][idx_img].save(os.path.join(save_path, mode, save_filename))
                                        images_to_combine.append(decoded_imgs[mode][idx_img])
                                    if len(mode_list) > 1:
                                        img_combined = combine_images_horizontally(images_to_combine)
                                        img_combined.save(os.path.join(save_path, 'combine', save_filename))

                                del decoded_imgs
                                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
