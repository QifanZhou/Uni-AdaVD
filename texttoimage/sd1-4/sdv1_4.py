import argparse
import copy
import csv
import hashlib
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
from einops import rearrange
from torch import nn
from tqdm import tqdm

from template import template_dict
from utils import seed_everything

class VisualAttentionProcess(nn.Module):
    def __init__(self, module_name=None, atten_type='original', target_records=None, record=False,
                 record_type=None, sigmoid_setting=None, decomp_timestep=0, cross_attention_dim=None):
        super().__init__()
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records = target_records
        self.record = record
        self.record_type = record_type.strip().split(',') if record_type is not None else []
        self.sigmoid_setting = sigmoid_setting
        self.decomp_timestep = decomp_timestep
        self.cross_attention_dim = cross_attention_dim

    def __call__(self, attn, hidden_states, encoder_hidden_states, *args, **kwargs):
        attn._modules.pop("processor")
        attn.processor = AttnProcessor(
            self.module_name, self.atten_type, self.target_records, self.record,
            self.record_type, self.sigmoid_setting, self.decomp_timestep, self.cross_attention_dim
        )
        return attn.processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)

class AttnProcessor:
    def __init__(self, module_name=None, atten_type='original', target_records=None, record=False,
                 record_type=None, sigmoid_setting=None, decomp_timestep=0, cross_attention_dim=None) -> None:
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records = copy.copy(target_records) if target_records else {}
        self.record = record
        if record_type is not None:
            if isinstance(record_type, list):
                self.record_type = [str(item).strip() for item in record_type]
            else:
                self.record_type = [str(record_type).strip()]
        else:
            self.record_type = []
        self.records = {key: {} for key in self.record_type} if self.record_type else {}
        self.sigmoid_setting = sigmoid_setting
        self.decomp_timestep = decomp_timestep
        self.cross_attention_dim = cross_attention_dim
        self.ORTHO_DECOMP_STORAGE = {}

    def sigmoid(self, x, setting):
        a, b, c = setting
        return c / (1 + torch.exp(-a * (x - b)))

    def cal_ortho_decomp(self, target_value, pro_record, ortho_basis=None, project_matrix=None):
        if ortho_basis is None and project_matrix is None:
            tar_record_ = target_value[0].permute(1, 0, 2).reshape(77, -1)
            pro_record_ = pro_record.permute(1, 0, 2).reshape(77, -1)
            cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)
            if self.sigmoid_setting:
                cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)
            weight = torch.nan_to_num(cos_sim * (torch.sum(tar_record_ * pro_record_, dim=-1) / torch.sum(tar_record_ * tar_record_, dim=-1)), nan=0.0)
            weight[0].fill_(0)
            era_record = weight.unsqueeze(0).unsqueeze(-1) * tar_record_.view((77, 16, -1)).permute(1, 0, 2)
        else:
            tar_record_ = rearrange(target_value, 'b h l d -> l b (h d)')
            pro_record_ = rearrange(pro_record, 'h l d -> l (h d)').unsqueeze(1)
            dot1 = (ortho_basis * pro_record_).sum(-1)
            dot2 = (ortho_basis * ortho_basis).sum(-1)
            weight = torch.nan_to_num((dot1 / dot2).unsqueeze(1), nan=0.0)
            weight[0].fill_(0)

            cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)
            if self.sigmoid_setting:
                cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)
            projected_basis = torch.bmm(project_matrix, cos_sim.unsqueeze(-1) * tar_record_)
            era_record = torch.bmm(weight, projected_basis).view((77, 16, -1)).permute(1, 0, 2)
        return era_record

    def record_ortho_decomp(self, target_record, current_record):
        current_name = next((k for k in target_record if k.endswith(self.module_name)), None)
        if not current_name:
            return current_record, current_record

        current_timestep, current_block = current_name.split('.', 1)
        target_value, project_matrix, ortho_basis = target_record.pop(current_name)

        if int(current_timestep) <= self.decomp_timestep:
            return current_record, current_record

        target_value = target_value.view((2, int(len(target_value)//16), -1) + target_value.size()[-2:])
        target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1) + target_value.size()[-2:])
        current_record = current_record.view((2, int(len(current_record)//16), -1) + target_value.size()[-2:])
        current_record = current_record.permute(1, 0, 2, 3, 4).contiguous().view((current_record.size()[1], -1) + target_value.size()[-2:])

        erase_record, retain_record = [], []
        for pro_record in current_record:
            era_record = self.cal_ortho_decomp(target_value, pro_record, ortho_basis, project_matrix)
            ret_record = pro_record - era_record
            erase_record.append(era_record.view((2, -1) + era_record.size()[-2:]))
            retain_record.append(ret_record.view((2, -1) + era_record.size()[-2:]))
        retain_record = rearrange(torch.stack(retain_record, dim=0), 'b n c l d -> (n b c) l d')
        erase_record = rearrange(torch.stack(erase_record, dim=0), 'b n c l d -> (n b c) l d')
        self.ORTHO_DECOMP_STORAGE[current_block] = (erase_record, retain_record)
        return self.ORTHO_DECOMP_STORAGE[current_block]

    def cal_gram_schmidt(self, target_value):
        target_value = target_value.view((2, int(len(target_value)//16), -1) + target_value.size()[-2:])
        target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1) + target_value.size()[-2:])
        target_value_ = rearrange(target_value, 'b h l d -> b l (h d)')
        results = [self.gram_schmidt(target_value_[:, i, :]) for i in range(target_value_.size()[1])]
        project_matrix = torch.stack([result[0] for result in results], dim=0)
        basis_ortho = torch.stack([result[1] for result in results], dim=0)
        return project_matrix, basis_ortho

    def gram_schmidt(self, V):
        n = V.size(0)
        project_matrix = torch.eye(n, dtype=V.dtype, device=V.device)
        for i in range(1, n):
            vi = V[i:i+1, :]
            for j in range(i):
                qj = V[j:j+1, :]
                proj = (qj @ vi.T) / (qj @ qj.T)
                project_matrix[i, j] = -proj.item()
        ortho_basis = project_matrix @ V
        return project_matrix, ortho_basis

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        if not self.record and encoder_hidden_states.shape[1] == 77:
            if 'queries' in self.target_records:
                erase_query, retain_query = self.record_ortho_decomp(
                    self.target_records['queries'], query
                )
                query = retain_query if self.atten_type == 'retain' else erase_query if self.atten_type == 'erase' else query
            if 'keys' in self.target_records:
                erase_key, retain_key = self.record_ortho_decomp(
                    self.target_records['keys'], key
                )
                key = retain_key if self.atten_type == 'retain' else erase_key if self.atten_type == 'erase' else key
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if not self.record and encoder_hidden_states.shape[1] == 77:
            if 'attn_maps' in self.target_records:
                erase_attn, retain_attn = self.record_ortho_decomp(
                    self.target_records['attn_maps'], attention_probs
                )
                attention_probs = retain_attn if self.atten_type == 'retain' else erase_attn if self.atten_type == 'erase' else attention_probs
        if encoder_hidden_states.shape[1] != 77:
            hidden_states = torch.bmm(attention_probs, value)
        else:
            if self.record:
                for kk, vv in {'queries': query, 'keys': key, 'values': value, 'attn_maps': attention_probs}.items():
                    if kk in self.record_type:
                        if vv.shape[0] // 16 == 1:
                            self.records[kk][self.module_name] = [vv] + [None, None]
                        else:
                            self.records[kk][self.module_name] = [vv] + list(self.cal_gram_schmidt(vv))
            elif 'values' in self.target_records:
                erase_value, retain_value = self.record_ortho_decomp(
                    self.target_records['values'], value
                )
                value = retain_value if self.atten_type == 'retain' else erase_value if self.atten_type == 'erase' else value
            hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor

def set_attenprocessor(unet, atten_type='original', target_records=None, record=False,
                       record_type=None, sigmoid_setting=None, decomp_timestep=0):
    for name, m in unet.named_modules():
        if name.endswith(('attn2', 'attn1')):
            cross_attention_dim = None
            if name.endswith("attn1"):  # self-attention
                cross_attention_dim = unet.config.cross_attention_dim
            else:  # cross-attention
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name.split('.')[1])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name.split('.')[1])
                    hidden_size = unet.config.block_out_channels[block_id]
                cross_attention_dim = hidden_size
            m.set_processor(VisualAttentionProcess(
                module_name=name,
                atten_type=atten_type,
                target_records=target_records,
                record=record,
                record_type=record_type,
                sigmoid_setting=sigmoid_setting,
                decomp_timestep=decomp_timestep,
                cross_attention_dim=cross_attention_dim
            ))
    return unet

def diffusion(unet, scheduler, latents, text_embeddings, total_timesteps, start_timesteps=0,
              guidance_scale=7.5, record=False, record_type=None, desc=None):
    visualize_map_withstep = {key: {} for key in record_type.split(',')} if record_type else {}
    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps], desc=desc):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        noise_pred = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample
        if record:
            for type_ in record_type.split(','):
                for processor in unet.attn_processors.values():
                    for k, v in processor.records[type_].items():
                        visualize_map_withstep[type_][f'{timestep.item()}.{k}'] = v
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample
    return (latents, visualize_map_withstep) if record else latents

def process_img(img_tensor):
    img_np = (img_tensor.cpu().numpy() + 1.0) / 2.0
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np.transpose(1, 2, 0))
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img_pil

ORTHO_DECOMP_STORAGE = {}


def slugify(text):
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", text.strip())
    return slug.strip("-") or "concept"


def expand_to_77(sot_emb, token_emb):
    return torch.cat([sot_emb] + [token_emb] * 76, dim=1)


def parse_target_concepts(target_concept_arg):
    target_concepts = [item.strip() for item in target_concept_arg.split(",") if item.strip()]
    if not target_concepts:
        raise ValueError("`--target_concept` must contain at least one non-empty concept.")
    return target_concepts


def stable_uint32_seed(*parts):
    raw = "||".join(str(part) for part in parts)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def build_template_prompt_df(args):
    erase_type = (args.erase_type or "").strip().lower()
    if erase_type not in template_dict:
        raise ValueError("Template mode requires `--erase_type` in {instance, style, celebrity}.")

    contents = [item.strip() for item in args.contents.split(",") if item.strip()]
    if not contents:
        raise ValueError("Template mode requires at least one concept in `--contents`.")

    rows = []
    for content in contents:
        for template_idx, prompt_template in enumerate(template_dict[erase_type]):
            prompt = prompt_template.format(content)
            rows.append(
                {
                    "prompt": prompt,
                    "content": content,
                    "template_idx": template_idx,
                    "case_number": f"{slugify(content)}_{template_idx:03d}",
                    "sd_seed": stable_uint32_seed(args.seed, erase_type, content, prompt),
                }
            )
    return pd.DataFrame(rows)


def build_target_embedding(concept, tokenizer, text_encoder, token_processing, device):
    token_output = tokenizer(
        concept,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    ).to(device)
    input_ids = token_output["input_ids"]
    encoding = text_encoder(input_ids)[0]
    eos_indices = (input_ids[0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    eot_idx = eos_indices[0].item() if len(eos_indices) > 0 else input_ids.shape[1] - 1

    if token_processing == "last_subject":
        if eot_idx == 0:
            raise ValueError("The EOT index cannot be 0.")
        subject_emb = encoding[:, eot_idx - 1, :].unsqueeze(1)
        return expand_to_77(encoding[:, :1], subject_emb)

    if token_processing == "last_subject_eot_mean":
        if eot_idx == 0:
            raise ValueError("The EOT index cannot be 0.")
        subject_emb = encoding[:, eot_idx - 1, :].unsqueeze(1)
        eot_emb = encoding[:, eot_idx:, :]
        combined_emb = torch.cat([subject_emb, eot_emb], dim=1)
        target_embedding = combined_emb.mean(dim=1, keepdim=True)
        return expand_to_77(encoding[:, :1], target_embedding)

    raise ValueError(f"Unsupported token processing mode: {token_processing}")

@torch.no_grad()
def main():
    global ORTHO_DECOMP_STORAGE
    parser = argparse.ArgumentParser(description="Uni-AdaVD inference-time concept erasure for SD v1_4.")
    parser.add_argument('--save_root', type=str, default='./outputs', help='Root directory for saved outputs.')
    parser.add_argument('--sd_ckpt', type=str, default='CompVis/stable-diffusion-v1-4', help='Local checkpoint path or Hugging Face model ID.')
    parser.add_argument('--mode', type=str, default='original,erase,retain', help='Comma-separated modes.')
    parser.add_argument('--erase_type', type=str, default='', help='Concept family. Use instance/style/celebrity for template mode or nsfw for CSV safety mode.')
    parser.add_argument('--contents', type=str, default='', help='Comma-separated concepts used to fill templates in template mode.')
    parser.add_argument('--prompt_file', type=str, default=None, help='CSV file with prompt metadata.')
    parser.add_argument('--batch_size', type=int, default=1, help='Reserved for backward compatibility. Use `--num_samples` to control outputs per prompt.')
    parser.add_argument('--num_samples', type=int, default=10, help='The number of samples per prompt to generate')
    parser.add_argument('--total_timesteps', type=int, default=20, help='Number of denoising steps.')
    parser.add_argument('--prompt_start', type=int, default=0, help='Start index in the CSV file.')
    parser.add_argument('--prompt_end', type=int, default=None, help='End index in the CSV file (exclusive).')
    parser.add_argument('--seed', type=int, default=0, help='Base seed used in template mode and as a fallback when the CSV does not provide one.')
    parser.add_argument('--target_concept', type=str, default='nudity', help='Target concept(s). Use comma separation for multi-concept erasure.')
    parser.add_argument('--token_processing', type=str, default='last_subject', choices=['last_subject', 'last_subject_eot_mean'], help='Strategy for constructing the target token representation.')
    parser.add_argument('--csv_encoding', type=str, default='iso-8859-1', help='CSV encoding.')
    parser.add_argument('--sigmoid_a', type=float, default=100, help='Sigmoid parameter a.')
    parser.add_argument('--sigmoid_b', type=float, default=0.93, help='Sigmoid parameter b.')
    parser.add_argument('--sigmoid_c', type=float, default=2, help='Sigmoid parameter c.')
    parser.add_argument('--use_coco_30k', action='store_true', help='Use the COCO-30k CSV and ignore `--prompt_file`.')
    parser.add_argument('--coco_30k_path', type=str, default=None, help='Path to the COCO-30k CSV file.')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Fallback guidance scale. Used for COCO and for I2P rows without a guidance column.')
    parser.add_argument('--device', type=str, default='cuda', help='Inference device, e.g. `cuda` or `cuda:0`.')

    args = parser.parse_args()

    mode_list = [m.strip() for m in args.mode.split(',') if m.strip()]
    valid_modes = {'original', 'erase', 'retain'}
    invalid_modes = [mode for mode in mode_list if mode not in valid_modes]
    if invalid_modes:
        raise ValueError(f"Unsupported mode(s): {invalid_modes}. Supported modes: {sorted(valid_modes)}")
    if args.batch_size < 1:
        raise ValueError("`--batch_size` must be at least 1.")
    if args.num_samples < 1:
        raise ValueError("`--num_samples` must be at least 1.")
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    ORTHO_DECOMP_STORAGE = {}
    use_template_mode = bool((args.contents or "").strip())
    if use_template_mode and (args.prompt_file or args.use_coco_30k):
        raise ValueError("Template mode via `--contents` cannot be combined with `--prompt_file` or `--use_coco_30k`.")
    if use_template_mode:
        if (args.erase_type or "").strip().lower() not in {"instance", "style", "celebrity"}:
            raise ValueError("Template mode requires `--erase_type` in {instance, style, celebrity}.")
    else:
        if (args.erase_type or "").strip().lower() != "nsfw":
            raise ValueError("CSV mode requires `--erase_type nsfw`.")

    COLUMN_MAPPINGS = {
        'guidance_scale': ['evaluation_guidance', 'sd_guidance_scale'],
        'seed': ['evaluation_seed', 'sd_seed']
    }

    def find_column(df, target_names, required=True, is_coco=False):
        for name in target_names:
            if name in df.columns:
                return name
        if is_coco and "guidance_scale" in target_names:
            return None
        if required:
            raise ValueError(f"Missing required column. Expected one of: {target_names}")
        return None

    os.makedirs(args.save_root, exist_ok=True)

    try:
        if use_template_mode:
            prompt_file = None
            df = build_template_prompt_df(args)
            seed_col = 'sd_seed'
            guidance_col = None
            dataset_type = "template"
            selected_prompts = df
            print(f"Generated {len(selected_prompts)} template prompt(s) from contents: {args.contents}")
        else:
            if args.use_coco_30k:
                if not args.coco_30k_path:
                    raise ValueError("`--coco_30k_path` is required when `--use_coco_30k` is enabled.")
                prompt_file = args.coco_30k_path
                print(f"Using COCO-30k prompts from: {prompt_file}")
            else:
                if not args.prompt_file:
                    raise ValueError("`--prompt_file` is required unless `--use_coco_30k` is enabled.")
                prompt_file = args.prompt_file
                print(f"Using prompts from: {prompt_file}")
            if not os.path.isfile(prompt_file):
                raise FileNotFoundError(f"Prompt CSV not found: {prompt_file}")

            df = pd.read_csv(
                prompt_file,
                encoding=args.csv_encoding,
                quoting=csv.QUOTE_ALL,
                engine='python',
                on_bad_lines='skip'
            )
            df = df[df['prompt'].notna()].reset_index(drop=True)
            is_coco_30k = args.use_coco_30k or 'coco_30k' in os.path.basename(prompt_file).lower()
            dataset_type = "coco" if is_coco_30k else "i2p"
            print(f"Detected dataset type: {dataset_type.upper()}")

            if dataset_type == "coco":
                guidance_col = None
                print(f"Using command-line guidance scale for COCO: {args.guidance_scale}")
            else:
                guidance_col = find_column(
                    df, COLUMN_MAPPINGS['guidance_scale'],
                    required=False,
                    is_coco=False
                )
                if guidance_col is None:
                    print(
                        "Warning: no guidance column found in the I2P CSV. "
                        f"Falling back to guidance_scale={args.guidance_scale}."
                    )

            seed_col = find_column(df, COLUMN_MAPPINGS['seed'], required=True)

            if is_coco_30k:
                if 'prompt' not in df.columns or seed_col is None:
                    raise ValueError("The COCO-30k CSV must contain `prompt` and a valid seed column.")
                print(f"Loaded COCO-30k CSV with {len(df)} rows.")
            else:
                if seed_col is None:
                    raise ValueError(f"The I2P CSV must contain one of {COLUMN_MAPPINGS['seed']}.")
                print(f"Loaded I2P CSV with {len(df)} rows.")

            if args.prompt_end is None:
                args.prompt_end = len(df)
            if args.prompt_end > len(df):
                print(f"Warning: prompt_end ({args.prompt_end}) exceeds CSV length ({len(df)}). Clamping to dataset size.")
                args.prompt_end = len(df)
            if args.prompt_start >= args.prompt_end:
                print("Invalid prompt index range.")
                return
            selected_prompts = df.iloc[args.prompt_start:args.prompt_end]
            print(f"Processing {len(selected_prompts)} prompt(s).")
        if dataset_type == "template":
            print(f"Using command-line guidance scale for templates: {args.guidance_scale}")
        elif dataset_type == "coco":
            guidance_col = None
    except Exception as e:
        print(f"Failed to load the CSV file: {e}")
        return

    try:
        weight_dtype = torch.float16 if args.device.startswith('cuda') else torch.float32
        print(f"Loading Stable Diffusion v1.4 pipeline from: {args.sd_ckpt}")
        pipe = DiffusionPipeline.from_pretrained(
            args.sd_ckpt, safety_checker=None, torch_dtype=weight_dtype
        )
        print(f"Moving pipeline to device: {args.device}")
        pipe = pipe.to(args.device)
        print("Pipeline ready.")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae
    except Exception as e:
        print(f"Failed to load the diffusion pipeline: {e}")
        return

    try:
        target_concepts = parse_target_concepts(args.target_concept)
        print(f"Building target embedding(s) for: {', '.join(target_concepts)}")
        target_embedding = torch.cat(
            [
                build_target_embedding(
                    concept,
                    tokenizer,
                    text_encoder,
                    args.token_processing,
                    pipe.device,
                )
                for concept in target_concepts
            ],
            dim=0,
        )
    except Exception as e:
        print(f"Failed to build target concept embeddings: {e}")
        return

    print(f"Loaded {len(target_concepts)} target concept(s): {', '.join(target_concepts)}")
    concept_tag = "__".join(slugify(concept) for concept in target_concepts)
    dataset_tag = dataset_type if dataset_type != "template" else f"template_{(args.erase_type or 'explicit').strip().lower()}"
    save_dir = os.path.join(
        args.save_root,
        f"{dataset_tag}_token_{args.token_processing}_target_{concept_tag}_a{args.sigmoid_a}_b{args.sigmoid_b}_c{args.sigmoid_c}",
    )
    os.makedirs(save_dir, exist_ok=True)

    config_path = os.path.join(save_dir, "run_config.txt")
    with open(config_path, "w", encoding="utf-8") as config_file:
        config_file.write(f"sd_ckpt: {args.sd_ckpt}\n")
        config_file.write(f"erase_type: {args.erase_type}\n")
        config_file.write(f"contents: {args.contents}\n")
        config_file.write(f"prompt_file: {prompt_file if prompt_file is not None else 'generated_from_contents'}\n")
        config_file.write(f"dataset_type: {dataset_type}\n")
        config_file.write(f"mode: {args.mode}\n")
        config_file.write(f"seed: {args.seed}\n")
        config_file.write(f"target_concept: {args.target_concept}\n")
        config_file.write(f"token_processing: {args.token_processing}\n")
        config_file.write(f"guidance_scale: {args.guidance_scale}\n")
        config_file.write(f"batch_size: {args.batch_size}\n")
        config_file.write(f"total_timesteps: {args.total_timesteps}\n")
        config_file.write(f"num_samples: {args.num_samples}\n")
        config_file.write(f"sigmoid_a: {args.sigmoid_a}\n")
        config_file.write(f"sigmoid_b: {args.sigmoid_b}\n")
        config_file.write(f"sigmoid_c: {args.sigmoid_c}\n")
        if dataset_type == "template":
            config_file.write("prompt_range: full_template_set\n")
        else:
            config_file.write(f"prompt_range: [{args.prompt_start}, {args.prompt_end})\n")
        config_file.write(f"device: {args.device}\n")

    target_records = {}
    if 'erase' in mode_list or 'retain' in mode_list:
        print("Preparing target-side value records for AdaVD retain/erase modes...")
        print("Copying reference UNet...")
        unet_temp = copy.deepcopy(unet)
        unet_temp = set_attenprocessor(unet_temp, atten_type='original', record=True, record_type='values')
        num_target_concepts = target_embedding.shape[0]
        latents_temp = torch.zeros(num_target_concepts, 4, 64, 64, device=pipe.device, dtype=pipe.unet.dtype)
        uncond_token = tokenizer(
            [""] * num_target_concepts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=77
        ).to(pipe.device)
        uncond_encoding_temp = text_encoder(uncond_token['input_ids'])[0]
        try:
            text_embeddings_temp = torch.cat([uncond_encoding_temp, target_embedding], dim=0)
        except Exception as e:
            raise RuntimeError("Failed to concatenate unconditional and target embeddings.") from e
        _, target_records = diffusion(
            unet_temp, pipe.scheduler, latents_temp, text_embeddings_temp,
            total_timesteps=1, guidance_scale=7.5, record=True, record_type='values'
        )
        pipe.scheduler.set_timesteps(args.total_timesteps)
        original_keys = list(target_records['values'].keys())
        target_records['values'].update({
            f"{timestep}.{'.'.join(key.split('.')[1:])}": target_records['values'][key]
            for timestep in pipe.scheduler.timesteps
            for key in original_keys
        })
        del unet_temp
        print("Reference target records prepared.")

    print(f"Creating per-mode UNet copies for: {', '.join(mode_list)}")
    models = {mode: copy.deepcopy(unet) for mode in mode_list}
    print("Per-mode UNet copies ready.")

    for idx, row in selected_prompts.iterrows():
        prompt = row['prompt']
        if seed_col in row and pd.notna(row[seed_col]):
            seed = int(row[seed_col])
        else:
            seed = stable_uint32_seed(args.seed, dataset_type, idx, prompt)
        coco_id = row.get('coco_id', 'unknown_coco_id')
        case_number = row.get('case_number', f'unknown_case_{idx}')
        content_name = str(row.get('content', '')).strip()

        if dataset_type in {"coco", "template"}:
            guidance_scale = args.guidance_scale
        else:
            guidance_scale = float(row[guidance_col]) if (guidance_col is not None and guidance_col in row) else args.guidance_scale
            if guidance_col is None:
                print(f"Notice: using fallback guidance scale {guidance_scale} for the current I2P sample.")

        seed_everything(seed, True)
        print(f"\nProcessing case {case_number}:")
        print(f" Prompt: {prompt[:80]}...")
        print(f" Seed: {seed}, Guidance: {guidance_scale}")
        print(f" Token processing: {args.token_processing}")
        print(f" Target concepts: {', '.join(target_concepts)}")
        if dataset_type == "template":
            print(f" Content: {content_name}")
        else:
            print(f" COCO ID: {coco_id}")

        token_prompt = tokenizer(
            prompt,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=77
        ).to(pipe.device)

        def combine_images_horizontally(images):
            widths, heights = zip(*(img.size for img in images))
            new_img = Image.new('RGB', (sum(widths), max(heights)))
            x_offset = 0
            for img in images:
                new_img.paste(img, (x_offset, 0))
                x_offset += img.width
            return new_img

        prompt_encoding = text_encoder(token_prompt['input_ids'])[0]
        current_save_dir = save_dir
        if dataset_type == "template" and content_name:
            current_save_dir = os.path.join(save_dir, slugify(content_name))
        for sample_start in range(0, args.num_samples, args.batch_size):
            current_batch_size = min(args.batch_size, args.num_samples - sample_start)
            text_emb = prompt_encoding.expand(current_batch_size, -1, -1)
            uncond_token = tokenizer(
                [""] * current_batch_size,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=77
            ).to(pipe.device)
            uncond_encoding = text_encoder(uncond_token['input_ids'])[0]
            text_emb = torch.cat([uncond_encoding, text_emb], dim=0)
            latents = torch.randn(current_batch_size, 4, 64, 64, device=pipe.device, dtype=pipe.unet.dtype)

            results = {}
            for mode in mode_list:
                if mode == 'original':
                    model = models['original']
                    out = diffusion(
                        model, pipe.scheduler, latents, text_emb,
                        total_timesteps=args.total_timesteps, guidance_scale=guidance_scale,
                        desc=f"{prompt[:30]} | original"
                    )
                else:
                    processor_mode = 'retain' if mode == 'retain' else 'erase'
                    model = set_attenprocessor(
                        models[mode],
                        atten_type=processor_mode,
                        target_records=copy.deepcopy(target_records),
                        sigmoid_setting=(args.sigmoid_a, args.sigmoid_b, args.sigmoid_c),
                        decomp_timestep=0
                    )
                    out = diffusion(
                        model, pipe.scheduler, latents, text_emb,
                        total_timesteps=args.total_timesteps, guidance_scale=guidance_scale,
                        desc=f"{prompt[:30]} | {processor_mode}"
                    )
                results[mode] = out

            batch_pil_imgs = {mode: [] for mode in mode_list}
            for mode, latent_batch in results.items():
                os.makedirs(os.path.join(current_save_dir, mode), exist_ok=True)
                images = vae.decode(latent_batch / vae.config.scaling_factor, return_dict=False)[0]

                for i, img in enumerate(images):
                    img_pil = process_img(img)
                    batch_pil_imgs[mode].append(img_pil)

                    sample_idx = sample_start + i
                    filename = f"case_{case_number}_s{seed}_g{guidance_scale:.1f}_i{sample_idx}.png"
                    img_pil.save(os.path.join(current_save_dir, mode, filename))
                    print(f" Saved [{mode}]: {filename}")

            if len(mode_list) > 1:
                combine_dir = os.path.join(current_save_dir, "combine")
                os.makedirs(combine_dir, exist_ok=True)

                for i in range(current_batch_size):
                    images_to_combine = []
                    for mode in mode_list:
                        if mode in batch_pil_imgs and len(batch_pil_imgs[mode]) > i:
                            images_to_combine.append(batch_pil_imgs[mode][i])

                    if len(images_to_combine) > 1:
                        combined_img = combine_images_horizontally(images_to_combine)
                        sample_idx = sample_start + i
                        combined_filename = f"case_{case_number}_s{seed}_g{guidance_scale:.1f}_i{sample_idx}_combined.png"
                        combined_img.save(os.path.join(combine_dir, combined_filename))
                        print(f" Generated combined image: {combined_filename}")

if __name__ == '__main__':
    main()
