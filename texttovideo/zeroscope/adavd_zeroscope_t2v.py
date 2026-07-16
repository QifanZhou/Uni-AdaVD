import argparse
import csv
import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from diffusers import TextToVideoSDPipeline
from PIL import Image
from tqdm import tqdm

FIXED_PROMPT_COL = "prompt"
FIXED_DEVICE = "cuda"
FIXED_DTYPE = torch.float16
FIXED_DTYPE_NAME = "fp16"
FIXED_DECOMP_TIMESTEP = 40


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism(enable: bool) -> None:
    if not enable:
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(True)


class AdaVDVideoAttnProcessor:
    def __init__(
        self,
        module_name: str,
        atten_type: str = "original",
        target_records: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        record: bool = False,
        record_type: Optional[str] = None,
        sigmoid_setting: Optional[tuple] = None,
        decomp_timestep: int = FIXED_DECOMP_TIMESTEP,
        text_seq_len: int = 77,
    ):
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records = target_records or {}
        self.record = record
        self.record_type = [x.strip() for x in (record_type or "").split(",") if x.strip()]
        self.records = {k: {} for k in self.record_type} if self.record_type else {}
        self.sigmoid_setting = sigmoid_setting
        self.decomp_timestep = int(decomp_timestep)
        self.text_seq_len = int(text_seq_len)
        self.current_timestep: Optional[int] = None

    @staticmethod
    def _safe_div(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return numer / torch.clamp(denom, min=eps)

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigmoid_setting is None:
            return x
        a, b, c = self.sigmoid_setting
        return c / (1.0 + torch.exp(-a * (x - b)))

    def _expand_target_to_runtime_batch(self, target_value: torch.Tensor, current_value: torch.Tensor) -> torch.Tensor:
        bt = target_value.shape[0]
        br = current_value.shape[0]

        if bt == br:
            return target_value

        if bt == 1:
            return target_value.expand(br, -1, -1)

        if bt == 2 and br % 2 == 0:
            half = br // 2
            target_uncond = target_value[0:1].expand(half, -1, -1)
            target_text = target_value[1:2].expand(half, -1, -1)
            return torch.cat([target_uncond, target_text], dim=0)

        return target_value.mean(dim=0, keepdim=True).expand(br, -1, -1)

    def _decompose_value(self, value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        input_dtype = value.dtype
        v = value.float()
        t = target_value.float()

        t = self._expand_target_to_runtime_batch(t, v)

        dot1 = (t * v).sum(dim=-1)
        dot2 = (t * t).sum(dim=-1)
        cos_sim = torch.nn.functional.cosine_similarity(t, v, dim=-1)
        cos_sim = self._sigmoid(cos_sim)

        weight = torch.nan_to_num(cos_sim * self._safe_div(dot1, dot2), nan=0.0, posinf=0.0, neginf=0.0)
        if weight.shape[1] > 0:
            weight[:, 0] = 0.0

        erase_value = weight.unsqueeze(-1) * t
        retain_value = v - erase_value

        if self.atten_type == "erase":
            out = erase_value
        elif self.atten_type == "retain":
            out = retain_value
        else:
            out = v

        return out.to(input_dtype)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        try:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        except TypeError:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        is_cross_attention = encoder_hidden_states is not None and encoder_hidden_states.shape[1] == self.text_seq_len

        if self.record and is_cross_attention and "values" in self.record_type:
            self.records["values"][self.module_name] = value.detach().clone()

        if (
            (not self.record)
            and is_cross_attention
            and (self.atten_type in {"erase", "retain"})
            and ("values" in self.target_records)
            and (self.module_name in self.target_records["values"])
        ):
            if self.current_timestep is None or int(self.current_timestep) > self.decomp_timestep:
                target_value = self.target_records["values"][self.module_name].to(device=value.device, dtype=value.dtype)
                value = self._decompose_value(value, target_value)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        return hidden_states / getattr(attn, "rescale_output_factor", 1.0)


def set_timestep_for_processors(unet: torch.nn.Module, timestep: int) -> None:
    for proc in unet.attn_processors.values():
        if isinstance(proc, AdaVDVideoAttnProcessor):
            proc.current_timestep = int(timestep)


def set_adavd_processors(
    unet: torch.nn.Module,
    atten_type: str = "original",
    target_records: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    record: bool = False,
    record_type: Optional[str] = None,
    sigmoid_setting: Optional[tuple] = None,
    decomp_timestep: int = FIXED_DECOMP_TIMESTEP,
    text_seq_len: int = 77,
) -> torch.nn.Module:
    for name, module in unet.named_modules():
        should_patch = name.endswith("attn2")
        if hasattr(module, "set_processor") and should_patch:
            module.set_processor(
                AdaVDVideoAttnProcessor(
                    module_name=name,
                    atten_type=atten_type,
                    target_records=target_records,
                    record=record,
                    record_type=record_type,
                    sigmoid_setting=sigmoid_setting,
                    decomp_timestep=decomp_timestep,
                    text_seq_len=text_seq_len,
                )
            )
    return unet


def restore_attn_processors(unet: torch.nn.Module, processors: Dict[str, torch.nn.Module]) -> None:
    unet.set_attn_processor(processors)


def sanitize_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\-. ]", "", s).strip().replace(" ", "_")
    return (s[:max_len] or "sample")


def load_prompt_items(prompt: str, prompt_file: str, prompt_start: int, prompt_end: int) -> List[Dict[str, object]]:
    if str(prompt).strip():
        return [{"index": None, "prompt": str(prompt).strip()}]

    if not str(prompt_file).strip():
        raise ValueError("Provide either --prompt or --prompt_file.")

    path = os.path.abspath(prompt_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    items: List[Dict[str, object]] = []

    if ext == ".csv":
        rows = None
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                with open(path, "r", encoding=encoding, newline="") as f:
                    rows = list(csv.DictReader(f))
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        if rows is None:
            raise RuntimeError(f"Failed to decode CSV file: {path}") from last_error
        if not rows:
            raise ValueError(f"No rows found in CSV: {path}")
        if FIXED_PROMPT_COL not in rows[0]:
            raise ValueError(f"Column '{FIXED_PROMPT_COL}' not found in CSV: {path}")
        for idx, row in enumerate(rows):
            value = str(row.get(FIXED_PROMPT_COL, "")).strip()
            if value:
                items.append({"index": idx, "prompt": value})
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                value = line.strip()
                if value:
                    items.append({"index": idx, "prompt": value})
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON prompt file must contain a list: {path}")
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                value = str(item.get(FIXED_PROMPT_COL, item.get("prompt", ""))).strip()
            else:
                value = str(item).strip()
            if value:
                items.append({"index": idx, "prompt": value})
    else:
        raise ValueError(f"Unsupported prompt file format: {path}")

    if prompt_start < 0:
        raise ValueError("--prompt_start must be >= 0.")

    end = len(items) if prompt_end < 0 else min(prompt_end, len(items))
    if prompt_start >= end:
        raise ValueError(f"Empty prompt range: start={prompt_start}, end={end}, total={len(items)}")

    return items[prompt_start:end]


def expand_to_length(sot_emb: torch.Tensor, token_emb: torch.Tensor, seq_len: int) -> torch.Tensor:
    return torch.cat([sot_emb] + [token_emb] * (seq_len - 1), dim=1)


def get_adavd_target_embedding(
    tokenizer,
    text_encoder,
    target_concept: str,
    token_processing: str,
    seq_len: int,
) -> torch.Tensor:
    device = text_encoder.device
    token_output = tokenizer(
        target_concept,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    ).to(device)
    input_ids = token_output["input_ids"].to(device)

    eos_token_id = tokenizer.eos_token_id
    eos_positions = (input_ids[0] == eos_token_id).nonzero(as_tuple=True)[0]
    eot_idx = eos_positions[0].item() if len(eos_positions) > 0 else input_ids.shape[1] - 1

    encoding = text_encoder(input_ids)[0]

    if token_processing == "last_subject":
        if eot_idx == 0:
            raise ValueError("EOT index cannot be 0 for last_subject")
        subject_emb = encoding[:, eot_idx - 1 : eot_idx, :]
        target_embedding = expand_to_length(encoding[:, :1, :], subject_emb, seq_len)
    elif token_processing == "eot_mean":
        if len(eos_positions) > 0:
            eot_emb = encoding[:, eos_positions, :]
            agg_emb = eot_emb.mean(dim=1, keepdim=True)
        else:
            agg_emb = encoding[:, :1, :]
        target_embedding = expand_to_length(encoding[:, :1, :], agg_emb, seq_len)
    elif token_processing == "last_subject_eot_mean":
        if eot_idx == 0:
            raise ValueError("EOT index cannot be 0 for last_subject_eot_mean")
        subject_emb = encoding[:, eot_idx - 1 : eot_idx, :]
        eot_emb = encoding[:, eot_idx:, :]
        combined = torch.cat([subject_emb, eot_emb], dim=1)
        agg_emb = combined.mean(dim=1, keepdim=True)
        target_embedding = expand_to_length(encoding[:, :1, :], agg_emb, seq_len)
    else:
        raise ValueError(f"Unsupported token_processing: {token_processing}")

    return target_embedding


@torch.no_grad()
def record_adavd_target_records(
    pipe: TextToVideoSDPipeline,
    target_concept: str,
    token_processing: str,
    num_frames: int,
    height: int,
    width: int,
    guidance_scale: float,
) -> Dict[str, Dict[str, torch.Tensor]]:
    device = pipe._execution_device
    dtype = pipe.unet.dtype
    seq_len = pipe.tokenizer.model_max_length

    target_embedding = get_adavd_target_embedding(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        target_concept=target_concept,
        token_processing=token_processing,
        seq_len=seq_len,
    ).to(device=device, dtype=dtype)

    uncond_input = pipe.tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    ).to(device)
    uncond_embedding = pipe.text_encoder(uncond_input.input_ids)[0].to(dtype=dtype)

    text_embeddings = torch.cat([uncond_embedding, target_embedding], dim=0)

    original_processors = dict(pipe.unet.attn_processors)
    try:
        set_adavd_processors(
            pipe.unet,
            atten_type="original",
            target_records=None,
            record=True,
            record_type="values",
            sigmoid_setting=None,
            decomp_timestep=0,
            text_seq_len=seq_len,
        )

        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.unet.config.in_channels,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=torch.Generator(device=device).manual_seed(0),
            latents=None,
        )
        latents = torch.zeros_like(latents)

        pipe.scheduler.set_timesteps(1, device=device)
        timestep = pipe.scheduler.timesteps[0]
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

        set_timestep_for_processors(pipe.unet, int(timestep.item()))
        _ = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )[0]

        target_records = {"values": {}}
        for proc in pipe.unet.attn_processors.values():
            if isinstance(proc, AdaVDVideoAttnProcessor):
                for k, v in proc.records.get("values", {}).items():
                    target_records["values"][k] = v.detach().clone()

        if len(target_records["values"]) == 0:
            raise RuntimeError("Failed to record AdaVD target records from UNet attention processors.")
    finally:
        restore_attn_processors(pipe.unet, original_processors)

    return target_records


@dataclass
class DenoiseOutput:
    latents: torch.Tensor


def export_to_video_opencv(frames, output_video_path: str, fps: int = 10) -> None:
    import cv2
    import numpy as np

    if isinstance(frames, np.ndarray):
        if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
            raise ValueError(f"Expected ndarray [F,H,W,C], got {frames.shape}")
        frame_iter = frames
    else:
        if not frames:
            raise ValueError("No frames to export.")
        frame_iter = frames

    first = frame_iter[0]
    if hasattr(first, "size") and not isinstance(first, np.ndarray):
        first = np.array(first)
    else:
        first = np.asarray(first)

    if first.ndim == 2:
        first = np.stack([first] * 3, axis=-1)
    if first.shape[-1] == 4:
        first = first[..., :3]
    if np.issubdtype(first.dtype, np.floating):
        first = np.clip(first * 255.0, 0, 255).astype(np.uint8)
    else:
        first = first.astype(np.uint8)

    height, width = first.shape[:2]
    writer = None
    for code in ("avc1", "H264", "mp4v"):
        trial = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*code),
            float(fps),
            (width, height),
        )
        if trial.isOpened():
            writer = trial
            break
        trial.release()
    if writer is None:
        raise RuntimeError(f"OpenCV VideoWriter failed to open: {output_video_path}")

    try:
        for frame in frame_iter:
            if hasattr(frame, "size") and not isinstance(frame, np.ndarray):
                arr = np.array(frame)
            else:
                arr = np.asarray(frame)

            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            writer.write(arr_bgr)
    finally:
        writer.release()


@torch.no_grad()
def decode_latents_to_uint8(
    pipe: TextToVideoSDPipeline,
    latents: torch.Tensor,
    vae_batch_size: int = 8,
) -> torch.Tensor:
    bsz, ch, frames, h_lat, w_lat = latents.shape
    latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, ch, h_lat, w_lat)

    decoded_batches = []
    for idx in range(0, latents_2d.shape[0], vae_batch_size):
        lat_batch = latents_2d[idx : idx + vae_batch_size].to(pipe.device, dtype=pipe.vae.dtype)
        lat_batch = lat_batch / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(lat_batch).sample
        decoded_batches.append(decoded.float().cpu())

    pixels = torch.cat(decoded_batches, dim=0)
    pixels = pixels.reshape(bsz, frames, pixels.shape[1], pixels.shape[2], pixels.shape[3]).permute(0, 1, 3, 4, 2)
    pixels = pixels.clamp(-1, 1).add(1).mul(127.5).to(torch.uint8)
    return pixels

@torch.no_grad()
def run_denoising(
    pipe: TextToVideoSDPipeline,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor],
    num_inference_steps: int,
    guidance_scale: float,
    generator,
    show_progress: bool = True,
) -> DenoiseOutput:
    device = latents.device
    do_classifier_free_guidance = guidance_scale > 1.0

    if do_classifier_free_guidance:
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    else:
        text_embeddings = prompt_embeds

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)

    iterator = timesteps
    if show_progress:
        iterator = tqdm(timesteps, desc="Denoising")

    for t in iterator:
        set_timestep_for_processors(pipe.unet, int(t.item()))

        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        bsz, channel, frames, h_lat, w_lat = latents.shape
        latents_4d = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, h_lat, w_lat)
        noise_pred_4d = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, h_lat, w_lat)

        latents_4d = pipe.scheduler.step(noise_pred_4d, t, latents_4d, **extra_step_kwargs).prev_sample
        latents = latents_4d[None, :].reshape(bsz, frames, channel, h_lat, w_lat).permute(0, 2, 1, 3, 4)

    return DenoiseOutput(latents=latents)


@torch.no_grad()
def decode_and_save(
    pipe: TextToVideoSDPipeline,
    latents: torch.Tensor,
    out_dir: str,
    fps: int,
    vae_batch_size: int = 8,
):
    os.makedirs(out_dir, exist_ok=True)

    videos = decode_latents_to_uint8(pipe, latents, vae_batch_size=vae_batch_size).numpy()

    for idx, video in enumerate(videos):
        sample_dir = out_dir if len(videos) == 1 else f"{out_dir}_{idx:02d}"
        os.makedirs(sample_dir, exist_ok=True)
        for frame_idx, frame in enumerate(video):
            Image.fromarray(frame).save(os.path.join(sample_dir, f"frame_{frame_idx:03d}.png"))


def parse_args():
    parser = argparse.ArgumentParser(description="AdaVD for ZeroScope Text-to-Video")
    parser.add_argument("--model_path", type=str, default="cerspense/zeroscope_v2_576w", help="Local checkpoint path or Hugging Face model ID for ZeroScope.")
    parser.add_argument("--save_root", type=str, default="./outputs/adavd_zeroscope", help="Root directory for saved frame sequences.")
    parser.add_argument("--prompt", type=str, default="", help="Single prompt used when --prompt_file is not provided.")
    parser.add_argument("--prompt_file", type=str, default="", help="Prompt file in CSV, TXT, or JSON format.")
    parser.add_argument("--prompt_start", type=int, default=0, help="Start index in the prompt file.")
    parser.add_argument("--prompt_end", type=int, default=-1, help="End index in the prompt file (exclusive). Use -1 to consume all remaining prompts.")
    parser.add_argument("--target_concept", type=str, required=True, help="Target concept to suppress.")
    parser.add_argument("--token_processing", type=str, default="last_subject", choices=["last_subject", "eot_mean", "last_subject_eot_mean"], help="Strategy for constructing the target token representation.")
    parser.add_argument("--mode", type=str, default="original,retain", help="comma-separated: original,erase,retain")
    parser.add_argument("--sigmoid_a", type=float, default=100.0, help="Sigmoid steepness used for AdaVD value scaling.")
    parser.add_argument("--sigmoid_b", type=float, default=0.43, help="Sigmoid center used for AdaVD value scaling.")
    parser.add_argument("--sigmoid_c", type=float, default=1.0, help="Sigmoid upper bound used for AdaVD value scaling.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of video frames to generate.")
    parser.add_argument("--height", type=int, default=320, help="Output frame height.")
    parser.add_argument("--width", type=int, default=576, help="Output frame width.")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Number of denoising steps. AdaVD intervention is restricted to the first 40 steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--num_videos", type=int, default=1, help="Number of videos to generate per prompt.")
    parser.add_argument("--fps", type=int, default=16, help="Frame rate metadata kept for downstream video assembly if needed.")
    parser.add_argument("--vae_batch_size", type=int, default=8, help="Number of latent frame batches decoded together during VAE export.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_determinism(True)
    seed_everything(args.seed)
    prompt_items = load_prompt_items(args.prompt, args.prompt_file, args.prompt_start, args.prompt_end)

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    allowed = {"original", "erase", "retain"}
    for m in modes:
        if m not in allowed:
            raise ValueError(f"Unsupported mode: {m}. Allowed: {sorted(allowed)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    print(f"Loading ZeroScope model from {args.model_path} on {FIXED_DEVICE} with dtype={FIXED_DTYPE_NAME}...", flush=True)
    load_kwargs = {"torch_dtype": FIXED_DTYPE}
    if os.path.isdir(args.model_path):
        load_kwargs["local_files_only"] = True
        load_kwargs["use_safetensors"] = False
    pipe = TextToVideoSDPipeline.from_pretrained(
        args.model_path,
        **load_kwargs,
    ).to(FIXED_DEVICE)
    print("ZeroScope model loaded.", flush=True)

    pipe.vae.enable_slicing()
    default_attn_processors = dict(pipe.unet.attn_processors)

    device = pipe._execution_device
    seq_len = pipe.tokenizer.model_max_length

    target_records = None
    if any(m in {"erase", "retain"} for m in modes):
        print(f"Recording AdaVD target records for concept: {args.target_concept}", flush=True)
        target_records = record_adavd_target_records(
            pipe=pipe,
            target_concept=args.target_concept,
            token_processing=args.token_processing,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
        )
        print(f"Recorded target attention values: {len(target_records['values'])} modules", flush=True)

    concept_slug = sanitize_filename(args.target_concept)
    do_cfg = args.guidance_scale > 1.0

    if args.prompt_file:
        end_text = "end" if args.prompt_end < 0 else str(args.prompt_end)
        print(f"Loaded {len(prompt_items)} prompts from {args.prompt_file} [{args.prompt_start}:{end_text}]", flush=True)

    for item_idx, item in enumerate(prompt_items, start=1):
        prompt_text = str(item["prompt"])
        prompt_idx = item["index"]
        prompt_tag = f"csv_idx={prompt_idx}" if prompt_idx is not None else "single_prompt"
        print(f"[Prompt {item_idx}/{len(prompt_items)}] {prompt_tag}", flush=True)
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=[prompt_text] * args.num_videos,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=None,
            clip_skip=None,
        )

        generator = torch.Generator(device=device).manual_seed(args.seed)
        base_latents = pipe.prepare_latents(
            batch_size=args.num_videos,
            num_channels_latents=pipe.unet.config.in_channels,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=None,
        )

        prompt_slug = sanitize_filename(prompt_text)
        file_stem = f"prompt_{prompt_idx:05d}_{prompt_slug}" if prompt_idx is not None else prompt_slug

        for mode in modes:
            print(f"[Prompt {item_idx}/{len(prompt_items)}] Running mode={mode}", flush=True)
            if mode == "original":
                restore_attn_processors(pipe.unet, default_attn_processors)
            else:
                set_adavd_processors(
                    pipe.unet,
                    atten_type=mode,
                    target_records=copy.deepcopy(target_records),
                    record=False,
                    record_type=None,
                    sigmoid_setting=(args.sigmoid_a, args.sigmoid_b, args.sigmoid_c),
                    decomp_timestep=FIXED_DECOMP_TIMESTEP,
                    text_seq_len=seq_len,
                )

            out = run_denoising(
                pipe=pipe,
                latents=base_latents.clone(),
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                show_progress=True,
            )

            save_dir = os.path.join(args.save_root, concept_slug, mode)
            os.makedirs(save_dir, exist_ok=True)

            frame_dir = os.path.join(save_dir, file_stem)
            decode_and_save(
                pipe,
                out.latents,
                frame_dir,
                fps=args.fps,
                vae_batch_size=args.vae_batch_size,
            )
            print(f"Done {mode}: {frame_dir}", flush=True)


if __name__ == "__main__":
    main()
