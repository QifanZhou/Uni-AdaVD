from __future__ import annotations

import torch
from torchvision.transforms import ToPILImage
from PIL.Image import Image as PILImage

from models.vqvae import VQVAEHF
from models.clip import FrozenCLIPEmbedder
from models.switti import SwittiHF, get_crop_condition
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng


TRAIN_IMAGE_SIZE = (512, 512)

TARGET_MODE_ALIASES = {
    "last_subject": "last_subject",
    "last_subject_eot_mean": "last_subject_eot_mean",
    "lastsubjectrepeat": "last_subject",
    "lastsubjecteotmeanrepeat": "last_subject_eot_mean",
    "lastsubjectandeotmeanrepeat": "last_subject_eot_mean",
}


def _normalize_target_mode(target_mode: str) -> str:
    mode = str(target_mode or "").strip().lower()
    if mode in TARGET_MODE_ALIASES:
        return TARGET_MODE_ALIASES[mode]
    raise ValueError(
        "Unsupported target_mode. Expected one of {last_subject, last_subject_eot_mean}."
    )

class SwittiPipeline:
    vae_path = "yresearch/VQVAE-Switti"
    text_encoder_path = "openai/clip-vit-large-patch14"
    text_encoder_2_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

    def __init__(self, switti, vae, text_encoder, text_encoder_2,
                 device, dtype=torch.float32,
                 ):
        self.switti = switti.to(dtype)
        self.vae = vae.to(dtype)
        self.text_encoder = text_encoder.to(dtype)
        self.text_encoder_2 = text_encoder_2.to(dtype)

        self.switti.eval()
        self.vae.eval()

        self.device = device

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        torch_dtype=torch.bfloat16,
                        device="cuda",
                        reso=1024,
                        ):
        switti = SwittiHF.from_pretrained(pretrained_model_name_or_path).to(device)
        vae = VQVAEHF.from_pretrained(cls.vae_path, reso=reso).to(device)
        text_encoder = FrozenCLIPEmbedder(cls.text_encoder_path, device=device)
        text_encoder_2 = FrozenCLIPEmbedder(cls.text_encoder_2_path, device=device)

        return cls(switti, vae, text_encoder, text_encoder_2, device, torch_dtype)

    @staticmethod
    def to_image(tensor):
        return [ToPILImage()(
            (255 * img.cpu().detach()).to(torch.uint8))
        for img in tensor]

    def _encode_prompt(self, prompt: str | list[str]):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        encodings = [
            self.text_encoder.encode(prompt),
            self.text_encoder_2.encode(prompt),
        ]
        prompt_embeds = torch.concat(
            [encoding.last_hidden_state for encoding in encodings], dim=-1
        )
        pooled_prompt_embeds = encodings[-1].pooler_output
        attn_bias = encodings[-1].attn_bias

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    @staticmethod
    def _attn_mask_to_bias(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Match FrozenCLIPEmbedder.forward(): 1 -> 0, 0 -> -inf
        attn_bias = attention_mask.to(dtype)
        attn_bias = attn_bias.clone()
        attn_bias[attn_bias == 0] = -float("inf")
        attn_bias[attn_bias == 1] = 0.0
        return attn_bias

    def _encode_with_input_ids(
        self, encoder: FrozenCLIPEmbedder, prompt: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns: (last_hidden_state [B,77,dim], input_ids [B,77])
        batch = encoder.tokenizer(
            prompt,
            truncation=True,
            max_length=encoder.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).to(encoder.device)
        outputs = encoder.transformer(**batch)
        return outputs["last_hidden_state"], batch["input_ids"]

    def _make_target_context_last_subject_repeat(
        self,
        target_concept: str,
        batch_size: int,
        device: str,
    ) -> torch.Tensor:
        """
        SD1-4 style target preprocessing:
        - encode `target_concept` with both CLIP encoders
        - pick the token embedding at (EOT index - 1), i.e. last_subject
        - repeat to length 77, then concat (CLIP-L + bigG) -> context_dim=2048
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        prompts = [target_concept] * batch_size

        h1, ids1 = self._encode_with_input_ids(self.text_encoder, prompts)  # [B,77,768]
        h2, ids2 = self._encode_with_input_ids(self.text_encoder_2, prompts)  # [B,77,1280]

        def last_subject_repeat(encoder: FrozenCLIPEmbedder, h: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            eos = encoder.tokenizer.eos_token_id
            if eos is None:
                # Fallback: last non-pad token (rare for CLIP)
                idx = (ids != encoder.tokenizer.pad_token_id).sum(dim=1) - 1
            else:
                # First EOS position, fallback to last position if not found
                eos_pos = (ids == eos).float()
                has_eos = eos_pos.sum(dim=1) > 0
                first_eos = eos_pos.argmax(dim=1)
                idx = torch.where(has_eos, first_eos - 1, torch.full_like(first_eos, ids.shape[1] - 1))
            idx = torch.clamp(idx, min=0, max=h.shape[1] - 1)
            subj = h[torch.arange(h.shape[0], device=h.device), idx]  # [B,dim]
            return subj[:, None, :].expand(-1, encoder.max_length, -1)

        t1 = last_subject_repeat(self.text_encoder, h1, ids1)  # [B,77,768]
        t2 = last_subject_repeat(self.text_encoder_2, h2, ids2)  # [B,77,1280]
        target_context = torch.cat([t1, t2], dim=-1).to(device)
        return target_context

    def _make_target_context_last_subject_eot_mean_repeat(
        self,
        target_concept: str,
        batch_size: int,
        device: str,
    ) -> torch.Tensor:
        """
        Target preprocessing aligned with the SD3 reference implementation:
        - encode `target_concept` with both CLIP encoders
        - take last-subject token embedding (first EOT index - 1)
        - collect all EOT token embeddings
        - average [last-subject, all-EOTs]
        - repeat to length 77, then concat (CLIP-L + bigG) -> context_dim=2048
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        prompts = [target_concept] * batch_size

        h1, ids1 = self._encode_with_input_ids(self.text_encoder, prompts)
        h2, ids2 = self._encode_with_input_ids(self.text_encoder_2, prompts)

        def last_subject_eot_mean_repeat(
            encoder: FrozenCLIPEmbedder, h: torch.Tensor, ids: torch.Tensor
        ) -> torch.Tensor:
            eos = encoder.tokenizer.eos_token_id
            if eos is None:
                idx = (ids != encoder.tokenizer.pad_token_id).sum(dim=1) - 1
                idx = torch.clamp(idx, min=0, max=h.shape[1] - 1)
                mixed = h[torch.arange(h.shape[0], device=h.device), idx]
            else:
                mixed_rows = []
                for bi in range(h.shape[0]):
                    eot_idxs = (ids[bi] == eos).nonzero(as_tuple=True)[0]
                    if len(eot_idxs) > 0:
                        first_eot = int(eot_idxs[0].item())
                        subj_idx = max(0, first_eot - 1)
                        subj_vec = h[bi, subj_idx : subj_idx + 1, :]
                        eot_vecs = h[bi, eot_idxs, :]
                        combined = torch.cat([subj_vec, eot_vecs], dim=0)
                        mixed_rows.append(combined.mean(dim=0))
                    else:
                        fallback_idx = int((ids[bi] != encoder.tokenizer.pad_token_id).sum().item() - 1)
                        fallback_idx = max(0, min(fallback_idx, h.shape[1] - 1))
                        mixed_rows.append(h[bi, fallback_idx, :])
                mixed = torch.stack(mixed_rows, dim=0)
            return mixed[:, None, :].expand(-1, encoder.max_length, -1)

        t1 = last_subject_eot_mean_repeat(self.text_encoder, h1, ids1)
        t2 = last_subject_eot_mean_repeat(self.text_encoder_2, h2, ids2)
        target_context = torch.cat([t1, t2], dim=-1).to(device)
        return target_context

    def _set_adavd_on_blocks(
        self,
        mode: str,
        target_context: torch.Tensor | None,
        sigmoid_setting: tuple[float, float, float] | None,
        record_target: bool,
        cfg_active: bool,
        debug_cos: bool = False,
        debug_print_limit: int = 1,
        cos_log_path: str | None = None,
        cos_log_dump_tokens: bool = False,
        target_contexts: list[torch.Tensor] | None = None,
        target_concept_names: list[str] | None = None,
        cos_log_dump_tokens_per_concept: bool = False,
        apply_to_uncond: bool = False,
    ) -> None:
        mode = (mode or "original").strip().lower()
        for b in self.switti.blocks:
            ca = getattr(b, "cross_attn", None)
            if ca is None:
                continue
            ca.set_adavd(
                mode=mode,
                target_context=target_context,
                target_contexts=target_contexts,
                target_concept_names=target_concept_names,
                sigmoid_setting=sigmoid_setting,
                record_target=record_target,
                cfg_active=cfg_active,
                apply_to_uncond=apply_to_uncond,
            )
            ca.set_adavd_debug(
                enable=debug_cos,
                print_limit=debug_print_limit,
                cos_log_path=cos_log_path,
                dump_tokens=cos_log_dump_tokens,
                dump_tokens_per_concept=cos_log_dump_tokens_per_concept,
            )

    def _set_adavd_cfg_active(self, cfg_active: bool) -> None:
        for b in self.switti.blocks:
            ca = getattr(b, "cross_attn", None)
            if ca is None:
                continue
            ca.set_adavd_cfg_active(cfg_active)

    def _set_adavd_target_context(self, target_context: torch.Tensor | None) -> None:
        for b in self.switti.blocks:
            ca = getattr(b, "cross_attn", None)
            if ca is None:
                continue
            ca.set_adavd_target_context(target_context)

    def _set_adavd_target_contexts(self, target_contexts: list[torch.Tensor] | None) -> None:
        for b in self.switti.blocks:
            ca = getattr(b, "cross_attn", None)
            if ca is None:
                continue
            ca.set_adavd_target_contexts(target_contexts)

    def encode_prompt(
        self,
        prompt: str | list[str],
        null_prompt: str = "",
        encode_null: bool = True,
    ):
        prompt_embeds, pooled_prompt_embeds, attn_bias = self._encode_prompt(prompt)
        if encode_null:
            B, L, hidden_dim = prompt_embeds.shape
            pooled_dim = pooled_prompt_embeds.shape[1]

            null_embeds, null_pooled_embeds, null_attn_bias = self._encode_prompt(null_prompt)
            
            null_embeds = null_embeds[:, :L].expand(B, L, hidden_dim).to(prompt_embeds.device)
            null_pooled_embeds = null_pooled_embeds.expand(B, pooled_dim).to(pooled_prompt_embeds.device)
            null_attn_bias = null_attn_bias[:, :L].expand(B, L).to(attn_bias.device)

            prompt_embeds = torch.cat([prompt_embeds, null_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, null_pooled_embeds], dim=0)
            attn_bias = torch.cat([attn_bias, null_attn_bias], dim=0)

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | list[str],
        null_prompt: str = "",
        # Optional precomputed text conditioning (for SAFREE / prompt editing baselines).
        # When provided, these must already include the unconditional branch (CFG), i.e. shape is (2*B, ...).
        prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        seed: int | None = None,
        cfg: float = 6.,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = False,
        return_pil: bool = True,
        smooth_start_si: int = 0,
        turn_off_cfg_start_si: int = 10,
        turn_on_cfg_start_si: int = 0,
        last_scale_temp: None | float = None,
        # AdaVD (value-space orthogonal decomposition) options
        adavd_mode: str = "original",  # original|retain|erase
        target_concept: str | None = None,
        target_concepts: str | list[str] | None = None,  # multi-concept span erasure
        target_mode: str = "last_subject",
        sigmoid_a: float = 100.0,
        sigmoid_b: float = 0.9,
        sigmoid_c: float = 1.0,
        adavd_record_target: bool = True,
        adavd_debug_cos: bool = False,
        adavd_debug_print_limit: int = 1,
        adavd_cos_log_path: str | None = None,
        adavd_cos_log_dump_tokens: bool = False,
        adavd_cos_log_dump_tokens_per_concept: bool = False,
        adavd_apply_to_uncond: bool = False,
    ) -> torch.Tensor | list[PILImage]:
        """
        only used for inference, on autoregressive mode
        :param prompt: text prompt to generate an image
        :param null_prompt: negative prompt for CFG
        :param seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: sampling using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if return_pil: list of PIL Images, else: torch.tensor (B, 3, H, W) in [0, 1]
        """
        assert not self.switti.training
        switti = self.switti
        vae = self.vae
        vae_quant = self.vae.quantize
        if seed is None:
            rng = None
        else:
            switti.rng.manual_seed(seed)
            rng = switti.rng

        if prompt_embeds is None or pooled_prompt_embeds is None or attn_bias is None:
            context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)
        else:
            context, cond_vector, context_attn_bias = prompt_embeds, pooled_prompt_embeds, attn_bias

        B = context.shape[0] // 2

        # AdaVD setup: target is static (CLIP cross-attn), so we can precompute and cache target_v per layer.
        adavd_mode = (adavd_mode or "original").strip().lower()
        if adavd_mode not in {"original", "retain", "erase"}:
            raise ValueError(f"Unsupported adavd_mode: {adavd_mode}")
        if adavd_mode != "original" and (not target_concept) and (not target_concepts):
            raise ValueError("adavd_mode requires target_concept or target_concepts")

        target_mode = _normalize_target_mode(target_mode)

        if adavd_mode == "original":
            self._set_adavd_on_blocks(
                mode="original",
                target_context=None,
                target_contexts=None,
                sigmoid_setting=None,
                record_target=adavd_record_target,
                cfg_active=True,
                debug_cos=adavd_debug_cos,
                debug_print_limit=adavd_debug_print_limit,
                cos_log_path=adavd_cos_log_path,
                cos_log_dump_tokens=adavd_cos_log_dump_tokens,
                cos_log_dump_tokens_per_concept=adavd_cos_log_dump_tokens_per_concept,
                apply_to_uncond=adavd_apply_to_uncond,
            )
            target_context_full = None
            target_contexts_full = None
        else:
            target_context_full = None
            target_contexts_full = None
            concepts = None
            if target_concepts is not None:
                if isinstance(target_concepts, str):
                    concepts = [c.strip() for c in target_concepts.split(",") if c.strip()]
                else:
                    concepts = [str(c).strip() for c in target_concepts if str(c).strip()]
                if len(concepts) == 0:
                    raise ValueError("target_concepts is empty")
                target_contexts = [
                    (
                        self._make_target_context_last_subject_eot_mean_repeat(
                            target_concept=c,
                            batch_size=B,
                            device=context.device,
                        )
                        if target_mode == "last_subject_eot_mean"
                        else self._make_target_context_last_subject_repeat(
                            target_concept=c,
                            batch_size=B,
                            device=context.device,
                        )
                    )
                    for c in concepts
                ]  # list of [B,77,2048]
                target_contexts_full = [tc.repeat(2, 1, 1) for tc in target_contexts]
            else:
                if target_mode == "last_subject_eot_mean":
                    target_context = self._make_target_context_last_subject_eot_mean_repeat(
                        target_concept=target_concept,
                        batch_size=B,
                        device=context.device,
                    )
                else:
                    target_context = self._make_target_context_last_subject_repeat(
                        target_concept=target_concept,
                        batch_size=B,
                        device=context.device,
                    )  # [B,77,2048]
                target_context_full = target_context.repeat(2, 1, 1)
            self._set_adavd_on_blocks(
                mode=adavd_mode,
                target_context=target_context_full,
                target_contexts=target_contexts_full,
                target_concept_names=concepts if target_concepts is not None else None,
                sigmoid_setting=(float(sigmoid_a), float(sigmoid_b), float(sigmoid_c)),
                record_target=adavd_record_target,
                cfg_active=True,
                debug_cos=adavd_debug_cos,
                debug_print_limit=adavd_debug_print_limit,
                cos_log_path=adavd_cos_log_path,
                cos_log_dump_tokens=adavd_cos_log_dump_tokens,
                cos_log_dump_tokens_per_concept=adavd_cos_log_dump_tokens_per_concept,
                apply_to_uncond=adavd_apply_to_uncond,
            )

        cond_vector = switti.text_pooler(cond_vector)

        if switti.use_crop_cond:
            crop_coords = get_crop_condition(2 * B * [TRAIN_IMAGE_SIZE[0]],
                                             2 * B * [TRAIN_IMAGE_SIZE[1]],
                                             ).to(cond_vector.device)
            crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
            crop_cond = switti.crop_proj(crop_embed)
        else:
            crop_cond = None

        sos = cond_BD = cond_vector

        lvl_pos = switti.lvl_embed(switti.lvl_1L)
        if not switti.rope:
            lvl_pos += switti.pos_1LC
        next_token_map = (
            sos.unsqueeze(1)
            + switti.pos_start.expand(2 * B, switti.first_l, -1)
            + lvl_pos[:, : switti.first_l]
        )
        cur_L = 0
        f_hat = sos.new_zeros(B, switti.Cvae, switti.patch_nums[-1], switti.patch_nums[-1])

        for b in switti.blocks:
            b.attn.kv_caching(switti.use_ar) # Use KV caching if switti is in the AR mode 
            b.cross_attn.kv_caching(True)

        cfg_is_active = True  # batch is 2B
        for si, pn in enumerate(switti.patch_nums):  # si: i-th segment
            ratio = si / switti.num_stages_minus_1
            x_BLC = next_token_map

            if switti.rope:
                freqs_cis = switti.freqs_cis[:, cur_L : cur_L + pn * pn]
            else:
                freqs_cis = switti.freqs_cis

            if si >= turn_off_cfg_start_si:
                apply_smooth = False
                x_BLC = x_BLC[:B]
                context = context[:B]
                context_attn_bias = context_attn_bias[:B]
                freqs_cis = freqs_cis[:B]
                cond_BD = cond_BD[:B]
                if crop_cond is not None:
                    crop_cond = crop_cond[:B]
                if cfg_is_active:
                    # Switch AdaVD to non-CFG mode and align target batch to B.
                    cfg_is_active = False
                    self._set_adavd_cfg_active(False)
                    if target_context_full is not None:
                        self._set_adavd_target_context(target_context_full[:B])
                    if target_contexts_full is not None:
                        self._set_adavd_target_contexts([tc[:B] for tc in target_contexts_full])
                for b in switti.blocks:
                    if b.attn.caching and b.attn.cached_k is not None:
                        b.attn.cached_k = b.attn.cached_k[:B]
                        b.attn.cached_v = b.attn.cached_v[:B]
                    if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                        b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                        b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
            else:
                apply_smooth = more_smooth

            for block in switti.blocks:
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=None,
                    context=context,
                    context_attn_bias=context_attn_bias,
                    freqs_cis=freqs_cis,
                    crop_cond=crop_cond,
                )
            cur_L += pn * pn

            logits_BlV = switti.get_logits(x_BLC, cond_BD)

            # Guidance
            if si < turn_on_cfg_start_si:
                logits_BlV = logits_BlV[:B]
            elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            elif last_scale_temp is not None:
                logits_BlV = logits_BlV / last_scale_temp

            if apply_smooth and si >= smooth_start_si:
                # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                idx_Bl = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng,
                )
                h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
            else:
                # default nucleus sampling
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1,
                )[:, :, 0]
                h_BChw = vae_quant.embedding(idx_Bl)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, switti.Cvae, pn, pn)
            f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                    si, len(switti.patch_nums), f_hat, h_BChw,
            )
            if si != switti.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    switti.word_embed(next_token_map)
                    + lvl_pos[:, cur_L : cur_L + switti.patch_nums[si + 1] ** 2]
                )
                # double the batch sizes due to CFG
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in switti.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)

        # de-normalize, from [-1, 1] to [0, 1]
        img = vae.fhat_to_img(f_hat).add(1).mul(0.5)
        if return_pil:
            img = self.to_image(img)

        return img
