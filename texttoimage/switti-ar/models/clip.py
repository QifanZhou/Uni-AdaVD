import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        dtype=torch.float32,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)

        # transformers>=4.55 may refuse to load PyTorch .bin weights unless torch>=2.6 due to a security CVE.
        # Prefer safetensors when available. Some local repos are sharded into
        # `pytorch_model-0000x-of-0000y.safetensors` (no `model.safetensors`), which still works with
        # `use_safetensors=True` as long as the *index* json is named `model.safetensors.index.json`.
        version_path = Path(str(version))
        if version_path.is_dir():
            idx_src = version_path / "pytorch_model.bin.index.json"
            idx_dst = version_path / "model.safetensors.index.json"
            # If we have safetensors shards but no safetensors index, create a compatible one.
            if (not idx_dst.exists()) and idx_src.exists() and any(version_path.glob("*.safetensors")):
                try:
                    obj = json.loads(idx_src.read_text(encoding="utf-8"))
                    wm = obj.get("weight_map", {})
                    # Rewrite shard filenames from .bin -> .safetensors when those files exist.
                    new_wm = {}
                    for k, v in wm.items():
                        if isinstance(v, str) and v.endswith(".bin"):
                            cand = v[:-4] + ".safetensors"
                            if (version_path / cand).exists():
                                new_wm[k] = cand
                            else:
                                new_wm[k] = v
                        else:
                            new_wm[k] = v
                    obj["weight_map"] = new_wm
                    idx_dst.write_text(json.dumps(obj, indent=2), encoding="utf-8")
                    print(f"[Switti] wrote safetensors index: {idx_dst}")
                except Exception as e:
                    print(f"[Switti] failed to write safetensors index for {version_path}: {e}")

        self.transformer = CLIPTextModel.from_pretrained(version, use_safetensors=True).to(device, dtype)
        self.device = device
        self.hidden_size = self.transformer.config.hidden_size
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        outputs = self.transformer(**batch_encoding)

        attn_bias = batch_encoding["attention_mask"].to(outputs["last_hidden_state"].dtype)
        attn_bias[attn_bias == 0] = -float("inf")
        attn_bias[attn_bias == 1] = 0.0
        outputs["attn_bias"] = attn_bias
        return outputs

    @torch.no_grad()
    def encode(self, text):
        return self(text)
