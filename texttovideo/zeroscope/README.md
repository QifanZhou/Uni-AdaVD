# Uni-AdaVD on ZeroScopeT2V

This directory contains the public Uni-AdaVD inference-time concept erasure code for ZeroScopeT2V.

## Setup

```bash
# create conda environment
conda create -n uniadavd_t2v python=3.9 -y
conda activate uniadavd_t2v

# install pytorch
pip install torch==2.0.1

# install remaining dependencies
cd texttovideo/zeroscope
pip install -r requirements.txt
```

`--model_path` accepts either a local checkpoint path or a Hugging Face model ID.

## Script

- `adavd_zeroscope_t2v.py`: main inference script

The public ZeroScope release saves decoded frame sequences by default rather than MP4 files.

## Argument Notes

- `--mode`: comma-separated subset of `original`, `retain`, and `erase`. `original` runs the unmodified model, `retain` suppresses the target concept while preserving unrelated content, and `erase` visualizes the removed target component.

## Implicit Concept Erasure

Implicit concepts such as `nudity` / `nsfw` are evaluated on SafeSora. For these concepts, use `--token_processing last_subject_eot_mean`. The prompt splits are provided under `../../datasets/safesora/`.

### Command Template

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python adavd_zeroscope_t2v.py \
  --model_path ${zeroscope_ckpt} \
  --prompt_file ${safesora_split_json} \
  --prompt_start ${start_idx} \
  --prompt_end ${end_idx} \
  --target_concept "${implicit_concept}" \
  --token_processing last_subject_eot_mean \
  --mode ${sample_mode} \
  --sigmoid_a 100 \
  --sigmoid_b 0.43 \
  --sigmoid_c 1 \
  --num_frames ${num_frames} \
  --height ${height} \
  --width ${width} \
  --num_inference_steps ${denoise_steps} \
  --guidance_scale ${cfg_scale} \
  --save_root ${your_save_path}
```

### Example

```bash
CUDA_VISIBLE_DEVICES=0 python adavd_zeroscope_t2v.py \
  --model_path cerspense/zeroscope_v2_576w \
  --prompt_file ../../datasets/safesora/safesora_sexual.json \
  --prompt_start 0 \
  --prompt_end 1 \
  --target_concept "nudity" \
  --token_processing last_subject_eot_mean \
  --mode original,retain \
  --sigmoid_a 100 \
  --sigmoid_b 0.43 \
  --sigmoid_c 1 \
  --num_frames 16 \
  --height 320 \
  --width 576 \
  --num_inference_steps 100 \
  --guidance_scale 7.5 \
  --save_root ./outputs_safesora_sexual
```

## Outputs

Results are saved under `--save_root/<target_concept>/<mode>/<prompt_name>/` as frame sequences.
