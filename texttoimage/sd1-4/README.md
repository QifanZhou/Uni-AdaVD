# Uni-AdaVD on SD v1_4

This directory contains the public Uni-AdaVD inference-time concept erasure code for SD v1_4.

## Setup

```bash
# create conda environment
conda create -n uniadavd_ti python=3.9 -y
conda activate uniadavd_ti

# install pytorch
pip install torch==2.0.1 torchvision==0.15.2

# install remaining dependencies
cd texttoimage/sd1-4
pip install -r ../requirements.txt
```

If the pinned PyTorch build does not match the local CUDA runtime, install a CUDA-compatible PyTorch build first and then install the remaining dependencies from `../requirements.txt`.

`--sd_ckpt` accepts either a local checkpoint path or a Hugging Face model ID.

## Script

- `sdv1_4.py`: main inference script
- `template.py`: template prompts for explicit concept erasure
- `utils.py`: utility helpers

## Argument Notes

- `--mode`: comma-separated subset of `original`, `retain`, and `erase`. `original` runs the unmodified model, `retain` suppresses the target concept while preserving unrelated content, and `erase` visualizes the removed target component.
- `--erase_type`: concept family used by the script. `instance`, `style`, and `celebrity` are explicit template-based settings; `nsfw` is the implicit safety setting used with benchmark prompt CSVs.
- `--target_concept`: comma-separated concept or concepts to erase.
- `--contents`: comma-separated concepts used to fill the evaluation templates. Include both erased concepts and non-target concepts that should be preserved, for example `--target_concept "Snoopy" --contents "Snoopy,dog,cat"`.

## Explicit Concept Erasure

Explicit concepts such as instance, style, and celebrity are handled in a template-based manner. These modes use `template.py` together with `--contents`; for explicit concepts, use `--erase_type {instance, style, celebrity}` and `--token_processing last_subject`. Every comma-separated item in `--contents` is expanded with the complete template set. Explicit mode does not use `--prompt_file`, `--prompt_start`, or `--prompt_end`.

### Command Template (Single Concept)

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python sdv1_4.py \
  --sd_ckpt ${sdv1_4_ckpt} \
  --erase_type ${erased_concept_type} \
  --target_concept "${erased_concept}" \
  --contents "${evaluate_concept}" \
  --token_processing last_subject \
  --mode ${sample_mode} \
  --sigmoid_a 100 \
  --sigmoid_b 0.93 \
  --sigmoid_c 2 \
  --num_samples ${img_per_prompt} \
  --batch_size ${sample_bs} \
  --save_root ${your_save_path}
```

### Command Template (Multi-Concept)

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python sdv1_4.py \
  --sd_ckpt ${sdv1_4_ckpt} \
  --erase_type ${erased_concept_type} \
  --target_concept "${erased_concept_1, erased_concept_2, ..., erased_concept_m}" \
  --contents "${evaluate_concept_1, evaluate_concept_2, ..., evaluate_concept_n}" \
  --token_processing last_subject \
  --mode ${sample_mode} \
  --sigmoid_a 100 \
  --sigmoid_b 0.93 \
  --sigmoid_c 2 \
  --num_samples ${img_per_prompt} \
  --batch_size ${sample_bs} \
  --save_root ${your_save_path}
```

### Example (Single Concept)

`contents = {Snoopy, dog, cat}`

```bash
CUDA_VISIBLE_DEVICES=0 python sdv1_4.py \
  --sd_ckpt CompVis/stable-diffusion-v1-4 \
  --erase_type instance \
  --target_concept "Snoopy" \
  --contents "Snoopy,dog,cat" \
  --token_processing last_subject \
  --mode original,retain \
  --sigmoid_a 100 \
  --sigmoid_b 0.93 \
  --sigmoid_c 2 \
  --num_samples 10 \
  --batch_size 1 \
  --save_root ./outputs_explicit
```

### Example (Multi-Concept)

`contents = {Snoopy, Spongebob, Mickey}`

```bash
CUDA_VISIBLE_DEVICES=0 python sdv1_4.py \
  --sd_ckpt CompVis/stable-diffusion-v1-4 \
  --erase_type instance \
  --target_concept "Snoopy,Mickey" \
  --contents "Snoopy,Spongebob,Mickey" \
  --token_processing last_subject \
  --mode original,retain \
  --sigmoid_a 100 \
  --sigmoid_b 0.93 \
  --sigmoid_c 2 \
  --num_samples 10 \
  --batch_size 1 \
  --save_root ./outputs_explicit_multi
```

## Implicit Concept Erasure

Implicit concepts such as `nudity` / `nsfw` are evaluated on I2P for unsafe-image suppression and on COCO-30k for preservation. For implicit concepts, use `--erase_type nsfw` and `--token_processing last_subject_eot_mean`.

### Command Template

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python sdv1_4.py \
  --sd_ckpt ${sdv1_4_ckpt} \
  --erase_type nsfw \
  --prompt_file ${benchmark_prompt_csv} \
  --prompt_start ${start_idx} \
  --prompt_end ${end_idx} \
  --target_concept "${implicit_concept}" \
  --token_processing last_subject_eot_mean \
  --mode ${sample_mode} \
  --sigmoid_a 100 \
  --sigmoid_b 0.43 \
  --sigmoid_c 1 \
  --num_samples ${img_per_prompt} \
  --batch_size ${sample_bs} \
  --save_root ${your_save_path}
```

### Example

```bash
CUDA_VISIBLE_DEVICES=0 python sdv1_4.py \
  --sd_ckpt CompVis/stable-diffusion-v1-4 \
  --erase_type nsfw \
  --prompt_file ../../datasets/i2p_benchmark.csv \
  --prompt_start 1275 \
  --prompt_end 1276 \
  --target_concept "nudity" \
  --token_processing last_subject_eot_mean \
  --mode original,retain \
  --sigmoid_a 100 \
  --sigmoid_b 0.43 \
  --sigmoid_c 1 \
  --num_samples 1 \
  --batch_size 1 \
  --save_root ./outputs_i2p_nudity
```

## Outputs

Results are written under `--save_root`, grouped by dataset, target concept, token processing mode, and sigmoid setting. If multiple modes are requested, the script also writes side-by-side comparison images under `combine/`.
