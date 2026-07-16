# Uni-AdaVD on SD v3

This directory contains the public Uni-AdaVD inference-time concept erasure code for SD v3.

## Setup

```bash
# create conda environment
conda create -n uniadavd_ti python=3.9 -y
conda activate uniadavd_ti

# install pytorch
pip install torch==2.0.1

# install remaining dependencies
cd texttoimage/sd3
pip install -r ../requirements.txt
```

`--sd_ckpt` accepts either a local checkpoint path or a Hugging Face model ID.

## Script

- `sd3_uniadavd.py`: main inference script
- `template.py`: template prompts for explicit concept erasure

## Argument Notes

- `--mode`: comma-separated subset of `original`, `retain`, and `erase`. `original` runs the unmodified model, `retain` suppresses the target concept while preserving unrelated content, and `erase` visualizes the removed target component.
- `--erase_type`: concept family used by the script. `instance`, `style`, and `celebrity` are explicit template-based settings; `nsfw` is the implicit safety setting used with benchmark prompt CSVs.

## Explicit Concept Erasure

Explicit concepts correspond to the template-based settings `instance`, `style`, and `celebrity`. These modes use `template.py` together with `--contents`, and the target construction should use `--token_processing last_subject`. In the public SD3 release, `sigmoid_b` is resolved internally in a layer-wise manner.

### Command Template

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python sd3_uniadavd.py \
  --sd_ckpt ${sd3_ckpt} \
  --erase_type ${erased_concept_type} \
  --target_concept "${erased_concept_1, erased_concept_2, ..., erased_concept_m}" \
  --contents "${evaluate_concept_1, evaluate_concept_2, ..., evaluate_concept_n}" \
  --token_processing last_subject \
  --mode ${sample_mode} \
  --sigmoid_a 100 \
  --sigmoid_c 2 \
  --num_samples ${img_per_prompt} \
  --batch_size ${sample_bs} \
  --save_root ${your_save_path}
```

### Example

```bash
CUDA_VISIBLE_DEVICES=0 python sd3_uniadavd.py \
  --sd_ckpt stabilityai/stable-diffusion-3-medium-diffusers \
  --erase_type instance \
  --target_concept "Snoopy" \
  --contents "Snoopy,dog,cat" \
  --token_processing last_subject \
  --mode original,retain \
  --sigmoid_a 100 \
  --sigmoid_c 2 \
  --num_samples 10 \
  --batch_size 1 \
  --save_root ./outputs_instance
```

## Implicit Concept Erasure

Implicit concepts such as `nudity` / `nsfw` are evaluated on I2P for unsafe-image suppression and on COCO-30k for preservation. For these concepts, use `--erase_type nsfw` and `--token_processing last_subject_eot_mean`.

### Command Template

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python sd3_uniadavd.py \
  --sd_ckpt ${sd3_ckpt} \
  --erase_type nsfw \
  --target_concept "${implicit_concept}" \
  --token_processing last_subject_eot_mean \
  --prompt_file ${benchmark_prompt_csv} \
  --prompt_start ${start_idx} \
  --prompt_end ${end_idx} \
  --mode ${sample_mode} \
  --sigmoid_a 100 \
  --sigmoid_c 1 \
  --batch_size ${sample_bs} \
  --save_root ${your_save_path}
```

### Example

```bash
CUDA_VISIBLE_DEVICES=0 python sd3_uniadavd.py \
  --sd_ckpt stabilityai/stable-diffusion-3-medium-diffusers \
  --erase_type nsfw \
  --target_concept "nudity" \
  --token_processing last_subject_eot_mean \
  --prompt_file ../../datasets/i2p_benchmark.csv \
  --prompt_start 0 \
  --prompt_end 4703 \
  --mode original,retain \
  --sigmoid_a 100 \
  --sigmoid_c 1 \
  --batch_size 1 \
  --save_root ./outputs_i2p_nudity
```

## Outputs

Template-mode outputs are organized by target concept, content concept, and generation mode. CSV-mode outputs are organized by target concept, prompt file, and mode.
