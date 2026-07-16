# Uni-AdaVD on FLUX

This directory contains the public Uni-AdaVD inference-time concept erasure code for FLUX.

## Setup

```bash
# create conda environment
conda create -n uniadavd_ti python=3.9 -y
conda activate uniadavd_ti

# install pytorch
pip install torch==2.0.1

# install remaining dependencies
cd texttoimage/flux
pip install -r ../requirements.txt
```

`--sd_ckpt` accepts either a local checkpoint path or a Hugging Face model ID.

## Script

- `flux_uniadavd.py`: main inference script
- `template.py`: template prompts for explicit concept erasure

## Argument Notes

- `--mode`: comma-separated subset of `original`, `retain`, and `erase`. `original` runs the unmodified model, `retain` suppresses the target concept while preserving unrelated content, and `erase` visualizes the removed target component.
- `--erase_type`: concept family used by the script. `instance`, `style`, and `celebrity` are explicit template-based settings; `nsfw` is the public implicit safety setting used with benchmark prompt CSVs.

## Explicit Concept Erasure

Explicit concepts correspond to the template-based settings `instance`, `style`, and `celebrity`. These modes use `template.py` together with `--contents`. In the public FLUX release, the T5-side target representation and layer-wise `sigmoid_b` are handled internally by the script.

### Command Template

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python flux_uniadavd.py \
  --sd_ckpt ${flux_ckpt} \
  --erase_type ${erased_concept_type} \
  --target_concept "${erased_concept_1, erased_concept_2, ..., erased_concept_m}" \
  --contents "${evaluate_concept_1, evaluate_concept_2, ..., evaluate_concept_n}" \
  --mode ${sample_mode} \
  --sigmoid_a 100 \
  --sigmoid_c 5 \
  --guidance_scale ${cfg_scale} \
  --num_inference_steps ${denoise_steps} \
  --num_samples ${img_per_prompt} \
  --batch_size ${sample_bs} \
  --save_root ${your_save_path}
```

### Example

```bash
CUDA_VISIBLE_DEVICES=0 python flux_uniadavd.py \
  --sd_ckpt black-forest-labs/FLUX.1-dev \
  --erase_type instance \
  --target_concept "Snoopy" \
  --contents "Snoopy,dog,cat" \
  --mode original,retain \
  --sigmoid_a 100 \
  --sigmoid_c 5 \
  --guidance_scale 3.5 \
  --num_inference_steps 30 \
  --num_samples 10 \
  --batch_size 1 \
  --save_root ./outputs_instance
```

## Implicit Concept Erasure

Implicit concepts such as `nudity` / `nsfw` are evaluated on I2P for unsafe-image suppression and on COCO-30k for preservation. In the public FLUX release, use `--erase_type nsfw`; the layer-wise `sigmoid_b` remains internal, while `sigmoid_c=2` is used for implicit concepts.

### Command Template

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python flux_uniadavd.py \
  --sd_ckpt ${flux_ckpt} \
  --erase_type nsfw \
  --target_concept "${implicit_concept}" \
  --prompt_file ${benchmark_prompt_csv} \
  --prompt_start ${start_idx} \
  --prompt_end ${end_idx} \
  --mode ${sample_mode} \
  --sigmoid_a 100 \
  --sigmoid_c 2 \
  --guidance_scale ${cfg_scale} \
  --num_inference_steps ${denoise_steps} \
  --batch_size ${sample_bs} \
  --save_root ${your_save_path}
```

### Example

```bash
CUDA_VISIBLE_DEVICES=0 python flux_uniadavd.py \
  --sd_ckpt black-forest-labs/FLUX.1-dev \
  --erase_type nsfw \
  --target_concept "nudity" \
  --prompt_file ../../datasets/i2p_benchmark.csv \
  --prompt_start 1275 \
  --prompt_end 1276 \
  --mode original,retain \
  --sigmoid_a 100 \
  --sigmoid_c 2 \
  --guidance_scale 3.5 \
  --num_inference_steps 30 \
  --batch_size 1 \
  --save_root ./outputs_i2p_nudity
```

## Outputs

Template-mode outputs are saved by target concept, content concept, and mode. CSV-mode outputs are saved by target concept, prompt index, and mode.
