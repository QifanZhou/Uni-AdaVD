# Uni-AdaVD on Switti-AR

This directory contains the public Uni-AdaVD inference-time concept erasure code for the autoregressive Switti-AR image generator.

## Setup

```bash
# create conda environment
conda create -n uniadavd_switti python=3.10 -y
conda activate uniadavd_switti

# install pytorch
pip install torch==2.5.1 torchvision==0.20.1

# install remaining dependencies
cd texttoimage/switti-ar
pip install -r requirements.txt
```

You also need to provide the auxiliary checkpoints used by the Switti pipeline:

```bash
export UNIDADAVD_SWITTI_VAE_PATH=/path/to/VQVAE-Switti
export UNIDADAVD_SWITTI_TEXT_ENCODER_PATH=/path/to/clip-vit-large-patch14
export UNIDADAVD_SWITTI_TEXT_ENCODER_2_PATH=/path/to/CLIP-ViT-bigG-14-laion2B-39B-b160k
```

`--model_path` should point to the Switti-AR checkpoint directory.

## Script

- `run_switti_adavd.py`: main inference script
- `template.py`: template prompts for explicit concept erasure
- `models/`, `utils/`, `dist.py`: minimal runtime package for the public release

## Argument Notes

- `--mode`: one of `original`, `retain`, or `erase`. `original` runs the unmodified model, `retain` suppresses the target concept while preserving unrelated content, and `erase` visualizes the removed target component.
- `--erase_type`: concept family used by the script. `instance`, `style`, and `celebrity` are explicit template-based settings; `nsfw` is the implicit safety setting used with benchmark CSVs.

## Explicit Concept Erasure

Explicit concepts correspond to the template-based settings `instance`, `style`, and `celebrity`. The public Switti-AR release uses `template.py` together with `--contents`; for these concepts, the target construction should use `--target_mode last_subject`.

### Command Template

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python run_switti_adavd.py \
  --model_path ${switti_ckpt} \
  --erase_type ${erased_concept_type} \
  --target_concept "${erased_concept_1, erased_concept_2, ..., erased_concept_m}" \
  --contents "${evaluate_concept_1, evaluate_concept_2, ..., evaluate_concept_n}" \
  --target_mode last_subject \
  --mode ${sample_mode} \
  --sigmoid_a 100 \
  --sigmoid_b 0.9 \
  --sigmoid_c 1 \
  --num_samples ${img_per_prompt} \
  --seed ${seed} \
  --cfg ${cfg_scale} \
  --save_root ${your_save_path} \
  --record_target
```

### Example

```bash
CUDA_VISIBLE_DEVICES=0 python run_switti_adavd.py \
  --model_path /path/to/Switti-1024-AR \
  --erase_type instance \
  --target_concept "Snoopy" \
  --contents "Snoopy,dog,cat" \
  --target_mode last_subject \
  --mode retain \
  --sigmoid_a 100 \
  --sigmoid_b 0.9 \
  --sigmoid_c 1 \
  --num_samples 10 \
  --seed 0 \
  --cfg 6.0 \
  --save_root ./outputs_instance \
  --record_target
```

## Implicit Concept Erasure

Implicit concepts such as `nudity` / `nsfw` are evaluated on I2P for unsafe-image suppression and on COCO-30k for preservation. For these concepts, use `--erase_type nsfw` and `--target_mode last_subject_eot_mean`.

### Command Template

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python run_switti_adavd.py \
  --model_path ${switti_ckpt} \
  --erase_type nsfw \
  --csv_path ${benchmark_prompt_csv} \
  --csv_start ${start_idx} \
  --csv_end ${end_idx} \
  --prompt_col ${prompt_column} \
  --seed_col ${seed_column} \
  --cfg_col ${cfg_column} \
  --target_concept "${implicit_concept}" \
  --target_mode last_subject_eot_mean \
  --mode ${sample_mode} \
  --sigmoid_a 100 \
  --sigmoid_b 0.4 \
  --sigmoid_c 1 \
  --save_root ${your_save_path} \
  --record_target
```

### Example

```bash
CUDA_VISIBLE_DEVICES=0 python run_switti_adavd.py \
  --model_path /path/to/Switti-1024-AR \
  --erase_type nsfw \
  --csv_path ../../datasets/i2p_benchmark.csv \
  --csv_start 0 \
  --csv_end 4703 \
  --prompt_col prompt \
  --seed_col sd_seed \
  --cfg_col sd_guidance_scale \
  --target_concept "nudity" \
  --target_mode last_subject_eot_mean \
  --mode retain \
  --sigmoid_a 100 \
  --sigmoid_b 0.4 \
  --sigmoid_c 1 \
  --save_root ./outputs_i2p_nudity \
  --record_target
```

## Outputs

Template-mode outputs are organized under `save_root/<target>/<content>/<mode>/`. NSFW CSV-mode outputs are organized under `save_root/<target>/<mode>/`.
