#!/usr/bin/env python3
import argparse
import csv
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

TEMPLATE_DEFAULT_SIGMOID = (100.0, 0.9, 1.0)
NSFW_DEFAULT_SIGMOID = (100.0, 0.4, 1.0)
FIXED_TOP_P = 1.0
FIXED_MORE_SMOOTH = False
FIXED_SMOOTH_START_SI = 0
FIXED_TURN_ON_CFG_START_SI = 0
FIXED_TURN_OFF_CFG_START_SI = 10
FIXED_LAST_SCALE_TEMP = None
TARGET_MODE_ALIASES = {
    "last_subject": "last_subject",
    "last_subject_eot_mean": "last_subject_eot_mean",
    "lastsubjectrepeat": "last_subject",
    "lastsubjecteotmeanrepeat": "last_subject_eot_mean",
    "lastsubjectandeotmeanrepeat": "last_subject_eot_mean",
}


def _set_reproducible(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if deterministic:
        torch.use_deterministic_algorithms(True)


def _normalize_target_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in TARGET_MODE_ALIASES:
        return TARGET_MODE_ALIASES[mode]
    raise argparse.ArgumentTypeError(
        "--target_mode must be one of {last_subject, last_subject_eot_mean}"
    )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("run_switti_adavd")
    p.add_argument("--model_path", type=str, required=True, help="Switti checkpoint dir, e.g. .../Switti-1024-AR")
    p.add_argument("--reso", type=int, default=1024, choices=[512, 1024], help="Generation resolution")
    p.add_argument("--prompt", type=str, default="", help="Single prompt used outside dataset mode.")
    p.add_argument("--batch_size", type=int, default=1, help="Number of images to sample for the same prompt (prompt will be repeated).")
    p.add_argument("--out_dir", type=str, default="./switti_adavd_out", help="Output directory for single-prompt mode.")
    p.add_argument("--out_name", type=str, default="", help="Optional filename prefix (no extension)")
    p.add_argument("--save_root", type=str, default="", help="Dataset mode output root. When set, enables template loop.")
    p.add_argument("--erase_type", type=str, default="", choices=["", "instance", "style", "celebrity", "nsfw"], help="Dataset mode erasure setting. Use instance/style/celebrity for templates or nsfw for CSV prompts.")
    p.add_argument("--contents", type=str, default="", help="Comma-separated concepts to fill templates.")
    p.add_argument("--num_samples", type=int, default=10, help="Dataset mode: number of samples per prompt (maps to --batch_size if --seeds is empty).")
    p.add_argument("--csv_path", type=str, default="", help="CSV prompt file for nsfw mode.")
    p.add_argument("--csv_start", type=int, default=0, help="Inclusive start row index for nsfw CSV mode.")
    p.add_argument("--csv_end", type=int, default=-1, help="Inclusive end row index for nsfw CSV mode. Use -1 to consume all remaining rows.")
    p.add_argument("--prompt_col", type=str, default="prompt", help="Prompt column name for nsfw CSV mode.")
    p.add_argument("--seed_col", type=str, default="", help="Seed column for nsfw CSV mode.")
    p.add_argument("--cfg_col", type=str, default="", help="Optional CFG column for nsfw CSV mode.")
    p.add_argument("--cfg_default", type=float, default=6.0, help="Fallback CFG scale used when the nsfw CSV does not provide one.")
    p.add_argument("--seed", type=int, default=0, help="Base random seed for single-prompt or template mode.")
    p.add_argument("--seeds", type=str, default="", help="Comma-separated seeds (overrides --seed and --batch_size). Example: --seeds 0,1 will generate two images with fixed per-sample seeds.")
    p.add_argument("--cfg", type=float, default=6.0, help="Classifier-free guidance scale for single-prompt or template mode.")
    p.add_argument("--mode", type=str, default="original", choices=["original", "retain", "erase"], help="Generation mode.")
    p.add_argument("--target_concept", type=str, default="", help="Required for retain/erase.")
    p.add_argument("--target_concepts", type=str, default="", help="Comma-separated multi-concepts for span-subspace erasure (e.g. 'nudity,violence').")
    p.add_argument("--target_mode", type=_normalize_target_mode, default="last_subject", help="Target-token construction mode. Use last_subject for explicit concepts and last_subject_eot_mean for implicit concepts.")
    p.add_argument("--sigmoid_a", type=float, default=100.0, help="Sigmoid steepness used for AdaVD value scaling.")
    p.add_argument("--sigmoid_b", type=float, default=0.9, help="Template default is 0.9; nsfw CSV mode falls back to 0.4 when left unchanged.")
    p.add_argument("--sigmoid_c", type=float, default=1.0, help="Sigmoid upper bound used for AdaVD value scaling.")
    p.add_argument("--record_target", action="store_true", help="Cache per-layer target_v (recommended).")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"], help="Inference dtype for the Switti pipeline.")
    p.add_argument("--device", type=str, default="cuda", help="Execution device, e.g. cuda or cuda:0.")
    p.add_argument("--deterministic", action="store_true", help="Enable strict deterministic algorithms.")
    return p


def _parse_dtype(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def _slug(s: str, max_len: int = 140) -> str:
    s = re.sub(r"[^\w\-. ]", "", s).strip().replace(" ", "_")
    return s[:max_len] or "sample"


def _cfg_tag(cfg: str) -> str:
    return str(cfg).replace(".", "p")


def _parse_int_like(value) -> int:
    if value is None:
        raise ValueError("seed value is None")
    s = str(value).strip()
    if not s:
        raise ValueError("seed value is empty")
    return int(float(s))


def _parse_cfg_like(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return s


def _iter_csv_rows(csv_path: Path, csv_start: int, csv_end: int, prompt_col: str, seed_col: str, cfg_col: str | None):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < csv_start:
                continue
            if idx > csv_end:
                break
            prompt = str(row[prompt_col])
            try:
                seed = _parse_int_like(row.get(seed_col))
            except Exception:
                seed = int(idx)
            cfg = _parse_cfg_like(row[cfg_col]) if cfg_col else None
            yield idx, prompt, seed, cfg


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size must be >= 1")

    if args.mode in {"retain", "erase"} and (not args.target_concept) and (not args.target_concepts):
        raise ValueError("--target_concept or --target_concepts is required for retain/erase.")

    ds_mode = bool((args.save_root or "").strip()) or bool((args.erase_type or "").strip()) or bool((args.contents or "").strip()) or bool((args.csv_path or "").strip())
    if ds_mode:
        if not (args.save_root or "").strip():
            raise ValueError("Dataset mode requires --save_root")
        erase_type = (args.erase_type or "").strip()
        if erase_type not in {"instance", "style", "celebrity", "nsfw"}:
            raise ValueError("Dataset mode requires --erase_type in {instance,style,celebrity,nsfw}")
        if erase_type == "nsfw":
            if not (args.csv_path or "").strip():
                raise ValueError("NSFW CSV mode requires --csv_path")
            if int(args.csv_end) < int(args.csv_start):
                raise ValueError("NSFW CSV mode requires --csv_end >= --csv_start")
            if not (args.seed_col or "").strip():
                raise ValueError("NSFW CSV mode requires --seed_col")
            if (args.seeds or "").strip():
                raise ValueError("NSFW CSV mode does not support --seeds; use --seed_col from CSV")
        else:
            if not (args.contents or "").strip():
                raise ValueError("Template mode requires --contents (comma-separated concepts)")
        if args.mode != "original" and (not args.target_concept) and (not args.target_concepts):
            raise ValueError("Dataset mode retain/erase requires --target_concept or --target_concepts")
    else:
        if not (args.prompt or "").strip():
            raise ValueError("Single-prompt mode requires --prompt. Alternatively, use dataset mode via --save_root/--erase_type/--contents/--csv_path.")


def _parse_seeds(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    args = _build_argparser().parse_args()
    args.target_mode = _normalize_target_mode(args.target_mode)
    _validate_args(args)

    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))

    from models.pipeline import SwittiPipeline

    env_vae_path = os.getenv("UNIDADAVD_SWITTI_VAE_PATH", "").strip()
    env_text_encoder_path = os.getenv("UNIDADAVD_SWITTI_TEXT_ENCODER_PATH", "").strip()
    env_text_encoder_2_path = os.getenv("UNIDADAVD_SWITTI_TEXT_ENCODER_2_PATH", "").strip()
    if env_vae_path:
        SwittiPipeline.vae_path = env_vae_path
    if env_text_encoder_path:
        SwittiPipeline.text_encoder_path = env_text_encoder_path
    if env_text_encoder_2_path:
        SwittiPipeline.text_encoder_2_path = env_text_encoder_2_path

    dtype = _parse_dtype(args.dtype)
    seeds_list = _parse_seeds(args.seeds)
    sigmoid_a, sigmoid_b, sigmoid_c = float(args.sigmoid_a), float(args.sigmoid_b), float(args.sigmoid_c)
    if (args.erase_type or "").strip() == "nsfw" and (sigmoid_a, sigmoid_b, sigmoid_c) == TEMPLATE_DEFAULT_SIGMOID:
        sigmoid_a, sigmoid_b, sigmoid_c = NSFW_DEFAULT_SIGMOID
    if seeds_list:
        _set_reproducible(seeds_list[0], args.deterministic)
    else:
        _set_reproducible(args.seed, args.deterministic)

    pipe = SwittiPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device=args.device,
        reso=args.reso,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_one(mode: str) -> Path:
        mode = mode.lower()
        out_prefix = args.out_name.strip() or "sample"
        out_path = out_dir / f"{out_prefix}_{mode}.png"
        out_path0 = out_dir / f"{out_prefix}_{mode}_00.png"
        eff_batch = len(seeds_list) if seeds_list else int(args.batch_size)
        if eff_batch > 1:
            out_path = out_path0
        if seeds_list:
            for i, s in enumerate(seeds_list):
                _set_reproducible(s, args.deterministic)
                imgs = pipe(
                    prompt=args.prompt,
                    seed=s,
                    cfg=args.cfg,
                    top_k=0,
                    top_p=FIXED_TOP_P,
                    more_smooth=FIXED_MORE_SMOOTH,
                    return_pil=True,
                    smooth_start_si=FIXED_SMOOTH_START_SI,
                    turn_off_cfg_start_si=FIXED_TURN_OFF_CFG_START_SI,
                    turn_on_cfg_start_si=FIXED_TURN_ON_CFG_START_SI,
                    last_scale_temp=FIXED_LAST_SCALE_TEMP,
                    adavd_mode=mode,
                    target_concept=(args.target_concept if mode != "original" and args.target_concepts == "" else None),
                    target_concepts=(args.target_concepts if mode != "original" and args.target_concepts != "" else None),
                    target_mode=str(args.target_mode),
                    sigmoid_a=sigmoid_a,
                    sigmoid_b=sigmoid_b,
                    sigmoid_c=sigmoid_c,
                    adavd_record_target=bool(args.record_target),
                )
                p = out_dir / f"{out_prefix}_{mode}_{i:02d}.png"
                imgs[0].save(p)
            print("saved:", str((out_dir / f"{out_prefix}_{mode}_00.png").resolve()))
            print("saved:", str((out_dir / f"{out_prefix}_{mode}_{len(seeds_list)-1:02d}.png").resolve()))
        else:
            imgs = pipe(
                prompt=[args.prompt] * int(args.batch_size),
                seed=args.seed,
                cfg=args.cfg,
                top_k=0,
                top_p=FIXED_TOP_P,
                more_smooth=FIXED_MORE_SMOOTH,
                return_pil=True,
                smooth_start_si=FIXED_SMOOTH_START_SI,
                turn_off_cfg_start_si=FIXED_TURN_OFF_CFG_START_SI,
                turn_on_cfg_start_si=FIXED_TURN_ON_CFG_START_SI,
                last_scale_temp=FIXED_LAST_SCALE_TEMP,
                adavd_mode=mode,
                target_concept=(args.target_concept if mode != "original" and args.target_concepts == "" else None),
                target_concepts=(args.target_concepts if mode != "original" and args.target_concepts != "" else None),
                target_mode=str(args.target_mode),
                sigmoid_a=sigmoid_a,
                sigmoid_b=sigmoid_b,
                sigmoid_c=sigmoid_c,
                adavd_record_target=bool(args.record_target),
            )
            if int(args.batch_size) == 1:
                imgs[0].save(out_path)
                print("saved:", str(out_path.resolve()))
            else:
                for i, im in enumerate(imgs):
                    p = out_dir / f"{out_prefix}_{mode}_{i:02d}.png"
                    im.save(p)
                print("saved:", str((out_dir / f"{out_prefix}_{mode}_00.png").resolve()))
                print("saved:", str((out_dir / f"{out_prefix}_{mode}_{int(args.batch_size)-1:02d}.png").resolve()))
        return out_path

    ds_mode = bool((args.save_root or "").strip()) or bool((args.erase_type or "").strip()) or bool((args.contents or "").strip()) or bool((args.csv_path or "").strip())
    if ds_mode:
        save_root = Path(args.save_root)
        target_dir = args.target_concept.replace(", ", "_").replace(" ", "_") if args.target_concept else "no_target"
        if args.erase_type == "nsfw":
            out_dir2 = save_root / target_dir / args.mode
            out_dir2.mkdir(parents=True, exist_ok=True)
            cfg_col = args.cfg_col.strip() or None
            csv_path = Path(args.csv_path)
            for idx, prompt, seed, cfg_from_csv in _iter_csv_rows(
                csv_path=csv_path,
                csv_start=int(args.csv_start),
                csv_end=int(args.csv_end),
                prompt_col=args.prompt_col,
                seed_col=args.seed_col,
                cfg_col=cfg_col,
            ):
                cfg_val = float(cfg_from_csv) if cfg_from_csv is not None else float(args.cfg_default)
                cfg_str = cfg_from_csv if cfg_from_csv is not None else str(args.cfg_default)
                out_name = f"{idx:05d}_{_slug(prompt)}_seed{seed}_cfg{_cfg_tag(cfg_str)}_{args.mode}.png"
                out_path = out_dir2 / out_name
                if out_path.exists():
                    print(f"[Skip] idx={idx} seed={seed} cfg={cfg_str}")
                    continue
                _set_reproducible(int(seed), args.deterministic)
                imgs = pipe(
                    prompt=prompt,
                    seed=int(seed),
                    cfg=float(cfg_val),
                    top_k=0,
                    top_p=FIXED_TOP_P,
                    more_smooth=FIXED_MORE_SMOOTH,
                    return_pil=True,
                    smooth_start_si=FIXED_SMOOTH_START_SI,
                    turn_off_cfg_start_si=FIXED_TURN_OFF_CFG_START_SI,
                    turn_on_cfg_start_si=FIXED_TURN_ON_CFG_START_SI,
                    last_scale_temp=FIXED_LAST_SCALE_TEMP,
                    adavd_mode=args.mode,
                    target_concept=(args.target_concept if args.mode != "original" and args.target_concepts == "" else None),
                    target_concepts=(args.target_concepts if args.mode != "original" and args.target_concepts != "" else None),
                    target_mode=str(args.target_mode),
                    sigmoid_a=sigmoid_a,
                    sigmoid_b=sigmoid_b,
                    sigmoid_c=sigmoid_c,
                    adavd_record_target=bool(args.record_target),
                )
                imgs[0].save(out_path)
            print("done csv mode. save_root:", str(save_root.resolve()))
        else:
            import importlib.util as _importlib_util

            _tpl_path = repo_root / "template.py"
            spec = _importlib_util.spec_from_file_location("adavd_templates", str(_tpl_path))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load templates from: {_tpl_path}")
            _mod = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(_mod)
            template_dict = _mod.template_dict
            templates = template_dict[args.erase_type]
            concepts = [c.strip() for c in args.contents.split(",") if c.strip()]
            num_samples = int(args.num_samples)
            if num_samples <= 0:
                raise ValueError("--num_samples must be >= 1")
            for concept in concepts:
                out_dir2 = save_root / target_dir / concept / args.mode
                out_dir2.mkdir(parents=True, exist_ok=True)
                expected = len(templates) * num_samples
                if args.mode == "retain":
                    existing = len([p for p in out_dir2.glob("*.png")])
                    if existing >= expected:
                        print(f"[Skip] concept={concept} ({existing}/{expected})")
                        continue
                prompt_list = [t.format(concept) for t in templates]
                for prompt in prompt_list:
                    prompt_slug = _slug(prompt)
                    if seeds_list:
                        for s in seeds_list:
                            _set_reproducible(s, args.deterministic)
                            imgs = pipe(
                                prompt=prompt,
                                seed=s,
                                cfg=args.cfg,
                                top_k=0,
                                top_p=FIXED_TOP_P,
                                more_smooth=FIXED_MORE_SMOOTH,
                                return_pil=True,
                                smooth_start_si=FIXED_SMOOTH_START_SI,
                                turn_off_cfg_start_si=FIXED_TURN_OFF_CFG_START_SI,
                                turn_on_cfg_start_si=FIXED_TURN_ON_CFG_START_SI,
                                last_scale_temp=FIXED_LAST_SCALE_TEMP,
                                adavd_mode=args.mode,
                                target_concept=(args.target_concept if args.mode != "original" and args.target_concepts == "" else None),
                                target_concepts=(args.target_concepts if args.mode != "original" and args.target_concepts != "" else None),
                                target_mode=str(args.target_mode),
                                sigmoid_a=sigmoid_a,
                                sigmoid_b=sigmoid_b,
                                sigmoid_c=sigmoid_c,
                                adavd_record_target=bool(args.record_target),
                            )
                            imgs[0].save(out_dir2 / f"{prompt_slug}_seed{s}.png")
                    else:
                        imgs = pipe(
                            prompt=[prompt] * num_samples,
                            seed=args.seed,
                            cfg=args.cfg,
                            top_k=0,
                            top_p=FIXED_TOP_P,
                            more_smooth=FIXED_MORE_SMOOTH,
                            return_pil=True,
                            smooth_start_si=FIXED_SMOOTH_START_SI,
                            turn_off_cfg_start_si=FIXED_TURN_OFF_CFG_START_SI,
                            turn_on_cfg_start_si=FIXED_TURN_ON_CFG_START_SI,
                            last_scale_temp=FIXED_LAST_SCALE_TEMP,
                            adavd_mode=args.mode,
                            target_concept=(args.target_concept if args.mode != "original" and args.target_concepts == "" else None),
                            target_concepts=(args.target_concepts if args.mode != "original" and args.target_concepts != "" else None),
                            target_mode=str(args.target_mode),
                            sigmoid_a=sigmoid_a,
                            sigmoid_b=sigmoid_b,
                            sigmoid_c=sigmoid_c,
                            adavd_record_target=bool(args.record_target),
                        )
                        for i, im in enumerate(imgs):
                            im.save(out_dir2 / f"{prompt_slug}_{i:02d}.png")
            print("done template mode. save_root:", str(save_root.resolve()))
    else:
        run_one(args.mode)

        print("prompt:", args.prompt)
        if seeds_list:
            print("seeds:", seeds_list)
        else:
            print("batch_size:", int(args.batch_size))
            print("seed:", args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
