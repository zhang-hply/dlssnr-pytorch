r"""Prepare teacher targets, distill a fixed pruned model, and evaluate exported weights.

    python scripts/distill.py prepare --input data/train --out out/teacher_train
    python scripts/distill.py train --pruning-report out/ablation.json \
        --train-cache out/teacher_train --val-cache out/teacher_val --out out/distilled
    python scripts/distill.py evaluate --checkpoint out/distilled/best.safetensors \
        --cache out/teacher_val --out out/distilled/evaluation.json

The original extraction weights are required to build the inference skeleton.
Defaults use full frames, batch size one, fixed Controls/seed, FP32 master weights,
gradient accumulation and whole-block checkpointing. Training does not resize images.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

from dlssnr.distill_data import prepare_cache
from dlssnr.distill_train import TrainConfig, evaluate_export, train_distillation


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="generate or resume full-model floating-point targets")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--out", required=True)
    train = commands.add_parser("train", help="recover the best pruning-report mask with all retained weights")
    train.add_argument("--pruning-report", required=True)
    train.add_argument("--train-cache", required=True)
    train.add_argument("--val-cache", required=True)
    train.add_argument("--out", default="out/distilled")
    train.add_argument("--resume", help="last.pt in the same output directory; resumes completed epochs")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument("--min-lr", type=float, default=1e-6)
    train.add_argument("--accumulation", type=int, default=4)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--seed", type=int, default=0, help="training shuffle seed; Controls.seed stays zero")
    train.add_argument("--no-checkpoint", action="store_true", help="disable block activation checkpointing")
    train.add_argument("--tensorboard", action="store_true",
                       help="record curves and full-resolution validation comparisons in <out>/tensorboard")
    evaluate = commands.add_parser("evaluate", help="evaluate best.safetensors via the ordinary inference graph")
    evaluate.add_argument("--checkpoint", required=True, help="exported safetensors plus its sibling JSON metadata")
    evaluate.add_argument("--cache", required=True)
    evaluate.add_argument("--out", help="optional JSON report")
    images = evaluate.add_mutually_exclusive_group()
    images.add_argument("--images-out", help="new or empty comparison directory; default: <out stem>_images, "
                        "or evaluation_images beside the checkpoint when --out is omitted")
    images.add_argument("--no-images", action="store_true", help="evaluate metrics without saving comparison PNGs")
    for command in (prepare, train, evaluate):
        command.add_argument("--weights", default="weights.safetensors")
        command.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_cache(args.input, args.out, args.weights, args.device, log)
            log(f"Teacher cache complete: {len(manifest['images'])} images, {args.out}")
        elif args.command == "train":
            config = TrainConfig(args.epochs, args.lr, args.min_lr, args.accumulation,
                                 args.grad_clip, args.patience, args.seed, not args.no_checkpoint)
            report = train_distillation(args.weights, args.pruning_report, args.train_cache,
                                        args.val_cache, args.out, args.device, config, args.resume, log,
                                        tensorboard=args.tensorboard)
            log(f"Training {report['status']}: best epoch={report['best_epoch']}, "
                f"export PSNR={report['export_validation']['psnr_db']:.4f}dB, "
                f"improved={report['improved']}; reports: {args.out}")
        else:
            images_out = None
            if not args.no_images:
                images_out = args.images_out
                if images_out is None:
                    images_out = (Path(args.out).with_name(Path(args.out).stem + "_images") if args.out
                                  else Path(args.checkpoint).parent / "evaluation_images")
                images_out = str(images_out)
            result = evaluate_export(args.weights, args.checkpoint, args.cache, args.out, args.device, log,
                                     images_out=images_out)
            log(f"Export evaluation: PSNR={result['metrics']['psnr_db']:.4f}dB, "
                f"uint8 PSNR={result['metrics']['psnr_uint8_db']:.4f}dB, SSIM={result['metrics']['ssim']}")
            if result.get("comparisons"):
                log(f"Comparison images: {result['comparisons']['directory']} (input | teacher | student)")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Completed cache frames and epoch-boundary checkpoints are preserved.", file=sys.stderr, flush=True)
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            print("GPU memory exhausted; input resolution was not changed.", file=sys.stderr, flush=True)
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
