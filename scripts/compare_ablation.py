"""Save baseline and pruned images using the best combination in an ablation report.

    python scripts/compare_ablation.py --report out/ablation.json --input frames/ --out out/comparison

--report also accepts ablation.md when the sibling ablation.json is present.
Without --input, use the original image paths recorded in the JSON report.
Side-by-side PNGs put the full-model baseline on the left and pruned output on the right.
"""
from __future__ import annotations

import argparse
import sys
import time

from dlssnr.compare_ablation import compare_ablation


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True, help="ablation JSON, or Markdown with a sibling JSON")
    parser.add_argument("--input", help="optional image or same-size image directory; overrides report input paths")
    parser.add_argument("--out", default="out/ablation_comparison", help="new or empty output directory")
    parser.add_argument("--weights", default="weights.safetensors", help="same original weights used for the search")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plan-method", choices=("best", "depth", "width", "hybrid"), default="best",
                        help="select the overall best or a method's best result from a comparison report")
    args = parser.parse_args(argv)
    try:
        result = compare_ablation(args.report, args.input, args.out, args.weights, args.device, log,
                                  plan_method=args.plan_method)
        log(f"Saved {len(result['images'])} image pairs to {args.out}; "
            f"aggregate PSNR={result['aggregate']['psnr_db']:.2f}dB")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed image pairs and comparison.json are preserved.", file=sys.stderr, flush=True)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
