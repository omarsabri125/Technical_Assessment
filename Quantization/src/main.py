import argparse
from pathlib import Path

import torch

from src.services.benchmark import run_benchmark


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-precision and 4-bit NF4 "
            "Qwen model variants."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["full", "quantized"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default="results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this comparison. "
            "In Google Colab choose Runtime > Change runtime type > T4 GPU."
        )

    result_path = run_benchmark(
        mode=args.mode,
        output_dir=Path(args.output_dir),
    )

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
