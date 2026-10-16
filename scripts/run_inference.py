"""
CLI utility to run inference on single local image files using AgriScan.

Supports:
- Close-up classification
- Aerial tiling stress detection
- EXIF GPS metadata inspection
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from agriscan.inference.engine import AgriScanInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Run AgriScan inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--mode",
        type=str,
        default="close_up",
        choices=["close_up", "aerial"],
        help="Analysis mode: 'close_up' or 'aerial'",
    )
    parser.add_argument("--crop", type=str, default="tomato", help="Crop type (e.g. 'tomato')")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained checkpoint (runs demo mode if missing)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Abstention threshold (default: 0.70)",
    )

    args = parser.parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Image not found: {img_path}")
        sys.exit(1)

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    engine = AgriScanInferenceEngine(
        close_up_checkpoint=args.checkpoint if Path(args.checkpoint).exists() else None,
        default_abstention_threshold=args.threshold,
    )

    if args.mode == "close_up":
        result = engine.run_close_up_inference(
            image=image,
            crop=args.crop,
            image_id=img_path.name,
            abstention_threshold=args.threshold,
        )
    else:
        result = engine.run_aerial_inference(
            image=image,
            crop=args.crop,
            image_id=img_path.name,
            confidence_threshold=args.threshold,
        )

    print("\n" + "=" * 80)
    print("AGRISCAN INFERENCE RESULT")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
