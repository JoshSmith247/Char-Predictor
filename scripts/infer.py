"""
Entry point: run inference on a set of character images.

Example:
    python scripts/infer.py \\
        --model models/final_model.keras \\
        --images data/raw/A/font1.png data/raw/A/font2.png data/raw/A/font3.png \\
        --output output/A_composite.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predict import CharPredictorInference


def parse_args():
    p = argparse.ArgumentParser(description="Predict a composite character image.")
    p.add_argument("--model", required=True, help="Path to saved .keras model file")
    p.add_argument("--images", nargs="+", required=True, help="Paths to input font images")
    p.add_argument("--output", required=True, help="Output path for the composite image (.png)")
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    predictor = CharPredictorInference(model_path=args.model, config_path=args.config)
    composite = predictor.predict_from_paths(args.images)
    predictor.save_output(composite, args.output)


if __name__ == "__main__":
    main()
