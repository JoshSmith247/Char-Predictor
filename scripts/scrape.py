"""Entry point: scrape fonts and render character images to data/raw/."""
import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.google_fonts_scraper import GoogleFontsScraper
from src.utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Scrape Google Fonts and render character images.")
    p.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    p.add_argument("--max-fonts", type=int, default=None, help="Override max_fonts from config")
    p.add_argument("--charset", default=None, help="Override charset from config (e.g. 'Aa0')")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.max_fonts is not None:
        cfg.setdefault("scraping", {})["max_fonts"] = args.max_fonts
    if args.charset is not None:
        cfg.setdefault("data", {})["charset"] = args.charset

    raw_dir = cfg["data"]["raw_dir"]
    scraper = GoogleFontsScraper(raw_dir=raw_dir, config=cfg)
    scraper.run()


if __name__ == "__main__":
    main()
