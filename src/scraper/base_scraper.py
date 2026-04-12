"""
Base scraper: shared font rendering logic and abstract interface.

Subclasses implement fetch_font_list() and any site-specific download
logic. The render_character() method is shared across all scrapers.
"""
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests
from PIL import Image, ImageDraw, ImageFont
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class FontMeta:
    name: str           # Human-readable font family name
    url: str            # Direct URL to a .ttf or .otf file
    style: str = "regular"


def _safe_char_dir(char: str) -> str:
    """Return a filesystem-safe directory name for a character.

    Printable ASCII chars that are valid in directory names are used as-is
    (e.g. 'A', '0', '@'). Characters that would be unsafe on common
    filesystems are encoded as 'char_<hex>' (e.g. '/' -> 'char_2f').
    """
    unsafe = set(r'/\:*?"<>|')
    if char not in unsafe and char.isprintable():
        return char
    return f"char_{ord(char):02x}"


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class BaseScraper(ABC):
    def __init__(self, raw_dir: str, config: dict):
        self.raw_dir = Path(raw_dir)
        self.config = config
        self.render_size: int = config.get("scraping", {}).get("render_size", 128)
        self.sleep: float = config.get("scraping", {}).get("sleep_between_requests", 0.5)
        self.max_fonts: int = config.get("scraping", {}).get("max_fonts", 1000)
        self.charset: str = config.get("data", {}).get("charset", "Aa0")
        self.session = _build_session()
        self.manifest_path = self.raw_dir / "manifest.json"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_font_list(self) -> list[FontMeta]:
        """Return a list of FontMeta objects representing downloadable fonts."""

    # ------------------------------------------------------------------
    # Shared rendering
    # ------------------------------------------------------------------

    def render_character(self, font_path: Path, char: str) -> Image.Image | None:
        """Render a single character using the given TTF/OTF font file.

        Renders at self.render_size onto a white square canvas. Returns None
        if the font does not contain a glyph for the character.
        """
        size = self.render_size
        try:
            font = ImageFont.truetype(str(font_path), size)
        except (OSError, IOError):
            return None

        # Measure actual glyph size using getbbox (Pillow >= 8.0)
        try:
            bbox = font.getbbox(char)
        except AttributeError:
            # Fallback for very old Pillow
            bbox = (0, 0, size, size)

        if bbox is None or (bbox[2] - bbox[0]) == 0 or (bbox[3] - bbox[1]) == 0:
            # Font has no glyph for this character
            return None

        # Render on a canvas 2× the font size to avoid clipping
        canvas_size = size * 2
        img = Image.new("L", (canvas_size, canvas_size), color=255)
        draw = ImageDraw.Draw(img)
        # Center the character on the canvas
        draw.text((canvas_size // 4, canvas_size // 4), char, font=font, fill=0)
        return img

    # ------------------------------------------------------------------
    # Manifest helpers (for resumable scraping)
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self, manifest: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Fetch fonts and render all characters in self.charset."""
        manifest = self._load_manifest()

        print(f"Fetching font list from {self.__class__.__name__}...")
        fonts = self.fetch_font_list()
        fonts = fonts[: self.max_fonts]
        print(f"  Found {len(fonts)} fonts (capped at {self.max_fonts})")

        font_cache_dir = self.raw_dir / "_fonts"
        font_cache_dir.mkdir(parents=True, exist_ok=True)

        for i, meta in enumerate(fonts):
            font_key = f"{meta.name}_{meta.style}"
            if font_key in manifest:
                continue  # Already processed

            print(f"  [{i+1}/{len(fonts)}] {meta.name} ({meta.style})")

            # Download font file
            font_path = font_cache_dir / f"{font_key.replace(' ', '_')}.ttf"
            if not font_path.exists():
                try:
                    resp = self.session.get(meta.url, timeout=10)
                    resp.raise_for_status()
                    font_path.write_bytes(resp.content)
                except Exception as e:
                    print(f"    Skipping {meta.name}: {e}")
                    time.sleep(self.sleep)
                    continue
                time.sleep(self.sleep)

            # Render each character
            rendered_count = 0
            for char in self.charset:
                char_dir = self.raw_dir / _safe_char_dir(char)
                char_dir.mkdir(parents=True, exist_ok=True)
                out_path = char_dir / f"{font_key.replace(' ', '_')}.png"
                if out_path.exists():
                    rendered_count += 1
                    continue

                img = self.render_character(font_path, char)
                if img is not None:
                    img.save(str(out_path))
                    rendered_count += 1

            manifest[font_key] = {"chars_rendered": rendered_count}
            self._save_manifest(manifest)

        print(f"Done. Manifest saved to {self.manifest_path}")
