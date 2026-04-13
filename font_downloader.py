"""
font_downloader.py

Usage from another file:
    from font_downloader import FontDownloader
    downloader = FontDownloader(api_key="YOUR_KEY")
    downloader.download(output_dir="fonts/", count=100)
"""

import os
import requests


GOOGLE_FONTS_API_URL = "https://www.googleapis.com/webfonts/v1/webfonts"


class FontDownloader:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def _fetch_font_list(self, count: int) -> list[dict]:
        """Fetch font metadata from the Google Fonts API."""
        response = self.session.get(
            GOOGLE_FONTS_API_URL,
            params={"key": self.api_key, "sort": "popularity"},
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        return items[:count]

    def _download_font_files(self, font: dict, output_dir: str) -> list[str]:
        """Download all variant files for a single font, return saved paths."""
        family_dir = os.path.join(output_dir, font["family"].replace(" ", "_"))
        os.makedirs(family_dir, exist_ok=True)

        saved = []
        for variant, url in font["files"].items():
            ext = url.split(".")[-1].split("?")[0]  # ttf or otf
            filename = f"{variant}.{ext}"
            dest = os.path.join(family_dir, filename)

            if os.path.exists(dest):
                saved.append(dest)
                continue

            file_response = self.session.get(url)
            file_response.raise_for_status()
            with open(dest, "wb") as f:
                f.write(file_response.content)
            saved.append(dest)

        return saved

    def download(self, output_dir: str = "fonts", count: int = 100) -> list[str]:
        """
        Download `count` Google Fonts into `output_dir`.

        Returns a list of all saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        fonts = self._fetch_font_list(count)
        if not fonts:
            raise RuntimeError("No fonts returned from the API. Check your API key.")

        all_paths = []
        for i, font in enumerate(fonts, start=1):
            print(f"[{i}/{len(fonts)}] Downloading {font['family']}...")
            paths = self._download_font_files(font, output_dir)
            all_paths.extend(paths)

        print(f"Done. {len(all_paths)} font files saved to '{output_dir}/'.")
        return all_paths
