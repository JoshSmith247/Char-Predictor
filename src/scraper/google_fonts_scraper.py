"""
Google Fonts scraper.

Uses the Google Fonts Developer API to get the list of all available font
families and their direct TTF download URLs. Requires a free API key from
https://developers.google.com/fonts/docs/developer_api

Set the key via the GOOGLE_FONTS_API_KEY environment variable or in
config/config.yaml under scraping.google_fonts_api_key.
"""
from .base_scraper import BaseScraper, FontMeta

_FONTS_API_URL = "https://www.googleapis.com/webfonts/v1/webfonts"


class GoogleFontsScraper(BaseScraper):
    def fetch_font_list(self) -> list[FontMeta]:
        api_key = self.config.get("scraping", {}).get("google_fonts_api_key", "")
        if not api_key:
            raise ValueError(
                "Google Fonts API key is required. Set GOOGLE_FONTS_API_KEY env var "
                "or scraping.google_fonts_api_key in config.yaml."
            )

        resp = self.session.get(
            _FONTS_API_URL,
            params={"key": api_key, "sort": "popularity"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        fonts: list[FontMeta] = []
        for item in data.get("items", []):
            family = item["family"]
            files = item.get("files", {})
            # Prefer "regular" variant; fall back to first available
            url = files.get("regular") or next(iter(files.values()), None)
            if url:
                # Google API returns http:// — upgrade to https
                url = url.replace("http://", "https://")
                fonts.append(FontMeta(name=family, url=url, style="regular"))

        return fonts
