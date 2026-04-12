import os
import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Allow API key to be overridden by environment variable
    env_key = os.environ.get("GOOGLE_FONTS_API_KEY", "")
    if env_key:
        cfg.setdefault("scraping", {})["google_fonts_api_key"] = env_key

    return cfg
