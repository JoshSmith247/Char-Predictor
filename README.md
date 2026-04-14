# Char-Predictor

Two models for working with character images rendered from Google Fonts:

- **VAE** (`scripts/vae.py`) — generates new images of a single character
- **Classifier** (`scripts/train.py`) — predicts which character is shown in an image

---

## Setup

```bash
pip install tensorflow pillow requests
```

---

## VAE — Generate character images

### Train

```bash
python scripts/vae.py --target-char A --epochs 40 --lr 3e-4 --latent-dim 16 --foreground-weight 2.0 --kl-weight 0.7 --generate 16 --output generated.png
```

If you haven't downloaded fonts yet, add your Google Fonts API key:

```bash
python scripts/vae.py --target-char A --api-key YOUR_KEY --count 100 --epochs 40 --lr 3e-4 --latent-dim 16 --foreground-weight 2.0 --kl-weight 0.7 --generate 16 --output generated.png
```

### Generate from a saved model

```bash
python scripts/vae.py --target-char A --load --generate 16 --output generated.png
```

---

## Classifier — Predict a character from an image

### Train

```bash
# Full 62-class classifier
python scripts/train.py --api-key YOUR_KEY --epochs 20

# Lightweight single-character binary classifier
python scripts/train.py --api-key YOUR_KEY --target-char A --epochs 10
```

### Predict

```python
from scripts.train import CharPredictor

predictor = CharPredictor()
predictor.load_model()
print(predictor.predict("image.png"))  # → "A"
```

---

## Getting a Google Fonts API key

1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Create or select a project
3. Enable the **Web Fonts Developer API**
4. Go to **APIs & Services → Credentials → Create Credentials → API key**
