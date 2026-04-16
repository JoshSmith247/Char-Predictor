# Char-Predictor

Two models for working with character images rendered from Google Fonts:

- **Dual-Encoder** (`scripts/model.py`) — predicts what a character looks like in a new font by learning content and style separately
- **Classifier** (`scripts/train.py`) — predicts which character is shown in an image

---

## Setup

```bash
pip install tensorflow pillow requests
```

---

## Dual-Encoder — Predict a character in a new font style

Given a target character and a font file, the model predicts what that character would look like rendered in that font. It uses a content encoder (the character across many fonts) and a style encoder (other characters in the same font) to synthesise the output.

### Train

Download fonts and train from scratch:

```bash
python scripts/model.py --target-char A --api-key YOUR_KEY --count 100
```

Additional training options (all optional):

```bash
python scripts/model.py --target-char A --api-key YOUR_KEY --count 100 \
  --epochs 50 --batch-size 32 --lr 3e-4 \
  --latent-dim 16 --n-content 8 --k-style 4 \
  --foreground-weight 5.0
```

If fonts are already downloaded, omit `--api-key`:

```bash
python scripts/model.py --target-char A --epochs 50
```

### Predict

Load saved weights and run inference on a font file:

```bash
python scripts/model.py --target-char A --load --font-path fonts/FontFamily/FontName.ttf --output predicted.png
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
