# Captcha Solver CNN

A deep learning system that breaks simple 4-character CAPTCHAs using a Convolutional Neural Network built with PyTorch. The pipeline covers the full workflow — segmenting CAPTCHAs into characters, training a classifier, and decoding unseen images end to end.

Inspired by Adam Geitgey's article: [How to break a CAPTCHA system in 15 minutes with Machine Learning](https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710), ported from Keras to PyTorch.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)

> **Educational project.** These are synthetic CAPTCHAs from a known generator, not a real-world security bypass — real systems use adversarial distortion, overlapping glyphs and behavioural signals that defeat this approach. The value here is the decomposition: turning "read a 4-character image" into "classify one character, four times", which is what makes a 700 KB model reach ~98% where end-to-end sequence prediction would need far more data.

---

## How It Works

The approach breaks a complex image classification problem into a simpler one by splitting each CAPTCHA into individual characters:

```
┌─────────────────────────────────────────────────────────────┐
│  CAPTCHA Image ("8X2N")                                     │
│  ┌──────┐                                                   │
│  │8X2N  │                                                   │
│  └──────┘                                                   │
│      │                                                      │
│      ▼                                                      │
│  1. Threshold + Contour Detection (OpenCV)                  │
│      │                                                      │
│      ▼                                                      │
│  ┌──┐ ┌──┐ ┌──┐ ┌──┐                                       │
│  │8 │ │X │ │2 │ │N │   ← Individual letter crops            │
│  └──┘ └──┘ └──┘ └──┘                                       │
│      │                                                      │
│      ▼                                                      │
│  2. Resize to 20×20 px → Normalize to [0, 1]               │
│      │                                                      │
│      ▼                                                      │
│  3. CNN Prediction → "8", "X", "2", "N"                    │
│      │                                                      │
│      ▼                                                      │
│  Result: "8X2N" ✓                                           │
└─────────────────────────────────────────────────────────────┘
```

Wide contours (`w/h > 1.25`) are assumed to be two touching characters and split down the middle — a cheap fix for the most common segmentation failure.

## Model Architecture

A compact CNN with two convolutional blocks followed by a fully connected classifier:

| Layer | Type | Output Shape |
|-------|------|-------------|
| Input | Grayscale image | 1 × 20 × 20 |
| Conv Block 1 | Conv2d(1→20, 5×5) + ReLU + MaxPool | 20 × 10 × 10 |
| Conv Block 2 | Conv2d(20→50, 5×5) + ReLU + MaxPool | 50 × 5 × 5 |
| Flatten | — | 1250 |
| FC1 | Linear(1250→500) + ReLU | 500 |
| FC2 (Output) | Linear(500→32) | 32 classes |

**32 classes:** digits `2-9` + letters `A-Z` (excluding `0`, `1`, `I`, `O` to avoid ambiguity). The network outputs raw logits — `CrossEntropyLoss` applies `log_softmax` internally, so no softmax layer is defined.

## Results

Measured against the shipped `captcha_model.pth`, not taken from the training log. Per-character accuracy is reproduced on the same 25% validation split the training script creates (`random_state=0`); end-to-end figures come from running the full inference pipeline over 3,000 sampled CAPTCHAs.

| Metric | Value |
|---|---|
| **Per-character validation accuracy** | **99.35%** (2,762 / 2,780) |
| **End-to-end, letters never seen in training** | **98.01%** (2,068 / 2,110) |
| End-to-end, letters seen in training | 97.44% (798 / 819) |
| End-to-end including segmentation failures | 95.53% of all sampled CAPTCHAs |
| Inference segmentation failure rate | 2.4% |

### Why per-character accuracy overstates the system

A CAPTCHA is only solved if **all four** characters are right. At 99.35% per character, the expected full-solve rate is `0.9935⁴ ≈ 97.4%` — and the measured held-out rate of 98.01% lands right there. That gap between "99.35% accurate" and "one in fifty CAPTCHAs wrong" is the number that matters for a system whose output is a string.

The held-out figure is the honest headline: those 2,110 CAPTCHAs contributed **no** letters to training, because segmentation skipped them (see below). They are genuinely unseen images from the same generator.

## The segmentation bottleneck

The most interesting finding in this project is that **the training set uses only 27.9% of the available data** — and it's caused by one line.

`extract_single_letters_from_captchas.py` discards contours smaller than 20 px² as noise:

```python
if cv.contourArea(contour) < 20:
    continue
```

`contourArea` measures the area enclosed by a contour outline, and thin glyph strokes enclose very little. So the filter throws away real characters, the image then fails the `len(regions) != 4` check, and the whole CAPTCHA is skipped. Measured over all 9,955 images:

| Extraction setting | CAPTCHAs segmenting to exactly 4 regions |
|---|---|
| With the area filter (current) | **27.9%** — 2,780 images → 11,120 letters |
| Without it | **97.4%** — 9,700 images → ~38,800 letters |

An ablation over padding and the filter confirms the filter alone is responsible; padding (8 px vs 20 px) moves the rate by under half a percent.

Two consequences:

1. **Removing that line yields ~3.5× more training data** for free.
2. **It skews the class distribution.** Characters with thin strokes get filtered most often, so class counts range from **20 samples (`7`) to 550 (`M`)** — a 27× imbalance that is an artifact of segmentation, not of the generator, which produces characters near-uniformly.

The inference script does *not* apply this filter, which is why it segments 97.4% of images successfully. The two halves of the pipeline disagree about what counts as a letter.

## Project Structure

```
captcha-solver-cnn/
├── extract_single_letters_from_captchas.py   # Step 1: Split CAPTCHAs into letter images
├── train_model.py                            # Step 2: Train the CNN on extracted letters
├── solve_captchas_with_model.py              # Step 3: Run inference on new CAPTCHAs
├── helper.py                                 # Utility: aspect-ratio-preserving resize
├── captcha_model.pth                         # Pre-trained model weights
├── model_labels.dat                          # Serialized label encoder (pickle)
├── generated_captcha_images/                 # 9,955 synthetic CAPTCHA images
├── extracted_letter_images/                  # Training data: 11,120 letters, one folder per class
│   ├── 2/ ... Z/                             # 32 character class folders
└── requirements.txt
```

## Getting Started

### Installation

```bash
git clone https://github.com/pop123-ux/captcha-solver-cnn.git
cd captcha-solver-cnn
pip install torch opencv-python numpy imutils scikit-learn
```

> ⚠️ Install the packages above rather than using `requirements.txt` — it is stale and lists the wrong framework. See limitations.

### Usage

**Option A — Use the pre-trained model** (fastest):

```bash
python solve_captchas_with_model.py
```

Loads `captcha_model.pth`, picks 10 random CAPTCHAs, and opens a window per image showing predictions with bounding boxes. Press any key to advance. Requires a desktop environment — see limitations for headless use.

**Option B — Train from scratch:**

```bash
python extract_single_letters_from_captchas.py   # Step 1: extract letters
python train_model.py                            # Step 2: train (10 epochs, ~1 min on CPU)
python solve_captchas_with_model.py              # Step 3: inference
```

## Limitations and scope

Some of these are bugs worth fixing; others are deliberate choices for a learning project. Both are listed.

**Known bugs**

* **`requirements.txt` describes a different project.** It lists `tensorflow` and `keras` — leftovers from the Keras original this was ported from — while the code is entirely PyTorch. It also lists `sklearn`, a deprecated stub package that now fails deliberately on install, and omits `torch`, `opencv-python` and `scikit-learn` entirely. Running `pip install -r requirements.txt` fails, and would install the wrong framework if it didn't.
* **The area filter silently discards 72% of the training data.** Detailed above. One line, ~3.5× more data.
* **`CaptchaCNN` is defined twice**, in `train_model.py` and `solve_captchas_with_model.py`. The copies have already diverged once — a comment in the inference copy reads *"Re-added this missing layer"*. A shared `model.py` would make that class of bug impossible.
* **Extraction and inference preprocess differently** — 8 px vs 20 px padding, and the area filter applied in one but not the other. Training and inference should see identical pixels.
* **`solve_captchas_with_model.py` samples from the training folder.** It draws its 10 demo CAPTCHAs from `generated_captcha_images/`, most of which contributed training letters. There is no held-out set in the repo; the held-out figures above were obtained by separating images whose letters segmentation had skipped.
* **Inference silently skips unsolvable CAPTCHAs.** When segmentation doesn't yield exactly 4 regions the loop `continue`s without printing anything, so a failed solve is indistinguishable from one that never ran. 2.4% of images hit this.
* **`cv.imshow` blocks and requires a display.** The script can't run on a headless server, in Colab, or over plain SSH, and needs a keypress per image. Writing annotated output to disk would make it scriptable.
* **The label encoder is a pickle**, so loading it executes arbitrary code from the file and is tied to the scikit-learn version that wrote it — loading `model_labels.dat` under a newer scikit-learn already emits an `InconsistentVersionWarning`. A JSON list of classes would be safer and version-proof.

**Deliberate scope choices**

* **Synthetic CAPTCHAs only.** The generator produces clean, non-overlapping glyphs on light noise. Real-world CAPTCHAs are adversarially designed against exactly this pipeline.
* **Segment-then-classify rather than sequence modelling.** A CRNN with CTC loss would handle touching characters without the `w/h > 1.25` hack, but needs far more data and would obscure the core lesson.
* **The train/test split is over letters, not CAPTCHAs.** Characters cropped from the same source image can land on both sides of the split, which slightly inflates per-character validation accuracy. The end-to-end held-out figure above is not affected.
* **Datasets are committed** (~104 MB). It makes the repo runnable immediately with no generation step, at the cost of clone size. `extracted_letter_images/` is fully derivable from `generated_captcha_images/`.

## Key Concepts Demonstrated

- **Image preprocessing** — thresholding, contour detection, and bounding box extraction with OpenCV
- **Data pipeline** — splitting multi-character images into single-character training samples
- **CNN architecture** — convolutional feature extraction + fully connected classification head
- **PyTorch training loop** — forward pass, loss computation, backpropagation, and validation
- **Inference pipeline** — end-to-end from raw image to decoded CAPTCHA text
- **Evaluation design** — why per-character accuracy and per-CAPTCHA accuracy differ, and how to construct a genuinely held-out set

## Roadmap

* Drop the `contourArea` filter and retrain on ~3.5× the data.
* Fix `requirements.txt`, or replace it with a `pyproject.toml`.
* Move `CaptchaCNN` into a shared `model.py` imported by both scripts.
* Hold out a proper test split of whole CAPTCHAs before extraction.
* Replace the pickled `LabelBinarizer` with a plain JSON class list.
* Add a `--no-display` flag that writes annotated images to disk instead of calling `cv.imshow`.
* Report per-class accuracy to quantify the effect of the 27× class imbalance.

## Disclaimer

This project is for **educational purposes only**. It demonstrates fundamental computer vision and deep learning concepts using synthetic CAPTCHAs. Do not use it to bypass security measures on real websites.

## License

MIT

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)
