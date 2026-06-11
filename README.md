# Captcha Solver CNN

A deep learning system that breaks simple 4-character CAPTCHAs using a Convolutional Neural Network built with PyTorch. The pipeline covers the full workflow — from generating training data to training a model and running inference on unseen CAPTCHAs.

Inspired by Adam Geitgey's article: [How to break a CAPTCHA system in 15 minutes with Machine Learning](https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710).

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)

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

**32 classes:** digits `2-9` + letters `A-Z` (excluding `0`, `1`, `I`, `O` to avoid ambiguity)

## Project Structure

```
captcha-solver-cnn/
├── extract_single_letters_from_captchas.py   # Step 1: Split CAPTCHAs into letter images
├── train_model.py                            # Step 2: Train the CNN on extracted letters
├── solve_captchas_with_model.py              # Step 3: Run inference on new CAPTCHAs
├── helper.py                                 # Utility: aspect-ratio-preserving resize
├── captcha_model.pth                         # Pre-trained model weights
├── model_labels.dat                          # Serialized label encoder (pickle)
├── generated_captcha_images/                 # ~9,955 synthetic CAPTCHA images
├── extracted_letter_images/                  # Training data: one folder per character
│   ├── 2/ ... Z/                             # 32 character class folders
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/pop123-ux/captcha-solver-cnn.git
cd captcha-solver-cnn
pip install -r requirements.txt
```

### Usage

**Option A — Use the pre-trained model** (fastest):

```bash
python solve_captchas_with_model.py
```

This loads `captcha_model.pth`, picks 10 random CAPTCHAs from `generated_captcha_images/`, and displays predictions with bounding boxes.

**Option B — Train from scratch:**

```bash
# Step 1: Extract individual letters from CAPTCHA images
python extract_single_letters_from_captchas.py

# Step 2: Train the CNN (10 epochs, ~1 min on CPU)
python train_model.py

# Step 3: Test on sample CAPTCHAs
python solve_captchas_with_model.py
```

## Training Results

The model converges quickly due to the simplicity of the character classification task:

| Metric | Value |
|--------|-------|
| Training Accuracy | ~99.8% |
| Validation Accuracy | ~99.5% |
| Epochs | 10 |
| Batch Size | 32 |
| Optimizer | Adam (lr=0.001) |

## Key Concepts Demonstrated

- **Image preprocessing** — thresholding, contour detection, and bounding box extraction with OpenCV
- **Data pipeline** — splitting multi-character images into single-character training samples
- **CNN architecture** — convolutional feature extraction + fully connected classification head
- **PyTorch training loop** — forward pass, loss computation, backpropagation, and validation
- **Inference pipeline** — end-to-end from raw image to decoded CAPTCHA text

## Disclaimer

This project is for **educational purposes only**. It demonstrates fundamental computer vision and deep learning concepts using synthetic CAPTCHAs. Do not use it to bypass security measures on real websites.

## License

MIT
