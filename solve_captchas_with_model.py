import os
import pickle
import cv2 as cv
import imutils
import numpy as np
import torch
import torch.nn as nn
from helper import resize_to_fit
from imutils import paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_FILENAME = os.path.join(SCRIPT_DIR, "captcha_model.pth")
MODEL_LABELS_FILENAME = os.path.join(SCRIPT_DIR, "model_labels.dat")
CAPTCHA_IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "generated_captcha_images")

class CaptchaCNN(nn.Module):
    def __init__(self, num_classes):
        super(CaptchaCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # <-- Re-added this missing layer
            
            # Block 2
            nn.Conv2d(20, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(50 * 5 * 5, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
with open(MODEL_LABELS_FILENAME, 'rb') as f:
    lb = pickle.load(f)

num_classes = len(lb.classes_)

# Initialize PyTorch model and map weights to CPU/GPU execution environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CaptchaCNN(num_classes = num_classes)
model.load_state_dict(torch.load(MODEL_WEIGHTS_FILENAME, map_location=device))
model.to(device)
model.eval()

# Grab 10 random test images
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
num_to_sample = min(10, len(captcha_image_files))

if num_to_sample == 0:
    print(f"[ERROR] No images found in {CAPTCHA_IMAGE_FOLDER}. Double check your folder contents!")
    exit()

captcha_image_files = np.random.choice(captcha_image_files, size=(num_to_sample,), replace=False)

for image_file in captcha_image_files:
    # Preprocessing layout (OpenCV steps)
    image = cv.imread(image_file)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    padded = cv.copyMakeBorder(gray, 20, 20, 20, 20, cv.BORDER_REPLICATE)
    thresh = cv.threshold(padded, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

    contours_raw = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours_raw[0] if len(contours_raw) == 2 else contours_raw[1]

    letter_image_regions = []

    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        if w / h > 1.25:
            half_width = int(w/2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x+half_width, y, half_width, h))

        else:
            letter_image_regions.append((x,y,w,h))

    if len(letter_image_regions) != 4:
        continue

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an RGB output image canvas for annotations
    output = cv.merge([padded] * 3)
    predictions = []

    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box

        # 1. Extract the letter from the original image with a 2-pixel margin
        y_start, y_end = max(0, y - 2), min(padded.shape[0], y + h + 2)
        x_start, x_end = max(0, x - 2), min(padded.shape[1], x + w + 2)
        letter_image = padded[y_start:y_end, x_start:x_end]
        
        # 2. Resize the character to a perfect 20x20 box preserving aspect ratio
        letter_image = resize_to_fit(letter_image, 20, 20)

        # 3. Normalize pixels to [0, 1] range
        letter_image = letter_image.astype('float32') / 255.0

        # 4. Reshape to PyTorch channels-first format: (Batch=1, Channel=1, H=20, W=20)
        letter_tensor = torch.from_numpy(letter_image).unsqueeze(0).unsqueeze(0)
        letter_tensor = letter_tensor.to(device)

        # 5. Run forward pass prediction through the model
        with torch.no_grad():
            outputs = model(letter_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            pred_numpy = probabilities.cpu().numpy()

        # 6. Translate the numerical index back into a text character
        letter = lb.inverse_transform(pred_numpy)[0]
        predictions.append(letter)

        # 7. Draw the bounding box annotation and prediction text on the visual canvas
        cv.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv.putText(output, letter, (x - 5, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Print decoded values out to terminal
    captcha_text = "".join(predictions)
    print(f"CAPTCHA text is {captcha_text}")

    # Render GUI Windows
    cv.imshow('Output', output)
    cv.waitKey(0)