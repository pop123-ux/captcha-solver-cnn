import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from helper import resize_to_fit


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LETTER_IMAGES_FOLDER = os.path.join(SCRIPT_DIR, "extracted_letter_images")
MODEL_WEIGHTS_FILENAME = os.path.join(SCRIPT_DIR, "captcha_model.pth")
MODEL_LABELS_FILENAME = os.path.join(SCRIPT_DIR, "model_labels.dat")

data = []
labels = []

# Data Ingestion and Preprocessing
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the letter to fit in a 20x20 pixel box
    image = resize_to_fit(image, 20, 20)

    # Grab the name of the letter based on the parent folder name
    label = image_file.split(os.path.sep)[-2]

    data.append(image)
    labels.append(label)

# Scale raw pixels to [0, 1] range and convert to numpy arrays
data = np.array(data, dtype='float32') / 255.0
labels = np.array(labels)

# Train/Test Split
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# One-Hot Encode the labels via scikit-learn
lb = LabelBinarizer().fit(Y_train)
Y_train_encoded = lb.transform(Y_train)
Y_test_encoded = lb.transform(Y_test)

# Save the label mapping to disk using pickle
with open(MODEL_LABELS_FILENAME, 'wb') as f:
    pickle.dump(lb, f)

X_train_pt = torch.from_numpy(X_train).unsqueeze(1) # Add Channel dimension at axis=1
X_test_pt = torch.from_numpy(X_test).unsqueeze(1)

Y_train_pt = torch.tensor(np.argmax(Y_train_encoded, axis=1), dtype=torch.long)
Y_test_pt = torch.tensor(np.argmax(Y_test_encoded, axis=1), dtype=torch.long)

# Wrap tensors into DataLoaders for clean batch generation
train_dataset = TensorDataset(X_train_pt, Y_train_pt)
test_dataset = TensorDataset(X_test_pt, Y_test_pt)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Network Architecture Definition
class CaptchaCNN(nn.Module):
    def __init__(self, num_classes):
        super(CaptchaCNN, self).__init__()

        # Convolutional Feature Extraction Blocks
        self.features = nn.Sequential(
            # Block 1: Input (1, 20, 20) -> Output (20, 20, 20) -> Pooled (20, 10, 10)
            nn.Conv2d(1, 20, kernel_size=5, padding=2), # padding=2 keeps dimensions 20x20
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: Input (20, 10, 10) -> Output (50, 10, 10) -> Pooled (50, 5, 5)
            nn.Conv2d(20, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dense Fully Connected Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(50*5*5, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes) # Outputs raw Logits (No explicit Softmax final layer here)
        )

    def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1) # Flatten dimensions except for batch
            x = self.classifier(x)
            return x
        
# Initialize training loop assets
num_classes = len(lb.classes_)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CaptchaCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Validation loops
epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Clear previous optimization gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass (Backpropagation)
        loss.backward()
        optimizer.step()

        # Track training metrics
        train_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()

    # Evaluate model performance against validation data
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()

    # Calculate average epoch losses and accuracies
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total

    print(f"Epoch {epoch+1:02d}/{epochs} - "
          f"loss: {epoch_train_loss:.4f} - accuracy: {epoch_train_acc:.4f} - "
          f"val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc:.4f}")

# Serialize final weights
torch.save(model.state_dict(), MODEL_WEIGHTS_FILENAME)
print(f"Successfully saved trained model weights to {MODEL_WEIGHTS_FILENAME}!")