import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import os
import time
import random
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "dataset/split_full/test"
MODEL_PATH = "best_model.pth"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# -----------------------------
# LOAD DATASET (RANDOM POOL)
# -----------------------------
classes = sorted(os.listdir(DATASET_PATH))

all_images = []
for cls in classes:
    class_path = os.path.join(DATASET_PATH, cls)
    for file in os.listdir(class_path):
        all_images.append((os.path.join(class_path, file), cls))

# -----------------------------
# DISPLAY FUNCTION
# -----------------------------
def show_image(img, true_label, pred_label=None, status="Predicting", confidence=0.0):
    display = cv2.resize(img, (500, 500))

    # bottom panel
    panel = 255 * np.ones((150, 500, 3), dtype=np.uint8)
    combined = np.vstack((display, panel))

    # TEXT
    cv2.putText(combined, f"True: {true_label}", (10, 520),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    if pred_label:
        cv2.putText(combined, f"Pred: {pred_label}", (10, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    # STATUS
    color = (0,165,255) if status=="Predicting" else (0,200,0) if status=="Passed" else (0,0,255)
    cv2.putText(combined, f"Status: {status}", (10, 580),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # CONFIDENCE BAR
    bar_x, bar_y = 10, 610
    bar_width = int(confidence * 480)

    cv2.rectangle(combined, (bar_x, bar_y), (bar_x+480, bar_y+20), (200,200,200), -1)
    cv2.rectangle(combined, (bar_x, bar_y), (bar_x+bar_width, bar_y+20), (0,200,0), -1)

    cv2.putText(combined, f"Confidence: {confidence:.2f}", (10, 645),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    cv2.imshow("Coral Demo", combined)


# -----------------------------
# MAIN LOOP
# -----------------------------
auto_mode = True
current_index = 0

while True:

    # RANDOM IMAGE
    img_path, true_label = random.choice(all_images)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # Predicting state
    show_image(img, true_label, status="Predicting", confidence=0)
    cv2.waitKey(300)

    # Preprocess
    input_img = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    pred_label = classes[pred]

    status = "Passed" if pred_label == true_label else "Failed"

    show_image(img, true_label, pred_label, status, confidence)

    # -----------------------------
    # KEY CONTROL
    # -----------------------------
    while True:
        key = cv2.waitKey(0 if not auto_mode else 1000)

        # ESC → exit
        if key == 27:
            cv2.destroyAllWindows()
            exit()

        # Toggle auto mode
        elif key == ord('w'):
            auto_mode = not auto_mode
            print("Auto mode:", auto_mode)
            break

        # Manual next (only if auto off)
        elif key == ord('n') and not auto_mode:
            break

        # Auto move
        elif auto_mode:
            break