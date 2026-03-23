import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

DATASETS = {
    "Turbid": "dataset/split_original_turbid",
    "Deep": "dataset/split_original_deep",
    "LowLight": "dataset/split_original_lowlight",
    "Robot": "dataset/split_original_robot"
}

results = {}

for name, path in DATASETS.items():
    print(f"\n=== Training on {name} dataset ===")

    train_data = datasets.ImageFolder(os.path.join(path, "train"), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(path, "test"), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TRAIN
    for epoch in range(5):  # keep small for speed
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, ", end = " ")

    # EVALUATE
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, output_dict=True)

    class_names = test_data.classes

    results[name] = {
        "accuracy": report["accuracy"],
        f"{class_names[0]}_recall": report["0"]["recall"],
        f"{class_names[1]}_recall": report["1"]["recall"]
    }

# SAVE RESULTS
with open("ablation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved results to ablation_results.json")