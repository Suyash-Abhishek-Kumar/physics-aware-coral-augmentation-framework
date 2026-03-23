import json
import matplotlib.pyplot as plt

with open("ablation_results.json") as f:
    data = json.load(f)

models = list(data.keys())
accuracy = [data[m]["accuracy"] for m in models]
bleached = [data[m]["bleached_corals_recall"] for m in models]
healthy = [data[m]["healthy_corals_recall"] for m in models]

# Accuracy bar chart
plt.figure()
plt.bar(models, accuracy)
plt.title("Ablation Study - Accuracy")
plt.xlabel("Dataset Variant")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.savefig("ablation_accuracy.png")

# Recall line chart
plt.figure()
plt.plot(models, bleached, marker='o', label="Bleached Recall")
plt.plot(models, healthy, marker='o', label="Healthy Recall")
plt.title("Ablation Study - Recall")
plt.xlabel("Dataset Variant")
plt.ylabel("Recall")
plt.legend()
plt.xticks(rotation=30)
plt.savefig("ablation_recall.png")

print("Graphs saved!")