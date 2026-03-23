import json

with open("ablation_results.json") as f:
    data = json.load(f)

print("\nAblation Table:\n")

print("Model\tAccuracy\tBleached Recall\tHealthy Recall")

for k, v in data.items():
    print(f"{k}\t{v['accuracy']:.2f}\t{v['bleached_corals_recall']:.2f}\t{v['healthy_corals_recall']:.2f}")