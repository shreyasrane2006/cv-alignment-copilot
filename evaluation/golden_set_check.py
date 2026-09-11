import json
from matcher import match_requirement
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

with open("golden_dataset_gemini.json", "r", encoding="utf-8") as f:
    golden_set = json.load(f)

results = []
for entry in golden_set:
    evidence = [] if entry["cv_evidence"].strip().lower() == "none" else [
        {"text": entry["cv_evidence"], "source": "golden_set", "similarity": 1.0}
    ]
    result = match_requirement(entry["requirement"], evidence)
    is_correct = result.match_label == entry["gold_label"]
    results.append({
        "source_file": entry["source_file"],
        "requirement": entry["requirement"],
        "gold": entry["gold_label"],
        "system": result.match_label,
        "correct": is_correct,
    })
    print(f"{'[OK]' if is_correct else '[X] '} [{entry['source_file']}] {entry['requirement'][:60]}...")
    print(f"   Gold: {entry['gold_label']} | System: {result.match_label}\n")

accuracy = sum(r["correct"] for r in results) / len(results) * 100
print(f"\nOverall agreement: {accuracy:.1f}% ({sum(r['correct'] for r in results)}/{len(results)})")

# Confusion breakdown — which label pairs get confused most
from collections import Counter
confusion = Counter((r["gold"], r["system"]) for r in results)
print("\nConfusion pairs (gold -> system):")
for (gold, system), count in confusion.most_common():
    marker = "[OK]" if gold == system else "    "
    print(f"{marker} {gold:15s} -> {system:15s} : {count}")

gold_labels = [r["gold"] for r in results]
system_labels = [r["system"] for r in results]

kappa = cohen_kappa_score(gold_labels, system_labels)
print(f"\nCohen's Kappa: {kappa:.3f}")

labels = ["Match", "Partial Match", "No Match"]

print("\nConfusion matrix:")
print(confusion_matrix(gold_labels, system_labels, labels=labels))

print("\nClassification report:")
print(classification_report(gold_labels, system_labels, labels=labels))