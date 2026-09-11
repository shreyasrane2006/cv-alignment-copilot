"""
export_disagreements_for_adjudication.py -- one-off script, NOT part of the
live app.

Prepares the golden-set disagreement pairs (PROJECT_CONTEXT.md section 8:
"raw agreement 67.6%... not yet done: full human adjudication of the 34
disagreement pairs") for a BLIND human review, so the adjudication numbers
are defensible in the report rather than "the student re-read the cases and
agreed with whichever label they already trusted."

For every (CV snippet, requirement) pair where the system's label disagreed
with golden_dataset_gemini.json's gold label, this writes TWO files:

  adjudication_blind.json  -- what you actually review. Each case shows the
      requirement + CV evidence + two candidate labels, randomly ordered as
      "Label A" / "Label B" with NO indication of which one is gold and
      which is the system's -- fill in "adjudicated_label" with "A", "B",
      "both" (genuinely both defensible), or "neither" (both wrong) as you
      go, plus optional "notes". Safe to send to a second reviewer too, for
      an inter-rater number -- it carries no information about which side
      is which.

  adjudication_key.json    -- DO NOT open this until adjudication_blind.json
      is fully filled in. Maps each case id to which side (A/B) was gold vs
      system, plus the system's own reasoning, for adjudication_report.py
      to reveal afterward.

Usage:
    python export_disagreements_for_adjudication.py
"""
import json
import random

from matcher import match_requirement

random.seed(42)  # reproducible A/B assignment, not that it matters for validity

with open("golden_dataset_gemini.json", "r", encoding="utf-8") as f:
    golden_set = json.load(f)

blind_cases = []
key = {}

disagreement_num = 0
for i, entry in enumerate(golden_set):
    evidence = [] if entry["cv_evidence"].strip().lower() == "none" else [
        {"text": entry["cv_evidence"], "source": "golden_set", "similarity": 1.0}
    ]
    result = match_requirement(entry["requirement"], evidence)

    if result.match_label == entry["gold_label"]:
        continue

    disagreement_num += 1
    case_id = f"case_{disagreement_num:02d}"
    print(f"[{disagreement_num}] {entry['source_file']} :: {entry['requirement'][:60]}")

    gold_is_a = random.random() < 0.5
    label_a = entry["gold_label"] if gold_is_a else result.match_label
    label_b = result.match_label if gold_is_a else entry["gold_label"]

    blind_cases.append({
        "id": case_id,
        "source_file": entry["source_file"],
        "requirement": entry["requirement"],
        "cv_evidence": entry["cv_evidence"],
        "label_a": label_a,
        "label_b": label_b,
        "adjudicated_label": None,  # fill in: "A", "B", "both", or "neither"
        "notes": "",
    })
    key[case_id] = {
        "gold_side": "A" if gold_is_a else "B",
        "system_side": "B" if gold_is_a else "A",
        "gold_label": entry["gold_label"],
        "system_label": result.match_label,
        "system_reason": result.reason,
    }

with open("adjudication_blind.json", "w", encoding="utf-8") as f:
    json.dump(blind_cases, f, indent=2)

with open("adjudication_key.json", "w", encoding="utf-8") as f:
    json.dump(key, f, indent=2)

print(f"\n{disagreement_num} disagreement(s) found.")
print("Review adjudication_blind.json -- fill in 'adjudicated_label' per case "
      "('A', 'B', 'both', or 'neither'). Do not open adjudication_key.json "
      "until you're done, then run adjudication_report.py.")
