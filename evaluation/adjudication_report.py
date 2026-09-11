"""
adjudication_report.py -- one-off script, NOT part of the live app.

Reveals the blind A/B assignment from export_disagreements_for_adjudication.py
against your filled-in adjudicated_label judgments, and computes the numbers
the RQ2 write-up actually needs:

  - adjudicated accuracy: of the disagreements, how many does an
    independent human review say the SYSTEM actually got right (i.e. gold
    was wrong), vs. gold actually being right, vs. genuinely ambiguous.
  - a revised overall accuracy figure (this run's raw-agreement correct +
    however many disagreements adjudication now credits to the system) /
    105, to cite alongside the raw agreement percentage.

Run this only after every case in adjudication_blind.json has a non-null
adjudicated_label. Accepts "A"/"B", the literal label text (whichever of
label_a/label_b you meant, e.g. "Partial Match" -- matched case-insensitively
against THAT case's own label_a/label_b text, since they're randomly
assigned per case), or "both"/"neither".

2026-08-19 bug note: an earlier version of this script only recognized a
literal "a"/"b" and silently treated anything else (e.g. real label text
like "Partial Match") as neither, falling through to "system_right" for
100% of cases -- a parsing bug, not a real finding. Rewritten to also match
against the actual label text, and to raise loudly on anything it can't
confidently classify instead of silently misclassifying it.

Usage:
    python adjudication_report.py
"""
import json

with open("adjudication_blind.json", "r", encoding="utf-8") as f:
    blind_cases = json.load(f)
with open("adjudication_key.json", "r", encoding="utf-8") as f:
    key = json.load(f)

unfilled = [c["id"] for c in blind_cases if not c["adjudicated_label"]]
if unfilled:
    print(f"{len(unfilled)} case(s) still unfilled in adjudication_blind.json: {unfilled}")
    print("Fill in every 'adjudicated_label' before running this report.")
    raise SystemExit(1)

TOTAL_GOLDEN_SET = 105  # fixed size of golden_dataset_gemini.json

# Every pair NOT in this export's disagreement set was, by construction, an
# agreement in THIS run -- deriving raw-correct from this run's own data
# instead of a separately-hardcoded number from a different (possibly
# non-deterministic, see PROJECT_CONTEXT.md 4.6) run of golden_set_check.py.
n = len(blind_cases)
RAW_CORRECT = TOTAL_GOLDEN_SET - n

system_credited = 0
gold_credited = 0
both_defensible = 0
neither_right = 0
unparseable = []
rows = []

for case in blind_cases:
    k = key[case["id"]]
    raw = case["adjudicated_label"].strip()
    verdict = raw.lower()

    label_a_norm = case["label_a"].strip().lower()
    label_b_norm = case["label_b"].strip().lower()

    if verdict in ("both", "neither"):
        outcome = verdict
        if verdict == "both":
            both_defensible += 1
        else:
            neither_right += 1
    elif verdict in ("a", "b"):
        chosen_side = verdict.upper()
        if chosen_side == k["gold_side"]:
            outcome = "gold_right"
            gold_credited += 1
        else:
            outcome = "system_right"
            system_credited += 1
    elif verdict == label_a_norm or verdict == label_b_norm:
        chosen_side = "A" if verdict == label_a_norm else "B"
        # If both label_a and label_b happen to be the identical text (rare,
        # but possible), the letter is genuinely ambiguous from text alone --
        # falls through to unparseable below instead of guessing.
        if label_a_norm == label_b_norm:
            unparseable.append((case["id"], raw))
            continue
        if chosen_side == k["gold_side"]:
            outcome = "gold_right"
            gold_credited += 1
        else:
            outcome = "system_right"
            system_credited += 1
    elif verdict in ("match", "partial match", "no match"):
        # A real, valid label, but not either of the two offered for this
        # case -- the human is asserting a specific THIRD label is correct
        # and both gold and system were wrong. More informative than a bare
        # "neither", so kept distinct in the outcome string.
        outcome = f"neither_right_answer_is_{verdict.replace(' ', '_')}"
        neither_right += 1
    else:
        unparseable.append((case["id"], raw))
        continue

    rows.append({
        "id": case["id"],
        "source_file": case["source_file"],
        "requirement": case["requirement"][:70],
        "gold_label": k["gold_label"],
        "system_label": k["system_label"],
        "human_notes": case["notes"],
        "outcome": outcome,
    })

if unparseable:
    print(f"{len(unparseable)} case(s) had an adjudicated_label that couldn't be "
          f"matched to 'A', 'B', 'both', 'neither', or that case's own label text:")
    for case_id, raw in unparseable:
        print(f"  {case_id}: {raw!r}")
    print("Fix these before the report can be trusted -- refusing to guess.")
    raise SystemExit(1)

print(f"Adjudicated {n} disagreement case(s):\n")
print(f"  System actually right (gold was wrong): {system_credited}")
print(f"  Gold actually right (system was wrong):  {gold_credited}")
print(f"  Both defensible / genuinely ambiguous:   {both_defensible}")
print(f"  Neither right:                           {neither_right}\n")

for r in rows:
    print(f"[{r['outcome']:>13s}] {r['id']} ({r['source_file']}) {r['requirement']}...")
    print(f"               gold={r['gold_label']} system={r['system_label']}"
          + (f" | notes: {r['human_notes']}" if r["human_notes"] else ""))

adjudicated_correct = RAW_CORRECT + system_credited
adjudicated_accuracy = adjudicated_correct / TOTAL_GOLDEN_SET * 100

print(f"\nRaw agreement (unadjudicated):  {RAW_CORRECT}/{TOTAL_GOLDEN_SET} = "
      f"{RAW_CORRECT / TOTAL_GOLDEN_SET * 100:.1f}%")
print(f"Adjudicated accuracy:           {adjudicated_correct}/{TOTAL_GOLDEN_SET} = "
      f"{adjudicated_accuracy:.1f}%  "
      f"(credits the system for {system_credited} case(s) where gold was wrong)")

with open("adjudication_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "total_disagreements": n,
        "system_credited": system_credited,
        "gold_credited": gold_credited,
        "both_defensible": both_defensible,
        "neither_right": neither_right,
        "raw_correct": RAW_CORRECT,
        "total_golden_set": TOTAL_GOLDEN_SET,
        "adjudicated_correct": adjudicated_correct,
        "adjudicated_accuracy_pct": round(adjudicated_accuracy, 1),
        "cases": rows,
    }, f, indent=2)

print("\nFull breakdown saved to adjudication_summary.json")
