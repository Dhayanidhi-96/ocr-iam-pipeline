from datasets import load_dataset
from collections import Counter
import os

#loading dataset from hugging face 
ds = load_dataset("Teklia/IAM-line")

def audit(split_name, dataset):
    texts = dataset["text"]
    print(f"SPLIT:{split_name.upper()} ({len(texts):,} samples)")

    empty       = [t for t in texts if not t or t.strip() == ""]
    very_short  = [t for t in texts if len(t.strip()) <= 2]
    very_long   = [t for t in texts if len(t.strip()) > 100]
    avg_len     = sum(len(t) for t in texts) / len(texts)
    duplicates  = {t: c for t, c in Counter(texts).items() if c > 1}

    print(f"  Empty labels      : {len(empty)}")
    print(f"  Very short (<=2)  : {len(very_short)} → {[t for t in very_short]}")
    print(f"  Very long  (>100) : {len(very_long)}")
    print(f"  Avg length        : {avg_len:.1f} chars")
    print(f"  Duplicates        : {len(duplicates)} unique texts repeated")

    return {
        "split"       : split_name,
        "total"       : len(texts),
        "empty"       : len(empty),
        "very_short"  : len(very_short),
        "very_long"   : len(very_long),
        "avg_len"     : round(avg_len, 1),
        "duplicates"  : len(duplicates),
    }

results = []
for split in ["train","validation","test"]:
    results.append(audit(split, ds[split]))

def normalize_text(text: str) -> str:
    """
    Rules:
    - Strip leading/trailing whitespace
    - Preserve case (real handwriting has capitals)
    - Keep punctuation (part of real sentences)
    - Keep digits (valid content)
    - Remove if only special chars with no alphanumeric content
    """
    text = text.strip()
    return text

def filter_sample(text: str) -> bool:
    """Return True if sample should be KEPT."""
    text = text.strip()
    if len(text) <= 2:
        return False  # too short, no learning value
    if not any(c.isalnum() for c in text):
        return False  # no alphanumeric content at all
    return True

print("\n\nFILTER SUMMARY")
print("-" * 50)
for split in ["train", "validation", "test"]:
    texts   = ds[split]["text"]
    kept    = [t for t in texts if filter_sample(t)]
    dropped = len(texts) - len(kept)
    print(f"  {split:12} : {len(texts):,} total → {dropped} dropped → {len(kept):,} kept")

# ── Save audit report ─────────────────────────────────────────
os.makedirs("reports", exist_ok=True)
with open("reports/data_audit.md", "w") as f:
    f.write("# Data Audit Report - IAM Handwriting Dataset\n\n")
    f.write("## Dataset Overview\n\n")
    f.write("| Split | Total | Empty | Very Short | Avg Length | Duplicates |\n")
    f.write("|---|---|---|---|---|---|\n")
    for r in results:
        f.write(f"| {r['split']} | {r['total']:,} | {r['empty']} | {r['very_short']} | {r['avg_len']} | {r['duplicates']} |\n")
    f.write("\n## Normalization Decisions\n\n")
    f.write("| Decision | Choice | Reason |\n")
    f.write("|---|---|---|\n")
    f.write("| Case | Preserve | Real handwriting has capitals. TrOCR trained with case. |\n")
    f.write("| Punctuation | Keep | Part of real sentences. Removing hurts model. |\n")
    f.write("| Digits | Keep | Rare but valid real-world content. |\n")
    f.write("| Whitespace | Strip | Clean hygiene, no information lost. |\n")
    f.write("| Samples <= 2 chars | Remove | No learning value, likely labelling errors. |\n")
    f.write("| Duplicates | Keep | Different images, same text = valid diversity. |\n")

print("\n✅ Audit report saved → reports/data_audit.md")

import matplotlib.pyplot as plt
import json

# ── Chart 1: Already done ─────────────────────────────────────
# reports/text_length_dist.png ✅

# ── Chart 2: Experiment Comparison Bar Chart ──────────────────
experiments = ['Broken\nBaseline', 'Fixed\nBaseline', 'Fine-tuned\n(Colab)', 'LLM\nPost-processed']
cer_values  = [55.31, 8.48, 18.29, 20.52]
wer_values  = [46.46, 30.72, 25.34, 47.04]

x = range(len(experiments))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar([i - width/2 for i in x], cer_values, width, label='CER %', color='steelblue')
bars2 = ax.bar([i + width/2 for i in x], wer_values, width, label='WER %', color='coral')

ax.set_title('Experiment Comparison — CER vs WER', fontsize=14)
ax.set_ylabel('Error Rate (%)')
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.legend()
ax.set_ylim(0, 70)

# add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height()}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height()}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('reports/experiment_comparison.png')
plt.close()
print("✅ Saved → reports/experiment_comparison.png")

# ── Chart 3: Error Type Distribution ─────────────────────────
error_types  = ['Char\nConfusion', 'Word\nSubstitution', 'Digit\nErrors', 'Repetition', 'Punctuation\nStyle']
frequencies  = [35, 25, 8, 20, 12]  # approximate % from our analysis
colors       = ['#e74c3c', '#e67e22', '#f1c40f', '#9b59b6', '#3498db']

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    frequencies,
    labels=error_types,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.85
)
ax.set_title('Error Type Distribution\n(from sample analysis)', fontsize=14)
plt.tight_layout()
plt.savefig('reports/error_distribution.png')
plt.close()
print("✅ Saved → reports/error_distribution.png")