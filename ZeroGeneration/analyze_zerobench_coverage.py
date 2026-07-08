import json
import re
import os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

JSONL_PATH = '/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench_output_v5/zerobench_v6_think.jsonl'
IMAGES_DIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/zerobench/images'
OUTPUT_DIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration'

# Count entries per image ID from jsonl
counts = defaultdict(int)
with open(JSONL_PATH) as f:
    for line in f:
        d = json.loads(line)
        imgs = d.get('images', [])
        for img in imgs:
            name = img.split('/')[-1]
            m = re.match(r'^(\d+)_(\d+)::', name)
            if m:
                image_id = int(m.group(1))
                counts[image_id] += 1
                break

# All 100 zerobench image IDs (1-100)
all_ids = list(range(1, 101))
entry_counts = [counts.get(i, 0) for i in all_ids]

covered = sum(1 for c in entry_counts if c > 0)
not_covered = 100 - covered
missing_ids = [i for i in all_ids if counts.get(i, 0) == 0]

print(f"Total image IDs (1-100): 100")
print(f"Covered (appear in jsonl): {covered}")
print(f"Not covered: {not_covered}")
print(f"Missing IDs: {missing_ids}")
print(f"Total jsonl entries: {sum(entry_counts)}")
print(f"Max entries per image: {max(entry_counts)}")
print(f"Min entries (covered): {min(c for c in entry_counts if c > 0)}")
print(f"Mean entries (covered): {sum(c for c in entry_counts if c > 0)/covered:.1f}")

# --- Figure layout ---
fig = plt.figure(figsize=(18, 12))
fig.suptitle('ZeroBench Images Coverage in zerobench_v6_think.jsonl', fontsize=16, fontweight='bold', y=0.98)

# ── Pie chart ──────────────────────────────────────────────
ax_pie = fig.add_axes([0.03, 0.45, 0.35, 0.48])
pie_labels = [f'Covered ({covered})', f'Not Covered ({not_covered})']
pie_sizes = [covered, not_covered]
pie_colors = ['#4CAF50', '#F44336']
explode = (0.05, 0.05)
wedges, texts, autotexts = ax_pie.pie(
    pie_sizes, labels=pie_labels, colors=pie_colors,
    autopct='%1.1f%%', startangle=90, explode=explode,
    textprops={'fontsize': 12},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight('bold')
ax_pie.set_title('Image Coverage (out of 100 images)', fontsize=13, pad=12)

# ── Histogram of entry counts ───────────────────────────────
ax_hist = fig.add_axes([0.45, 0.45, 0.52, 0.48])
covered_counts = [c for c in entry_counts if c > 0]
bins = range(0, max(entry_counts) + 10, 8)
ax_hist.hist(covered_counts, bins=bins, color='#2196F3', edgecolor='white', linewidth=0.8)
ax_hist.set_xlabel('Number of Entries in JSONL', fontsize=12)
ax_hist.set_ylabel('Number of Images', fontsize=12)
ax_hist.set_title('Distribution of Entry Counts per Covered Image', fontsize=13)
ax_hist.axvline(np.mean(covered_counts), color='#FF5722', linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(covered_counts):.1f}')
ax_hist.axvline(np.median(covered_counts), color='#9C27B0', linestyle=':', linewidth=2,
                label=f'Median = {np.median(covered_counts):.1f}')
ax_hist.legend(fontsize=11)
ax_hist.grid(axis='y', alpha=0.3)

# ── Bar chart: entry count per image ID ─────────────────────
ax_bar = fig.add_axes([0.05, 0.06, 0.92, 0.33])
colors = ['#F44336' if c == 0 else '#2196F3' for c in entry_counts]
bars = ax_bar.bar(all_ids, entry_counts, color=colors, edgecolor='none', width=0.8)
ax_bar.set_xlabel('Image ID (1–100)', fontsize=12)
ax_bar.set_ylabel('Entry Count', fontsize=12)
ax_bar.set_title('Entry Count per Image ID in JSONL', fontsize=13)
ax_bar.set_xlim(0, 101)
ax_bar.set_xticks(range(0, 101, 5))
ax_bar.grid(axis='y', alpha=0.3)

# Annotate missing
for i in missing_ids:
    ax_bar.annotate(f'{i}', xy=(i, 1), ha='center', va='bottom',
                    fontsize=7, color='#F44336', fontweight='bold')

legend_handles = [
    mpatches.Patch(color='#2196F3', label='In JSONL'),
    mpatches.Patch(color='#F44336', label='Missing from JSONL'),
]
ax_bar.legend(handles=legend_handles, fontsize=11, loc='upper right')

out_path = os.path.join(OUTPUT_DIR, 'zerobench_coverage_analysis.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved to: {out_path}")
