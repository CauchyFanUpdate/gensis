import matplotlib.pyplot as plt
import os

data = {
    "Zero-Bench": 1773,
    "Babyvision": 1229,
    "Charxiv": 864,
    "EMMA": 220,
    "MMMU": 513,
    "MMMU-Pro": 869,
    "DocVQA": 1000,
    "InfoVQA": 1499,
    "logitVista": 458,
    "realwordQA": 961,
    "mmstar": 1032,
    "hallusionbench": 823,
    "MathVista_MINI": 1352,
    "MM-Bench-EN": 1149,
}

labels = list(data.keys())
sizes = list(data.values())
total = sum(sizes)

colors = plt.cm.tab20.colors[: len(labels)]

fig, ax = plt.subplots(figsize=(12, 9), subplot_kw=dict(aspect="equal"))

def autopct_fmt(pct):
    count = int(round(pct * total / 100.0))
    return f"{pct:.1f}%\n({count})"

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=autopct_fmt,
    startangle=140,
    pctdistance=0.78,
    labeldistance=1.05,
    wedgeprops=dict(edgecolor="white", linewidth=1.2),
    textprops=dict(fontsize=10),
)

for t in autotexts:
    t.set_fontsize(8)
    t.set_color("black")

ax.set_title(
    f"Reinforcement Learning Data Distribution (Total: {total})",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

legend_labels = [f"{k}: {v} ({v / total * 100:.1f}%)" for k, v in data.items()]
ax.legend(
    wedges,
    legend_labels,
    title="Datasets",
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    fontsize=9,
    title_fontsize=10,
)

plt.tight_layout()

out_dir = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration"
out_path = os.path.join(out_dir, "rl_data_distribution.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved pie chart to: {out_path}")
print(f"Total samples: {total}")
