import ast
import json
import re
from collections import Counter
from pathlib import Path

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

SRC = Path("/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench_output_v5/parquet/merged_with_images.jsonl")
OUT_DIR = Path("/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench_output_v5/wordcloud_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOILERPLATE = (
    "You are a helpful, conversational assistant tasked with answering a question about an image. "
    "First provide concise reasoning inside <think></think>, then give the final response inside <answer></answer>."
)

TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

EXTRA_STOPWORDS = {
    "image", "images", "question", "answer", "think", "user", "assistant",
    "shown", "show", "shows", "given", "based", "following", "please",
    "let", "lets", "us", "also", "would", "could", "should", "may",
    "one", "two", "three", "four", "five", "many", "much", "see",
    "use", "using", "used", "make", "made", "get", "got", "go", "goes",
    "still", "even", "every", "another", "first", "second", "third",
    "look", "looks", "looking", "find", "finding", "find", "know",
    "thing", "things", "way", "ways", "well", "like", "want",
}
STOP = set(STOPWORDS) | EXTRA_STOPWORDS

counter = Counter()
question_chunks = []
n = 0
with SRC.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt_field = rec.get("prompt", "")
        try:
            msgs = ast.literal_eval(prompt_field) if isinstance(prompt_field, str) else prompt_field
        except (ValueError, SyntaxError):
            msgs = [{"content": prompt_field}]
        for m in msgs:
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            content = content.replace(BOILERPLATE, " ")
            content = TAG_RE.sub(" ", content)
            question_chunks.append(content)
            for tok in TOKEN_RE.findall(content.lower()):
                if tok in STOP or len(tok) <= 2:
                    continue
                counter[tok] += 1
        n += 1

print(f"Records processed: {n}")
print(f"Unique tokens: {len(counter)}")
print("Top 30:", counter.most_common(30))

freq_path = OUT_DIR / "word_frequencies.txt"
with freq_path.open("w", encoding="utf-8") as f:
    for w, c in counter.most_common(500):
        f.write(f"{w}\t{c}\n")
print(f"Wrote frequencies -> {freq_path}")

wc = WordCloud(
    width=1600,
    height=900,
    background_color="white",
    max_words=400,
    collocations=False,
    prefer_horizontal=0.9,
).generate_from_frequencies(dict(counter.most_common(400)))

png_path = OUT_DIR / "wordcloud.png"
wc.to_file(str(png_path))
print(f"Wrote word cloud -> {png_path}")

fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
ax.set_title(f"Word Cloud — {SRC.name} ({n} records)", fontsize=14)
fig.tight_layout()
preview = OUT_DIR / "wordcloud_preview.png"
fig.savefig(preview, bbox_inches="tight")
print(f"Wrote preview -> {preview}")
