import os
import pandas as pd
from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

dataset_path = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/zerobench"

# Load data on startup
print("Loading dataset...")
main_df = pd.read_parquet(f"{dataset_path}/data/zerobench-00000-of-00001.parquet")
sub_df = pd.read_parquet(f"{dataset_path}/data/zerobench_subquestions-00000-of-00001.parquet")

main_records = main_df.drop(columns=['question_images_decoded'], errors='ignore').to_dict(orient='records')
sub_records = sub_df.drop(columns=['question_images_decoded'], errors='ignore').to_dict(orient='records')

tasks = {}
for r in main_records:
    tid = str(r['question_id'])
    tasks[tid] = {
        'main': r,
        'subs': []
    }

for r in sub_records:
    tid = str(r['question_id'])
    main_id = tid.split('_')[0]
    if main_id in tasks:
        tasks[main_id]['subs'].append(r)

sorted_task_ids = sorted(tasks.keys(), key=lambda x: int(x) if x.isdigit() else x)

def process_images(record):
    imgs = record.get('question_images', [])
    if type(imgs).__name__ == 'ndarray':
        imgs = imgs.tolist()
    elif not isinstance(imgs, (list, tuple)):
        imgs = []
    # Strip prefix so we can serve them from the flask route
    return [img.replace('zerobench/images/', '').replace('images/', '') for img in imgs if isinstance(img, str)]

for tid in sorted_task_ids:
    tasks[tid]['main']['processed_images'] = process_images(tasks[tid]['main'])
    for sub in tasks[tid]['subs']:
        sub['processed_images'] = process_images(sub)
        
    tasks[tid]['subs'] = sorted(tasks[tid]['subs'], key=lambda x: int(str(x['question_id']).split('_')[1]) if '_' in str(x['question_id']) else str(x['question_id']))

@app.route('/')
def index():
    return render_template('index.html', tasks=tasks, sorted_task_ids=sorted_task_ids)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(dataset_path, 'images'), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=True)
