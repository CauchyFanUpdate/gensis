import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, send_file, request

app = Flask(__name__)

# Paths
JSONL_PATH = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_babyvision/clear_babyvision_stage2_think.jsonl"
PARQUET_PATH = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/BabyVision/data/train-00000-of-00001.parquet"
ORIGINAL_IMAGES_DIR = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/BabyVision/images"

# Load original data
print("Loading original dataset...")
original_df = pd.read_parquet(PARQUET_PATH)
original_data_map = {}
for _, row in original_df.iterrows():
    task_id = str(row['taskId'])
    ans = row['choiceAns'] if pd.notna(row['choiceAns']) else row['blankAns']
    
    options = row.get('options', [])
    if isinstance(options, np.ndarray):
        options = options.tolist()
        
    original_data_map[task_id] = {
        'image_path': os.path.join(ORIGINAL_IMAGES_DIR, row['image']['path']),
        'question': row['question'],
        'options': options,
        'answer': ans,
        'cot': row['coT'] if pd.notna(row['coT']) else ""
    }
print("Original dataset loaded.")

# Load generated data
print("Loading generated dataset...")
generated_data = []
with open(JSONL_PATH, 'r') as f:
    for line in f:
        data = json.loads(line)
        gen_img_path = data['images'][0]
        # Extract taskId from filename e.g. "4173::e2::p1::s0.png"
        filename = os.path.basename(gen_img_path)
        task_id = filename.split("::")[0]
        
        user_msg = ""
        assistant_msg = ""
        for msg in data['messages']:
            if msg['role'] == 'user':
                user_msg = msg['content'].replace('<image>', '').strip()
            elif msg['role'] == 'assistant':
                assistant_msg = msg['content']
                
        generated_data.append({
            'task_id': task_id,
            'gen_img_path': gen_img_path,
            'gen_question': user_msg,
            'gen_answer': assistant_msg,
            'original': original_data_map.get(task_id, {})
        })
print(f"Generated dataset loaded: {len(generated_data)} items.")

@app.route('/')
def index():
    page = int(request.args.get('page', 1))
    per_page = 10
    total_pages = (len(generated_data) + per_page - 1) // per_page
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    items = generated_data[start_idx:end_idx]
    
    return render_template('index.html', items=items, page=page, total_pages=total_pages)

@app.route('/image/original/<task_id>')
def serve_original_image(task_id):
    if task_id in original_data_map:
        return send_file(original_data_map[task_id]['image_path'])
    return "Not found", 404

@app.route('/image/generated')
def serve_generated_image():
    path = request.args.get('path')
    if path and os.path.exists(path):
        return send_file(path)
    return "Not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
