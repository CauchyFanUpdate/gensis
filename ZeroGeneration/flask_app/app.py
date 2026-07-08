import json
import os
import re
import math
import pandas as pd
from flask import Flask, render_template, send_file, request, abort

app = Flask(__name__)

# Load Original Data from Parquet
original_data = {}
orig_base_dir = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/BabyVision/images"
parquet_path = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/BabyVision/data/train-00000-of-00001.parquet"

if os.path.exists(parquet_path):
    df = pd.read_parquet(parquet_path)
    for _, row in df.iterrows():
        task_id = str(row['taskId'])
        
        # Get image path
        img_path = None
        if isinstance(row.get('image'), dict) and 'path' in row['image']:
            img_filename = row['image']['path']
            img_path = os.path.join(orig_base_dir, img_filename)
        
        # Determine original answer
        ans = row.get('choiceAns')
        if pd.isna(ans) or ans == '' or ans is None:
            ans = row.get('blankAns')
        
        original_data[task_id] = {
            'images': [img_path] if img_path and os.path.exists(img_path) else [],
            'question': row.get('question', ''),
            'answer': str(ans)
        }

# Load Generated Data
generated_data = []
gen_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_babyvision/babyvision_stage1_think.jsonl"

if os.path.exists(gen_file):
    with open(gen_file, 'r') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            images = item.get('images', [])
            if not images:
                continue
            
            basename = os.path.basename(images[0])
            match = re.match(r'^(\d+)::', basename)
            if match:
                orig_id = match.group(1)
            else:
                orig_id = "unknown"
                
            orig_info = original_data.get(orig_id, {})
            
            # Safe extraction from generated messages
            messages = item.get('messages', [])
            gen_question = ""
            gen_answer = ""
            if len(messages) > 0:
                gen_question = messages[0].get('content', '').replace('<image>', '').replace('<image>\n', '').strip()
            if len(messages) > 1:
                gen_answer = messages[1].get('content', '')
            
            generated_data.append({
                'id': idx + 1,
                'orig_id': orig_id,
                'orig_images': orig_info.get('images', []),
                'orig_question': orig_info.get('question', 'N/A'),
                'orig_answer': orig_info.get('answer', 'N/A'),
                'gen_images': images,
                'gen_question': gen_question,
                'gen_answer': gen_answer
            })

@app.route('/')
def index():
    page = int(request.args.get('page', 1))
    per_page = 20
    total_items = len(generated_data)
    total_pages = math.ceil(total_items / per_page) if total_items > 0 else 1
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    items = generated_data[start_idx:end_idx]
    
    return render_template('index.html', items=items, page=page, total_pages=total_pages, title="BabyVision Generation Visualization")

@app.route('/image')
def serve_image():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        abort(404)
    return send_file(path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
