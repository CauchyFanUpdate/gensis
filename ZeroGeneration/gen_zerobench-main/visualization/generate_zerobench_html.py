import pandas as pd
import json
import os
import numpy as np

dataset_path = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/zerobench"
output_path = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench-main/visualization/zerobench_tasks.html"

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
    else:
        print(f"Warning: subquestion {tid} has no main question.")

sorted_task_ids = sorted(tasks.keys(), key=lambda x: int(x) if x.isdigit() else x)

html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZeroBench Tasks Visualization</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .sidebar { position: sticky; top: 20px; height: calc(100vh - 40px); overflow-y: auto; }
        .q-img { max-width: 100%; max-height: 400px; object-fit: contain; margin: 10px 0; border: 1px solid #dee2e6; border-radius: 4px; padding: 4px; background: white;}
        .sub-task { margin-left: 20px; border-left: 3px solid #0d6efd; padding-left: 15px; }
        .nav-pills .nav-link { cursor: pointer; text-align: left; }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <h1 class="mb-4 text-center">ZeroBench Main & Sub Tasks</h1>
        <div class="row">
            <div class="col-md-2">
                <div class="sidebar bg-white p-3 rounded shadow-sm">
                    <h5 class="mb-3">Tasks List</h5>
                    <div class="nav flex-column nav-pills" id="task-tabs" role="tablist" aria-orientation="vertical">
"""

for i, tid in enumerate(sorted_task_ids):
    active = "active" if i == 0 else ""
    html_content += f'<button class="nav-link {active} mb-1" id="tab-{tid}" data-bs-toggle="pill" data-bs-target="#content-{tid}" type="button" role="tab">Task {tid}</button>\n'

html_content += """
                    </div>
                </div>
            </div>
            <div class="col-md-10">
                <div class="tab-content" id="task-tabContent">
"""

def render_images(image_paths):
    if not isinstance(image_paths, (list, tuple)):
        if type(image_paths).__name__ == 'ndarray':
            image_paths = image_paths.tolist()
        else:
            return ""
    if not image_paths: return ""
    imgs = []
    for p in image_paths:
        if isinstance(p, str):
            # Resolve image relative path from visualization dir to datasets dir
            # visualization dir: project/benchmark/ZeroGeneration/gen_zerobench-main/visualization/
            # datasets dir: datasets/zerobench/
            rel_p = f"../../../../../datasets/zerobench/{p}"
            imgs.append(f'<img src="{rel_p}" class="q-img" alt="Image">')
    return "".join(imgs)

for i, tid in enumerate(sorted_task_ids):
    active = "show active" if i == 0 else ""
    task = tasks[tid]
    main_q = task['main']
    
    def get_sub_id(x):
        try:
            return int(str(x['question_id']).split('_')[1])
        except:
            return str(x['question_id'])
            
    subs = sorted(task['subs'], key=get_sub_id)
    
    html_content += f"""
                    <div class="tab-pane fade {active}" id="content-{tid}" role="tabpanel" tabindex="0">
                        <div class="card mb-4 shadow-sm border-primary">
                            <div class="card-header bg-primary text-white d-flex justify-content-between align-items-center">
                                <h4 class="mb-0">Main Task: {main_q.get('question_id', tid)}</h4>
                                <span class="badge bg-light text-dark">{main_q.get('category', '')} / {main_q.get('sub_category', '')}</span>
                            </div>
                            <div class="card-body">
                                <h5>Question:</h5>
                                <p style="white-space: pre-wrap; font-size: 1.1em;">{main_q.get('question_text', '')}</p>
                                {render_images(main_q.get('question_images', []))}
                                <h5 class="mt-3 text-success">Answer:</h5>
                                <p class="fw-bold fs-5">{main_q.get('question_answer', '')}</p>
                            </div>
                        </div>
    """
    
    if subs:
        html_content += f'<h4 class="mb-3 text-secondary">Sub Tasks ({len(subs)})</h4>'
        for sub in subs:
            html_content += f"""
                        <div class="card mb-3 shadow-sm sub-task">
                            <div class="card-header bg-light d-flex justify-content-between align-items-center">
                                <h5 class="mb-0">Sub Task: {sub.get('question_id', '')}</h5>
                                <span class="badge bg-secondary">{sub.get('category', '')} / {sub.get('sub_category', '')}</span>
                            </div>
                            <div class="card-body">
                                <h5>Question:</h5>
                                <p style="white-space: pre-wrap;">{sub.get('question_text', '')}</p>
                                {render_images(sub.get('question_images', []))}
                                <h5 class="mt-3 text-success">Answer:</h5>
                                <p class="fw-bold">{sub.get('question_answer', '')}</p>
                            </div>
                        </div>
            """
    html_content += "</div>\n"

html_content += """
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated visualization at {output_path}")
