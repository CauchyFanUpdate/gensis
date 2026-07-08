import os
import json
import html
import shutil
from collections import defaultdict

TASKS = ['chess', 'connect4', 'game2048', 'graph', 'maze', 'nonogram', 'othello', 'pong', 'sokoban', 'sudoku', 'track_a']

def get_cat_and_task(file_path):
    # Category
    if 'outputs/rl' in file_path:
        category = 'rl'
    elif 'outputs/sft' in file_path:
        category = 'sft'
    elif 'outputs/hopchain' in file_path:
        category = 'hopchain'
    else:
        category = 'other'
        
    # Task
    filename = os.path.basename(file_path)
    task = 'other'
    for t in TASKS:
        if t in filename:
            task = t
            break
            
    if category == 'hopchain' and task == 'other':
        task = 'queries' # default group for verified_queries
        
    return category, task

def main():
    base_dir = '/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench-main'
    vis_dir = os.path.join(base_dir, 'visualization')
    img_dir = os.path.join(vis_dir, 'images')
    
    os.makedirs(img_dir, exist_ok=True)
    
    # Find all jsonl files
    jsonl_files = []
    for root, dirs, files in os.walk(os.path.join(base_dir, 'outputs')):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
                
    # Group files by Category -> Task -> File
    categories = defaultdict(lambda: defaultdict(list))
    for f in sorted(jsonl_files):
        cat, task = get_cat_and_task(f)
        categories[cat][task].append(f)
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ZeroGeneration JSONL Visualization</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
            .card { margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .q-img { max-width: 100%; max-height: 500px; object-fit: contain; margin: 10px 0; border: 1px solid #dee2e6; border-radius: 4px; padding: 4px; background: white;}
            .role-badge { font-size: 0.9rem; margin-bottom: 10px; display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
            .role-user { background-color: #e9ecef; color: #495057; }
            .role-assistant { background-color: #cff4fc; color: #055160; }
            .think-block { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; font-style: italic; white-space: pre-wrap; }
            .answer-block { background-color: #d1e7dd; border-left: 4px solid #198754; padding: 10px; margin: 10px 0; font-weight: bold; white-space: pre-wrap; }
            .message-content { white-space: pre-wrap; }
            .sidebar { position: sticky; top: 20px; height: calc(100vh - 40px); overflow-y: auto; }
            
            /* Accordion Styling */
            .accordion-button { padding: 0.75rem 1rem; font-weight: 600; }
            .accordion-button:not(.collapsed) { background-color: #e7f1ff; color: #0c63e4; }
            .list-group-item { cursor: pointer; border-left: 4px solid transparent; transition: all 0.2s; }
            .list-group-item:hover { background-color: #f8f9fa; border-left-color: #dee2e6; }
            .list-group-item.active { background-color: #e9ecef; border-left-color: #0d6efd; color: #212529; font-weight: bold; }
            .sub-list { padding-left: 1rem; font-size: 0.9rem; }
            .sub-list .list-group-item { padding: 0.4rem 1rem; }
            .category-title { text-transform: uppercase; font-size: 0.8rem; letter-spacing: 1px; color: #6c757d; margin: 15px 0 5px 10px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container-fluid py-4">
            <h1 class="mb-4 text-center">ZeroGeneration JSONL Visualization</h1>
            
            <div class="row">
                <!-- Sidebar for navigation -->
                <div class="col-md-3">
                    <div class="sidebar">
                        <button class="btn btn-primary w-100 mb-3" onclick="filterCards('All', 'All', 'All')">Show All Files</button>
                        
                        <div class="accordion" id="accordionSidebar">
    """
    
    # Generate Accordion HTML for Sidebar
    for cat_idx, cat in enumerate(sorted(categories.keys())):
        cat_id = f"collapse-cat-{cat}"
        html_content += f"""
                            <div class="accordion-item">
                                <h2 class="accordion-header">
                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#{cat_id}">
                                        {cat.upper()}
                                    </button>
                                </h2>
                                <div id="{cat_id}" class="accordion-collapse collapse" data-bs-parent="#accordionSidebar">
                                    <div class="accordion-body p-0">
                                        <div class="accordion accordion-flush" id="accordion-{cat}">
        """
        for task_idx, task in enumerate(sorted(categories[cat].keys())):
            task_id = f"collapse-{cat}-{task}"
            html_content += f"""
                                            <div class="accordion-item">
                                                <h2 class="accordion-header">
                                                    <button class="accordion-button collapsed py-2" type="button" data-bs-toggle="collapse" data-bs-target="#{task_id}">
                                                        {task}
                                                    </button>
                                                </h2>
                                                <div id="{task_id}" class="accordion-collapse collapse" data-bs-parent="#accordion-{cat}">
                                                    <ul class="list-group list-group-flush sub-list">
            """
            for file_path in sorted(categories[cat][task]):
                file_name = os.path.basename(file_path)
                html_content += f"""
                                                        <li class="list-group-item nav-file" onclick="filterCards('{cat}', '{task}', '{file_name}', this)">
                                                            {file_name}
                                                        </li>
                """
            html_content += """
                                                    </ul>
                                                </div>
                                            </div>
            """
        html_content += """
                                        </div>
                                    </div>
                                </div>
                            </div>
        """
        
    html_content += """
                        </div>
                    </div>
                </div>
                
                <!-- Main content area -->
                <div class="col-md-9">
                    <div id="cards-container">
    """
    
    total_samples = 0
    
    # Generate Cards
    for file_path in sorted(jsonl_files):
        file_name = os.path.basename(file_path)
        cat, task = get_cat_and_task(file_path)
        print(f"Processing {cat}/{task}/{file_name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # Handle different JSON formats
                if "messages" not in data:
                    if "query" in data:
                        # Hopchain format (e.g., verified_queries.jsonl)
                        messages = [
                            {"role": "user", "content": "<image>\\n" + str(data.get("query", ""))},
                            {"role": "assistant", "content": str(data.get("answer", ""))}
                        ]
                        img_path = data.get("image_path")
                        images = [img_path] if img_path else []
                    elif "original_question" in data:
                        # Track A format
                        messages = [
                            {"role": "user", "content": "<image>\\n" + str(data.get("original_question", ""))},
                            {"role": "assistant", "content": "Original Answer: " + str(data.get("original_answer", "")) + "\\nNew Answer: " + str(data.get("new_answer", "")) + "\\nEdit: " + str(data.get("edit_description", ""))}
                        ]
                        img_path = data.get("image_path")
                        images = [img_path] if img_path else []
                    else:
                        messages = []
                        images = []
                else:
                    messages = data.get("messages", [])
                    images = data.get("images", [])
                
                # Copy or symlink images
                img_relative_paths = []
                for img_idx, img_path in enumerate(images):
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(base_dir, img_path)
                    if os.path.exists(img_path):
                        ext = os.path.splitext(img_path)[1]
                        # Create unique filename: filename_lineidx_imgidx
                        safe_name = f"{file_name}_L{line_idx}_I{img_idx}{ext}"
                        dest_path = os.path.join(img_dir, safe_name)
                        
                        # Use copy instead of symlink to ensure it works anywhere
                        if not os.path.exists(dest_path):
                            shutil.copy2(img_path, dest_path)
                            
                        img_relative_paths.append(f"images/{safe_name}")
                    else:
                        print(f"Warning: Image not found {img_path}")
                        img_relative_paths.append("") # Not found
                
                # Build card for this sample
                card_html = f"""
                    <div class="card jsonl-card" data-cat="{cat}" data-task="{task}" data-file="{file_name}">
                        <div class="card-header d-flex justify-content-between align-items-center bg-light">
                            <span class="badge bg-primary me-2">{cat.upper()}</span>
                            <span class="badge bg-secondary me-2">{task}</span>
                            <span class="text-muted small ms-auto">{file_name} (Line {line_idx+1})</span>
                        </div>
                        <div class="card-body">
                """
                
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    role_class = "role-user" if role == "user" else "role-assistant" if role == "assistant" else "bg-light"
                    
                    # If it's user, replace <image> with actual images
                    if role == "user":
                        for img_rel_path in img_relative_paths:
                            if img_rel_path:
                                img_tag = f'<br><img src="{img_rel_path}" class="q-img" alt="Image"><br>'
                                content = content.replace("<image>", img_tag, 1) # Replace one by one
                        
                        formatted_content = content # already contains HTML now
                    elif role == "assistant":
                        # Format <think> and <answer> blocks
                        formatted_content = html.escape(content)
                        # Re-enable the blocks
                        formatted_content = formatted_content.replace("&lt;think&gt;", '<div class="think-block"><strong>Think:</strong>\\n')
                        formatted_content = formatted_content.replace("&lt;/think&gt;", '</div>')
                        formatted_content = formatted_content.replace("&lt;answer&gt;", '<div class="answer-block"><strong>Answer:</strong>\\n')
                        formatted_content = formatted_content.replace("&lt;/answer&gt;", '</div>')
                    else:
                        formatted_content = html.escape(content)
                        
                    card_html += f"""
                        <div class="mb-3">
                            <div class="role-badge {role_class}">{role.capitalize()}</div>
                            <div class="message-content">{formatted_content}</div>
                        </div>
                    """
                    
                card_html += """
                        </div>
                    </div>
                """
                
                html_content += card_html
                total_samples += 1

    html_content += """
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            function filterCards(cat, task, file, el) {
                // Update active state in sidebar
                document.querySelectorAll('.nav-file').forEach(e => e.classList.remove('active'));
                if (el) el.classList.add('active');
                
                // Filter cards
                const cards = document.querySelectorAll('.jsonl-card');
                cards.forEach(card => {
                    let show = true;
                    if (cat !== 'All' && card.getAttribute('data-cat') !== cat) show = false;
                    if (task !== 'All' && card.getAttribute('data-task') !== task) show = false;
                    if (file !== 'All' && card.getAttribute('data-file') !== file) show = false;
                    
                    card.style.display = show ? 'block' : 'none';
                });
            }
        </script>
    </body>
    </html>
    """
    
    out_html = os.path.join(vis_dir, 'index.html')
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"\\nVisualization generated successfully at: {out_html}")
    print(f"Total samples processed: {total_samples}")

if __name__ == "__main__":
    main()
