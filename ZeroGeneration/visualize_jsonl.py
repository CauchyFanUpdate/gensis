import os
import json
import glob
import html
import shutil

def main():
    base_dir = '/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration'
    vis_dir = os.path.join(base_dir, 'visualization')
    img_dir = os.path.join(vis_dir, 'images')
    
    os.makedirs(img_dir, exist_ok=True)
    
    # Find all jsonl files
    jsonl_files = glob.glob(os.path.join(base_dir, 'gen_pipeline_*/*think.jsonl'))
    
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
            .nav-pills .nav-link { cursor: pointer; margin: 0 5px; }
            .sidebar { position: sticky; top: 20px; height: calc(100vh - 40px); overflow-y: auto; }
        </style>
    </head>
    <body>
        <div class="container-fluid py-4">
            <h1 class="mb-4 text-center">ZeroGeneration JSONL Visualization</h1>
            
            <div class="row">
                <!-- Sidebar for navigation -->
                <div class="col-md-2">
                    <div class="sidebar">
                        <h5>Files</h5>
                        <ul class="nav nav-pills flex-column" id="file-tabs">
                            <li class="nav-item"><a class="nav-link active" onclick="filterFile('All')">All Files</a></li>
    """
    
    # Add navigation tabs
    for file_path in sorted(jsonl_files):
        file_name = os.path.basename(file_path)
        html_content += f'<li class="nav-item"><a class="nav-link" onclick="filterFile(\'{file_name}\')">{file_name}</a></li>\n'
        
    html_content += """
                        </ul>
                    </div>
                </div>
                
                <!-- Main content area -->
                <div class="col-md-10">
                    <div id="cards-container">
    """
    
    total_samples = 0
    
    for file_path in sorted(jsonl_files):
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                messages = data.get("messages", [])
                images = data.get("images", [])
                
                # Copy or symlink images
                img_relative_paths = []
                for img_idx, img_path in enumerate(images):
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
                    <div class="card jsonl-card" data-file="{file_name}">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <span class="badge bg-secondary">{file_name} (Line {line_idx+1})</span>
                        </div>
                        <div class="card-body">
                """
                
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    role_class = "role-user" if role == "user" else "role-assistant" if role == "assistant" else "bg-light"
                    
                    # If it's user, replace <image> with actual images
                    if role == "user":
                        # For each <image> tag, replace with corresponding image
                        for img_rel_path in img_relative_paths:
                            if img_rel_path:
                                img_tag = f'<br><img src="{img_rel_path}" class="q-img" alt="Image"><br>'
                                content = content.replace("<image>", img_tag, 1) # Replace one by one
                        
                        formatted_content = content # already contains HTML now
                    elif role == "assistant":
                        # Format <think> and <answer> blocks
                        formatted_content = html.escape(content)
                        # Re-enable the blocks
                        formatted_content = formatted_content.replace("&lt;think&gt;", '<div class="think-block"><strong>Think:</strong>\n')
                        formatted_content = formatted_content.replace("&lt;/think&gt;", '</div>')
                        formatted_content = formatted_content.replace("&lt;answer&gt;", '<div class="answer-block"><strong>Answer:</strong>\n')
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
        
        <script>
            function filterFile(fileName) {
                // Update active tab
                document.querySelectorAll('#file-tabs .nav-link').forEach(el => el.classList.remove('active'));
                event.target.classList.add('active');
                
                // Filter cards
                const cards = document.querySelectorAll('.jsonl-card');
                cards.forEach(card => {
                    if (fileName === 'All' || card.getAttribute('data-file') === fileName) {
                        card.style.display = 'block';
                    } else {
                        card.style.display = 'none';
                    }
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
