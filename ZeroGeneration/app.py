import json
import os
import glob
from flask import Flask, render_template_string, send_from_directory, jsonify

app = Flask(__name__)

JSONL_PATH = '/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench_output_v5/clear_zerobench_stage1_think.jsonl'
ORIG_JSONL_PATH = '/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/zerobench/zerobench_converted.jsonl'
ORIGINAL_IMG_DIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/zerobench/images'
GENERATED_IMG_DIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/code_chenlifeng/AMath_train_dataset/zerobench_pipeline/_archive/output_v5/stage1_images'

orig_data_map = {}
gen_data_list = []

def load_data():
    global orig_data_map, gen_data_list
    orig_data_map.clear()
    gen_data_list.clear()
    
    # Load Original Data
    if os.path.exists(ORIG_JSONL_PATH):
        with open(ORIG_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                images = d.get('images', [])
                if not images: continue
                # zerobench/images/1_0.png -> 1_0.png
                first_img = os.path.basename(images[0])
                base_id = first_img.split('_')[0]
                
                q, a = "", ""
                for msg in d.get('messages', []):
                    if msg.get('role') == 'user': q = msg.get('content', '')
                    elif msg.get('role') == 'assistant': a = msg.get('content', '')
                
                # Remove <image> tag from question
                q = q.replace('<image>\n', '').replace('<image>', '').strip()
                
                orig_data_map[base_id] = {
                    'question': q,
                    'answer': a,
                    'images': [os.path.basename(img) for img in images]
                }

    # Load Generated Data
    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                images = d.get('images', [])
                gen_img_path = images[0] if images else ""
                base_id = ""
                if gen_img_path:
                    base_id = os.path.basename(gen_img_path).split('_')[0]
                    
                q, a_raw = "", ""
                for msg in d.get('messages', []):
                    if msg.get('role') == 'user': q = msg.get('content', '')
                    elif msg.get('role') == 'assistant': a_raw = msg.get('content', '')
                
                q = q.replace('<image>\n', '').replace('<image>', '').strip()
                    
                think = ""
                answer = a_raw
                if '<think>' in a_raw and '</think>' in a_raw:
                    parts = a_raw.split('</think>')
                    think = parts[0].replace('<think>', '').strip()
                    answer = parts[1]
                    if '<answer>' in answer:
                        answer = answer.replace('<answer>', '').replace('</answer>', '').strip()
                elif '<answer>' in a_raw:
                    answer = a_raw.replace('<answer>', '').replace('</answer>', '').strip()
                    
                gen_data_list.append({
                    'base_id': base_id,
                    'gen_images': [os.path.basename(img) for img in images],
                    'question': q,
                    'think': think,
                    'answer': answer
                })

load_data()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ZeroBench Comparison</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f4f5f7; margin: 0; padding: 0; display: flex; height: 100vh; overflow: hidden; }
        .sidebar { width: 250px; background: #fff; border-right: 1px solid #ddd; overflow-y: auto; display: flex; flex-direction: column; }
        .sidebar-header { padding: 15px; background: #2c3e50; color: #fff; font-weight: bold; position: sticky; top: 0; display: flex; justify-content: space-between; align-items: center; }
        .nav-item { padding: 12px 15px; border-bottom: 1px solid #eee; cursor: pointer; color: #333; font-size: 0.95em; }
        .nav-item:hover { background: #f0f4f8; }
        .nav-item.active { background: #e3effd; border-left: 4px solid #0366d6; font-weight: bold; }
        .main-content { flex: 1; overflow-y: auto; padding: 20px; }
        .header-controls { display: flex; justify-content: space-between; margin-bottom: 20px; align-items: center; background: #fff; padding: 15px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .btn { padding: 8px 16px; background: #0366d6; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .btn:hover { background: #0255b3; }
        .btn:disabled { background: #ccc; cursor: not-allowed; }
        .comparison-container { display: flex; gap: 20px; }
        .col { flex: 1; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); padding: 20px; display: flex; flex-direction: column; gap: 15px; overflow-x: hidden; }
        .col-title { font-size: 1.2em; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 0; text-align: center; }
        .img-container { background: #f8f9fa; border: 1px solid #e1e4e8; border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 15px; }
        .img-container img { max-width: 100%; height: auto; max-height: 400px; object-fit: contain; cursor: zoom-in; transition: transform 0.2s; }
        .img-container img:hover { transform: scale(1.02); }
        .img-label { font-size: 0.85em; color: #666; margin-bottom: 5px; font-family: monospace; background: #e1e4e8; display: inline-block; padding: 2px 6px; border-radius: 4px; }
        .qa-section { background: #f8f9fa; padding: 15px; border-radius: 6px; border: 1px solid #e1e4e8; }
        .qa-section h4 { margin-top: 0; color: #444; font-size: 0.9em; text-transform: uppercase; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
        .qa-section pre { white-space: pre-wrap; margin: 0; font-family: inherit; font-size: 0.95em; line-height: 1.6; }
        .think-box { background: #fffbdd; border-color: #ffe08a; }
        .think-box h4 { color: #b07d00; border-bottom-color: #ffe08a; }
        .think-box pre { font-family: monospace; font-size: 0.9em; }
        .answer-box { background: #dcffe4; border-color: #b7ebc5; }
        .answer-box h4 { color: #28a745; border-bottom-color: #b7ebc5; }
        .answer-box pre { font-weight: bold; font-size: 1.1em; }
        .badge { background: #e1e4e8; color: #333; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-left: 10px; font-weight: normal; }
        
        /* Modal Styles */
        .modal { display: none; position: fixed; z-index: 9999; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.85); justify-content: center; align-items: center; flex-direction: column; }
        .modal-content { max-width: 95%; max-height: 90vh; object-fit: contain; cursor: zoom-out; border-radius: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
        .close-btn { position: absolute; top: 20px; right: 30px; color: #fff; font-size: 40px; font-weight: bold; cursor: pointer; z-index: 10000; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
        .close-btn:hover { color: #ccc; }
    </style>
</head>
<body>
    <!-- Image Modal -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close-btn" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImg" onclick="event.stopPropagation(); closeModal();">
    </div>

    <div class="sidebar">
        <div class="sidebar-header">
            <span>Cases (<span id="total-cases">{{ total }}</span>)</span>
        </div>
        <div id="nav-list"></div>
    </div>
    <div class="main-content">
        <div class="header-controls">
            <button class="btn" id="btn-prev" onclick="nav(-1)">&#8592; Previous</button>
            <h2 id="case-title" style="margin:0; color:#333;">Loading...</h2>
            <button class="btn" id="btn-next" onclick="nav(1)">Next &#8594;</button>
        </div>
        <div class="comparison-container">
            <!-- Original Column -->
            <div class="col">
                <h3 class="col-title">Original Data</h3>
                <div id="orig-images"></div>
                <div class="qa-section">
                    <h4>Original Question</h4>
                    <pre id="orig-q"></pre>
                </div>
                <div class="qa-section answer-box">
                    <h4>Original Answer</h4>
                    <pre id="orig-a"></pre>
                </div>
            </div>
            <!-- Generated Column -->
            <div class="col">
                <h3 class="col-title">Generated Data</h3>
                <div id="gen-images"></div>
                <div class="qa-section">
                    <h4>Generated Question</h4>
                    <pre id="gen-q"></pre>
                </div>
                <div class="qa-section think-box" id="gen-think-container">
                    <h4>Reasoning (Think)</h4>
                    <pre id="gen-think"></pre>
                </div>
                <div class="qa-section answer-box">
                    <h4>Generated Answer</h4>
                    <pre id="gen-a"></pre>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentIndex = 0;
        const total = {{ total }};
        
        function openModal(imgSrc) {
            document.getElementById('imageModal').style.display = 'flex';
            document.getElementById('modalImg').src = imgSrc;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        function loadList() {
            const list = document.getElementById('nav-list');
            for (let i = 0; i < total; i++) {
                const div = document.createElement('div');
                div.className = 'nav-item';
                div.id = 'nav-' + i;
                div.innerText = 'Case #' + (i + 1);
                div.onclick = () => loadCase(i);
                list.appendChild(div);
            }
        }

        async function loadCase(index) {
            if (index < 0 || index >= total) return;
            currentIndex = index;
            
            // Update active state
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            const activeNav = document.getElementById('nav-' + index);
            if (activeNav) {
                activeNav.classList.add('active');
                // Scroll the sidebar to keep active item in view if needed
                const sidebar = document.querySelector('.sidebar');
                const offset = activeNav.offsetTop - sidebar.scrollTop;
                if (offset < 50 || offset > sidebar.clientHeight - 50) {
                    activeNav.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
            
            // Update buttons
            document.getElementById('btn-prev').disabled = (index === 0);
            document.getElementById('btn-next').disabled = (index === total - 1);
            
            try {
                const res = await fetch(`/api/case/${index}`);
                const data = await res.json();
                
                document.getElementById('case-title').innerHTML = `Case #${index + 1} <span class="badge">Base ID: ${data.base_id}</span>`;
                
                // Render Original
                const origImagesContainer = document.getElementById('orig-images');
                origImagesContainer.innerHTML = '';
                if (data.orig_data && data.orig_data.images) {
                    data.orig_data.images.forEach(img => {
                        origImagesContainer.innerHTML += `<div class="img-container"><div class="img-label">${img}</div><img src="/images/original/${img}" onclick="openModal(this.src)" /></div>`;
                    });
                } else {
                    origImagesContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No original images found</div>';
                }
                
                document.getElementById('orig-q').innerText = data.orig_data ? data.orig_data.question : 'N/A';
                document.getElementById('orig-a').innerText = data.orig_data ? data.orig_data.answer : 'N/A';
                
                // Render Generated
                const genImagesContainer = document.getElementById('gen-images');
                genImagesContainer.innerHTML = '';
                if (data.gen_images) {
                    data.gen_images.forEach(img => {
                        genImagesContainer.innerHTML += `<div class="img-container"><div class="img-label">${img}</div><img src="/images/generated/${img}" onclick="openModal(this.src)" /></div>`;
                    });
                }
                
                document.getElementById('gen-q').innerText = data.question || 'N/A';
                
                const thinkContainer = document.getElementById('gen-think-container');
                if (data.think) {
                    thinkContainer.style.display = 'block';
                    document.getElementById('gen-think').innerText = data.think;
                } else {
                    thinkContainer.style.display = 'none';
                }
                
                document.getElementById('gen-a').innerText = data.answer || 'N/A';
                
                // Scroll to top of main content
                document.querySelector('.main-content').scrollTop = 0;
                
            } catch (e) {
                console.error(e);
                alert("Error loading case data.");
            }
        }

        function nav(dir) {
            loadCase(currentIndex + dir);
        }

        // Add keyboard navigation
        document.addEventListener('keydown', function(event) {
            if (event.key === 'ArrowLeft') {
                nav(-1);
            } else if (event.key === 'ArrowRight') {
                nav(1);
            } else if (event.key === 'Escape') {
                closeModal();
            }
        });

        window.onload = () => {
            loadList();
            loadCase(0);
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, total=len(gen_data_list))

@app.route('/api/case/<int:idx>')
def get_case(idx):
    if idx < 0 or idx >= len(gen_data_list):
        return jsonify({'error': 'Index out of bounds'}), 400
    
    gen_item = gen_data_list[idx]
    base_id = gen_item['base_id']
    orig_item = orig_data_map.get(base_id, None)
    
    result = {
        'base_id': base_id,
        'gen_images': gen_item['gen_images'],
        'question': gen_item['question'],
        'think': gen_item['think'],
        'answer': gen_item['answer'],
        'orig_data': orig_item
    }
    return jsonify(result)

@app.route('/images/original/<path:filename>')
def serve_orig_img(filename):
    return send_from_directory(ORIGINAL_IMG_DIR, filename)

@app.route('/images/generated/<path:filename>')
def serve_gen_img(filename):
    return send_from_directory(GENERATED_IMG_DIR, filename)

import sys

if __name__ == '__main__':
    port = 8891
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    app.run(host='0.0.0.0', port=port)
