import json
import os
import glob
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import sys

JSONL_PATH = '/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench_output_v5/clear_zerobench_stage1_think.jsonl'
ORIGINAL_IMG_DIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/zerobench/images'
GENERATED_IMG_DIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/code_chenlifeng/AMath_train_dataset/zerobench_pipeline/_archive/output_v5/stage1_images'

class VisualizationHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed_path.path)

        if path == '/':
            self.serve_html()
        elif path.startswith('/images/original/'):
            filename = os.path.basename(path)
            filepath = os.path.join(ORIGINAL_IMG_DIR, filename)
            self.serve_file(filepath)
        elif path.startswith('/images/generated/'):
            filename = os.path.basename(path)
            filepath = os.path.join(GENERATED_IMG_DIR, filename)
            self.serve_file(filepath)
        else:
            self.send_error(404, "File not found")

    def serve_file(self, filepath):
        if not os.path.exists(filepath):
            self.send_error(404, "File not found")
            return
            
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.send_header('Content-length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Server Error: {str(e)}")

    def serve_html(self):
        try:
            html_content = self.generate_html()
            encoded = html_content.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Content-length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except Exception as e:
            self.send_error(500, f"Server Error generating HTML: {str(e)}")

    def generate_html(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>ZeroBench Visualization</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f4f5f7; padding: 20px; color: #333; margin: 0; }
                .container { max-width: 1400px; margin: 0 auto; }
                h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
                .card { background: #fff; margin-bottom: 30px; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); display: flex; gap: 20px; align-items: stretch; }
                .images-col { flex: 1; display: flex; flex-direction: column; gap: 15px; border-right: 1px solid #eaeaea; padding-right: 20px; }
                .text-col { flex: 1; display: flex; flex-direction: column; gap: 15px; overflow-wrap: break-word; }
                .img-container { background: #f8f9fa; border: 1px solid #e1e4e8; border-radius: 8px; padding: 10px; text-align: center; }
                .img-container img { max-width: 100%; height: auto; max-height: 400px; border-radius: 4px; object-fit: contain; }
                .img-title { font-weight: 600; margin-bottom: 8px; color: #495057; font-size: 0.9em; }
                .qa-box { background: #fff; padding: 15px; border: 1px solid #e1e4e8; border-left: 4px solid #0366d6; border-radius: 6px; }
                .qa-box h4 { margin-top: 0; margin-bottom: 10px; color: #0366d6; font-size: 1.1em; }
                .qa-box p { margin: 0; white-space: pre-wrap; line-height: 1.5; }
                .think-box { background: #fffbdd; padding: 15px; border: 1px solid #e1e4e8; border-left: 4px solid #f9a03f; border-radius: 6px; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.9em; line-height: 1.5; }
                .answer-box { background: #dcffe4; padding: 15px; border: 1px solid #e1e4e8; border-left: 4px solid #28a745; border-radius: 6px; font-weight: 600; white-space: pre-wrap; font-size: 1em; }
                .tag { display: inline-block; background: #e1e4e8; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-left: 10px; font-weight: normal; color: #586069; }
                .section-title { font-size: 0.85em; text-transform: uppercase; color: #6a737d; font-weight: 600; margin-bottom: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ZeroBench Generation Output</h1>
        """

        with open(JSONL_PATH, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                
                messages = data.get('messages', [])
                question = ""
                answer_raw = ""
                for msg in messages:
                    if msg.get('role') == 'user':
                        question = msg.get('content', '')
                    elif msg.get('role') == 'assistant':
                        answer_raw = msg.get('content', '')
                        
                think = ""
                answer = answer_raw
                if '<think>' in answer_raw and '</think>' in answer_raw:
                    parts = answer_raw.split('</think>')
                    think = parts[0].replace('<think>', '').strip()
                    answer = parts[1]
                    if '<answer>' in answer:
                        answer = answer.replace('<answer>', '').replace('</answer>', '').strip()
                elif '<answer>' in answer_raw:
                    answer = answer_raw.replace('<answer>', '').replace('</answer>', '').strip()
                        
                gen_images = data.get('images', [])
                
                img_html = ""
                
                if gen_images:
                    gen_img_path = gen_images[0]
                    gen_img_name = os.path.basename(gen_img_path)
                    
                    # Try to extract base ID (e.g. 1_1::e2.png -> 1)
                    base_id = gen_img_name.split('_')[0]
                    
                    orig_pattern = os.path.join(ORIGINAL_IMG_DIR, f"{base_id}_*.png")
                    orig_images = sorted(glob.glob(orig_pattern))
                    
                    if orig_images:
                        for o_img in orig_images:
                            o_name = os.path.basename(o_img)
                            img_html += f'''
                            <div class="img-container">
                                <div class="img-title">Original Image <span class="tag">{o_name}</span></div>
                                <img src="/images/original/{o_name}" loading="lazy">
                            </div>
                            '''
                    else:
                        img_html += f'<p>No original image found for base ID {base_id}</p>'
                    
                    for g_img in gen_images:
                        g_name = os.path.basename(g_img)
                        img_html += f'''
                        <div class="img-container" style="background: #eef4ff; border-color: #cce0ff;">
                            <div class="img-title">Generated Stage1 Image <span class="tag">{g_name}</span></div>
                            <img src="/images/generated/{g_name}" loading="lazy">
                        </div>
                        '''
                else:
                    img_html += "<p>No images associated with this entry.</p>"
                
                text_html = f'''
                <div class="qa-box">
                    <h4>Question {idx+1}</h4>
                    <p>{question}</p>
                </div>
                '''
                if think:
                    text_html += f'''
                    <div>
                        <div class="section-title">Reasoning (Think)</div>
                        <div class="think-box">{think}</div>
                    </div>
                    '''
                text_html += f'''
                <div>
                    <div class="section-title">Final Answer</div>
                    <div class="answer-box">{answer}</div>
                </div>
                '''
                
                html += f'''
                <div class="card">
                    <div class="images-col">{img_html}</div>
                    <div class="text-col">{text_html}</div>
                </div>
                '''

        html += """
            </div>
        </body>
        </html>
        """
        return html

def run(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, VisualizationHandler)
    print(f"Server running at http://localhost:{port}/")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

if __name__ == '__main__':
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run(port)
