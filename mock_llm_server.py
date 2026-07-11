from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class MockLLM(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            post_data = self.rfile.read(content_length)
        
        # Fireworks/OpenAI chat completion response format
        # We wrap the JSON inside markdown ticks because that's what FireworksProvider expects to parse out.
        response = {
            "id": "mock-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "kimi-2.6-mock",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '```json\n{\n  "kernel_name": "vector_mul",\n  "surface_code": "parallel_for(i, N) {\\n  z[i] = x[i] * y[i];\\n}",\n  "assumptions": ["x, y, z are vectors of length N"]\n}\n```'
                },
                "finish_reason": "stop"
            }]
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    # Suppress verbose logging to keep output clean
    def log_message(self, format, *args):
        pass

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8080), MockLLM)
    print("Starting mock LLM server on port 8080...")
    server.serve_forever()
