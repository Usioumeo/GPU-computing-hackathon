import http.server
import socketserver
import json

PORT = 7700
DATA_FILE = "gpu-computing-hackathon-results.jsonl"  # Line-delimited JSON

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/append":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            json_data = json.loads(body)  # Validate it's JSON
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid JSON")
            return

        with open(DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_data) + "\n")

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"JSON appended")

    def do_GET(self):
        if self.path == "/data":
            try:
                with open(DATA_FILE, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"[\n" + b",\n".join(data.strip().splitlines()) + b"\n]")
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"No data found")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()
