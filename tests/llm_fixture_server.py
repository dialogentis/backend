"""Isolated OpenAI-compatible transport for browser acceptance tests."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.dialogue_fixtures import fixture_completion


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        result = json.dumps(fixture_completion(body)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        self.wfile.write(result)


if __name__ == "__main__":
    with ThreadingHTTPServer(("0.0.0.0", 8000), Handler) as server:
        server.serve_forever()
