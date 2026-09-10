"""
src/ui/serve_dashboard.py
=========================
Stage 7: Decision-Support Dashboard Backend Server.

Lightweight local HTTP server:
  - Serves static web app assets from web/
  - Exposes REST API endpoints:
      GET  /api/state         -> Returns full multi-day route & hazard states
      POST /api/recalculate   -> Re-runs pipeline for user input coordinates
"""

import os
import sys
import json
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

WEB_DIR = os.path.join(PROJECT_ROOT, "web")
MULTI_DAY_STATE_PATH = os.path.join(PROJECT_ROOT, "src/contracts/multi_day_route_state.json")
INITIAL_STATE_PATH = os.path.join(PROJECT_ROOT, "src/contracts/initial_state.json")


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom request handler serving web UI and route/hazard APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/state":
            self._handle_get_state()
        else:
            super().do_GET()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/recalculate":
            self._handle_recalculate()
        else:
            self.send_error(404, "Endpoint not found")

    def _handle_get_state(self):
        try:
            if os.path.exists(MULTI_DAY_STATE_PATH):
                with open(MULTI_DAY_STATE_PATH, "r") as f:
                    data = json.load(f)
            else:
                data = []

            payload = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            self.send_error(500, f"Error reading state: {e}")

    def _handle_recalculate(self):
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            params = json.loads(body) if body else {}

            start_coords = params.get("start_coords", [-63.5, -58.2])
            dest_coords = params.get("dest_coords", [-60.8, -52.4])
            date_str = params.get("date", "2026-09-10T00:00:00Z")
            days_raw = params.get("days", 7)
            try:
                days = max(1, min(14, int(days_raw)))
            except (ValueError, TypeError):
                days = 7

            from run_pipeline import run_pipeline
            res = run_pipeline(
                start_coords=start_coords,
                dest_coords=dest_coords,
                forecast_start_date=date_str,
                forecast_days=days
            )

            payload = json.dumps(res["step3"]).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ERROR] /api/recalculate:\n{tb}")
            err_msg = json.dumps({"error": str(e), "traceback": tb}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_msg)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(err_msg)


def run_server(port=8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"🌊 Antarctic Navigation Dashboard server running at:")
    print(f"   👉 http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Navigation Decision-Support Dashboard Server")
    parser.add_argument("--port", "-p", type=int, default=8080, help="HTTP Port (default: 8080)")
    args = parser.parse_args()
    run_server(args.port)
