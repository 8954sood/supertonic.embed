"""Loopback-only web UI for the native MPS extractor. Run with python src/serve_mps.py."""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import signal
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "styles/mps-web"
FILES = {"best.json": "application/json", "style.json": "application/json",
         "report.json": "application/json", "preview.wav": "audio/wav"}
JOBS = {}
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, data, content_type="application/json", filename=None):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def valid_host(self):
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_GET(self):
        if not self.valid_host():
            return self.respond(403, {"error": "Loopback host required"})
        url = urlparse(self.path)
        if url.path in ("/", "/mps.mjs"):
            name = "mps.html" if url.path == "/" else "mps.mjs"
            return self.respond(200, (ROOT / "web" / name).read_bytes(),
                                "text/html; charset=utf-8" if name.endswith("html") else "text/javascript; charset=utf-8")
        parts = url.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "files" and parts[1] in JOBS and parts[2] in FILES:
            path = JOBS[parts[1]]["directory"] / parts[2]
            if path.is_file():
                return self.respond(200, path.read_bytes(), FILES[parts[2]], filename=parts[2])
        if url.path == "/api/status":
            job = JOBS.get(parse_qs(url.query).get("id", [""])[0])
            if job is None:
                return self.respond(404, {"error": "Unknown run"})
            path = job["directory"] / "progress.json"
            progress = json.loads(path.read_text()) if path.exists() else {}
            code = job["process"].poll()
            log = job["log_path"].read_text(errors="replace")[-6000:]
            files = [name for name in FILES if (job["directory"] / name).is_file()]
            return self.respond(200, {"progress": progress, "done": code is not None,
                "exit_code": code, "cancel_requested": job["cancel_requested"], "log": log, "files": files})
        self.respond(404, {"error": "Not found"})

    def do_POST(self):
        # A custom header rejects cross-site simple requests, and the origin check
        # protects against a foreign page causing local work. No CORS is enabled.
        allowed_origins = (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}")
        if not self.valid_host() or self.headers.get("Origin") not in allowed_origins or self.headers.get("X-Supertonic-Request") != "1":
            return self.respond(403, {"error": "Same-origin request required"})
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/api/cancel":
            job = JOBS.get(query.get("id", [""])[0])
            if job is None:
                return self.respond(404, {"error": "Unknown run"})
            if job["process"].poll() is None and not job["cancel_requested"]:
                job["cancel_requested"] = True
                try:
                    job["process"].send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
            return self.respond(200, {"ok": True})
        if url.path != "/api/start":
            return self.respond(404, {"error": "Not found"})
        try:
            iterations = int(query.get("iterations", ["0"])[0])
            duration = int(query.get("duration", ["100"])[0])
            threshold = float(query.get("threshold", ["0.24"])[0])
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= iterations <= 10000 or not 1 <= duration <= 1000 or not 0 < length <= 32 * 1024 * 1024:
                raise ValueError()
            if not math.isfinite(threshold) or threshold < 0:
                return self.respond(400, {"error": "목표 손실은 0 이상의 유한한 숫자여야 합니다."})
        except ValueError:
            return self.respond(400, {"error": "Invalid settings or file size (max 32 MiB)"})
        with LOCK:
            if any(job["process"].poll() is None for job in JOBS.values()):
                return self.respond(409, {"error": "Another extraction is running"})
            identifier = uuid.uuid4().hex
            directory = OUTPUTS / identifier
            directory.mkdir(parents=True)
            audio = directory / "reference.audio"
            body = self.rfile.read(length)
            if len(body) != length:
                return self.respond(400, {"error": "Incomplete audio upload"})
            audio.write_bytes(body)
            command = [sys.executable, "-u", str(ROOT / "src/run_extract_style_mps.py"), str(audio),
                       "--out", str(directory), "--iterations", str(iterations), "--duration-iterations", str(duration),
                       "--threshold", str(threshold)]
            if query.get("skip_duration", ["0"])[0] == "1":
                command.append("--skip-duration")
            log_path = directory / "run.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            JOBS[identifier] = {"process": process, "directory": directory, "log_path": log_path, "cancel_requested": False}
        self.respond(200, {"id": identifier})

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"MPS extraction: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in JOBS.values():
            if job["process"].poll() is None:
                try:
                    job["process"].send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
        server.server_close()


if __name__ == "__main__":
    main()
