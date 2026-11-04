"""Neural Chat desktop app: a local server for the page + the Lab model, shown in a native window."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import lab
import sysmon

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8766
_generating = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/lab/status":
            self._json(lab.info())
        elif self.path == "/sys":
            self._json(sysmon.snapshot())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/lab/chat":
            return self._json({"error": "not found"}, 404)
        messages = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))["messages"]
        if not _generating.acquire(blocking=False):
            return self._json({"error": "Une génération est déjà en cours."}, 409)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            send = lambda obj: (self.wfile.write((json.dumps(obj) + "\n").encode()), self.wfile.flush())
            if not lab._model:
                send({"status": "Chargement du modèle Labo…"})
            model = lab.get()
            closed = [False]
            for event in model.generate(messages, lambda: closed[0]):
                try:
                    send(event)
                except (BrokenPipeError, ConnectionResetError):
                    closed[0] = True    # the page pressed Esc: stop generating
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self.wfile.write((json.dumps({"error": str(e)}) + "\n").encode())
            except Exception:
                pass
        finally:
            _generating.release()


def start_server():
    for port in (PORT, 0):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def main():
    port = start_server()
    url = f"http://127.0.0.1:{port}/"
    if lab.available():
        threading.Thread(target=lab.get, daemon=True).start()   # warm up the Lab model in the background
    if "--server" in sys.argv:
        print(url, flush=True)
        threading.Event().wait()
    import webview
    webview.create_window("Neural Chat", url, width=1400, height=880, min_size=(420, 600), background_color="#070918")
    webview.start()


if __name__ == "__main__":
    main()
