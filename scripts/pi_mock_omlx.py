"""A stand-in oMLX server for developing the Pi extension without loading a model.

Serves /v1/models, a streaming /v1/chat/completions that replays recorded
replies, and /scope/api/live, which publishes the recorded trace events of the
reply being streamed (with the requesting client's id), at a realistic pace.

    python scripts/pi_mock_omlx.py REC.sse [REC2.sse ...] [--port 8099] [--tps 30]
    METALSCOPE_URL=http://127.0.0.1:8099 pi --model metalscope-mock/Qwen3.8-Flash-Next-oQ4e-mtp
"""
import argparse
import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("recordings", nargs="+")
ap.add_argument("--port", type=int, default=8099)
ap.add_argument("--tps", type=float, default=30.0)
ap.add_argument("--model", default="Qwen3.8-Flash-Next-oQ4e-mtp")
ap.add_argument("--dump", help="write each chat request body (JSON) to this path")
args = ap.parse_args()

# Recorded replies: [(events for one rid, start..end)], full level preferred.
replies, cur = [], {}
for path in args.recordings:
    for line in open(path):
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except json.JSONDecodeError:  # a recording cut mid-line
            continue
        cur.setdefault(ev["rid"], []).append(ev)
        if ev["type"] == "end":
            replies.append(cur.pop(ev["rid"]))
seen, uniq = set(), []
for r in replies:
    key = r[-1]["text"]
    if key not in seen:
        seen.add(key)
        uniq.append(r)
replies = uniq


print(f"{len(replies)} recorded replies", flush=True)
subscribers: list[queue.Queue] = []
lock = threading.Lock()
counter = [0]


def publish(ev):
    with lock:
        for q in subscribers:
            q.put(ev)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._json({"data": [{"id": args.model}]})
        if self.path.startswith("/scope/api/live"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = queue.Queue()
            with lock:
                subscribers.append(q)
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        ev = q.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    subscribers.remove(q)
            return
        if self.path.startswith("/scope/api/status"):     # what Pi reads to pick and register the server
            return self._json({"mock": True, "engine": "mock", "caps": [], "models": [{"name": args.model}],
                               "chat": {args.model: {"reasoning": True, "thinking_format": "qwen-chat-template"}}})
        if self.path.startswith("/scope"):
            return self._json({"mock": True})
        self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if args.dump:
            with open(args.dump, "w") as f:
                json.dump(body, f, indent=1)
        if not self.path.startswith("/v1/chat/completions"):
            return self.send_error(404)
        client = self.headers.get("X-Scope-Trace-Client")
        traced = self.headers.get("X-Scope-Trace", "off") != "off"
        rec = replies[counter[0] % len(replies)]
        counter[0] += 1
        rid = str(uuid.uuid4())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def chunk(delta, finish=None, usage=None):
            obj = {"id": rid, "object": "chat.completion.chunk", "model": args.model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage:
                obj["usage"] = usage
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        def ev(e, **extra):
            if traced:
                publish(dict(e, rid=rid, client=client, **extra))

        toks = [e for e in rec if e["type"] == "token"]
        end = rec[-1]
        ev(rec[0], level=self.headers.get("X-Scope-Trace"))
        time.sleep(0.3)
        for e in rec:
            if e["type"] == "prefill":
                ev(e)
        chunk({"role": "assistant", "content": ""})
        for e in toks[: end["n_tokens"]]:
            time.sleep(1.0 / args.tps)
            ev(e)
            chunk({"content": e["text"]})
        ev(end, trace_id=f"mock-{rid[:8]}")
        chunk({}, "stop", {"prompt_tokens": 30, "completion_tokens": len(toks), "total_tokens": 30 + len(toks)})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


print(f"mock oMLX on http://127.0.0.1:{args.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
