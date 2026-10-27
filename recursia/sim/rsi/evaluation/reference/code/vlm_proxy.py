#!/usr/bin/env python3
"""Loopback-only VLM proxy for the VB1 arm (runs in .venv-vlmproxy; never inside a rollout worker).

The provider credential stays in this process (the Anthropic SDK resolves it from this process's environment); rollout workers
reach the model only through POST http://127.0.0.1:<port>/v1/vlm and never see the key. A hard USD cap is enforced from a
persistent ledger: once spent + the worst-case cost of the next call would exceed the cap, requests are refused (HTTP 402) and
the policy treats that as an abstention.

Request JSON:  {"tag": str, "prompt": str, "images": [base64 PNG, ...], "schema": {JSON schema}, "max_tokens": int (optional)}
Response JSON: {"status": "ok", "data": {...}, "usage": {...}, "cost_usd": float, "id": str}
               | {"status": "refusal" | "truncated" | "error" | "budget_exhausted", "detail": str}
Model: claude-opus-5-5 (fixed), effort "low", adaptive thinking (the model default), structured JSON output. No server-side
fallback: a refusal is returned as an abstention so that every answer in the evaluation comes from one model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic

MODEL = "claude-opus-5-5"
EFFORT = "low"
PRICE_IN, PRICE_OUT = 4.00 / 1e6, 20.00 / 1e6          # USD per token (claude-opus-5-5 list price)
PRICE_CACHE_READ, PRICE_CACHE_WRITE = 0.20 / 1e6, 5.00 / 1e6
DEFAULT_MAX_TOKENS = 3000
MAX_IMAGES = 4


class Ledger:
    def __init__(self, path: Path, cap_usd: float):
        self.path, self.cap = path, float(cap_usd)
        self.lock = threading.Lock()
        self.spent = 0.0
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    self.spent += float(json.loads(line).get("cost_usd", 0.0))

    def reserve_ok(self, worst_case: float) -> bool:
        with self.lock:
            return self.spent + worst_case <= self.cap

    def record(self, entry: dict):
        with self.lock:
            self.spent += float(entry.get("cost_usd", 0.0))
            entry["spent_total_usd"] = round(self.spent, 6)
            with self.path.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")


def usage_cost(u) -> float:
    return (int(getattr(u, "input_tokens", 0) or 0) * PRICE_IN + int(getattr(u, "output_tokens", 0) or 0) * PRICE_OUT
            + int(getattr(u, "cache_read_input_tokens", 0) or 0) * PRICE_CACHE_READ
            + int(getattr(u, "cache_creation_input_tokens", 0) or 0) * PRICE_CACHE_WRITE)


def _new_client():
    # N15 amendment v2: SDK retries off (0) and a 40 s per-attempt timeout; the handler's own 4-attempt loop retries and rebuilds the
    # client after a connection failure (a stuck long-lived connection through the HTTP proxy made every later call fail).
    return anthropic.Anthropic(max_retries=0, timeout=anthropic.Timeout(40.0, connect=10.0))


def make_handler(client: anthropic.Anthropic, ledger: Ledger):
    holder = {"c": client}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):   # quiet; the ledger is the record
            pass

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": MODEL, "effort": EFFORT, "spent_usd": round(ledger.spent, 6), "cap_usd": ledger.cap})
            else:
                self._send(404, {"status": "error", "detail": "not found"})

        def do_POST(self):
            if self.path != "/v1/vlm":
                return self._send(404, {"status": "error", "detail": "not found"})
            t0 = time.time()
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                prompt, images, schema = str(req["prompt"]), list(req.get("images") or []), dict(req["schema"])
                tag, max_tokens = str(req.get("tag", "")), int(req.get("max_tokens", DEFAULT_MAX_TOKENS))
                assert 0 < len(images) <= MAX_IMAGES and 0 < max_tokens <= 8000
            except (KeyError, ValueError, TypeError, AssertionError, json.JSONDecodeError) as e:
                return self._send(400, {"status": "error", "detail": f"bad request: {e!r}"})
            # worst case: generous input estimate (images <= 1600 tokens each + text) + the full output budget
            worst = (len(images) * 1600 + len(prompt) // 2 + 500) * PRICE_IN + max_tokens * PRICE_OUT
            entry = {"t": round(t0, 3), "tag": tag, "model": MODEL, "effort": EFFORT, "n_images": len(images),
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16]}
            if not ledger.reserve_ok(worst):
                entry.update(status="budget_exhausted", cost_usd=0.0); ledger.record(entry)
                return self._send(402, {"status": "budget_exhausted", "detail": f"cap {ledger.cap} USD reached"})
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b}} for b in images]
            content.append({"type": "text", "text": prompt})
            try:
                # N15 infrastructure amendment (2026-10-09): transient provider failures (connection, rate limit, 5xx/overloaded) are
                # retried up to 4 attempts with 2/4/8 s backoff before the request is answered as an error (abstention). Answers,
                # schema, model, effort and the USD cap are unchanged; retries cost nothing for failed attempts.
                for _attempt in range(4):
                    try:
                        resp = holder["c"].messages.create(
                            model=MODEL, max_tokens=max_tokens,
                            messages=[{"role": "user", "content": content}],
                            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
                        )
                        break
                    except (anthropic.APIConnectionError, anthropic.RateLimitError) as _e:
                        if _attempt == 3:
                            raise
                        if isinstance(_e, anthropic.APIConnectionError):
                            holder["c"] = _new_client()   # fresh connection pool after a connection failure
                    except anthropic.APIStatusError as _e:
                        if _attempt == 3 or int(getattr(_e, "status_code", 0) or 0) < 500:
                            raise
                    entry["retries"] = _attempt + 1
                    time.sleep(2 ** (_attempt + 1))
            except anthropic.RateLimitError as e:
                entry.update(status="error", detail="rate_limited", cost_usd=0.0); ledger.record(entry)
                return self._send(200, {"status": "error", "detail": f"rate limited: {e.status_code}"})
            except anthropic.APIStatusError as e:
                entry.update(status="error", detail=f"api_status_{e.status_code}", cost_usd=0.0); ledger.record(entry)
                return self._send(200, {"status": "error", "detail": f"api status {e.status_code}"})
            except anthropic.APIConnectionError as e:
                entry.update(status="error", detail="connection", cost_usd=0.0); ledger.record(entry)
                return self._send(200, {"status": "error", "detail": f"connection: {e!r}"})
            cost = usage_cost(resp.usage)
            u = {k: int(getattr(resp.usage, k, 0) or 0) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
            entry.update(id=resp.id, stop_reason=resp.stop_reason, usage=u, cost_usd=round(cost, 6), latency_s=round(time.time() - t0, 2))
            if resp.stop_reason == "refusal":
                entry["status"] = "refusal"; ledger.record(entry)
                return self._send(200, {"status": "refusal", "detail": "model declined", "cost_usd": cost, "id": resp.id})
            if resp.stop_reason == "max_tokens":
                entry["status"] = "truncated"; ledger.record(entry)
                return self._send(200, {"status": "truncated", "detail": "max_tokens", "cost_usd": cost, "id": resp.id})
            text = next((b.text for b in resp.content if b.type == "text"), None)
            try:
                data = json.loads(text) if text is not None else None
                assert isinstance(data, dict)
            except (json.JSONDecodeError, AssertionError):
                entry["status"] = "error"; entry["detail"] = "unparsable"; ledger.record(entry)
                return self._send(200, {"status": "error", "detail": "unparsable output", "cost_usd": cost, "id": resp.id})
            entry["status"] = "ok"; ledger.record(entry)
            return self._send(200, {"status": "ok", "data": data, "usage": u, "cost_usd": cost, "id": resp.id})
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8117)
    p.add_argument("--cap-usd", type=float, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    a = p.parse_args()
    client = _new_client()
    ledger = Ledger(a.ledger, a.cap_usd)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(client, ledger))   # loopback only
    print(json.dumps({"listening": f"127.0.0.1:{a.port}", "model": MODEL, "effort": EFFORT, "cap_usd": a.cap_usd, "spent_usd": ledger.spent}), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
