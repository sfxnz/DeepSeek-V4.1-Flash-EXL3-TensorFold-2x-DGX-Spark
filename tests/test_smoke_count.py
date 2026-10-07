#!/usr/bin/env python3
"""smoke_count.py against a stub /v1/chat/completions: the request it sends and its pass/fail rule."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Handler(BaseHTTPRequestHandler):
    content = ""
    last_body: dict = {}

    def log_message(self, fmt: str, *args) -> None:  # noqa: ARG002
        return

    def do_POST(self) -> None:  # noqa: N802
        _Handler.last_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        raw = json.dumps({"choices": [{"message": {"content": _Handler.content}, "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class SmokeCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        host, port = cls.httpd.server_address
        cls.url = f"http://{host}:{port}/v1/chat/completions"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()

    def run_smoke(self, content: str) -> subprocess.CompletedProcess[str]:
        _Handler.content = content
        return subprocess.run([sys.executable, str(ROOT / "smoke_count.py"), "--url", self.url],
                              capture_output=True, text=True, check=False)

    def test_full_count_passes_and_request_is_thinking_off(self) -> None:
        proc = self.run_smoke(", ".join(str(i) for i in range(1, 201)))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = _Handler.last_body
        self.assertEqual(body["model"], "deepseek-ai/DeepSeek-V4.1-Flash")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["chat_template_kwargs"], {"thinking": False, "reasoning_effort": "low"})

    def test_a_gap_fails(self) -> None:
        nums = [i for i in range(1, 201) if i != 120]
        proc = self.run_smoke(", ".join(map(str, nums)))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("result=fail", proc.stderr)

    def test_a_late_start_fails(self) -> None:
        proc = self.run_smoke(", ".join(str(i) for i in range(2, 202)))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("result=fail", proc.stderr)

    def test_a_jump_after_one_fails(self) -> None:
        # 1, then a gapless run of 200 from 5: the longest run is long enough, but the count skipped 2 to 4.
        proc = self.run_smoke(", ".join(map(str, [1] + list(range(5, 205)))))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("result=fail", proc.stderr)


if __name__ == "__main__":
    unittest.main()
