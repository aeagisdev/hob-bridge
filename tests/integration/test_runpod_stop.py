"""Integration: the real runpod SDK's per-job stop signal reaches A1111 as /interrupt.

Drives runpod.serverless.modules.rp_scale.JobScaler.run() through the SDK's own
job path (rp_job.handle_job -> run_job -> worker handler); only the platform
long-polls (job take, stop channel) are injected, which the SDK allows while
RUNPOD_WEBHOOK_GET_JOB is unset. A fake A1111 stands in for the real one.

    pip install -r worker/requirements.txt pytest
    pytest tests/integration/test_runpod_stop.py

Re-run when bumping the runpod pin in worker/requirements.txt.
"""
import asyncio
import http.server
import json
import os
import sys
import threading
import time

import pytest

pytest.importorskip("runpod")
if os.environ.get("RUNPOD_WEBHOOK_GET_JOB"):
    pytest.skip("needs the SDK's local-test mode (RUNPOD_WEBHOOK_GET_JOB unset)", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "worker"))
import handler as worker  # noqa: E402
from runpod.serverless.modules.rp_scale import JobScaler  # noqa: E402


def _fake_a1111():
    interrupted, hits = threading.Event(), []

    class A1111(http.server.BaseHTTPRequestHandler):
        def _reply(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            path = self.path.split("?")[0]
            hits.append(path)
            if path == "/sdapi/v1/interrupt":
                interrupted.set()
            elif path == "/sdapi/v1/txt2img":
                interrupted.wait(15)  # a long generation, until interrupted
            body = json.dumps({"images": [], "info": "{}", "progress": 0.3}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = do_POST = _reply

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), A1111)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, interrupted, hits


def test_stop_signal_interrupts_a1111(monkeypatch):
    server, interrupted, hits = _fake_a1111()
    monkeypatch.setattr(worker, "A1111_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(worker, "PROGRESS_INTERVAL", 0.2)
    state = {"job_given": False, "stop_sent": False}

    async def jobs_fetcher(session, num_jobs):
        if not state["job_given"]:
            state["job_given"] = True
            return [{"id": "job-1", "input": {"type": "txt2img", "params": {}}}]
        await asyncio.sleep(0.3)
        return []

    async def stop_fetcher(session):
        await asyncio.sleep(1.0)  # the job is generating by now
        if not state["stop_sent"]:
            state["stop_sent"] = True
            return ["job-1"]  # what RunPod sends on /cancel or a timeout
        return []

    async def main():
        scaler = JobScaler({"handler": worker.handler, "jobs_fetcher": jobs_fetcher,
                            "stop_signals_fetcher": stop_fetcher})
        run = asyncio.create_task(scaler.run())
        started = time.time()
        while not interrupted.is_set() and time.time() - started < 10:
            await asyncio.sleep(0.1)
        scaler.kill_worker()
        await asyncio.wait_for(run, 20)

    try:
        asyncio.run(main())
    finally:
        server.shutdown()
    assert "/sdapi/v1/txt2img" in hits
    assert interrupted.is_set(), "stop signal did not reach A1111"
