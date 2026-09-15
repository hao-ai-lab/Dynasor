"""Real SDK/HTTP transport coverage, without a model or external API call."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import DefaultHttpxClient, OpenAI

from dynasor.core.cot import openai_chat_completion_stream


@pytest.mark.parametrize("termination", ["natural_stop", "consumer_close"])
def test_real_sse_responses_close_without_consuming_probe(termination):
    probe_disconnected = threading.Event()
    main_disconnected = threading.Event()
    responses = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            is_probe = request["max_tokens"] == 20
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Test-Stream", "probe" if is_probe else "main")
            self.end_headers()
            self.wfile.flush()

            if not is_probe:
                data = {
                    "id": "completion-test", "object": "text_completion",
                    "created": 1, "model": "test",
                    "choices": [{
                        "index": 0, "text": "first ", "logprobs": None,
                        "finish_reason": "stop" if termination == "natural_stop" else None,
                    }],
                }
                payload = ("data: " + json.dumps(data) + "\n\n").encode()
                if termination == "natural_stop":
                    payload += b"data: [DONE]\n\n"
                self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                if termination == "natural_stop":
                    self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                if termination == "natural_stop":
                    return

            # Leave the body unfinished. A retained, unread probe cannot finish
            # by iteration or garbage collection; only transport close ends it.
            self.connection.settimeout(5.0)
            try:
                disconnected = self.connection.recv(1) == b""
            except ConnectionResetError:
                disconnected = True
            except socket.timeout:
                disconnected = False
            if disconnected:
                (probe_disconnected if is_probe else main_disconnected).set()
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = DefaultHttpxClient(
        trust_env=False, timeout=2.0,
        event_hooks={"response": [responses.append]},
    )
    client = OpenAI(
        api_key="local-test", base_url=f"http://127.0.0.1:{server.server_port}/v1",
        http_client=transport, max_retries=0,
    )
    iterator = openai_chat_completion_stream(
        client, "test", "prompt", max_tokens=32, dynasor_saving_effort=(3, 32),
    )
    try:
        assert next(iterator) == "first "
        assert len(responses) == 2
        main, probe = responses
        assert main.headers["X-Test-Stream"] == "main"
        assert probe.headers["X-Test-Stream"] == "probe"
        assert not probe.is_closed
        assert not probe.is_stream_consumed
        if termination == "natural_stop":
            assert list(iterator) == []
        else:
            iterator.close()
        # Assertions precede client.close(): client teardown must not hide a
        # leaked SDK Stream. The peer independently observes the socket close.
        assert main.is_closed
        assert probe.is_closed
        assert not probe.is_stream_consumed
        assert probe_disconnected.wait(timeout=2.0)
        if termination == "consumer_close":
            assert main_disconnected.wait(timeout=2.0)
    finally:
        iterator.close()
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
