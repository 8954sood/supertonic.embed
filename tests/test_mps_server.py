import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serve_mps import Handler, ThreadingHTTPServer, JOBS


class LocalServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def test_page_is_served_and_foreign_host_is_rejected(self):
        with urllib.request.urlopen(self.base) as response:
            self.assertIn("Mac MPS".encode(), response.read())
        request = urllib.request.Request(self.base, headers={"Host": "foreign.example"})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_foreign_or_simple_post_cannot_start_work(self):
        for headers in ({}, {"Origin": "http://foreign.example", "X-Supertonic-Request": "1"}):
            request = urllib.request.Request(self.base + "/api/start", data=b"audio", headers=headers)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 403)
        request = urllib.request.Request(self.base + "/api/start?iterations=-1", data=b"audio",
            headers={"Origin": self.base, "X-Supertonic-Request": "1"})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 400)

    def test_json_download_returns_actual_file_bytes_and_disallows_other_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            payload = b'{"style_ttl":{"dims":[1,50,256]}}'
            (directory / "best.json").write_bytes(payload)
            JOBS["test"] = {"directory": directory}
            try:
                with urllib.request.urlopen(self.base + "/files/test/best.json") as response:
                    self.assertEqual(response.headers["Content-Type"], "application/json")
                    self.assertIn('attachment; filename="best.json"', response.headers["Content-Disposition"])
                    self.assertEqual(response.read(), payload)
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(self.base + "/files/test/reference.audio")
                self.assertEqual(error.exception.code, 404)
            finally:
                JOBS.pop("test")

    def test_custom_threshold_reaches_extractor_command_and_invalid_values_are_rejected(self):
        headers = {"Origin": self.base, "X-Supertonic-Request": "1"}
        for value in ("-1", "nan", "inf"):
            request = urllib.request.Request(self.base + f"/api/start?threshold={value}", data=b"audio", headers=headers)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 400)
        with tempfile.TemporaryDirectory() as temp, patch('serve_mps.OUTPUTS', Path(temp)), \
                patch('serve_mps.subprocess.Popen', return_value=Mock(poll=lambda: 0)) as launch:
            request = urllib.request.Request(self.base + "/api/start?threshold=0.075", data=b"audio", headers=headers)
            with urllib.request.urlopen(request) as response:
                identifier = json.load(response)["id"]
            try:
                command = launch.call_args.args[0]
                self.assertEqual(command[command.index("--threshold") + 1], "0.075")
            finally:
                JOBS.pop(identifier)


if __name__ == "__main__":
    unittest.main()
