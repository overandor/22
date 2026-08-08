import unittest
import sys
import types
from unittest.mock import patch

if "sounddevice" not in sys.modules:
    sd = types.ModuleType("sounddevice")
    sd.InputStream = object
    sd.play = lambda *args, **kwargs: None
    sd.wait = lambda *args, **kwargs: None
    sys.modules["sounddevice"] = sd

if "soundfile" not in sys.modules:
    sf = types.ModuleType("soundfile")
    sf.read = lambda *args, **kwargs: ([], 16000)
    sys.modules["soundfile"] = sf

if "faster_whisper" not in sys.modules:
    fw = types.ModuleType("faster_whisper")

    class DummyWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, *args, **kwargs):
            return [], {}

    fw.WhisperModel = DummyWhisperModel
    sys.modules["faster_whisper"] = fw

import local_voice_agent as agent


class AgentCoreTests(unittest.TestCase):
    def test_detect_whisper_runtime_respects_explicit_env(self):
        with patch.object(agent, "WHISPER_DEVICE", "cpu"), patch.object(
            agent, "WHISPER_COMPUTE_TYPE", "int8"
        ):
            runtime = agent.detect_whisper_runtime()
            self.assertEqual(runtime["device"], "cpu")
            self.assertEqual(runtime["compute_type"], "int8")

    def test_load_json_file_returns_default_on_bad_json(self):
        path = agent.DATA_DIR / "tmp_bad_json_test.json"
        path.write_text("{this-is-bad-json", encoding="utf-8")
        try:
            val = agent.load_json_file(path, default={"ok": True})
            self.assertEqual(val, {"ok": True})
        finally:
            path.unlink(missing_ok=True)

    def test_execute_tool_bad_numeric_input_falls_back(self):
        with patch.object(agent, "cut_latest_video", return_value="ok") as mocked:
            out = agent.execute_tool(
                {
                    "tool": "cut_latest_video",
                    "tool_input": {"start_sec": "nanx", "duration_sec": None},
                }
            )
            self.assertEqual(out, "ok")
            mocked.assert_called_once_with(start_sec=0.0, duration_sec=30.0)


if __name__ == "__main__":
    unittest.main()
