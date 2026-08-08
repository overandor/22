import gc
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import requests

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

# =========================
# Configuration
# =========================

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "qwen3:8b"

WHISPER_MODEL = "base"
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # "auto", "cpu", "cuda", or "mps"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")  # "auto", "int8", "float16", etc.

PIPER_EXE = "piper"
PIPER_VOICE = "voices/en_US-amy-medium.onnx"

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
BLOCKSIZE = 1024

MAIN_LOOP_SLEEP_MS = 100
GC_EVERY_N_CYCLES = 50

DATA_DIR = Path("agent_data")
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
EVENTS_LOG_FILE = LOGS_DIR / "events.jsonl"

NOTES_FILE = DATA_DIR / "notes.txt"
IDEAS_FILE = DATA_DIR / "video_ideas.txt"
QUEUE_FILE = DATA_DIR / "publish_queue.json"

TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)

CLIPS_DIR = Path("clips")
CLIPS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class AgentConfig:
    ollama_chat_url: str
    ollama_tags_url: str
    ollama_model: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    piper_exe: str
    piper_voice: str


def get_config() -> AgentConfig:
    return AgentConfig(
        ollama_chat_url=OLLAMA_CHAT_URL,
        ollama_tags_url=OLLAMA_TAGS_URL,
        ollama_model=OLLAMA_MODEL,
        whisper_model=WHISPER_MODEL,
        whisper_device=WHISPER_DEVICE,
        whisper_compute_type=WHISPER_COMPUTE_TYPE,
        piper_exe=PIPER_EXE,
        piper_voice=PIPER_VOICE,
    )


# =========================
# Utility helpers
# =========================


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_append_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def load_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def save_json_file(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_latest_file(directory: Path, suffixes: Optional[List[str]] = None) -> Optional[Path]:
    if not directory.exists():
        return None

    files = [p for p in directory.iterdir() if p.is_file()]
    if suffixes:
        suffixes_lower = {s.lower() for s in suffixes}
        files = [p for p in files if p.suffix.lower() in suffixes_lower]

    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime)


def short_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def log_event(event_type: str, payload: Dict[str, Any]) -> None:
    event = {
        "ts": now_ts(),
        "event_type": event_type,
        "payload": payload,
    }
    safe_append_line(EVENTS_LOG_FILE, json.dumps(event, ensure_ascii=False))


def require_sounddevice():
    try:
        import sounddevice as sd  # type: ignore

        return sd
    except Exception as e:
        raise RuntimeError(f"sounddevice dependency unavailable: {e}") from e


def require_soundfile():
    try:
        import soundfile as sf  # type: ignore

        return sf
    except Exception as e:
        raise RuntimeError(f"soundfile dependency unavailable: {e}") from e


def require_whisper_model():
    try:
        from faster_whisper import WhisperModel  # type: ignore

        return WhisperModel
    except Exception as e:
        raise RuntimeError(f"faster-whisper dependency unavailable: {e}") from e


def detect_whisper_runtime() -> Dict[str, str]:
    """
    Select the best available Whisper runtime based on local hardware.
    Respects explicit WHISPER_DEVICE / WHISPER_COMPUTE_TYPE values when not set to 'auto'.
    """
    configured_device = WHISPER_DEVICE.strip().lower()
    configured_compute = WHISPER_COMPUTE_TYPE.strip().lower()

    if configured_device != "auto":
        device = configured_device
    else:
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            device = "cpu"

    if configured_compute != "auto":
        compute_type = configured_compute
    elif device == "cuda":
        compute_type = "float16"
    else:
        compute_type = "int8"

    return {"device": device, "compute_type": compute_type}


def get_system_profile() -> Dict[str, Any]:
    runtime = detect_whisper_runtime()
    profile: Dict[str, Any] = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
        "whisper_device": runtime["device"],
        "whisper_compute_type": runtime["compute_type"],
    }
    return profile


# =========================
# Environment checks
# =========================


def check_environment() -> None:
    cfg = get_config()
    problems = []

    try:
        r = requests.get(cfg.ollama_tags_url, timeout=5)
        r.raise_for_status()
    except Exception as e:
        problems.append(f"Ollama not reachable at {cfg.ollama_tags_url}: {e}")

    if shutil.which(cfg.piper_exe) is None:
        problems.append(f"Piper executable not found in PATH: {cfg.piper_exe}")

    if not Path(cfg.piper_voice).exists():
        problems.append(f"Piper voice file not found: {cfg.piper_voice}")

    if problems:
        raise RuntimeError("Environment check failed:\n- " + "\n- ".join(problems))


def environment_report() -> Dict[str, Any]:
    cfg = get_config()
    problems: List[str] = []
    try:
        requests.get(cfg.ollama_tags_url, timeout=5).raise_for_status()
    except Exception as e:
        problems.append(f"Ollama: {e}")
    if shutil.which(cfg.piper_exe) is None:
        problems.append(f"Piper executable not found: {cfg.piper_exe}")
    if not Path(cfg.piper_voice).exists():
        problems.append(f"Piper voice file not found: {cfg.piper_voice}")

    return {
        "ok": len(problems) == 0,
        "problems": problems,
        "system_profile": get_system_profile(),
    }


# =========================
# Audio recorder
# =========================


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        dtype: str = DTYPE,
        blocksize: int = BLOCKSIZE,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize

    def record_until_enter(self) -> np.ndarray:
        sd = require_sounddevice()
        print("Recording... press Enter to stop.")
        frames: List[np.ndarray] = []
        stop_flag = {"stop": False}

        def wait_for_enter():
            input()
            stop_flag["stop"] = True

        listener = threading.Thread(target=wait_for_enter, daemon=True)
        listener.start()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
        ) as stream:
            while not stop_flag["stop"]:
                data, overflowed = stream.read(self.blocksize)
                if overflowed:
                    print("[audio] input overflow detected")
                frames.append(data.copy())

        if not frames:
            return np.zeros((0, self.channels), dtype=np.int16)

        return np.concatenate(frames, axis=0)

    def save_wav(self, audio: np.ndarray, path: str) -> None:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(np.dtype(audio.dtype).itemsize)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())


# =========================
# Whisper transcription
# =========================


class LocalTranscriber:
    def __init__(self):
        WhisperModel = require_whisper_model()
        cfg = get_config()
        runtime = detect_whisper_runtime()
        self.device = runtime["device"]
        self.compute_type = runtime["compute_type"]
        print(
            f"[init] loading faster-whisper model (device={self.device}, compute={self.compute_type})..."
        )
        self.model = WhisperModel(
            cfg.whisper_model,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, wav_path: str) -> str:
        segments, _info = self.model.transcribe(
            wav_path,
            vad_filter=True,
            beam_size=5,
        )
        parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(parts).strip()


# =========================
# Local content / video tools
# =========================


def save_note(text: str) -> str:
    text = text.strip()
    if not text:
        return "No note text found."
    safe_append_line(NOTES_FILE, f"[{now_ts()}] {text}")
    return f"Saved note: {text}"


def append_video_idea(text: str) -> str:
    text = text.strip()
    if not text:
        return "No video idea text found."
    safe_append_line(IDEAS_FILE, f"[{now_ts()}] {text}")
    return f"Added video idea: {text}"


def queue_publish(title: str, description: str) -> str:
    items = load_json_file(QUEUE_FILE, [])
    item = {
        "title": title.strip() or "Untitled draft",
        "description": description.strip(),
        "created_at": now_ts(),
        "status": "queued_local",
    }
    items.append(item)
    save_json_file(QUEUE_FILE, items)
    return f"Queued publish draft: {item['title']}"


def archive_transcript(text: str, prefix: str = "transcript") -> str:
    text = text.strip()
    if not text:
        return "No transcript text to archive."

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = TRANSCRIPTS_DIR / f"{prefix}_{timestamp}.txt"
    path.write_text(text, encoding="utf-8")
    return f"Archived transcript: {path}"


def get_latest_transcript() -> Optional[Path]:
    return get_latest_file(TRANSCRIPTS_DIR, [".txt"])


def summarize_latest_transcript_local_stub() -> str:
    latest = get_latest_transcript()
    if latest is None:
        return "No archived transcript found."

    text = latest.read_text(encoding="utf-8").strip()
    if not text:
        return f"Latest transcript is empty: {latest.name}"

    preview = text[:500].replace("\n", " ")
    return f"Latest transcript preview from {latest.name}: {preview}"


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def auto_editor_exists() -> bool:
    return shutil.which("auto-editor") is not None


def get_latest_recording() -> Optional[Path]:
    return get_latest_file(RECORDINGS_DIR, [".mp4", ".mov", ".mkv", ".webm"])


def cut_latest_video(start_sec: float = 0.0, duration_sec: float = 30.0) -> str:
    latest = get_latest_recording()
    if latest is None:
        return "No recording found in recordings/."

    if not ffmpeg_exists():
        return "ffmpeg is not installed or not in PATH."

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = CLIPS_DIR / f"{latest.stem}_clip_{timestamp}.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        str(latest),
        "-t",
        str(duration_sec),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(out_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"Created clip: {out_path}"
    except subprocess.CalledProcessError as e:
        return "ffmpeg clip cut failed:\n" + e.stderr.decode("utf-8", errors="ignore")


def auto_cut_latest_video() -> str:
    latest = get_latest_recording()
    if latest is None:
        return "No recording found in recordings/."

    if auto_editor_exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = CLIPS_DIR / f"{latest.stem}_auto_{timestamp}.mp4"
        cmd = [
            "auto-editor",
            str(latest),
            "--output",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return f"Auto-cut video created: {out_path}"
        except subprocess.CalledProcessError as e:
            return "auto-editor failed:\n" + e.stderr.decode("utf-8", errors="ignore")

    if ffmpeg_exists():
        return "auto-editor not found. Install it for silence-aware cutting."
    return "Neither auto-editor nor ffmpeg is available."


def tool_registry() -> Dict[str, str]:
    return {
        "none": "Do nothing.",
        "answer_only": "Reply only, no file or system action.",
        "save_note": "Save a note into agent_data/notes.txt",
        "append_video_idea": "Append a content idea into agent_data/video_ideas.txt",
        "queue_publish": "Queue a post draft into agent_data/publish_queue.json",
        "archive_transcript": "Save the current transcript into agent_data/transcripts/",
        "summarize_latest_transcript": "Read the latest archived transcript and return a short preview",
        "cut_latest_video": "Cut a simple clip from the newest file in recordings/",
        "auto_cut_latest_video": "Use auto-editor on the newest file in recordings/",
    }


def execute_tool(plan: Dict[str, Any]) -> str:
    tool = (plan.get("tool") or "answer_only").strip()
    tool_input = plan.get("tool_input") or {}

    text = str(tool_input.get("text") or "").strip()
    title = str(tool_input.get("title") or "").strip()
    description = str(tool_input.get("description") or "").strip()
    try:
        start_sec = float(tool_input.get("start_sec") or 0.0)
    except (TypeError, ValueError):
        start_sec = 0.0
    try:
        duration_sec = float(tool_input.get("duration_sec") or 30.0)
    except (TypeError, ValueError):
        duration_sec = 30.0

    if tool == "save_note":
        return save_note(text or title or description)

    if tool == "append_video_idea":
        return append_video_idea(text or title or description)

    if tool == "queue_publish":
        return queue_publish(title or text, description or text)

    if tool == "archive_transcript":
        return archive_transcript(text)

    if tool == "summarize_latest_transcript":
        return summarize_latest_transcript_local_stub()

    if tool == "cut_latest_video":
        return cut_latest_video(start_sec=start_sec, duration_sec=duration_sec)

    if tool == "auto_cut_latest_video":
        return auto_cut_latest_video()

    if tool in ("none", "answer_only"):
        return "No tool executed."

    return f"Unknown tool '{tool}'."


# =========================
# Ollama structured planning
# =========================

TOOLS = tool_registry()

SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": list(TOOLS.keys()),
        },
        "tool_input": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "start_sec": {"type": "number"},
                "duration_sec": {"type": "number"},
            },
            "required": ["text", "title", "description", "start_sec", "duration_sec"],
            "additionalProperties": False,
        },
        "assistant_reply": {"type": "string"},
    },
    "required": ["tool", "tool_input", "assistant_reply"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""
You are a fully local desktop voice agent.

You receive the user's transcribed speech.
You decide what local tool to run, if any.
You must return STRICT JSON only.

Available tools and meanings:
{json.dumps(TOOLS, indent=2, ensure_ascii=False)}

Output MUST match this JSON schema exactly:
{json.dumps(SCHEMA, indent=2, ensure_ascii=False)}

Behavior rules:
- Use save_note when user says note, remember, save this thought.
- Use append_video_idea when user is brainstorming content/video ideas.
- Use queue_publish when user wants a title/post/draft/caption saved for later publishing.
- Use archive_transcript when the user wants the speech preserved as text.
- Use summarize_latest_transcript if user asks about the last transcript.
- Use cut_latest_video when the user asks to cut the latest recording.
- Use auto_cut_latest_video when the user asks to automatically trim the newest recording.
- Use answer_only when no action is needed.
- assistant_reply should be natural, short, and spoken aloud.
- tool_input fields must always exist.
- If a field is not needed, use:
  text = ""
  title = ""
  description = ""
  start_sec = 0
  duration_sec = 30

Output JSON only.
""".strip()


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("No valid JSON found in Ollama response.")


def call_ollama_structured(user_text: str) -> Dict[str, Any]:
    cfg = get_config()
    payload = {
        "model": cfg.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0.2},
    }

    data = None
    last_error: Optional[Exception] = None
    for _attempt in range(2):
        try:
            response = requests.post(cfg.ollama_chat_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            break
        except requests.RequestException as e:
            last_error = e
            time.sleep(0.4)
    if data is None:
        raise requests.RequestException(f"Ollama request failed after retry: {last_error}")

    content = data["message"]["content"]
    parsed = extract_json_block(content)

    parsed.setdefault("tool", "answer_only")
    parsed.setdefault("tool_input", {})
    parsed["tool_input"].setdefault("text", "")
    parsed["tool_input"].setdefault("title", "")
    parsed["tool_input"].setdefault("description", "")
    parsed["tool_input"].setdefault("start_sec", 0)
    parsed["tool_input"].setdefault("duration_sec", 30)
    parsed.setdefault("assistant_reply", "Done.")

    return parsed


# =========================
# Piper TTS
# =========================


def speak_with_piper(text: str) -> None:
    cfg = get_config()
    sf = require_soundfile()
    sd = require_sounddevice()
    text = text.strip()
    if not text:
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        subprocess.run(
            [
                cfg.piper_exe,
                "--model",
                cfg.piper_voice,
                "--output_file",
                wav_path,
            ],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        audio, sr = sf.read(wav_path, dtype="float32")
        sd.play(audio, sr)
        sd.wait()

    except FileNotFoundError:
        print("[tts] Piper executable not found.")
    except subprocess.CalledProcessError as e:
        print("[tts] Piper failed.")
        print(e.stderr.decode("utf-8", errors="ignore"))
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


# =========================
# Main loop
# =========================


def process_transcript(transcript: str, speak_reply: bool = True) -> Dict[str, Any]:
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("Transcript is empty.")

    started = time.time()
    archive_msg = archive_transcript(transcript, prefix="live")
    plan = call_ollama_structured(transcript)
    tool_result = execute_tool(plan)
    reply = plan.get("assistant_reply", "").strip() or "Done."
    final_reply = f"{reply} {tool_result}".strip() if tool_result != "No tool executed." else reply

    if speak_reply:
        speak_with_piper(final_reply)

    result = {
        "transcript": transcript,
        "archive_message": archive_msg,
        "plan": plan,
        "tool_result": tool_result,
        "reply": final_reply,
        "latency_sec": round(time.time() - started, 3),
    }
    log_event(
        "transcript_processed",
        {
            "tool": plan.get("tool", "answer_only"),
            "latency_sec": result["latency_sec"],
            "transcript_hash": short_text_hash(transcript),
            "transcript_len": len(transcript),
        },
    )
    return result


def run_cycle(recorder: AudioRecorder, transcriber: LocalTranscriber) -> None:
    audio = recorder.record_until_enter()

    if len(audio) == 0:
        print("[warn] No audio captured.")
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        recorder.save_wav(audio, wav_path)

        print("[stt] Transcribing...")
        transcript = transcriber.transcribe(wav_path)
        print(f"[you] {transcript}")

        if not transcript:
            print("[warn] Empty transcript.")
            return

        print("[llm] Thinking locally...")
        result = process_transcript(transcript, speak_reply=False)
        print(f"[archive] {result['archive_message']}")
        print("[plan]")
        print(json.dumps(result["plan"], indent=2, ensure_ascii=False))
        print(f"[tool] {result['tool_result']}")
        print(f"[agent] {result['reply']}")
        speak_with_piper(result["reply"])

    except requests.RequestException as e:
        print(f"[error] Ollama request failed: {e}")
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def run_cli() -> None:
    check_environment()

    recorder = AudioRecorder()
    transcriber = LocalTranscriber()

    print("\nLocal voice/content agent ready.")
    print("Press Enter to start recording, then Enter again to stop.")
    print("Type 'q' and Enter to quit.\n")

    cycle = 0
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            print("Goodbye.")
            break

        run_cycle(recorder, transcriber)
        cycle += 1

        if cycle % GC_EVERY_N_CYCLES == 0:
            gc.collect()

        time.sleep(MAIN_LOOP_SLEEP_MS / 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Voice + Content Agent")
    parser.add_argument(
        "--mode",
        choices=["cli", "healthcheck"],
        default="cli",
        help="Run interactive CLI loop or print environment health report.",
    )
    args = parser.parse_args()

    if args.mode == "healthcheck":
        report = environment_report()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report["ok"]:
            raise SystemExit(1)
        return

    run_cli()


if __name__ == "__main__":
    main()
