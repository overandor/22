# Local Voice + Content Agent

A local, keyboard-controlled voice agent that combines:
- offline speech-to-text via `faster-whisper`
- structured local planning via Ollama chat JSON output
- local text-to-speech via Piper
- utility actions for notes, idea capture, transcript archival, and basic video clip workflows

## Features

- Push-to-record style loop (`Enter` to start, `Enter` to stop)
- Automatic transcript archival to `agent_data/transcripts/`
- Structured JSON tool routing with a strict schema
- Local tools:
  - save note
  - append content/video idea
  - queue publish drafts
  - summarize latest transcript
  - cut latest recording clip (ffmpeg)
  - auto-cut latest recording (auto-editor)

## Repository layout

- `local_voice_agent.py` → main implementation
- `agent_data/` → notes, ideas, queue, transcripts (runtime generated)
- `recordings/` → source media for clip tooling (runtime generated)
- `clips/` → generated clips (runtime generated)

## Onboarding (quick start)

1) Install Python dependencies (Python 3.10+ recommended).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) Install and run local model services.

- Ollama must be running at `http://localhost:11434`
- Pull model (default config):

```bash
ollama pull qwen3:8b
```

3) Install Piper and download a voice model.

- Ensure `piper` is in `PATH`
- Place voice model at:

```text
voices/en_US-amy-medium.onnx
```

4) Optional video tools.

- Install `ffmpeg` for clip cutting
- Install `auto-editor` for silence-aware cuts

5) Run.

```bash
python local_voice_agent.py
```

Optional: if you want to run without Piper configured yet, use:

```bash
python local_voice_agent.py --skip-tts-check
```

## Configuration knobs

Edit constants in `local_voice_agent.py`:
- model endpoints: `OLLAMA_CHAT_URL`, `OLLAMA_TAGS_URL`, `OLLAMA_MODEL`
- STT model/runtime: `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`
- TTS executable/voice: `PIPER_EXE`, `PIPER_VOICE`
- audio settings: `SAMPLE_RATE`, `CHANNELS`, `DTYPE`, `BLOCKSIZE`
- loop hygiene: `MAIN_LOOP_SLEEP_MS`, `GC_EVERY_N_CYCLES`

Environment variable overrides are also supported for:
- `OLLAMA_CHAT_URL`, `OLLAMA_TAGS_URL`, `OLLAMA_MODEL`
- `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`
- `PIPER_EXE`, `PIPER_VOICE`

## USD appraisal (full buyout, code + integration baseline)

Scope assumption: this repository currently includes one production-ready prototype script plus onboarding documentation, with no proprietary training data, no SLA, and no transfer of brand/channel distribution.

Indicative buyout band (USD):
- Asset-only code transfer: **$8,000–$20,000**
- Code + setup support + hardening sprint (2–4 weeks): **$20,000–$55,000**
- Expanded transfer with custom tooling adapters, tests, and deployment runbook: **$55,000–$120,000**

Primary value drivers:
- reliability hardening and observability
- portability across machines/OS/audio devices
- extensible tool graph and safety constraints
- acceptance tests + benchmark baselines
- post-transfer maintenance obligations

## Notes

- This project is local-first and does not require cloud APIs.
- The script performs startup environment checks and fails fast if critical dependencies are missing.
