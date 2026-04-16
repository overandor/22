# Local Voice + Content Agent

A local, keyboard-controlled voice agent that combines:
- offline speech-to-text via `faster-whisper`
- structured local planning via Ollama chat JSON output
- local text-to-speech via Piper
- utility actions for notes, idea capture, transcript archival, and basic video clip workflows
- optional Streamlit UI for operator-friendly workflows and demos

## Features

- Push-to-record style loop (`Enter` to start, `Enter` to stop)
- Web UI (`ui_app.py`) for non-technical users
- Automatic transcript archival to `agent_data/transcripts/`
- Structured JSON tool routing with a strict schema
- Automatic local hardware selection for Whisper runtime (`cuda`/`mps`/`cpu`)
- JSONL audit log for traceable operations (`agent_data/logs/events.jsonl`)
- Local tools:
  - save note
  - append content/video idea
  - queue publish drafts
  - summarize latest transcript
  - cut latest recording clip (ffmpeg)
  - auto-cut latest recording (auto-editor)

## Repository layout

- `local_voice_agent.py` → main implementation
- `ui_app.py` → Streamlit UI for local operations
- `agent_data/` → notes, ideas, queue, transcripts (runtime generated)
- `recordings/` → source media for clip tooling (runtime generated)
- `clips/` → generated clips (runtime generated)

## Onboarding (quick start)

1) Install Python dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy requests sounddevice soundfile faster-whisper streamlit
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

6) Optional UI mode.

```bash
streamlit run ui_app.py
```

7) Health check mode (CI / deployment validation).

```bash
python local_voice_agent.py --mode healthcheck
```

## Configuration knobs

Edit constants in `local_voice_agent.py`:
- model endpoints: `OLLAMA_CHAT_URL`, `OLLAMA_TAGS_URL`, `OLLAMA_MODEL`
- STT model/runtime: `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`
- TTS executable/voice: `PIPER_EXE`, `PIPER_VOICE`
- audio settings: `SAMPLE_RATE`, `CHANNELS`, `DTYPE`, `BLOCKSIZE`
- loop hygiene: `MAIN_LOOP_SLEEP_MS`, `GC_EVERY_N_CYCLES`

Hardware behavior:
- `WHISPER_DEVICE=auto` (default) will choose the best available runtime (`cuda` → `mps` → `cpu`).
- `WHISPER_COMPUTE_TYPE=auto` (default) uses `float16` on GPU and `int8` on CPU.
- You can force settings with environment variables, e.g.:

```bash
WHISPER_DEVICE=cpu WHISPER_COMPUTE_TYPE=int8 python local_voice_agent.py
```

## Sell-ready packaging checklist

Use this list to prepare a client handoff or paid deployment:

1. **Branding**
   - Rename app title and voice persona.
   - Replace default Piper voice with client-approved voice assets/licensing.
2. **Operational hardening**
   - Add health checks (Ollama, Piper, model files).
   - Add structured logs and a rotating log policy.
   - Add smoke tests for tool actions.
3. **Deployment**
   - Pin Python + dependency versions.
   - Package with a startup script/service wrapper (systemd, Docker, or platform equivalent).
   - Include backup/restore instructions for `agent_data/`.
   - Gate rollouts with `--mode healthcheck` in CI/CD.
4. **Security & privacy**
   - Keep all data local by default.
   - Add local role/account controls if used by multiple operators.
5. **Commercial docs**
   - Include statement of work (SOW), support SLA, and maintenance terms.
   - Provide admin runbook and end-user quickstart.

## Documentation bundle recommendation

For production sales, include:
- `docs/architecture.md` (runtime components and data flow)
- `docs/operations.md` (monitoring, backup, incident steps)
- `docs/security.md` (local data handling policy)
- `docs/handoff.md` (installation + acceptance checklist)

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
