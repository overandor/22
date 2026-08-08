# Operations Runbook

## Startup checks
1. Verify Ollama is reachable at `http://localhost:11434/api/tags`.
2. Verify `piper` executable exists in `PATH`.
3. Verify voice model file exists at configured path.

## Start commands
- CLI mode: `python local_voice_agent.py`
- UI mode: `streamlit run ui_app.py`
- Healthcheck mode: `python local_voice_agent.py --mode healthcheck`

## Backups
- Backup `agent_data/` daily.
- Backup `voices/` and runtime config files on changes.

## Incident triage
- **No transcription**: verify microphone access, model presence, and runtime memory.
- **No speech output**: verify Piper executable and selected ONNX voice model path.
- **Planner failure**: verify Ollama process/model availability.
- **Forensics**: inspect `agent_data/logs/events.jsonl` for recent tool/action traces.
