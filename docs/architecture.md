# Architecture

## Components
- **Capture layer**: `sounddevice` records microphone input into temporary WAV data.
- **STT layer**: `faster-whisper` transcribes locally with automatic runtime selection (`cuda`, `mps`, `cpu`).
- **Planning layer**: Ollama chat endpoint returns strict JSON tool plans.
- **Action layer**: local file-based tools save notes, ideas, queue items, transcripts, and optional video cuts.
- **TTS layer**: Piper synthesizes spoken confirmations.
- **UI layer**: Streamlit app exposes non-technical controls and status.

## Data flow
1. User audio or uploaded file enters the app.
2. Transcript is generated locally.
3. Planner selects tool + reply.
4. Tool executes against local filesystem.
5. Result is displayed (and optionally spoken).

## Storage paths
- `agent_data/notes.txt`
- `agent_data/video_ideas.txt`
- `agent_data/publish_queue.json`
- `agent_data/transcripts/*.txt`
- `recordings/*` and `clips/*`
