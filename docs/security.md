# Security and Privacy

## Default posture
- Local-first processing: transcription, planning, and storage run on local infrastructure.
- No cloud API keys are required for default operation.

## Data handling
- Store only required operational artifacts under `agent_data/`.
- Define retention windows for transcripts and queue drafts.
- Restrict filesystem access permissions to operational users.

## Hardening recommendations
- Run with least-privilege OS user.
- Segment machine/network used for production operation.
- Add audit logging for operator actions if required by compliance.
