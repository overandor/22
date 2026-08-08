import json
import tempfile
from pathlib import Path

import streamlit as st

from local_voice_agent import (
    EVENTS_LOG_FILE,
    LocalTranscriber,
    append_video_idea,
    environment_report,
    get_latest_transcript,
    get_system_profile,
    load_json_file,
    process_transcript,
    queue_publish,
    save_note,
    summarize_latest_transcript_local_stub,
    QUEUE_FILE,
)

st.set_page_config(page_title="Local Voice + Content Agent", layout="wide")
st.title("Local Voice + Content Agent UI")
st.caption("Runs fully local and automatically selects available hardware for transcription.")

profile = get_system_profile()
with st.expander("Hardware & Runtime", expanded=True):
    st.json(profile)
with st.expander("Environment Health", expanded=True):
    health = environment_report()
    if health["ok"]:
        st.success("Environment checks passed.")
    else:
        st.error("Environment checks failed.")
        st.write(health["problems"])

@st.cache_resource
def get_transcriber() -> LocalTranscriber:
    return LocalTranscriber()

with st.expander("Quick Actions", expanded=True):
    note_text = st.text_input("Save a note")
    if st.button("Save note", use_container_width=True):
        st.success(save_note(note_text))

    idea_text = st.text_input("Add content/video idea")
    if st.button("Add idea", use_container_width=True):
        st.success(append_video_idea(idea_text))

    col1, col2 = st.columns(2)
    with col1:
        q_title = st.text_input("Queue title")
    with col2:
        q_desc = st.text_input("Queue description")
    if st.button("Queue publish draft", use_container_width=True):
        st.success(queue_publish(q_title, q_desc))

st.subheader("Audio-to-Action")
uploaded = st.file_uploader("Upload WAV/MP3/M4A audio", type=["wav", "mp3", "m4a"])
if uploaded is not None:
    st.audio(uploaded)
    if st.button("Transcribe and run local planner", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        with st.spinner("Transcribing and running plan..."):
            transcriber = get_transcriber()
            transcript = transcriber.transcribe(tmp_path)
            result = process_transcript(transcript, speak_reply=False)

        st.markdown("### Transcript")
        st.write(result["transcript"])
        st.markdown("### Tool Plan")
        st.code(json.dumps(result["plan"], indent=2, ensure_ascii=False), language="json")
        st.markdown("### Tool Result")
        st.write(result["tool_result"])
        st.markdown("### Assistant Reply")
        st.success(result["reply"])
        st.caption(f"Pipeline latency: {result['latency_sec']}s")

st.subheader("Operations")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Preview latest transcript"):
        st.info(summarize_latest_transcript_local_stub())
with col_b:
    if st.button("Show queued drafts"):
        st.code(json.dumps(load_json_file(QUEUE_FILE, []), indent=2, ensure_ascii=False), language="json")

latest = get_latest_transcript()
if latest:
    st.caption(f"Latest transcript file: {latest}")

st.subheader("Recent Audit Events")
if EVENTS_LOG_FILE.exists():
    lines = EVENTS_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()[-15:]
    st.code("\n".join(lines), language="json")
else:
    st.caption("No events logged yet.")
