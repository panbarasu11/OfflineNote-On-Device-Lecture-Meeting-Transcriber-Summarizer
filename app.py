"""OfflineNote: on-device lecture/meeting transcriber + summarizer (Streamlit)."""
import json, tempfile, urllib.request
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/generate"   # local LLM server (Ollama)
LLM_MODEL = "llama3.2:3b"                              # any small local model

def available_backend() -> str:
    """Report which ONNX Runtime provider will be used (QNN = Snapdragon NPU)."""
    try:
        import onnxruntime as ort
        if "QNNExecutionProvider" in ort.get_available_providers():
            return "Snapdragon NPU (QNN)"
    except Exception:
        pass
    return "CPU"

@st.cache_resource
def load_whisper(size="base"):
    # CPU-friendly Whisper for the prototype. On Snapdragon, swap in the Qualcomm AI Hub
    # Whisper ONNX export and run it with ONNX Runtime's QNNExecutionProvider.
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")

def transcribe(path: str) -> str:
    segments, _ = load_whisper().transcribe(path)
    return "\n".join(f"[{s.start:6.1f}s] {s.text.strip()}" for s in segments)

def summarize(transcript: str) -> str:
    prompt = ("Summarize this transcript as: 1) Short summary 2) Key points "
              "3) Action items.\n\n" + transcript[:12000])
    req = urllib.request.Request(OLLAMA_URL,
        data=json.dumps({"model": LLM_MODEL, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]

st.title("OfflineNote")
st.caption(f"Runs fully on your device. Inference backend: {available_backend()}")
audio = st.file_uploader("Upload lecture/meeting audio", type=["wav", "mp3", "m4a"])
if audio and st.button("Transcribe & summarize"):
    with tempfile.NamedTemporaryFile(suffix=audio.name[-4:], delete=False) as f:
        f.write(audio.read()); path = f.name
    with st.spinner("Transcribing..."):
        text = transcribe(path)
    st.subheader("Transcript"); st.text_area("", text, height=250)
    with st.spinner("Summarizing with local LLM..."):
        try:
            summary = summarize(text)
        except Exception as e:
            summary = f"Summarizer unavailable (start Ollama): {e}"
    st.subheader("Summary"); st.markdown(summary)
    st.download_button("Download notes (.md)", f"# Summary\n{summary}\n\n# Transcript\n{text}", "notes.md")
