# OfflineNote
On-device lecture & meeting transcriber + summarizer, designed to be optimized for Snapdragon-powered PCs.
Submission for the Snapdragon® AI Lab Build & Present Challenge.

## Features
- Upload audio -> timestamped transcript (Whisper) -> summary, key points, action items (local LLM)
- 100% offline: no audio or text leaves the device
- Backend selector shows whether the Snapdragon NPU (ONNX Runtime QNN execution provider) is available; falls back to CPU

## Run
```
pip install -r requirements.txt
ollama pull llama3.2:3b      # local summarizer (https://ollama.com)
streamlit run app.py
```

## Status (honest disclosure)
- Prototype built and tested on CPU on a non-Snapdragon laptop.
- Snapdragon NPU port (Qualcomm AI Hub Whisper + ONNX Runtime QNN EP) and benchmarks are planned; results will be added here.

## Roadmap
Snapdragon X benchmarks, speaker labels, live captions, Tamil/English support.
