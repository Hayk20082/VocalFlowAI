#VocalFlowAI

Real-time AI Voice Enhancement & Confidence Scoring
VocalFlowAI is a web application that enhances your voice in real-time: removes background noise, normalizes volume, and makes your speech sound more confident and professional. Perfect for calls, streaming, podcasts, and voice recording.
✨ Key Features

Noise Suppression powered by ONNX model (SpeechDenoiser)
Volume Normalization (AGC – Automatic Gain Control)
Confidence Boost – makes voice warmer and more "radio-like"
Preview Mode – listen to the enhanced version after recording
Fully browser-based (React + Vite)
Backend built with FastAPI and WebSocket for real-time processing

🚀 Live Demo
Live version: https://trust-voice-ai.vercel.app
Backend API docs: https://vocalflowai-production.up.railway.app/docs (Swagger UI)
🛠 Tech Stack
Frontend:

React + Vite
Tailwind CSS
Web Audio API

Backend:

FastAPI (Python)
Uvicorn
ONNX Runtime (SpeechDenoiser model)
NumPy, SciPy

Deployment:

Frontend: Vercel
Backend: Railway


