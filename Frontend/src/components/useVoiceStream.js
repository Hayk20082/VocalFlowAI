// Frontend/src/components/useVoiceStream.js

import { useRef, useState } from "react";


const WS_URL = "wss://vocalflowai-production.up.railway.app/ws/audio";
const SAMPLE_RATE = 16000;
const FRAME_SAMPLES = 320; // 20ms при 16kHz

export function useVoiceStream() {
  const [status, setStatus] = useState("idle"); // idle, recording
  const [previewUrl, setPreviewUrl] = useState(null);

  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);

  const accumulatedInput = useRef([]); // для отправки на сервер
  const enhancedChunks = useRef([]); // собираем улучшенный звук от сервера

  const start = async () => {
    try {
      setStatus("recording");
      setPreviewUrl(null);
      accumulatedInput.current = [];
      enhancedChunks.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;

      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      // Глушим сырой звук полностью
      const gain = audioCtx.createGain();
      gain.gain.value = 0;
      processor.connect(gain);
      gain.connect(audioCtx.destination);

      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const ws = wsRef.current;

        // Накапливаем для отправки по 320 сэмплов
        for (let i = 0; i < input.length; i++) {
          accumulatedInput.current.push(input[i]);
        }

        while (accumulatedInput.current.length >= FRAME_SAMPLES) {
          const frame = accumulatedInput.current.splice(0, FRAME_SAMPLES);
          const int16 = new Int16Array(frame.map(sample => {
            const s = Math.max(-1, Math.min(1, sample));
            return s < 0 ? s * 0x8000 : s * 0x7FFF;
          }));
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(int16.buffer);
          }
        }
      };

      source.connect(processor);

      // WebSocket
      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => console.log("WS connected");
      ws.onmessage = (msg) => {
        if (msg.data instanceof ArrayBuffer && msg.data.byteLength > 0) {
          enhancedChunks.current.push(new Int16Array(msg.data));
        }
      };
      ws.onclose = () => console.log("WS closed");
      ws.onerror = (e) => console.error("WS error", e);

      wsRef.current = ws;
    } catch (err) {
      console.error(err);
      setStatus("idle");
    }
  };

  const stop = () => {
    // Очистка
    if (wsRef.current) wsRef.current.close();
    if (processorRef.current) processorRef.current.disconnect();
    if (sourceRef.current) sourceRef.current.disconnect();
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    if (audioCtxRef.current) audioCtxRef.current.close();

    wsRef.current = null;
    processorRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;
    audioCtxRef.current = null;

    setStatus("idle");

    // Автоматически создаём preview
    makePreview();
  };

  const makePreview = () => {
    if (enhancedChunks.current.length === 0) {
      setPreviewUrl(null);
      return;
    }

    const totalSamples = enhancedChunks.current.reduce((sum, chunk) => sum + chunk.length, 0);
    const combined = new Int16Array(totalSamples);
    let offset = 0;
    for (const chunk of enhancedChunks.current) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    // Простой WAV
    const buffer = new ArrayBuffer(44 + combined.length * 2);
    const view = new DataView(buffer);
    const writeString = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + combined.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, SAMPLE_RATE, true);
    view.setUint32(28, SAMPLE_RATE * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, combined.length * 2, true);

    const pcmOffset = 44;
    for (let i = 0; i < combined.length; i++) {
      view.setInt16(pcmOffset + i * 2, combined[i], true);
    }

    const blob = new Blob([buffer], { type: "audio/wav" });
    setPreviewUrl(URL.createObjectURL(blob));
  };

  return { status, previewUrl, start, stop, makePreview };
}
