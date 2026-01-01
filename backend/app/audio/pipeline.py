import time
import numpy as np
from pathlib import Path  # ← Добавлен импорт
from app.audio.models import AudioFrame
from app.core.config import settings
from app.core.logging import logger
from app.audio.dsp.gain import normalize_gain  # ← Добавлен импорт normalize_gain
import onnxruntime as ort

# Глобальные переменные
ort_session = None
stream_states = {}  # stream_id -> (states, atten_lim_db)

from pathlib import Path
def load_speech_denoiser():
    global ort_session
    if ort_session is None:
        # Use the configured model path relative to the app directory
        model_path = Path(__file__).parent.parent.parent / settings.onnx_model_path
        logger.info(f"Loading SpeechDenoiser model from {model_path}")
        try:
            ort_session = ort.InferenceSession(str(model_path))
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            ort_session = None
    return ort_session

def speech_denoiser_enhance(frame: AudioFrame) -> AudioFrame:
    session = load_speech_denoiser()
    if session is None:
        return frame

    pcm_float = frame.pcm_data.astype(np.float32) / 32768.0

    # Handle frame size mismatch: model expects 480 samples, pad if necessary
    expected_size = 480
    if len(pcm_float) < expected_size:
        # Pad with zeros if frame is shorter than expected
        padding = np.zeros(expected_size - len(pcm_float), dtype=np.float32)
        input_frame = np.concatenate([pcm_float, padding])
        logger.debug(f"Padded frame from {len(pcm_float)} to {expected_size} samples")
    elif len(pcm_float) > expected_size:
        # Truncate if frame is longer than expected
        input_frame = pcm_float[:expected_size]
        logger.debug(f"Truncated frame from {len(pcm_float)} to {expected_size} samples")
    else:
        input_frame = pcm_float

    stream_id = frame.stream_id

    # Инициализация состояний для нового стрима
    if stream_id not in stream_states:
        initial_states = np.zeros(45304, dtype=np.float32)  # Model expects rank 1 with 45304 elements
        initial_atten = np.array(-20.0, dtype=np.float32)    # Model expects scalar
        stream_states[stream_id] = (initial_states, initial_atten)

    states, atten_lim_db = stream_states[stream_id]

    inputs = {
        "input_frame": input_frame,
        "states": states,
        "atten_lim_db": atten_lim_db
    }

    try:
        outputs = session.run(None, inputs)
        enhanced = outputs[0]
        new_states = outputs[1]
        new_atten_lim_db = outputs[2]

        stream_states[stream_id] = (new_states, new_atten_lim_db)

        enhanced_int16 = np.clip(enhanced.flatten() * 32767, -32768, 32767).astype(np.int16)

        # If we padded the input, truncate output back to original size
        original_size = len(frame.pcm_data)
        if len(enhanced_int16) > original_size:
            enhanced_int16 = enhanced_int16[:original_size]

        return AudioFrame(
            pcm_data=enhanced_int16,
            sample_rate=frame.sample_rate,
            timestamp=frame.timestamp,
            stream_id=frame.stream_id
        )
    except Exception as e:
        logger.error(f"ONNX inference error: {e}")
        return frame

def process_audio_frame(frame: AudioFrame, confidence: float | None = None) -> AudioFrame:
    processed = frame

    if settings.enable_ml_enhancement:
        processed = speech_denoiser_enhance(processed)

    processed = normalize_gain(processed)  # Теперь импорт есть — работает!

    return processed

# Очистка состояний
def cleanup_stream_states(stream_id: str):
    if stream_id in stream_states:
        del stream_states[stream_id]
        logger.debug(f"Cleaned ONNX states for stream {stream_id}")