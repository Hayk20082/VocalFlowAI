"""Configuration settings for the AI Voice Confidence Backend."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Audio settings
    sample_rate: int = 16000  # Hz, default 16 kHz
    frame_size_ms: int = 20  # milliseconds per frame
    channels: int = 1  # mono
    bit_depth: int = 16  # 16-bit PCM
    
    # DSP pipeline settings
    enable_rnnoise: bool = True  # Enable RNNoise-based noise reduction (Phase 3)
    enable_gain_normalization: bool = True
    enable_noise_reduction: bool = True  # Basic noise reduction (used if RNNoise disabled)
    enable_confidence_boost: bool = True  # Enable voice confidence booster (Phase 3)
    enable_pitch_adjustment: bool = False  # Phase 2+
    
    # ML enhancement settings (Phase 3)
    onnx_model_path: Optional[str] = "app/models/denoiser_model.onnx"
    enable_ml_enhancement: bool = True # Enable ML-based voice enhancement
    # onnx_model_path: Optional[str] = None  # Path to ONNX model file

    ml_frame_size: int = 320  # Frame size for ML model (samples, default: 20ms at 16kHz)
    
    # Confidence scoring settings
    confidence_window_seconds: float = 0.5  # Window for feature aggregation
    confidence_update_interval_ms: int = 100  # How often to send updates
    confidence_smoothing_alpha: float = 0.7  # Smoothing factor (0.0-1.0, higher = less smoothing)
    confidence_phase: int = 3  # Confidence scoring phase (1, 2, or 3)
    
    # Performance settings
    max_concurrent_streams: int = 10
    processing_timeout_ms: int = 20  # Target per-frame processing time

    # Memory management settings
    max_buffer_frames_per_stream: int = 500  # Maximum frames to buffer per stream
    max_memory_mb: int = 512  # Maximum memory usage in MB
    cleanup_interval_seconds: int = 300  # How often to run cleanup (5 minutes)
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

