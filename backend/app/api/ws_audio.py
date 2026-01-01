"""WebSocket endpoint for audio ingestion, confidence streaming, and processed audio output."""
import asyncio
import json
import uuid
from fastapi import WebSocket, WebSocketDisconnect
from app.audio.ingestion import bytes_to_audio_frame, validate_audio_data
from app.audio.buffers import buffer_manager
from app.audio.pipeline import process_audio_frame
from app.audio.ml.features import extract_features
from app.audio.ml.confidence import score_confidence
from app.audio.streaming import frame_to_bytes
from app.services.confidence_service import confidence_service
from app.services.stream_router import stream_router
from app.core.config import settings
from app.core.logging import logger


async def process_stream(stream_id: str, websocket: WebSocket) -> None:
    """
    Process audio stream: receive frames, process, compute confidence, send updates.
    
    Args:
        stream_id: Unique identifier for this stream
        websocket: WebSocket connection
    """
    buffer = await buffer_manager.get_or_create_buffer(stream_id)
    await stream_router.register_stream(stream_id)
    
    last_confidence_update = 0.0
    import time
    
    try:
        while True:
            # Receive binary audio data
            data = await websocket.receive_bytes()
            
            # Validate audio data
            if not validate_audio_data(data):
                logger.warning(f"Invalid audio data from stream {stream_id}")
                continue
            
            # Convert bytes to AudioFrame
            try:
                frame = bytes_to_audio_frame(data, stream_id)
            except Exception as e:
                logger.error(f"Error converting audio data for stream {stream_id}: {e}")
                continue
            
            # Get current confidence for confidence boost (use previous value if available)
            current_confidence = None
            if settings.enable_confidence_boost:
                # Try to get latest confidence for this stream
                current_confidence = await confidence_service.get_confidence(stream_id)
                # Update global confidence state for confidence boost module
                if current_confidence is not None:
                    from app.audio.dsp.confidence_boost import set_confidence
                    set_confidence(current_confidence)
            
            # Process frame through DSP + ML pipeline (synchronous, low-latency)
            processed_frame = process_audio_frame(frame, confidence=current_confidence)
            
            # Add processed frame to buffer (for confidence calculation on enhanced audio)
            await buffer.add_frame(processed_frame)
            
            # Get recent processed frames for feature extraction (Phase 3: use enhanced audio)
            recent_frames = buffer.get_recent_frames(settings.confidence_window_seconds)
            
            # Extract features and compute confidence (Phase 3: advanced features)
            confidence = None
            features = None
            if recent_frames:
                features = extract_features(recent_frames)
                # Phase 3: Use advanced confidence scoring with smoothing
                confidence = score_confidence(
                    features,
                    phase=settings.confidence_phase,
                    stream_id=stream_id,
                    apply_smoothing=True
                )
                
                # Update confidence service
                await confidence_service.update_confidence(stream_id, float(confidence))
            
            # Send responses to client
            current_time = time.time()
            update_interval = settings.confidence_update_interval_ms / 1000.0
            
            # Protocol: Send confidence (JSON) first, then processed audio (binary)
            # Only send confidence updates at the specified interval
            send_confidence = False
            if confidence is not None and (current_time - last_confidence_update >= update_interval):
                send_confidence = True
                last_confidence_update = current_time
            
            try:
                # Step 1: Send confidence score as JSON (if it's time for an update)
                # Phase 3: Include rich telemetry (features) in response
                if send_confidence:
                    confidence_response = {
                        "confidence": confidence,  # Already an integer from Phase 3
                        "timestamp": frame.timestamp,
                        "stream_id": stream_id
                    }
                    
                    # Phase 3: Add feature summary for debugging/analysis
                    if features:
                        confidence_response["features"] = {
                            "rms_mean": round(features.get("rms_mean", 0), 1),
                            "pitch_mean": round(features.get("pitch_mean", 0), 1),
                            "silence_ratio": round(features.get("silence_ratio", 1.0), 2),
                            "spectral_centroid": round(features.get("spectral_centroid_mean", 0), 1),
                            "zcr": round(features.get("zcr_mean", 0), 3)
                        }
                    
                    await websocket.send_json(confidence_response)
                
                # Step 2: Always send processed audio frame as binary
                # This ensures real-time audio streaming with minimal latency
                processed_audio_bytes = frame_to_bytes(processed_frame)
                await websocket.send_bytes(processed_audio_bytes)
                
            except Exception as e:
                logger.error(f"Error sending response for stream {stream_id}: {e}")
                break
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for stream {stream_id}")
    except Exception as e:
        logger.error(f"Error processing stream {stream_id}: {e}")
    finally:

        await buffer_manager.remove_buffer(stream_id)
        await stream_router.unregister_stream(stream_id)
        await confidence_service.remove_stream(stream_id)
        from app.audio.pipeline import cleanup_stream_states
        cleanup_stream_states(stream_id)
        logger.info(f"Cleaned up stream {stream_id}")


async def websocket_audio_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint handler for /ws/audio.
    
    Accepts binary PCM audio frames and sends JSON confidence updates.
    """
    await websocket.accept()
    
    # Generate unique stream ID
    stream_id = f"ws-{uuid.uuid4().hex[:8]}"
    logger.info(f"New WebSocket connection: {stream_id}")
    
    try:
        await process_stream(stream_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error for {stream_id}: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass