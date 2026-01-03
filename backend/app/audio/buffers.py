"""Audio buffering and queue management per stream."""
import asyncio
from typing import Dict, Optional
from collections import deque
import numpy as np
from app.audio.models import AudioFrame, StreamState
from app.core.config import settings
from app.core.logging import logger


class StreamBuffer:
    """Manages audio frame buffering for a single stream."""

    def __init__(self, stream_id: str, max_frames: Optional[int] = None):
        """
        Initialize buffer for a stream.

        Args:
            stream_id: Unique identifier for the stream
            max_frames: Maximum number of frames to keep in buffer (uses config if None)
        """
        if max_frames is None:
            from app.core.config import settings
            max_frames = settings.max_buffer_frames_per_stream

        self.stream_id = stream_id
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_frames)
        self.frame_history: deque = deque(maxlen=max_frames)
        self.state = StreamState(
            stream_id=stream_id,
            created_at=asyncio.get_event_loop().time(),
            last_frame_time=0.0,
            frame_count=0
        )
    
    async def add_frame(self, frame: AudioFrame) -> None:
        """Add a frame to the buffer."""
        try:
            self.queue.put_nowait(frame)
            self.frame_history.append(frame)  # deque automatically handles maxlen
            self.state.frame_count += 1
            self.state.last_frame_time = frame.timestamp
        except asyncio.QueueFull:
            logger.warning(f"Buffer full for stream {self.stream_id}, dropping oldest frame")
            try:
                # Remove oldest frame from both queue and history
                oldest_frame = self.queue.get_nowait()
                # Remove the oldest frame from history (deque handles maxlen, but we need to sync)
                if self.frame_history and self.frame_history[0] == oldest_frame:
                    self.frame_history.popleft()

                # Add new frame to both
                self.queue.put_nowait(frame)
                self.frame_history.append(frame)
                self.state.frame_count += 1
                self.state.last_frame_time = frame.timestamp
            except asyncio.QueueEmpty:
                # Queue became empty, just add the new frame
                self.queue.put_nowait(frame)
                self.frame_history.append(frame)
                self.state.frame_count += 1
                self.state.last_frame_time = frame.timestamp
    
    async def get_frame(self, timeout: Optional[float] = None) -> Optional[AudioFrame]:
        """Get the next frame from the buffer."""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def get_recent_frames(self, window_seconds: float) -> list[AudioFrame]:
        """
        Get frames from the last N seconds.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            List of frames within the time window
        """
        if not self.frame_history:
            return []
        
        current_time = self.frame_history[-1].timestamp
        cutoff_time = current_time - window_seconds
        
        return [f for f in self.frame_history if f.timestamp >= cutoff_time]


class StreamBufferManager:
    """Manages buffers for all active streams."""

    def __init__(self):
        """Initialize the buffer manager."""
        self._buffers: Dict[str, StreamBuffer] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = 0.0
    
    async def get_or_create_buffer(self, stream_id: str) -> StreamBuffer:
        """Get existing buffer or create a new one for a stream."""
        async with self._lock:
            if stream_id not in self._buffers:
                self._buffers[stream_id] = StreamBuffer(stream_id)
                logger.info(f"Created buffer for stream {stream_id}")
            return self._buffers[stream_id]
    
    async def remove_buffer(self, stream_id: str) -> None:
        """Remove buffer for a stream (on disconnect)."""
        async with self._lock:
            if stream_id in self._buffers:
                del self._buffers[stream_id]
                logger.info(f"Removed buffer for stream {stream_id}")
    
    async def get_buffer(self, stream_id: str) -> Optional[StreamBuffer]:
        """Get buffer for a stream if it exists."""
        async with self._lock:
            return self._buffers.get(stream_id)
    
    async def list_stream_ids(self) -> list[str]:
        """Get list of all active stream IDs."""
        async with self._lock:
            return list(self._buffers.keys())
    
    async def get_stream_count(self) -> int:
        """Get number of active streams."""
        async with self._lock:
            return len(self._buffers)

    async def get_memory_usage_mb(self) -> float:
        """Estimate current memory usage in MB."""
        async with self._lock:
            total_frames = 0
            for buffer in self._buffers.values():
                total_frames += len(buffer.frame_history)

            # Rough estimate: each frame is ~640 bytes (320 samples * 2 bytes/sample)
            # Plus overhead for metadata
            return (total_frames * 640 + len(self._buffers) * 1024) / (1024 * 1024)

    async def cleanup_old_buffers(self, max_age_seconds: int = 3600) -> int:
        """
        Remove buffers for streams that haven't received frames recently.

        Args:
            max_age_seconds: Maximum age of buffers to keep (default 1 hour)

        Returns:
            Number of buffers removed
        """
        from app.core.config import settings
        import time

        current_time = time.time()
        removed_count = 0

        async with self._lock:
            streams_to_remove = []
            for stream_id, buffer in self._buffers.items():
                age = current_time - buffer.state.last_frame_time
                if age > max_age_seconds:
                    streams_to_remove.append(stream_id)

            for stream_id in streams_to_remove:
                del self._buffers[stream_id]
                removed_count += 1
                logger.info(f"Cleaned up old buffer for stream {stream_id} (age: {age:.0f}s)")

        return removed_count

    async def enforce_memory_limits(self) -> bool:
        """
        Enforce memory limits by removing oldest buffers if needed.

        Returns:
            True if cleanup was performed, False otherwise
        """
        from app.core.config import settings

        memory_mb = await self.get_memory_usage_mb()
        if memory_mb > settings.max_memory_mb:
            logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {settings.max_memory_mb}MB, cleaning up old buffers")
            removed = await self.cleanup_old_buffers(max_age_seconds=300)  # Remove buffers older than 5 minutes
            if removed == 0:
                # If no old buffers, remove least recently used
                logger.warning("No old buffers to clean, removing least recently used buffers")
                await self._remove_least_recently_used()
            return True
        return False

    async def _remove_least_recently_used(self, count: int = 5) -> None:
        """Remove the specified number of least recently used buffers."""
        async with self._lock:
            if len(self._buffers) <= count:
                return

            # Sort buffers by last frame time (oldest first)
            sorted_buffers = sorted(
                self._buffers.items(),
                key=lambda x: x[1].state.last_frame_time
            )

            for stream_id, _ in sorted_buffers[:count]:
                del self._buffers[stream_id]
                logger.warning(f"Removed least recently used buffer for stream {stream_id}")

    async def periodic_cleanup(self) -> None:
        """Perform periodic cleanup tasks."""
        from app.core.config import settings
        import time

        current_time = time.time()
        if current_time - self._last_cleanup > settings.cleanup_interval_seconds:
            logger.debug("Running periodic buffer cleanup")
            await self.enforce_memory_limits()
            await self.cleanup_old_buffers()
            self._last_cleanup = current_time


# Global buffer manager instance
buffer_manager = StreamBufferManager()

