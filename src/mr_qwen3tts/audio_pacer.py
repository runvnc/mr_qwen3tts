"""Audio pacing for proper timing of TTS output to SIP.

Copied from mr_eleven_stream with minimal modifications.
"""

import asyncio
import logging
import time
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)


class AudioPacer:
    """Paces audio chunk delivery to match real-time playback rate.
    
    This ensures audio is sent at the correct rate for SIP/telephony systems,
    preventing buffer underruns or overruns.
    """
    
    def __init__(self, sample_rate: int = 8000, bytes_per_sample: int = 1):
        """
        Args:
            sample_rate: Audio sample rate in Hz (default 8000 for ulaw)
            bytes_per_sample: Bytes per sample (default 1 for ulaw)
        """
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        
        # Buffer for audio chunks
        self.buffer: list = []
        self.buffer_lock = asyncio.Lock()
        
        # State tracking
        self.is_running = False
        self.is_finished = False
        self.interrupted = False
        self.bytes_sent = 0
        
        # Timing
        self.start_time: Optional[float] = None
        self.current_timestamp: float = 0.0
        
        # Callback and context
        self._send_callback: Optional[Callable] = None
        self._context: Optional[Any] = None
        self._pacing_task: Optional[asyncio.Task] = None
        
        # Done event
        self._done_event = asyncio.Event()
    
    async def start_pacing(self, send_callback: Callable, context: Any = None):
        """Start the pacing loop.
        
        Args:
            send_callback: Async function to call with (chunk, timestamp, context)
            context: Context to pass to callback
        """
        self._send_callback = send_callback
        self._context = context
        self.is_running = True
        self.is_finished = False
        self.interrupted = False
        self.bytes_sent = 0
        self.start_time = time.perf_counter()
        self.current_timestamp = 0.0
        self._done_event.clear()
        
        self._pacing_task = asyncio.create_task(self._pacing_loop())
    
    async def add_chunk(self, chunk: bytes):
        """Add a chunk to the buffer."""
        async with self.buffer_lock:
            self.buffer.append(chunk)
    
    def mark_finished(self):
        """Mark that no more chunks will be added."""
        self.is_finished = True
    
    async def wait_until_done(self):
        """Wait until all buffered audio has been sent."""
        await self._done_event.wait()
    
    async def stop(self):
        """Stop the pacing loop."""
        self.is_running = False
        
        if self._pacing_task:
            self._pacing_task.cancel()
            try:
                await self._pacing_task
            except asyncio.CancelledError:
                pass
            self._pacing_task = None
    
    def _chunk_duration(self, chunk_size: int) -> float:
        """Calculate duration of a chunk in seconds."""
        samples = chunk_size / self.bytes_per_sample
        return samples / self.sample_rate
    
    async def _pacing_loop(self):
        """Main pacing loop that sends chunks at the correct rate."""
        try:
            while self.is_running:
                chunk = None
                
                # Get next chunk from buffer
                async with self.buffer_lock:
                    if self.buffer:
                        chunk = self.buffer.pop(0)
                
                if chunk:
                    # Calculate when this chunk should be sent
                    chunk_duration = self._chunk_duration(len(chunk))
                    target_time = self.start_time + self.current_timestamp
                    
                    # Wait until it's time to send
                    now = time.perf_counter()
                    if target_time > now:
                        await asyncio.sleep(target_time - now)
                    
                    # Send the chunk
                    if self._send_callback:
                        result = await self._send_callback(
                            chunk,
                            timestamp=self.current_timestamp,
                            context=self._context
                        )
                        
                        # Check if we were interrupted
                        if result is False:
                            self.interrupted = True
                            break
                    
                    self.bytes_sent += len(chunk)
                    self.current_timestamp += chunk_duration
                    
                elif self.is_finished:
                    # No more chunks and we're done
                    break
                else:
                    # Buffer empty but not finished, wait a bit
                    await asyncio.sleep(0.005)
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in pacing loop: {e}")
        finally:
            self._done_event.set()
