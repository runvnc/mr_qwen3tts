"""Audio pacing for proper timing of TTS output to SIP.

Adapted from mr_eleven_stream with interrupt support.
Pre-buffer support for jitter absorption.
"""

import asyncio
import logging
import os
import time
from collections import deque
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)


# Default pre-buffer: 40ms of audio before starting playback (absorbs HTTP chunked jitter)
DEFAULT_PRE_BUFFER_MS = int(os.environ.get('MR_QWEN3TTS_PRE_BUFFER_MS', '40'))

class AudioPacer:
    """Paces audio chunks to real-time speed with small buffer.
    
    Buffers incoming audio chunks and sends them at the correct rate
    with precise timestamps for proper playback timing.
    """

    def __init__(self, sample_rate: int = 8000, pre_buffer_ms: int = None):
        """
        Args:
            sample_rate: Audio sample rate in Hz (default 8000 for ulaw telephony)
            pre_buffer_ms: Milliseconds of audio to buffer before starting playback.
                           Absorbs network jitter from HTTP chunked transfer.
                           Default: 60ms (env: MR_QWEN3TTS_PRE_BUFFER_MS)
        """
        self.sample_rate = sample_rate
        self.pre_buffer_ms = pre_buffer_ms if pre_buffer_ms is not None else DEFAULT_PRE_BUFFER_MS
        self.buffer = deque()
        self.pacer_task: Optional[asyncio.Task] = None
        self.on_audio_chunk: Optional[Callable] = None
        self.context: Any = None
        self._running = False
        
        # Absolute timing for precise pacing
        self.start_time: Optional[float] = None
        self.bytes_sent = 0
        self.audio_start_time: Optional[float] = None
        self._finished_adding = False
        self._interrupted = False
        
        # Pre-buffer state
        self._pre_buffering = True
        self._pre_buffer_bytes = int(self.sample_rate * self.pre_buffer_ms / 1000)
        self._pre_buffer_accumulated = 0
        self._underrun_count = 0

    async def add_chunk(self, audio_bytes: bytes):
        """Add audio chunk to buffer."""
        if self._running:
            self.buffer.append(audio_bytes)
            
            # Accumulate pre-buffer before we start playing
            if self._pre_buffering:
                self._pre_buffer_accumulated += len(audio_bytes)
                if self._pre_buffer_accumulated >= self._pre_buffer_bytes:
                    self._pre_buffering = False
                    self.audio_start_time = time.perf_counter()
                    logger.info(f"AudioPacer: pre-buffer full ({self._pre_buffer_accumulated} bytes, "
                                f"{self._pre_buffer_accumulated/self.sample_rate*1000:.0f}ms), starting playback")

    @property
    def interrupted(self) -> bool:
        """Check if pacer was interrupted."""
        return self._interrupted

    def interrupt(self):
        """Interrupt the pacer externally (e.g. from on_interrupt hook)."""
        self._interrupted = True
        self.buffer.clear()
        logger.debug("AudioPacer: externally interrupted")

    def _set_interrupted(self):
        """Mark pacer as interrupted (from callback returning False)."""
        self._interrupted = True

    def mark_finished(self):
        """Mark that all chunks have been added."""
        self._finished_adding = True

    async def clear(self):
        """Clear buffer and reset state for interruption."""
        self.buffer.clear()
        self.audio_start_time = None
        self.bytes_sent = 0
        self.start_time = time.perf_counter()
        self._pre_buffering = True
        self._pre_buffer_accumulated = 0
        self._underrun_count = 0
        self._interrupted = False
        self._finished_adding = False
        logger.debug("AudioPacer cleared and reset")

    async def start_pacing(self, on_audio_chunk: Callable, context: Any):
        """Start real-time pacing task.
        
        Args:
            on_audio_chunk: Async callback function(chunk, timestamp, context)
            context: Context to pass to callback
        """
        self.on_audio_chunk = on_audio_chunk
        self.context = context
        self._running = True
        self._finished_adding = False
        self._interrupted = False
        
        self.start_time = time.perf_counter()
        self.bytes_sent = 0
        self.audio_start_time = None
        self._pre_buffering = True
        self._pre_buffer_accumulated = 0
        self._underrun_count = 0
        
        self.pacer_task = asyncio.create_task(self._pace_loop())

    async def _pace_loop(self):
        """Send buffered chunks at real-time intervals using absolute timing."""
        while self._running:
            if self._interrupted:
                break

            # During pre-buffering, wait for buffer to fill before sending
            if self._pre_buffering:
                await asyncio.sleep(0.005)
                continue

            if len(self.buffer) > 0:
                chunk = self.buffer.popleft()
                
                # Calculate timestamp for this chunk
                if self.audio_start_time:
                    chunk_timestamp = self.audio_start_time + (self.bytes_sent / self.sample_rate)
                else:
                    chunk_timestamp = None
                
                try:
                    result = await self.on_audio_chunk(chunk, timestamp=chunk_timestamp, context=self.context)
                    if result is False:
                        logger.debug("AudioPacer: callback requested stop")
                        self._set_interrupted()
                        break
                except Exception as e:
                    logger.error(f"AudioPacer: error in callback: {e}")
                    break
                
                self.bytes_sent += len(chunk)
                
                # Calculate target time based on total bytes sent
                base_time = self.audio_start_time if self.audio_start_time else self.start_time
                target_time = base_time + (self.bytes_sent / self.sample_rate)
                
                current_time = time.perf_counter()
                sleep_duration = target_time - current_time
                
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
                else:
                    # We're behind - potential underrun just happened or is about to
                    self._underrun_count += 1
                    if self._underrun_count <= 3:
                        logger.warning(f"AudioPacer: underrun #{self._underrun_count}, "
                                      f"{self.bytes_sent} bytes sent, buffer={len(self.buffer)}")
                    # Small yield to prevent busy-loop when catching up
                    await asyncio.sleep(0)
                    
            else:
                if self._finished_adding:
                    break
                # Buffer empty but not finished - short sleep
                self._underrun_count += 1
                if self._underrun_count <= 3:
                    logger.warning(f"AudioPacer: buffer empty (underrun #{self._underrun_count})")
                await asyncio.sleep(0.001)
        
        logger.info(f"AudioPacer: finished, sent {self.bytes_sent} bytes, "
                    f"underruns={self._underrun_count}")

    async def stop(self):
        """Stop pacing and clear buffer."""
        self._running = False
        if self.pacer_task:
            self.pacer_task.cancel()
            try:
                await self.pacer_task
            except asyncio.CancelledError:
                pass
        self.buffer.clear()

    async def wait_until_done(self):
        """Wait for all buffered audio to be sent."""
        if self.pacer_task:
            try:
                await self.pacer_task
            except asyncio.CancelledError:
                pass
