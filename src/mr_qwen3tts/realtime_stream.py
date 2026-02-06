"""Realtime streaming TTS using Qwen3-TTS WebSocket server.

This module provides the ability to stream text to the TTS server as it comes in
from the LLM, rather than waiting for complete sentences.

Uses a persistent WebSocket connection that is reused across sessions.
"""

import os
import asyncio
import json
import logging
import base64
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from lib.pipelines.pipe import pipe
from lib.providers.services import service_manager
from .audio_pacer import AudioPacer

logger = logging.getLogger(__name__)

# Debug log file
DEBUG_LOG_FILE = "/tmp/qwen3tts_debug.log"

def debug_log(msg):
    """Write debug message to dedicated log file."""
    import datetime
    with open(DEBUG_LOG_FILE, 'a') as f:
        f.write(f"{datetime.datetime.now().isoformat()} | {msg}\n")

def is_realtime_streaming_enabled() -> bool:
    """Check if realtime streaming is enabled via environment variable."""
    val = os.environ.get('MR_QWEN3TTS_REALTIME_STREAM', '0').lower()
    return val in ('1', 'true', 'yes', 'on')


# Configuration
DEFAULT_WS_URL = os.environ.get('MR_QWEN3TTS_WS_URL', 'ws://localhost:8765')
DEFAULT_REF_AUDIO = os.environ.get('MR_QWEN3TTS_REF_AUDIO', '')
DEFAULT_REF_TEXT = os.environ.get('MR_QWEN3TTS_REF_TEXT', '')


def _load_ref_audio_base64(audio_path: str) -> str:
    if not audio_path or not os.path.exists(audio_path):
        logger.warning(f"Reference audio not found: {audio_path}")
        return ""
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode('utf-8')


async def _get_voice_path_from_context(context) -> Optional[str]:
    if not context or not hasattr(context, 'agent_name'):
        return None
    try:
        agent_data = await service_manager.get_agent_data(context.agent_name)
        persona = agent_data.get("persona", {})
        voice_path = persona.get("voice_id", "")
        if voice_path and os.path.isabs(voice_path) and os.path.exists(voice_path):
            return voice_path
    except Exception as e:
        logger.warning(f"Could not get voice path from persona: {e}")
    return None


# --- Persistent WebSocket connection for realtime sessions ---

class _PersistentRealtimeWS:
    """Module-level persistent WebSocket connection for realtime TTS sessions.
    
    Maintains a single long-lived connection to the Qwen3-TTS server,
    with cached voice initialization state so we don't re-init on every request.
    """
    
    def __init__(self):
        self._ws = None
        self._ws_url = DEFAULT_WS_URL
        self._lock = asyncio.Lock()
        self._voice_initialized = False
        self._voice_id: Optional[str] = None
        self._current_voice_path: Optional[str] = None
        self._connection_id = 0
    
    async def get_connection(self, voice_path: str = None, context=None):
        """Get the persistent WebSocket, connecting and initializing voice if needed.
        
        Returns the raw websocket object for direct use by sessions.
        """
        async with self._lock:
            await self._ensure_connected()
            await self._ensure_voice_ready(voice_path, context)
            return self._ws
    
    async def _ensure_connected(self):
        """Ensure we have a live WebSocket connection."""
        import websockets
        
        if self._ws is not None:
            try:
                pong = await asyncio.wait_for(self._ws.ping(), timeout=2.0)
                await pong
                logger.debug(f"Realtime WS: reusing connection {self._connection_id}")
                return
            except Exception as e:
                logger.warning(f"Realtime WS: connection {self._connection_id} dead: {e}")
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
                self._voice_initialized = False
                self._voice_id = None
        
        # Create new connection
        self._connection_id += 1
        logger.info(f"Realtime WS: creating connection {self._connection_id} to {self._ws_url}")
        
        self._ws = await websockets.connect(
            self._ws_url,
            max_size=50 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        )
        
        # Wait for connected message
        msg = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
        data = json.loads(msg)
        if data.get("type") == "connected":
            logger.info(f"Realtime WS: connected {self._connection_id}, model: {data.get('model')}")
        
        self._voice_initialized = False
        self._voice_id = None
    
    async def _ensure_voice_ready(self, voice_path: str = None, context=None):
        """Ensure voice is initialized on the current connection."""
        if not voice_path:
            voice_path = await _get_voice_path_from_context(context)
        if not voice_path:
            voice_path = DEFAULT_REF_AUDIO
        
        # Already initialized with this voice on this connection
        if self._voice_initialized and voice_path == self._current_voice_path:
            logger.debug(f"Realtime WS: voice already initialized ({self._voice_id})")
            return
        
        if not voice_path:
            logger.error("Realtime WS: no reference audio available")
            return
        
        # Try voice_id-only fast re-init if we have a cached ID
        # (works if the server still has it in its voice cache)
        if self._voice_id and voice_path == self._current_voice_path:
            try:
                await self._ws.send(json.dumps({
                    "type": "init",
                    "voice_id": self._voice_id,
                }))
                msg = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
                data = json.loads(msg)
                if data.get("type") == "ready" and data.get("voice_loaded"):
                    self._voice_initialized = True
                    logger.info(f"Realtime WS: voice re-init from cache ({self._voice_id})")
                    return
            except Exception as e:
                logger.warning(f"Realtime WS: voice_id re-init failed: {e}")
        
        # Full init with audio
        ref_audio_b64 = _load_ref_audio_base64(voice_path)
        if not ref_audio_b64:
            logger.error(f"Realtime WS: could not load reference audio: {voice_path}")
            return
        
        logger.info(f"Realtime WS: initializing voice from {voice_path}")
        await self._ws.send(json.dumps({
            "type": "init",
            "ref_audio_base64": ref_audio_b64,
            "ref_text": DEFAULT_REF_TEXT or "",
            "auto_transcribe": True if not DEFAULT_REF_TEXT else False,
            "x_vector_only": False,
        }))
        
        msg = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
        data = json.loads(msg)
        if data.get("type") == "ready" and data.get("voice_loaded"):
            self._voice_initialized = True
            self._voice_id = data.get("voice_id")
            self._current_voice_path = voice_path
            logger.info(f"Realtime WS: voice ready (id={self._voice_id}, cached={data.get('cached')})")
        else:
            logger.error(f"Realtime WS: voice init failed: {data}")
    
    async def close(self):
        """Explicitly close the connection (e.g. on shutdown)."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._voice_initialized = False


# Module-level singleton
_persistent_ws = _PersistentRealtimeWS()


class RealtimeSpeakSession:
    """Manages a realtime streaming TTS session.
    
    Tracks text deltas and streams them to the Qwen3-TTS server,
    while simultaneously streaming audio output to SIP or local playback.
    
    Uses the module-level persistent WebSocket connection instead of
    creating/destroying connections per session.
    """
    
    def __init__(self, context: Any):
        self.context = context
        
        # State tracking
        self.previous_text = ""
        self.is_active = False
        self.is_finished = False
        
        # Text buffer for word boundary detection
        self._text_buffer = ""
        self._pending_text = ""
        
        # Reference to the shared WebSocket (set during start)
        self._ws = None
        
        # Audio processing
        self._pacer: Optional[AudioPacer] = None
        self._audio_task: Optional[asyncio.Task] = None
        
        # Synchronization
        self._generation_complete = asyncio.Event()
    
    async def _process_audio(self):
        """Async task that receives audio from WebSocket and sends to SIP."""
        try:
            debug_log("_process_audio: Starting audio processor")
            sip_available = service_manager.functions.get('sip_audio_out_chunk') is not None
            debug_log(f"_process_audio: SIP available={sip_available}")
            
            if sip_available:
                self._pacer = AudioPacer(sample_rate=8000)
                
                async def send_to_sip(chunk, timestamp=None, context=None):
                    result = await service_manager.sip_audio_out_chunk(
                        chunk, timestamp=timestamp, context=context
                    )
                    return result
                
                await self._pacer.start_pacing(send_to_sip, self.context)
            
            chunk_count = 0
            while True:
                try:
                    msg = await asyncio.wait_for(self._ws.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    if self.is_finished and self._generation_complete.is_set():
                        break
                    continue
                
                if isinstance(msg, bytes):
                    # Audio chunk
                    chunk_count += 1
                    debug_log(f"_process_audio: Got chunk {chunk_count}, size={len(msg)}")
                    
                    if sip_available and self._pacer:
                        await self._pacer.add_chunk(msg)
                        
                        if self._pacer.interrupted:
                            logger.debug("Pacer interrupted, stopping audio processing")
                            break
                else:
                    # JSON message
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    
                    if msg_type == "audio_start":
                        debug_log("_process_audio: Audio generation started")
                    elif msg_type == "audio_end":
                        debug_log("_process_audio: Audio generation complete")
                        self._generation_complete.set()
                        break
                    elif msg_type == "error":
                        logger.error(f"Server error: {data.get('message')}")
                        break
            
            logger.info(f"Audio processor completed. Total chunks: {chunk_count}")
            
        except Exception as e:
            logger.error(f"Error in audio processor: {e}")
    
    async def _finalize_pacer(self):
        """Finalize the audio pacer."""
        debug_log("_finalize_pacer: Starting pacer finalization")
        if self._pacer:
            self._pacer.mark_finished()
            
            if not self._pacer.interrupted:
                await self._pacer.wait_until_done()
            
            await self._pacer.stop()
            debug_log(f"_finalize_pacer: Pacer stopped, bytes_sent={self._pacer.bytes_sent}")
    
    async def start(self):
        """Start the realtime TTS session."""
        if self.is_active:
            debug_log("RealtimeSpeakSession.start: Session already active")
            return
        
        self.is_active = True
        self.is_finished = False
        self.previous_text = ""
        self._text_buffer = ""
        self._generation_complete.clear()
        
        # Get the persistent shared WebSocket (connects + inits voice if needed)
        voice_path = await _get_voice_path_from_context(self.context)
        self._ws = await _persistent_ws.get_connection(
            voice_path=voice_path, 
            context=self.context
        )
        
        debug_log(f"RealtimeSpeakSession.start: Session started (conn {_persistent_ws._connection_id})")
        logger.info("Started realtime TTS session")
    
    async def feed_text(self, delta: str):
        """Feed a text delta to the TTS stream."""
        if not self.is_active:
            debug_log("feed_text: Session not active!")
            return
        
        if delta:
            self._text_buffer += delta
            debug_log(f"feed_text: Buffer now: '{self._text_buffer}'")
    
    async def finish(self):
        """Signal that no more text will be added and wait for completion."""
        logger.info("Finishing realtime TTS session...")
        self.is_finished = True
        
        # Send accumulated text for generation
        if self._text_buffer and self._ws:
            debug_log(f"finish: Sending final text: '{self._text_buffer}'")
            
            await self._ws.send(json.dumps({
                "type": "generate_stream",
                "text": self._text_buffer,
                "language": "Auto",
                "voice_id": _persistent_ws._voice_id,
                "initial_chunk_tokens": 8,
                "stream_chunk_tokens": 8,
            }))
            
            # Start audio processing
            self._audio_task = asyncio.create_task(self._process_audio())
            
            # Wait for audio processing to complete
            try:
                await asyncio.wait_for(self._audio_task, timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning("Audio task timed out")
                self._audio_task.cancel()
        
        # Finalize pacer
        await self._finalize_pacer()
        
        # NOTE: We do NOT close the WebSocket here anymore.
        # The persistent connection stays open for reuse.
        self._ws = None  # Release our reference (connection stays alive in _persistent_ws)
        
        self.is_active = False
        logger.info("Realtime TTS session finished")
    
    async def cancel(self):
        """Cancel the session immediately."""
        logger.info("Cancelling realtime TTS session...")
        self.is_finished = True
        self.is_active = False
        
        # Cancel on server
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "cancel"}))
            except:
                pass
        
        # Stop pacer
        if self._pacer:
            await self._pacer.stop()
        
        # Cancel audio task
        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
        
        # NOTE: We do NOT close the WebSocket here anymore.
        self._ws = None
        
        logger.info("Realtime TTS session cancelled")


# Global registry of active sessions per log_id
_realtime_sessions: Dict[str, RealtimeSpeakSession] = {}


def get_session(log_id: str) -> Optional[RealtimeSpeakSession]:
    """Get the active session for a log_id, if any."""
    return _realtime_sessions.get(log_id)


def has_active_session(log_id: str) -> bool:
    """Check if there's an active realtime session for this log_id."""
    session = _realtime_sessions.get(log_id)
    return session is not None and session.is_active


async def cleanup_session(log_id: str):
    """Clean up and remove a session."""
    if log_id in _realtime_sessions:
        session = _realtime_sessions[log_id]
        if session.is_active:
            await session.cancel()
        del _realtime_sessions[log_id]


@pipe(name='partial_command', priority=10)
async def handle_speak_partial(data: dict, context=None) -> dict:
    """Intercepts partial_command calls to detect speak commands
    and stream text deltas to Qwen3-TTS in realtime.
    """
    # Check if realtime streaming is enabled
    if not is_realtime_streaming_enabled():
        return data
    
    debug_log(f"handle_speak_partial: data={data}")
    command = data.get('command')
    
    # Only handle speak commands
    if command != 'speak':
        return data
    
    log_id = context.log_id if context else None
    if not log_id:
        debug_log("handle_speak_partial: No log_id in context")
        return data
    
    params = data.get('params', {})
    new_text = params.get('text', '')
    
    if not new_text:
        return data
    
    # Get or create session for this log_id
    debug_log(f"handle_speak_partial: log_id={log_id}, new_text length={len(new_text)}")
    if log_id not in _realtime_sessions:
        debug_log(f"handle_speak_partial: Creating new session for log_id {log_id}")
        session = RealtimeSpeakSession(context=context)
        _realtime_sessions[log_id] = session
        await session.start()
    
    session = _realtime_sessions[log_id]
    
    # Calculate delta (new text since last update)
    debug_log(f"handle_speak_partial: previous_text length={len(session.previous_text)}, new_text length={len(new_text)}")
    if len(new_text) > len(session.previous_text):
        delta = new_text[len(session.previous_text):]
        if delta:
            debug_log(f"handle_speak_partial: Feeding delta: '{delta}'")
            await session.feed_text(delta)
            session.previous_text = new_text
    
    return data
