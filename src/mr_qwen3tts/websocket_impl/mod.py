"""
mr_qwen3tts - Qwen3-TTS streaming TTS plugin for MindRoot.

Drop-in replacement for mr_eleven_stream that connects to a remote
Qwen3-TTS WebSocket server for voice cloning and streaming TTS.
"""

import os
import asyncio
import base64
import json
import logging
import time
from typing import AsyncGenerator, Optional, Dict, Any

import dotenv
dotenv.load_dotenv()
import datetime

from lib.providers.services import service, service_manager
from lib.providers.commands import command
from lib.providers.hooks import hook

# Import realtime streaming support
from .realtime_stream import (
    has_active_session, 
    get_session, 
    cleanup_session, 
    is_realtime_streaming_enabled
)
from .audio_pacer import AudioPacer

logger = logging.getLogger(__name__)

# Dedicated debug log for speak() tracing
SPEAK_DEBUG_FILE = "/tmp/qwen3tts_speak_debug.log"

def _speak_debug(msg):
    """Write to dedicated speak debug log with timestamp."""
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"{ts} | {msg}"
    with open(SPEAK_DEBUG_FILE, 'a') as f:
        f.write(line + "\n")

# Configuration
DEFAULT_WS_URL = os.environ.get('MR_QWEN3TTS_WS_URL', 'ws://localhost:8765')
DEFAULT_REF_AUDIO = os.environ.get('MR_QWEN3TTS_REF_AUDIO', '')
DEFAULT_REF_TEXT = os.environ.get('MR_QWEN3TTS_REF_TEXT', '')
DEFAULT_LANGUAGE = os.environ.get('MR_QWEN3TTS_LANGUAGE', 'Auto')

# Reduced initial tokens for faster first audio (was 8)
INITIAL_CHUNK_TOKENS = int(os.environ.get('MR_QWEN3TTS_INITIAL_TOKENS', '4'))
STREAM_CHUNK_TOKENS = int(os.environ.get('MR_QWEN3TTS_STREAM_TOKENS', '4'))

# Global dictionary to track active speak() calls per log_id
_active_speak_locks: Dict[str, asyncio.Lock] = {}

# Track warmed up voices by path
_warmed_voices: Dict[str, str] = {}  # path -> voice_id
# Track active AudioPacer instances per log_id (for interrupt support)
_active_pacers: Dict[str, Any] = {}


def _get_local_playback_enabled() -> bool:
    """Check if local playback is enabled (no SIP available)."""
    return service_manager.functions.get('sip_audio_out_chunk', None) is None


def _load_ref_audio_base64(audio_path: str) -> str:
    """Load reference audio file and encode as base64."""
    if not audio_path or not os.path.exists(audio_path):
        logger.warning(f"Reference audio not found: {audio_path}")
        return ""
    
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    
    logger.info(f"Loaded reference audio: {audio_path} ({len(audio_bytes)} bytes)")
    return base64.b64encode(audio_bytes).decode('utf-8')


class Qwen3TTSClient:
    """WebSocket client for Qwen3-TTS server with persistent connection."""
    
    def __init__(
        self,
        ws_url: str = DEFAULT_WS_URL,
        language: str = DEFAULT_LANGUAGE,
    ):
        self.ws_url = ws_url
        self.language = language
        
        self._ws = None
        self._voice_initialized = False
        self._voice_id: Optional[str] = None
        self._current_audio_path: Optional[str] = None
        
        # Single lock for all operations
        self._lock = asyncio.Lock()
        
        # Connection tracking
        self._connection_id = 0
        self._created_at = time.time()
        
        # Track whether previous generation completed cleanly
        self._needs_drain = False
        
        logger.info(f"Qwen3TTSClient created at {self._created_at}")
    
    async def force_reconnect(self):
        """Force disconnect and reconnect with voice re-init.
        
        Used after interruptions to ensure a fresh connection.
        """
        _speak_debug("force_reconnect: starting")
        async with self._lock:
            try:
                await self.disconnect()
                await self.connect()
                # Re-init voice if we had one
                if self._current_audio_path:
                    await self.initialize_voice(audio_path=self._current_audio_path)
                _speak_debug(f"force_reconnect: done, conn={self._connection_id}")
            except Exception as e:
                _speak_debug(f"force_reconnect: error: {e}")
                logger.error(f"Error during force_reconnect: {e}")
    
    def schedule_reconnect(self):
        """Schedule a background reconnect (non-blocking)."""
        _speak_debug("schedule_reconnect: scheduling background reconnect")
        self._needs_drain = False  # Fresh connection won't need drain
        asyncio.create_task(self.force_reconnect())
    
    async def connect(self):
        """Connect to the WebSocket server."""
        import websockets
        
        # Check if existing connection is alive
        if self._ws is not None:
            try:
                pong = await asyncio.wait_for(self._ws.ping(), timeout=2.0)
                await pong
                logger.debug(f"Reusing connection {self._connection_id}")
                return
            except Exception as e:
                logger.warning(f"Connection {self._connection_id} dead: {e}")
                try:
                    await self._ws.close()
                except:
                    pass
                self._ws = None
                self._voice_initialized = False
                self._needs_drain = False
        
        # Create new connection
        self._connection_id += 1
        logger.info(f"Creating connection {self._connection_id} to {self.ws_url}")
        
        self._ws = await websockets.connect(
            self.ws_url,
            max_size=50 * 1024 * 1024,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=10,
        )
        
        # Wait for connected message
        msg = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
        data = json.loads(msg)
        if data.get("type") == "connected":
            logger.info(f"Connected {self._connection_id}, model: {data.get('model')}")
        
        self._voice_initialized = False
        self._needs_drain = False
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._voice_initialized = False
            self._needs_drain = False
    
    async def _drain_pending(self):
        """Drain any pending messages from a previous interrupted generation.
        
        Sends a cancel message and reads/discards messages until we get
        audio_end or the buffer is empty (timeout). This ensures the
        WebSocket is in a clean state before starting a new request.
        """
        if not self._needs_drain or not self._ws:
            return
        
        logger.info("Draining pending messages from previous generation...")
        
        # Send cancel in case server is still generating
        try:
            await self._ws.send(json.dumps({"type": "cancel"}))
        except Exception as e:
            logger.warning(f"Failed to send cancel during drain: {e}")
            self._needs_drain = False
            return
        
        # Read and discard messages until audio_end or timeout
        drained_count = 0
        while True:
            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=0.05)
                drained_count += 1
                
                if isinstance(msg, bytes):
                    continue  # Discard audio chunk
                
                data = json.loads(msg)
                msg_type = data.get("type")
                
                if msg_type == "audio_end":
                    logger.info(f"Drain complete: discarded {drained_count} messages (got audio_end)")
                    break
                elif msg_type == "error":
                    logger.info(f"Drain complete: discarded {drained_count} messages (got error)")
                    break
                    
            except asyncio.TimeoutError:
                # No more pending messages
                logger.info(f"Drain complete: discarded {drained_count} messages (buffer empty)")
                break
            except Exception as e:
                logger.warning(f"Error during drain: {e}")
                break
        
        self._needs_drain = False
    
    async def initialize_voice(
        self, 
        audio_path: str = None,
        ref_audio_b64: str = None, 
        ref_text: str = None,
        auto_transcribe: bool = True,
    ) -> dict:
        """Initialize voice clone with reference audio."""
        global _warmed_voices
        
        t0 = time.time()
        
        # If warmed previously, do a fast init using cached voice_id
        if audio_path and not ref_audio_b64 and audio_path in _warmed_voices:
            cached_voice_id = _warmed_voices[audio_path]
            try:
                logger.info(f"Trying voice_id-only init with {cached_voice_id}")
                await self._ws.send(json.dumps({
                    "type": "init",
                    "voice_id": cached_voice_id,
                    "ref_text": ref_text or "",
                }))

                msg = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
                data = json.loads(msg)

                if data.get("type") == "ready" and data.get("voice_loaded"):
                    self._voice_initialized = True
                    self._voice_id = data.get("voice_id") or cached_voice_id
                    self._current_audio_path = audio_path
                    logger.info(f"Voice re-init from cache in {(time.time()-t0)*1000:.0f}ms")
                    return data

                logger.warning(f"voice_id-only init failed: {data}")
            except Exception as e:
                logger.warning(f"voice_id-only init exception: {e}")
        
        # Load audio if needed
        if audio_path and not ref_audio_b64:
            ref_audio_b64 = _load_ref_audio_base64(audio_path)
        
        if not ref_audio_b64:
            logger.warning("No reference audio provided")
            return {"voice_loaded": False, "error": "No reference audio"}
        
        # Full init with audio
        await self._ws.send(json.dumps({
            "type": "init",
            "ref_audio_base64": ref_audio_b64,
            "ref_text": ref_text or "",
            "auto_transcribe": auto_transcribe and not ref_text,
            "x_vector_only": False,
        }))
        
        msg = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
        data = json.loads(msg)
        if data.get("type") == "ready" and data.get("voice_loaded"):
            self._voice_initialized = True
            self._voice_id = data.get("voice_id")
            self._current_audio_path = audio_path
            
            if audio_path:
                _warmed_voices[audio_path] = self._voice_id
            
            logger.info(f"Voice initialized in {(time.time()-t0)*1000:.0f}ms: id={self._voice_id}")
            return data
        elif data.get("type") == "error":
            raise RuntimeError(f"Voice init failed: {data.get('message')}")
        
        return data
    
    async def warmup_voice(self, audio_path: str, ref_text: str = None) -> dict:
        """Warm up a voice by path."""
        async with self._lock:
            try:
                await self.connect()
                await self._drain_pending()
                result = await self.initialize_voice(
                    audio_path=audio_path,
                    ref_text=ref_text,
                    auto_transcribe=True
                )
                
                if result.get("voice_loaded"):
                    return {
                        "warmed_up": True,
                        "voice_id": result.get("voice_id"),
                        "was_cached": result.get("cached", False),
                    }
                return {"warmed_up": False, "reason": result.get("error", "init_failed")}
                    
            except Exception as e:
                logger.error(f"Warmup failed: {e}")
                return {"warmed_up": False, "reason": str(e)}
    
    async def ensure_voice_ready(self, audio_path: str = None, ref_text: str = None):
        """Ensure voice is initialized before generating."""
        if audio_path and audio_path != self._current_audio_path:
            logger.info(f"Switching voice to {audio_path}")
            self._voice_initialized = False
        
        if not self._voice_initialized:
            await self.initialize_voice(audio_path=audio_path, ref_text=ref_text)
    
    async def generate_stream(
        self,
        text: str,
        language: str = None,
        audio_path: str = None,
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio and stream chunks."""
        t0 = time.time()
        completed_cleanly = False
        
        async with self._lock:
            logger.info(f"generate_stream starting (conn {self._connection_id})")
            
            await self.connect()
            
            # Drain any leftover messages from a previous interrupted generation
            await self._drain_pending()
            
            await self.ensure_voice_ready(audio_path=audio_path)
            
            language = language or self.language
            
            # Mark that we're starting a generation that needs draining if interrupted
            self._needs_drain = True
            
            # Send generate_stream request
            await self._ws.send(json.dumps({
                "type": "generate_stream",
                "text": text,
                "language": language,
                "voice_id": self._voice_id,
                "initial_chunk_tokens": INITIAL_CHUNK_TOKENS,
                "stream_chunk_tokens": STREAM_CHUNK_TOKENS,
            }))
            
            chunk_count = 0
            first_chunk_time = None
            
            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(self._ws.recv(), timeout=60.0)
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for audio")
                        break
                    
                    if isinstance(msg, bytes):
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                            logger.info(f"First audio in {(first_chunk_time-t0)*1000:.0f}ms")
                        chunk_count += 1
                        yield msg
                    else:
                        data = json.loads(msg)
                        msg_type = data.get("type")
                        
                        if msg_type == "audio_end":
                            logger.info(f"Audio complete: {chunk_count} chunks in {(time.time()-t0)*1000:.0f}ms")
                            completed_cleanly = True
                            break
                        elif msg_type == "audio_start":
                            continue
                        elif msg_type == "error":
                            logger.error(f"Generation error: {data.get('message')}")
                            completed_cleanly = True  # No leftover audio to drain
                            break
            except GeneratorExit:
                # Consumer stopped iterating (e.g., interrupted)
                logger.info(f"generate_stream interrupted after {chunk_count} chunks")
            finally:
                if completed_cleanly:
                    self._needs_drain = False
                else:
                    # Generation was interrupted - next call needs to drain
                    self._needs_drain = True
                    logger.info("Generation did not complete cleanly, will drain on next call")
    
    async def cancel(self):
        """Cancel current generation."""
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "cancel"}))
            except Exception:
                pass


# Global client instance - SINGLETON
_client: Optional[Qwen3TTSClient] = None
_client_lock = None


def _get_client_lock():
    """Get or create the client lock."""
    global _client_lock
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    return _client_lock


async def get_client() -> Qwen3TTSClient:
    """Get or create the global Qwen3-TTS client instance."""
    global _client
    
    lock = _get_client_lock()
    async with lock:
        if _client is None:
            logger.info("Creating new global Qwen3TTSClient")
            _client = Qwen3TTSClient()
        else:
            logger.debug(f"Reusing global client (created at {_client._created_at})")
        return _client


async def get_voice_path_from_context(context) -> Optional[str]:
    """Get the voice audio path from the agent's persona."""
    if not context or not hasattr(context, 'agent_name'):
        return None
    
    try:
        agent_data = await service_manager.get_agent_data(context.agent_name)
        persona = agent_data.get("persona", {})
        voice_path = persona.get("voice_id", "")
        
        if voice_path and os.path.isabs(voice_path) and os.path.exists(voice_path):
            return voice_path
    except Exception as e:
        logger.warning(f"Could not get voice path: {e}")
    
    return None


async def get_all_agent_voice_paths() -> Dict[str, str]:
    """Scan all agent personas and return their voice paths."""
    import glob
    
    voice_paths = {}
    cwd = os.getcwd()
    personas_dir = os.path.join(cwd, 'personas', 'local')
    
    if not os.path.exists(personas_dir):
        return voice_paths
    
    for persona_dir in glob.glob(os.path.join(personas_dir, '*')):
        if not os.path.isdir(persona_dir):
            continue
        
        persona_name = os.path.basename(persona_dir)
        persona_file = os.path.join(persona_dir, 'persona.json')
        
        if not os.path.exists(persona_file):
            continue
        
        try:
            with open(persona_file, 'r') as f:
                persona_data = json.load(f)
            
            voice_id = persona_data.get('voice_id', '')
            
            if voice_id and os.path.isabs(voice_id) and os.path.exists(voice_id):
                voice_paths[persona_name] = voice_id
        except Exception as e:
            logger.warning(f"Error reading persona {persona_name}: {e}")
    
    return voice_paths


@hook()
async def startup(app, context):
    """Warm up all agent voices on plugin startup."""
    client = await get_client()
    
    voice_paths = await get_all_agent_voice_paths()
    
    if DEFAULT_REF_AUDIO and os.path.exists(DEFAULT_REF_AUDIO):
        voice_paths['_default_'] = DEFAULT_REF_AUDIO
    
    if not voice_paths:
        logger.info("No voice files found to warm up")
        return
    
    logger.info(f"Warming up {len(voice_paths)} voice(s)")
    
    for agent_name, audio_path in voice_paths.items():
        try:
            ref_text = DEFAULT_REF_TEXT if agent_name == '_default_' else None
            result = await client.warmup_voice(audio_path, ref_text)
            if result.get("warmed_up"):
                logger.info(f"Voice ready for {agent_name}")
        except Exception as e:
            logger.error(f"Voice warmup failed for {agent_name}: {e}")
    
    logger.info("Voice warmup complete")


@service()
async def stream_tts(
    text: str,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
    output_format: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> AsyncGenerator[bytes, None]:
    """Stream text-to-speech audio using Qwen3-TTS."""
    client = await get_client()
    
    audio_path = voice_id
    
    if not audio_path or not os.path.exists(audio_path or ""):
        audio_path = await get_voice_path_from_context(context)
    
    if not audio_path:
        audio_path = DEFAULT_REF_AUDIO
    
    logger.info(f"TTS stream: {text[:50]}... (voice: {audio_path})")
    
    async for chunk in client.generate_stream(text=text, audio_path=audio_path):
        yield chunk


@command()
async def speak(
    text: str,
    voice_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Convert text to speech using Qwen3-TTS streaming."""
    try:
        log_id = None
        if context and hasattr(context, 'log_id'):
            log_id = context.log_id

        _speak_debug(f"speak() CALLED text='{text[:60]}...' log_id={log_id}")
        
        if log_id:
            if log_id not in _active_speak_locks:
                _active_speak_locks[log_id] = asyncio.Lock()
            
            lock = _active_speak_locks[log_id]
            
            if lock.locked():
                _speak_debug(f"speak() REJECTED - lock already held for {log_id}")
                logger.warning(f"speak() already running for {log_id}")
                return "ERROR: Speech already in progress."
            
            await lock.acquire()
        
        # Check for active realtime streaming session
        realtime_enabled = is_realtime_streaming_enabled()
        
        if realtime_enabled and log_id and has_active_session(log_id):
            session = get_session(log_id)
            if session:
                await session.finish()
                await cleanup_session(log_id)
            _speak_debug(f"speak() handled realtime session, returning")
            if log_id and log_id in _active_speak_locks and _active_speak_locks[log_id].locked():
                _active_speak_locks[log_id].release()
            return None
        
        audio_path = voice_id
        if not audio_path or not os.path.exists(audio_path or ""):
            audio_path = await get_voice_path_from_context(context)
        if not audio_path:
            audio_path = DEFAULT_REF_AUDIO
        
        # Warm up voice if not already done
        if audio_path and audio_path not in _warmed_voices:
            client = await get_client()
            logger.info(f"Warming up voice: {audio_path}")
            await client.warmup_voice(audio_path)
        
        chunk_count = 0
        local_playback = _get_local_playback_enabled()
        
        if not local_playback:
            try:
                is_halted = await service_manager.sip_is_audio_halted(context=context)
                if is_halted:
                    _speak_debug(f"speak() SKIPPED - sip_is_audio_halted=True text='{text[:60]}'")
                    logger.info("Audio halted, skipping speak")
                    return None
            except Exception:
                pass
        
        _speak_debug(f"speak() STARTING stream text='{text[:60]}' local={local_playback}")

        pacer = None
        if not local_playback:
            pacer = AudioPacer(sample_rate=8000)
            
            # Track this pacer for interrupt support
            if log_id:
                _active_pacers[log_id] = pacer
            
            async def send_to_sip(chunk, timestamp=None, context=None):
                try:
                    return await service_manager.sip_audio_out_chunk(chunk, timestamp=timestamp, context=context)
                except Exception as e:
                    logger.error(f"Error sending to SIP: {e}")
                    return False
            
            await pacer.start_pacing(send_to_sip, context)
        
        local_audio_buffer = b"" if local_playback else None
        
        async for chunk in stream_tts(text=text, voice_id=audio_path, context=context):
            chunk_count += 1
            
            if local_playback:
                local_audio_buffer += chunk
            else:
                if pacer.interrupted:
                    _speak_debug(f"speak() INTERRUPTED by pacer after {chunk_count} chunks")
                    break
                await pacer.add_chunk(chunk)
        
        _speak_debug(f"speak() stream done, {chunk_count} chunks, pacer.interrupted={pacer.interrupted if pacer else 'N/A'}")

        if not local_playback and pacer:
            pacer.mark_finished()
            
            if not pacer.interrupted:
                await pacer.wait_until_done()
            
            await pacer.stop()
            
            # Remove from active pacers
            if log_id and log_id in _active_pacers:
                del _active_pacers[log_id]
            
            # After interruption, force reconnect in background so next speak()
            # gets a fresh connection instead of a potentially dying one
            if pacer.interrupted:
                client = await get_client()
                client.schedule_reconnect()
            
            if pacer.interrupted and chunk_count < 2:
                _speak_debug(f"speak() returning WARNING - interrupted with <2 chunks")
                return "SYSTEM: WARNING - Command interrupted!\n\n"
        
        _speak_debug(f"speak() COMPLETE text='{text[:60]}' chunks={chunk_count}")
        logger.info(f"Speech complete: {len(text)} chars, {chunk_count} chunks")
        return None
        
    except Exception as e:
        _speak_debug(f"speak() EXCEPTION: {e}")
        logger.error(f"Error in speak: {e}")
        return None

    finally:
        if log_id and log_id in _active_speak_locks:
            lock = _active_speak_locks[log_id]
            if lock.locked():
                lock.release()
        _speak_debug(f"speak() FINALLY - lock released for {log_id}")


@hook()
async def on_interrupt(context=None):
    """Handle interruption signal from the system.
    
    Called when the user interrupts the AI (e.g., starts speaking during TTS).
    Cancels any active TTS streams for the current session.
    """
    log_id = None
    if context and hasattr(context, 'log_id'):
        log_id = context.log_id
    
    if not log_id:
        logger.debug("on_interrupt called without log_id")
        return
    
    _speak_debug(f"on_interrupt FIRED log_id={log_id} pacer_active={log_id in _active_pacers}")
    # Cancel active pacer for this session
    if log_id in _active_pacers:
        pacer = _active_pacers[log_id]
        logger.info(f"on_interrupt: Interrupting TTS stream for session {log_id}")
        pacer.interrupt()
    
    # Also cleanup any realtime streaming session
    if is_realtime_streaming_enabled() and has_active_session(log_id):
        logger.info(f"on_interrupt: Cleaning up realtime session for {log_id}")
        await cleanup_session(log_id)
