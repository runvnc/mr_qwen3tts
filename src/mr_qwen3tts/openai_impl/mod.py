"""
mr_qwen3tts openai_impl - Qwen3-TTS via groxaxo OpenAI-FastAPI server.

Connects to groxaxo/Qwen3-TTS-Openai-Fastapi server's /v1/audio/speech endpoint.
Uses OpenAI-compatible API with PCM streaming.

Default backend (MR_QWEN3TTS_BACKEND=openai).
"""

import os
import asyncio
import audioop
import hashlib
import logging
import time
import datetime
from typing import AsyncGenerator, Optional, Dict, Any

import dotenv
dotenv.load_dotenv()

import socket
import httpx

from lib.providers.services import service, service_manager
from lib.providers.commands import command
from lib.providers.hooks import hook

from ..audio_pacer import AudioPacer

logger = logging.getLogger(__name__)

# Dedicated debug log for speak() tracing
SPEAK_DEBUG_FILE = "/tmp/qwen3tts_speak_debug.log"


def _speak_debug(msg):
    """Write to dedicated speak debug log with timestamp."""
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"{ts} | {msg}"
    with open(SPEAK_DEBUG_FILE, 'a') as f:
        f.write(line + "\n")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.environ.get('MR_QWEN3TTS_OPENAI_URL', 'http://localhost:8880')
VOICE = os.environ.get('MR_QWEN3TTS_VOICE', 'Vivian')
LANGUAGE = os.environ.get('MR_QWEN3TTS_LANGUAGE', 'Auto')

# TCP_NODELAY socket option tuple: (IPPROTO_TCP, TCP_NODELAY, 1)
# Disables Nagle's algorithm to avoid up to 40ms buffering delay on small writes.
# Critical for streaming audio chunks on localhost.
_TCP_NODELAY_OPT = (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

MODEL = os.environ.get('MR_QWEN3TTS_MODEL', 'qwen3-tts')
REF_AUDIO = os.environ.get('MR_QWEN3TTS_REF_AUDIO', '')
REF_TEXT = os.environ.get('MR_QWEN3TTS_REF_TEXT', '')

# Qwen speak() already paces chunks before handing them to SIP.
# For live RTP, forwarding absolute timestamps can trigger PySIP catch-up bursts
# when the final sender sees frames as late, which Linphone may render as
# flutter/fast-echo artifacts. Default off for Qwen only; ElevenLabs is untouched.
SEND_TIMESTAMPS = os.environ.get(
    'MR_QWEN3TTS_SEND_TIMESTAMPS', '0'
).lower() in ('1', 'true', 'yes', 'on')

# Per-session speak() locks
_active_speak_locks: Dict[str, asyncio.Lock] = {}
# Active AudioPacer instances per log_id (for interrupt support)
_active_pacers: Dict[str, Any] = {}

# Voice URL -> clone name cache
# When voice_id is a URL, we auto-register it on first use and cache the
# resulting clone:Name so subsequent requests skip registration.
_voice_url_cache: Dict[str, str] = {}
# Lock to prevent concurrent registration of the same URL
_voice_register_locks: Dict[str, asyncio.Lock] = {}


def _get_local_playback_enabled() -> bool:
    """Check if local playback is enabled (no SIP available)."""
    return service_manager.functions.get('sip_audio_out_chunk', None) is None


def _is_voice_url(voice: str) -> bool:
    """Check if a voice identifier is a URL (needs auto-registration)."""
    return voice.startswith('http://') or voice.startswith('https://')


async def _register_voice_url(voice_url: str, name: str = None) -> str:
    """
    Register a voice from a URL with the groxaxo server's voice-register endpoint.

    Downloads the audio, creates a voice library profile, and returns the
    clone:Name identifier for use in subsequent TTS requests.

    Args:
        voice_url: URL to the reference audio file
        name: Optional profile name. If not provided, derived from URL hash.

    Returns:
        clone:Name string for use as the voice parameter in TTS requests
    """
    # Check cache first
    if voice_url in _voice_url_cache:
        return _voice_url_cache[voice_url]

    # Prevent concurrent registration of the same URL
    if voice_url not in _voice_register_locks:
        _voice_register_locks[voice_url] = asyncio.Lock()
    lock = _voice_register_locks[voice_url]

    async with lock:
        # Double-check cache after acquiring lock
        if voice_url in _voice_url_cache:
            return _voice_url_cache[voice_url]

        # Derive a profile name from the URL if not provided
        if not name:
            # Use a short hash of the URL as the profile name
            url_hash = hashlib.md5(voice_url.encode()).hexdigest()[:8]
            name = f"auto_{url_hash}"

        # Sanitize name for the server
        name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)

        register_url = f"{API_URL.rstrip('/')}/v1/audio/voice-register"
        payload = {
            "name": name,
            "ref_audio_url": voice_url,
            "ref_text": REF_TEXT or None,
            "language": LANGUAGE,
            "x_vector_only_mode": False,  # ICL mode - server auto-transcribes with Whisper for best quality
        }

        logger.info(f"Auto-registering voice from URL: {voice_url} -> clone:{name}")
        _speak_debug(f"Auto-registering voice: {voice_url} -> clone:{name}")

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
                transport=httpx.AsyncHTTPTransport(socket_options=[_TCP_NODELAY_OPT])
            ) as client:
                response = await client.post(register_url, json=payload)

                if response.status_code == 200:
                    result = response.json()
                    clone_name = result.get("voice_id", f"clone:{name}")
                    _voice_url_cache[voice_url] = clone_name
                    logger.info(f"Voice registered: {clone_name}")
                    _speak_debug(f"Voice registered: {clone_name}")
                    return clone_name
                else:
                    logger.error(f"Voice registration failed ({response.status_code}): {response.text[:500]}")
                    _speak_debug(f"Voice registration FAILED: {response.status_code}")
                    # Fall back to using the URL directly (may not work but better than crashing)
                    return voice_url

        except Exception as e:
            logger.error(f"Voice registration error: {e}")
            _speak_debug(f"Voice registration ERROR: {e}")
            return voice_url


async def _resolve_voice(voice: str) -> str:
    """
    Resolve a voice identifier to a form the groxaxo server understands.

    - If voice is a URL, auto-register it and return clone:Name
    - If voice starts with clone:, return as-is
    - Otherwise return as-is (built-in voice name)

    Args:
        voice: Voice identifier (name, clone:Name, or URL)

    Returns:
        Resolved voice identifier for the TTS request
    """
    if _is_voice_url(voice):
        return await _register_voice_url(voice)
    return voice


# ---------------------------------------------------------------------------
# PCM 24kHz -> ulaw 8kHz conversion
# ---------------------------------------------------------------------------

def _pcm24k_to_ulaw8k(pcm_bytes: bytes, state) -> tuple:
    """
    Convert raw 16-bit signed PCM at 24kHz to ulaw 8kHz.
    Returns (ulaw_bytes, new_state).
    """
    if not pcm_bytes:
        return b'', state
    # Resample 24000 -> 8000 (factor 1/3)
    resampled, new_state = audioop.ratecv(pcm_bytes, 2, 1, 24000, 8000, state)
    # Encode to ulaw
    ulaw = audioop.lin2ulaw(resampled, 2)
    return ulaw, new_state


# ---------------------------------------------------------------------------
# Core streaming function
# ---------------------------------------------------------------------------

async def _stream_tts_openai(
    text: str,
    voice: str = None,
    language: str = None,
) -> AsyncGenerator[bytes, None]:
    """
    Stream ulaw 8kHz audio chunks from groxaxo OpenAI-FastAPI server.

    The groxaxo server streams raw 16-bit signed PCM at 24kHz when stream=true
    and response_format=pcm. We resample and encode to ulaw 8kHz on the fly.
    Yields 160-byte chunks (20ms at 8kHz) suitable for SIP.
    """
    url = f"{API_URL.rstrip('/')}/v1/audio/speech"

    voice = voice or VOICE

    # Resolve voice: if it's a URL, auto-register and get clone:Name
    voice = await _resolve_voice(voice)

    # Build the OpenAI-compatible payload
    payload = {
        "model": MODEL,
        "voice": voice,
        "input": text,
        "response_format": "pcm",  # Raw PCM for streaming
        "stream": True,
        "speed": 1.0,
    }

    resample_state = None
    ulaw_buffer = b''
    pcm_align_buffer = b''  # Ensure even-length PCM for audioop (16-bit samples)
    CHUNK_SIZE = 160  # 20ms at 8kHz ulaw
    t0 = time.time()
    first_chunk_logged = False

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        transport=httpx.AsyncHTTPTransport(socket_options=[_TCP_NODELAY_OPT])
    ) as client:
        async with client.stream('POST', url, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                logger.error(f"TTS server error {response.status_code}: {body[:500]}")
                response.raise_for_status()

            async for pcm_chunk in response.aiter_bytes(chunk_size=512):
                if not pcm_chunk:
                    continue

                # Ensure PCM is even-length (complete 16-bit samples for audioop)
                pcm_chunk = pcm_align_buffer + pcm_chunk
                if len(pcm_chunk) % 2 != 0:
                    pcm_align_buffer = pcm_chunk[-1:]
                    pcm_chunk = pcm_chunk[:-1]
                else:
                    pcm_align_buffer = b''

                if not first_chunk_logged:
                    logger.info(f"First PCM chunk in {(time.time()-t0)*1000:.0f}ms")
                    first_chunk_logged = True

                # Convert PCM 24kHz -> ulaw 8kHz
                ulaw_chunk, resample_state = _pcm24k_to_ulaw8k(pcm_chunk, resample_state)
                ulaw_buffer += ulaw_chunk

                # Yield complete 160-byte frames
                while len(ulaw_buffer) >= CHUNK_SIZE:
                    yield ulaw_buffer[:CHUNK_SIZE]
                    ulaw_buffer = ulaw_buffer[CHUNK_SIZE:]

    # Flush any remaining PCM alignment bytes (shouldn't normally happen)
    if pcm_align_buffer:
        logger.debug(f"Discarding {len(pcm_align_buffer)} trailing PCM byte")

    # Flush any remaining audio
    if ulaw_buffer:
        # Pad to 160 bytes with silence (0x7f = ulaw silence)
        ulaw_buffer = ulaw_buffer.ljust(CHUNK_SIZE, b'\x7f')
        yield ulaw_buffer[:CHUNK_SIZE]

    logger.info(f"TTS stream complete in {(time.time()-t0)*1000:.0f}ms")


# ---------------------------------------------------------------------------
# MindRoot service
# ---------------------------------------------------------------------------

@service()
async def stream_tts(
    text: str,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
    output_format: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> AsyncGenerator[bytes, None]:
    """Stream text-to-speech audio using groxaxo Qwen3-TTS OpenAI-FastAPI server."""
    voice = voice_id or VOICE
    logger.info(f"stream_tts: '{text[:60]}...' voice={voice}")
    async for chunk in _stream_tts_openai(text=text, voice=voice):
        yield chunk


# ---------------------------------------------------------------------------
# MindRoot command
# ---------------------------------------------------------------------------

@command()
async def speak(
    text: str,
    voice_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Convert text to speech using groxaxo Qwen3-TTS streaming."""
    log_id = None
    if context and hasattr(context, 'log_id'):
        log_id = context.log_id

    try:
        _speak_debug(f"speak() CALLED text='{text[:60]}...' log_id={log_id}")

        # Per-session serialization
        if log_id:
            if log_id not in _active_speak_locks:
                _active_speak_locks[log_id] = asyncio.Lock()
            lock = _active_speak_locks[log_id]
            if lock.locked():
                _speak_debug(f"speak() REJECTED - lock already held for {log_id}")
                logger.warning(f"speak() already running for {log_id}")
                return "ERROR: Speech already in progress."
            await lock.acquire()

        # Resolve voice: explicit arg > persona voice_id > env var default
        voice = voice_id
        if not voice and context and hasattr(context, 'agent_name'):
            try:
                agent_data = await service_manager.get_agent_data(context.agent_name)
                persona = agent_data.get("persona", {})
                persona_voice = persona.get("voice_id", "")
                if persona_voice:
                    voice = persona_voice
                    _speak_debug(f"speak() using persona voice_id='{persona_voice}'")
                    logger.info(f"speak(): using persona voice_id='{persona_voice}'")
            except Exception as e:
                logger.warning(f"speak(): could not get persona voice_id: {e}")
        if not voice:
            voice = VOICE

        # Guard: if voice is not a URL and not a clone: profile, refuse to speak
        # (prevents silently falling back to a built-in voice when clone is expected)
        is_clone = voice.startswith('clone:') or _is_voice_url(voice)
        if not is_clone:
            err = f"ERROR: No voice clone configured. voice='{voice}' is a built-in voice name. Set persona voice_id to a URL or clone:Name."
            logger.error(err)
            _speak_debug(err)
            return err

        local_playback = _get_local_playback_enabled()

        if not local_playback:
            try:
                is_halted = await service_manager.sip_is_audio_halted(context=context)
                if is_halted:
                    _speak_debug(f"speak() SKIPPED - sip_is_audio_halted=True")
                    logger.info("Audio halted, skipping speak")
                    return None
            except Exception:
                pass

        _speak_debug(f"speak() STARTING stream text='{text[:60]}' local={local_playback}")

        pacer = None
        if not local_playback:
            pacer = AudioPacer(sample_rate=8000)
            if log_id:
                _active_pacers[log_id] = pacer

            async def send_to_sip(chunk, timestamp=None, context=None):
                try:
                    send_ts = timestamp if SEND_TIMESTAMPS else None
                    return await service_manager.sip_audio_out_chunk(chunk, timestamp=send_ts, context=context)
                except Exception as e:
                    logger.error(f"Error sending to SIP: {e}")
                    return False

            await pacer.start_pacing(send_to_sip, context)

        local_audio_buffer = b'' if local_playback else None
        chunk_count = 0

        async for chunk in _stream_tts_openai(text=text, voice=voice):
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
            if log_id and log_id in _active_pacers:
                del _active_pacers[log_id]
            if pacer.interrupted and chunk_count < 2:
                _speak_debug(f"speak() returning WARNING - interrupted with <2 chunks")
                return "SYSTEM: WARNING - Command interrupted!\n"

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


# ---------------------------------------------------------------------------
# Interrupt hook
# ---------------------------------------------------------------------------

@hook()
async def on_interrupt(context=None):
    """Cancel active TTS stream on user interrupt."""
    log_id = None
    if context and hasattr(context, 'log_id'):
        log_id = context.log_id
    if log_id and log_id in _active_pacers:
        logger.info(f"on_interrupt: interrupting TTS for {log_id}")
        _active_pacers[log_id].interrupt()
