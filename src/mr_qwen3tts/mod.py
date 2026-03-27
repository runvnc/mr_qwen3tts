"""
mr_qwen3tts - Qwen3-TTS streaming TTS plugin for MindRoot.

Drop-in replacement for mr_eleven_stream that connects to a vllm-omni
server's /v1/audio/speech endpoint for streaming TTS.

Streaming: HTTP chunked transfer, stream=true, response_format=pcm
Output: raw 16-bit signed PCM at 24kHz, converted to ulaw 8kHz for SIP.
"""

import os
import asyncio
import audioop
import logging
import time
from typing import AsyncGenerator, Optional, Dict, Any

import dotenv
dotenv.load_dotenv()

import httpx

from lib.providers.services import service, service_manager
from lib.providers.commands import command
from lib.providers.hooks import hook
from .audio_pacer import AudioPacer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.environ.get('MR_QWEN3TTS_API_URL', 'http://localhost:8091')
VOICE = os.environ.get('MR_QWEN3TTS_VOICE', 'vivian')
LANGUAGE = os.environ.get('MR_QWEN3TTS_LANGUAGE', 'Auto')

# PCM conversion state (audioop.ratecv is stateful)
_RESAMPLE_STATE = None

# Active AudioPacer instances per log_id (for interrupt support)
_active_pacers: Dict[str, Any] = {}
# Per-session speak() locks
_active_speak_locks: Dict[str, asyncio.Lock] = {}


# ---------------------------------------------------------------------------
# PCM 24kHz -> ulaw 8kHz conversion
# ---------------------------------------------------------------------------

def _pcm24k_to_ulaw8k(pcm_bytes: bytes, state) -> tuple[bytes, Any]:
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

async def _stream_tts_http(
    text: str,
    voice: str = None,
    language: str = None,
) -> AsyncGenerator[bytes, None]:
    """
    Stream ulaw 8kHz audio chunks from vllm-omni /v1/audio/speech.

    vllm-omni streams raw 16-bit signed PCM at 24kHz when stream=true
    and response_format=pcm. We resample and encode to ulaw 8kHz on the fly.
    Yields 160-byte chunks (20ms at 8kHz) suitable for SIP.
    """
    url = f"{API_URL.rstrip('/')}/v1/audio/speech"
    payload = {
        "input": text,
        "voice": voice or VOICE,
        "language": language or LANGUAGE,
        "stream": True,
        "response_format": "pcm",
    }

    resample_state = None
    ulaw_buffer = b''
    CHUNK_SIZE = 160  # 20ms at 8kHz ulaw
    t0 = time.time()
    first_chunk_logged = False

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)) as client:
        async with client.stream('POST', url, json=payload) as response:
            response.raise_for_status()

            async for pcm_chunk in response.aiter_bytes(chunk_size=4096):
                if not pcm_chunk:
                    continue

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
    """Stream text-to-speech audio using vllm-omni Qwen3-TTS."""
    voice = voice_id or VOICE
    logger.info(f"stream_tts: '{text[:60]}...' voice={voice}")
    async for chunk in _stream_tts_http(text=text, voice=voice):
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
    """Convert text to speech using vllm-omni Qwen3-TTS streaming."""
    log_id = None
    if context and hasattr(context, 'log_id'):
        log_id = context.log_id

    try:
        # Per-session serialization
        if log_id:
            if log_id not in _active_speak_locks:
                _active_speak_locks[log_id] = asyncio.Lock()
            lock = _active_speak_locks[log_id]
            if lock.locked():
                logger.warning(f"speak() already running for {log_id}, skipping")
                return "ERROR: Speech already in progress."
            await lock.acquire()

        voice = voice_id or VOICE
        local_playback = service_manager.functions.get('sip_audio_out_chunk', None) is None

        if not local_playback:
            try:
                is_halted = await service_manager.sip_is_audio_halted(context=context)
                if is_halted:
                    logger.info("Audio halted, skipping speak")
                    return None
            except Exception:
                pass

        pacer = None
        if not local_playback:
            pacer = AudioPacer(sample_rate=8000)
            if log_id:
                _active_pacers[log_id] = pacer

            async def send_to_sip(chunk, timestamp=None, context=None):
                try:
                    return await service_manager.sip_audio_out_chunk(chunk, timestamp=timestamp, context=context)
                except Exception as e:
                    logger.error(f"Error sending to SIP: {e}")
                    return False

            await pacer.start_pacing(send_to_sip, context)

        local_audio_buffer = b'' if local_playback else None
        chunk_count = 0

        async for chunk in _stream_tts_http(text=text, voice=voice):
            chunk_count += 1
            if local_playback:
                local_audio_buffer += chunk
            else:
                if pacer.interrupted:
                    logger.info(f"speak() interrupted after {chunk_count} chunks")
                    break
                await pacer.add_chunk(chunk)

        if not local_playback and pacer:
            pacer.mark_finished()
            if not pacer.interrupted:
                await pacer.wait_until_done()
            await pacer.stop()
            if log_id and log_id in _active_pacers:
                del _active_pacers[log_id]
            if pacer.interrupted and chunk_count < 2:
                return "SYSTEM: WARNING - Command interrupted!\n\n"

        logger.info(f"speak() complete: {len(text)} chars, {chunk_count} chunks")
        return None

    except Exception as e:
        logger.error(f"Error in speak: {e}")
        return None

    finally:
        if log_id and log_id in _active_speak_locks:
            lock = _active_speak_locks[log_id]
            if lock.locked():
                lock.release()


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
