# mr_qwen3tts

Qwen3-TTS streaming TTS plugin for MindRoot - drop-in replacement for `mr_eleven_stream`.

## Overview

This plugin provides text-to-speech using Qwen3-TTS with multiple backend options:

- **openai (default)** - groxaxo OpenAI-FastAPI server - lowest latency (~97ms TTFA), recommended for production
- **websocket** - custom Qwen3-TTS WebSocket server - voice cloning with auto-transcription
- **vllm** - vllm-omni HTTP API - integrated with vLLM inference engine

All backends output streaming ulaw 8kHz audio suitable for SIP/telephony.

## Installation

```bash
pip install -e .
```

## Backend Configuration

Set `MR_QWEN3TTS_BACKEND` to select the implementation:

```bash
# Default: groxaxo OpenAI-FastAPI (recommended, lowest latency)
MR_QWEN3TTS_BACKEND=openai
MR_QWEN3TTS_OPENAI_URL=http://localhost:8880
MR_QWEN3TTS_VOICE=Vivian

# Alternative: WebSocket server
MR_QWEN3TTS_BACKEND=websocket
MR_QWEN3TTS_WS_URL=ws://localhost:8765

# Alternative: vllm-omni
MR_QWEN3TTS_BACKEND=vllm
```

## Voice Configuration (OpenAI Backend)

### Built-in Voices

The groxaxo server includes built-in voices like `Vivian`, `Ethan`, etc.

### Voice Library (Recommended for Production)

Register voice profiles on the server, then use `clone:VoiceName`:

```bash
MR_QWEN3TTS_VOICE=clone:Alice
```

The server caches speaker embeddings, saving ~0.7s per repeated clone request.

### Auto-Registration from URL (Recommended)

Set `voice_id` to a URL pointing to a reference audio file. The plugin will automatically:

1. Register the voice with the groxaxo server on first use
2. Cache the URL -> `clone:Name` mapping locally
3. Use `clone:Name` for all subsequent requests (no re-registration)

```json
{
  "name": "MyAgent",
  "persona": {
    "voice_id": "https://example.com/agent_voices/alice.wav"
  }
}
```

The first utterance from a new voice has ~1-2s overhead for registration.
All subsequent utterances use the cached profile with no overhead.

This requires the groxaxo server to be running with the voice registration
router mounted (the `run_groxaxo.py` wrapper handles this automatically).

### Per-Agent Voice

Set `voice_id` in the agent's persona:

```json
{
  "name": "MyAgent",
  "persona": {
    "voice_id": "clone:Alice"
  }
}
```

## Voice Configuration (WebSocket Backend)

### Option 1: Per-Agent Voice (Recommended)

Set the `voice_id` in the agent's persona to an **absolute path** to a reference audio file:

```json
{
  "name": "MyAgent",
  "persona": {
    "voice_id": "/path/to/voice_sample.wav"
  }
}
```

The server will automatically transcribe the audio using Whisper.

### Option 2: Default Voice (Fallback)

Set environment variables for a default voice:

```bash
MR_QWEN3TTS_REF_AUDIO=/path/to/default_voice.wav
MR_QWEN3TTS_REF_TEXT="Optional transcript"  # Will auto-transcribe if not set
```

## Environment Variables

```bash
# Backend selection
MR_QWEN3TTS_BACKEND=openai          # openai (default) | websocket | vllm

# OpenAI backend settings
MR_QWEN3TTS_OPENAI_URL=http://localhost:8880  # groxaxo server URL
MR_QWEN3TTS_VOICE=Vivian            # Default voice name
MR_QWEN3TTS_MODEL=qwen3-tts         # Model name for API

# WebSocket backend settings
MR_QWEN3TTS_WS_URL=ws://localhost:8765
MR_QWEN3TTS_REF_AUDIO=/path/to/default_voice.wav
MR_QWEN3TTS_REF_TEXT="Optional transcript"

# Common settings
MR_QWEN3TTS_LANGUAGE=Auto           # Auto, Chinese, English, Japanese, Korean, etc.
MR_QWEN3TTS_REALTIME_STREAM=1        # Enable realtime streaming
```

## Usage

### As a Command

```json
{ "speak": { "text": "Hello, this is a test message" } }
```

### As a Service

```python
from lib.providers.services import service_manager

async for chunk in service_manager.stream_tts(text="Hello world", context=context):
    # chunk is ulaw 8kHz audio (160 bytes = 20ms)
    await send_to_sip(chunk)
```

## Container Setup

The RunPod container supports multiple TTS backends via `TTS_BACKEND` env var:

```bash
# Default: groxaxo OpenAI-FastAPI (lowest latency)
TTS_BACKEND=qwen3tts_openai

# Alternative: vllm-omni
TTS_BACKEND=qwen3tts

# Alternative: Custom WebSocket server
TTS_BACKEND=qwen3tts_custom

# Alternative: CosyVoice3
TTS_BACKEND=cosyvoice3
```

## Latency Comparison

| Backend | TTFA | RTF | Notes |
|--------|------|-----|-------|
| openai (groxaxo) | ~97ms | 0.65-0.70 | Optimized backend with torch.compile + CUDA graphs |
| vllm-omni | ~205ms | ~0.83 | Integrated with vLLM |
| websocket | ~100ms+ | varies | Custom server, voice cache helps |

Note: First 2-3 requests with the openai backend are slow (~10-30s) during torch.compile warmup.
Set `TTS_WARMUP_ON_START=true` to warm up at container start.

## API Compatibility

This plugin is designed as a drop-in replacement for `mr_eleven_stream`:
- Same `speak` command interface
- Same `stream_tts` service interface  
- Same audio output format (ulaw 8kHz)
- Same realtime streaming support via `partial_command` pipe

## GPU Memory (H200)

| Component | Memory |
|-----------|--------|
| LLM (Qwen3.5-35B-A3B) | ~122 GiB |
| Qwen3-TTS 0.6B | ~4 GiB |
| Qwen3-TTS 1.7B | ~8 GiB |
| Free buffer | ~5-21 GiB |

The groxaxo server manages its own GPU memory via PyTorch/transformers (not vllm),
so no `gpu_memory_utilization` setting is needed.
