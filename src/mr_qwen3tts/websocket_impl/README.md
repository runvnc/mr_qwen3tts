# WebSocket Implementation (Legacy)

This directory contains the original implementation of `mr_qwen3tts` that connects
to the custom `qwen3tts-server` WebSocket server (at `/files/qwen3tts`).

That server runs Qwen3-TTS directly via the Hugging Face transformers pipeline
and exposes a WebSocket API with voice cloning support (ref_audio / ref_text).

## Files

- `mod.py` - Main plugin module with WebSocket client, voice warmup, realtime streaming
- `realtime_stream.py` - Realtime incremental text streaming support
- `audio_pacer.py` - Audio pacing for SIP output timing

## Why replaced

The main `mod.py` was rewritten to use the **vllm-omni** HTTP API instead
(`/v1/audio/speech` on port 8091). This gives:
- Standard OpenAI-compatible API
- Better throughput via vLLM's batching engine
- No separate WebSocket server to maintain
- Predefined voices (vivian, ryan, aiden, etc.) via CustomVoice model

The tradeoff is no voice cloning (CustomVoice uses predefined speakers).
If you need voice cloning, use the Base model variant and switch back to
this WebSocket implementation or adapt the HTTP client to use `task_type=Base`
with `ref_audio`.

## To restore

Copy `mod.py` from this directory back to the parent and update `plugin_info.json`
env vars to include `MR_QWEN3TTS_WS_URL` etc.
