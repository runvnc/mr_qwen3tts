"""
mr_qwen3tts - Qwen3-TTS streaming TTS plugin for MindRoot.

Backend switcher: set MR_QWEN3TTS_BACKEND env var to select implementation.
  'websocket' (default) - custom Qwen3-TTS WebSocket server (ws://)
  'vllm'               - vllm-omni HTTP API (/v1/audio/speech)
"""

import os

_BACKEND = os.environ.get('MR_QWEN3TTS_BACKEND', 'websocket').lower()

if _BACKEND == 'vllm':
    from .vllm_impl.mod import *
else:
    from .websocket_impl.mod import *
