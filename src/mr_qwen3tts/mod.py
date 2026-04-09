"""
mr_qwen3tts - Qwen3-TTS streaming TTS plugin for MindRoot.

Backend switcher: set MR_QWEN3TTS_BACKEND env var to select implementation.
  'openai' (default)    - groxaxo OpenAI-FastAPI server (http://localhost:8880/v1)
  'vllm'                - vllm-omni HTTP API (/v1/audio/speech on port 8091)
  'websocket'           - custom Qwen3-TTS WebSocket server (ws://)
"""

import os

_BACKEND = os.environ.get('MR_QWEN3TTS_BACKEND', 'openai').lower()

if _BACKEND == 'openai':
    from .openai_impl.mod import *
elif _BACKEND == 'vllm':
    from .vllm_impl.mod import *
else:
    from .websocket_impl.mod import *