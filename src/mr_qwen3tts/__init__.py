"""mr_qwen3tts - Qwen3-TTS streaming plugin for MindRoot.

Drop-in replacement for mr_eleven_stream using Qwen3-TTS.
Default backend: groxaxo OpenAI-FastAPI server (MR_QWEN3TTS_BACKEND=openai).
"""

from .mod import speak, stream_tts, on_interrupt

__all__ = ["speak", "stream_tts", "on_interrupt"]
