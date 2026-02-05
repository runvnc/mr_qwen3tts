"""mr_qwen3tts - Qwen3-TTS streaming plugin for MindRoot.

Drop-in replacement for mr_eleven_stream using Qwen3-TTS via remote WebSocket server.
"""

from .mod import speak, stream_tts

__all__ = ["speak", "stream_tts"]
