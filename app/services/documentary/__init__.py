"""Documentary Mode services.

This package is intentionally isolated from the legacy MoneyPrinterTurbo short-video
pipeline. Shared lower-level services (TTS, subtitles, FFmpeg/MoviePy, LLMs) are
reused from app.services as the documentary workflow is implemented.
"""
