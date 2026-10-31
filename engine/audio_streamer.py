"""
Audio Streamer, Chunk Synthesizer & Turn-Taking Latency Tracker.
Optimizes Time-to-First-Audio (TTFA) via token-to-speech streaming chunk buffers.
"""

from dataclasses import dataclass
import time
from typing import Generator, List, Optional


@dataclass
class AudioChunk:
    chunk_index: int
    duration_ms: float
    is_first_chunk: bool
    synthesized_text: str


@dataclass
class TurnTakingMetrics:
    vad_endpoint_latency_ms: float
    asr_latency_ms: float
    llm_ttft_ms: float
    tts_first_chunk_latency_ms: float
    total_time_to_first_audio_ms: float
    barge_in_cancelled: bool = False


class AudioStreamer:
    """
    Simulates streaming neural TTS (e.g., Kokoro / FastSpeech / Whisper-TTS)
    piped directly over WebSocket/WebRTC audio tracks.
    """

    def __init__(self, tts_first_chunk_delay_ms: float = 85.0):
        self.tts_first_chunk_delay_ms = tts_first_chunk_delay_ms
        self.is_streaming = False
        self.active_chunks: List[AudioChunk] = []

    def stream_speech_chunks(
        self,
        token_stream: Generator[str, None, any],
        vad_endpoint_latency_ms: float = 120.0,
        asr_latency_ms: float = 140.0,
        barge_in_signal: bool = False
    ) -> TurnTakingMetrics:
        """
        Stream tokens from LLM directly into audio chunk synthesizer.
        Computes the complete conversational turn-taking latency budget.
        """
        start_time = time.perf_counter()
        first_token_received = False
        first_audio_synthesized = False
        llm_ttft = 95.0
        cancelled = False

        chunk_idx = 0
        current_phrase = []

        for token in token_stream:
            # Check for immediate barge-in interruption
            if barge_in_signal:
                cancelled = True
                break

            current_phrase.append(token)
            if not first_token_received:
                first_token_received = True

            # Synthesize upon reaching clause boundary (comma, period, or 3 tokens)
            if any(punct in token for punct in [",", ".", "!"]) or len(current_phrase) >= 3:
                phrase_text = "".join(current_phrase)
                is_first = (chunk_idx == 0)
                chunk = AudioChunk(
                    chunk_index=chunk_idx,
                    duration_ms=len(phrase_text) * 45.0,  # ~45ms audio per syllable/char
                    is_first_chunk=is_first,
                    synthesized_text=phrase_text
                )
                self.active_chunks.append(chunk)
                chunk_idx += 1
                current_phrase = []

                if is_first:
                    first_audio_synthesized = True

        total_ttfa = vad_endpoint_latency_ms + asr_latency_ms + llm_ttft + self.tts_first_chunk_delay_ms

        return TurnTakingMetrics(
            vad_endpoint_latency_ms=vad_endpoint_latency_ms,
            asr_latency_ms=asr_latency_ms,
            llm_ttft_ms=llm_ttft,
            tts_first_chunk_latency_ms=self.tts_first_chunk_delay_ms,
            total_time_to_first_audio_ms=total_ttfa,
            barge_in_cancelled=cancelled
        )
