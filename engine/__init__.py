"""
Vocalis Core Engine Package.
Real-Time Streaming Voice Agent & ASR Evaluation Engine.
"""

from .vad import EnergyVAD, VADState, VADFrameResult
from .asr_pipeline import StreamingASRPipeline, TranscriptionResult
from .agent_runtime import SecurityAgentRuntime, AgentTurnResponse, ToolCallResult
from .audio_streamer import AudioStreamer, TurnTakingMetrics, AudioChunk
from .evaluation import VoiceAgentEvaluator, BenchmarkReport

__all__ = [
    "EnergyVAD",
    "VADState",
    "VADFrameResult",
    "StreamingASRPipeline",
    "TranscriptionResult",
    "SecurityAgentRuntime",
    "AgentTurnResponse",
    "ToolCallResult",
    "AudioStreamer",
    "TurnTakingMetrics",
    "AudioChunk",
    "VoiceAgentEvaluator",
    "BenchmarkReport",
]
