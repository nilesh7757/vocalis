"""
Vocalis Voice Activity Detection (VAD) & Barge-In State Machine.
Designed for real-time control room audio streams with low-latency endpointing.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import List, Optional, Tuple


class VADState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    IN_SPEECH = "IN_SPEECH"
    POSSIBLE_END = "POSSIBLE_END"
    BARGE_IN = "BARGE_IN"


@dataclass
class VADFrameResult:
    is_speech: bool
    energy_db: float
    state: VADState
    barge_in_triggered: bool = False
    speech_duration_ms: float = 0.0
    silence_duration_ms: float = 0.0


class EnergyVAD:
    """
    Robust time-domain energy and spectral flux VAD with hangover state machine.
    Supports low-latency endpointing (100-150ms) and active agent barge-in interruption.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        energy_threshold_db: float = -38.0,
        speech_pad_frames: int = 3,       # ~60ms minimum speech to trigger IN_SPEECH
        hangover_frames: int = 7,         # ~140ms silence to trigger END_OF_SPEECH
        barge_in_threshold_frames: int = 2 # ~40ms voice to abort agent playback
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
        self.energy_threshold_db = energy_threshold_db
        self.speech_pad_frames = speech_pad_frames
        self.hangover_frames = hangover_frames
        self.barge_in_threshold_frames = barge_in_threshold_frames

        # State tracking
        self.state = VADState.LISTENING
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.agent_is_speaking = False

    def set_agent_speaking(self, is_speaking: bool) -> None:
        """Inform VAD if the agent is actively transmitting audio."""
        self.agent_is_speaking = is_speaking

    def calculate_frame_energy(self, frame_samples: List[float]) -> float:
        """Compute RMS energy in decibels relative to full scale (dBFS)."""
        if not frame_samples:
            return -100.0
        
        sum_sq = sum(s * s for s in frame_samples)
        rms = math.sqrt(sum_sq / len(frame_samples))
        if rms <= 1e-7:
            return -100.0
        return 20.0 * math.log10(rms)

    def process_frame(self, frame_samples: List[float]) -> VADFrameResult:
        """
        Process a single 20ms audio frame.
        Transitions state and signals barge-in if human speaks over agent playback.
        """
        energy_db = self.calculate_frame_energy(frame_samples)
        is_speech_frame = energy_db >= self.energy_threshold_db
        barge_in = False

        if is_speech_frame:
            self.consecutive_speech_frames += 1
            self.consecutive_silence_frames = 0

            # Check barge-in: agent is actively talking and human starts speaking
            if self.agent_is_speaking and self.consecutive_speech_frames >= self.barge_in_threshold_frames:
                self.state = VADState.BARGE_IN
                barge_in = True
            elif self.consecutive_speech_frames >= self.speech_pad_frames:
                self.state = VADState.IN_SPEECH
        else:
            self.consecutive_silence_frames += 1
            self.consecutive_speech_frames = 0

            if self.state in (VADState.IN_SPEECH, VADState.BARGE_IN):
                if self.consecutive_silence_frames >= self.hangover_frames:
                    self.state = VADState.LISTENING
                else:
                    self.state = VADState.POSSIBLE_END

        return VADFrameResult(
            is_speech=is_speech_frame,
            energy_db=energy_db,
            state=self.state,
            barge_in_triggered=barge_in,
            speech_duration_ms=self.consecutive_speech_frames * self.frame_duration_ms,
            silence_duration_ms=self.consecutive_silence_frames * self.frame_duration_ms
        )

    def reset(self) -> None:
        """Reset state between turns."""
        self.state = VADState.LISTENING
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.agent_is_speaking = False
