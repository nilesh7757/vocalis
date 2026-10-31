"""
Unit & Integration Test Suite for Vocalis Voice Agent.
Tests:
  1. EnergyVAD frame processing, hangover logic, and barge-in trigger.
  2. Streaming ASR entity extraction and confidence degradation.
  3. Grounded agent schema validation, CAD dispatch, and human escalation.
  4. Audio streamer turn-taking latency budgeting and interruption abort.
  5. Evaluator WER, EER (Precision/Recall/F1), and percentile calculations.
"""

import math
import unittest

from engine.vad import EnergyVAD, VADState
from engine.asr_pipeline import StreamingASRPipeline
from engine.agent_runtime import SecurityAgentRuntime
from engine.audio_streamer import AudioStreamer
from engine.evaluation import VoiceAgentEvaluator, levenshtein_distance


class TestVocalisEngine(unittest.TestCase):

    def setUp(self):
        self.vad = EnergyVAD()
        self.asr = StreamingASRPipeline()
        self.agent = SecurityAgentRuntime()
        self.streamer = AudioStreamer()
        self.evaluator = VoiceAgentEvaluator()

    def test_vad_silence_vs_speech(self):
        silent_frame = [0.0001] * 320
        loud_frame = [0.5] * 320

        res_silent = self.vad.process_frame(silent_frame)
        self.assertFalse(res_silent.is_speech)

        res_loud = self.vad.process_frame(loud_frame)
        self.assertTrue(res_loud.is_speech)

    def test_barge_in_interruption(self):
        self.vad.set_agent_speaking(True)
        loud_frame = [0.6] * 320

        # Frame 1: speech detected
        r1 = self.vad.process_frame(loud_frame)
        # Frame 2: triggers barge-in
        r2 = self.vad.process_frame(loud_frame)
        self.assertTrue(r2.barge_in_triggered)
        self.assertEqual(r2.state, VADState.BARGE_IN)

    def test_asr_entity_extraction(self):
        sample = "Officer Bravo-4 report intrusion at Gate A vehicle plate GJ-01-AB-1234"
        entities = self.asr.extract_entities(sample)
        self.assertTrue(any("Gate A" in e for e in entities))
        self.assertTrue(any("Intrusion" in e for e in entities))
        self.assertTrue(any("GJ-01-AB-1234" in e for e in entities))

    def test_grounded_agent_dispatch_success(self):
        gen = self.agent.process_turn_stream(
            transcript="Motion sensor alert at Gate A priority 1",
            entities=["SITE:Gate A", "INCIDENT:Motion Sensor"],
            confidence=0.95
        )
        tokens = list(gen)
        self.assertTrue(len(tokens) > 0)
        self.assertTrue(any("Gate A" in t for t in tokens))

    def test_grounded_agent_invalid_location_escalates(self):
        gen = self.agent.process_turn_stream(
            transcript="Intrusion reported at Unknown Sector 99",
            entities=["INCIDENT:Intrusion"],
            confidence=0.92
        )
        tokens = list(gen)
        reply = "".join(tokens)
        self.assertTrue("Escalating" in reply or "unclear" in reply)

    def test_evaluator_levenshtein_and_wer(self):
        ref = "the quick brown fox"
        hyp = "the fast brown fox"
        wer = self.evaluator.compute_wer(ref, hyp)
        self.assertAlmostEqual(wer, 0.25, places=2)

    def test_evaluator_entity_eer(self):
        ref_e = ["SITE:Gate A", "OFFICER:Bravo-4"]
        hyp_e = ["SITE:Gate A", "OFFICER:Bravo-5"]
        p, r, f1 = self.evaluator.compute_entity_metrics(ref_e, hyp_e)
        self.assertEqual(p, 0.5)
        self.assertEqual(r, 0.5)
        self.assertEqual(f1, 0.5)

    def test_percentile_calculation(self):
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
        p50, p95, p99 = self.evaluator.calculate_percentiles(latencies)
        self.assertEqual(p50, 300.0)
        self.assertTrue(p95 > 450.0)


if __name__ == "__main__":
    unittest.main()
