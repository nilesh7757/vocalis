"""
First-Principles Evaluation Benchmark Framework for Voice Agents.
Calculates Word Error Rate (WER), Entity Error Rate (EER), Tool Accuracy,
and Latency Distributions (p50, p95, p99) across clean, telephony, and noisy channels.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Tuple


def levenshtein_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    """Calculate minimum edit distance (insertions, deletions, substitutions)."""
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1].lower() == hyp_tokens[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1]   # Substitution
                )
    return dp[m][n]


@dataclass
class BenchmarkReport:
    total_samples: int
    mean_wer: float
    entity_precision: float
    entity_recall: float
    entity_f1: float
    tool_selection_accuracy: float
    procedure_compliance_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    channel_stratification: Dict[str, Dict[str, float]] = field(default_factory=dict)


class VoiceAgentEvaluator:
    """
    Evaluator implementing Callease AI domain-specific metrics:
    Separates general WER from Entity Error Rate (EER) to safeguard high-stakes identifiers.
    """

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate (WER = (S + D + I) / N)."""
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        dist = levenshtein_distance(ref_words, hyp_words)
        return min(1.0, dist / len(ref_words))

    def compute_entity_metrics(
        self,
        ref_entities: List[str],
        hyp_entities: List[str]
    ) -> Tuple[float, float, float]:
        """
        Compute Precision, Recall, and F1 on critical security domain entities.
        EER = 1.0 - F1.
        """
        ref_set = set(e.lower() for e in ref_entities)
        hyp_set = set(e.lower() for e in hyp_entities)

        if not ref_set and not hyp_set:
            return 1.0, 1.0, 1.0
        if not ref_set or not hyp_set:
            return 0.0, 0.0, 0.0

        true_positives = len(ref_set.intersection(hyp_set))
        precision = true_positives / len(hyp_set) if hyp_set else 0.0
        recall = true_positives / len(ref_set) if ref_set else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def calculate_percentiles(self, latencies: List[float]) -> Tuple[float, float, float]:
        """Calculate p50, p95, and p99 from measured latency distribution."""
        if not latencies:
            return 0.0, 0.0, 0.0
        sorted_l = sorted(latencies)
        n = len(sorted_l)

        def get_p(p: float) -> float:
            k = (n - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_l[int(k)]
            d0 = sorted_l[int(f)] * (c - k)
            d1 = sorted_l[int(c)] * (k - f)
            return d0 + d1

        return get_p(0.50), get_p(0.95), get_p(0.99)

    def run_benchmark(self, test_cases: List[Dict]) -> BenchmarkReport:
        """
        Execute full evaluation suite across test interactions.
        """
        wer_scores = []
        precisions = []
        recalls = []
        tool_correct = 0
        compliance_correct = 0
        latencies = []
        channel_metrics: Dict[str, Dict[str, List[float]]] = {}

        for case in test_cases:
            ref = case["reference_text"]
            hyp = case["transcribed_text"]
            chan = case.get("channel", "clean_16khz")
            lat = case.get("latency_ms", 450.0)
            latencies.append(lat)

            # WER
            wer = self.compute_wer(ref, hyp)
            wer_scores.append(wer)

            # EER
            p, r, f1 = self.compute_entity_metrics(
                case.get("reference_entities", []),
                case.get("transcribed_entities", [])
            )
            precisions.append(p)
            recalls.append(r)

            # Tool & Procedure
            if case.get("tool_selected_correctly", True):
                tool_correct += 1
            if case.get("procedure_compliant", True):
                compliance_correct += 1

            if chan not in channel_metrics:
                channel_metrics[chan] = {"wer": [], "lat": []}
            channel_metrics[chan]["wer"].append(wer)
            channel_metrics[chan]["lat"].append(lat)

        total = len(test_cases)
        avg_wer = sum(wer_scores) / total if total > 0 else 0.0
        avg_p = sum(precisions) / total if total > 0 else 0.0
        avg_r = sum(recalls) / total if total > 0 else 0.0
        avg_f1 = (2.0 * avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

        p50, p95, p99 = self.calculate_percentiles(latencies)

        stratification = {}
        for c, data in channel_metrics.items():
            stratification[c] = {
                "mean_wer": sum(data["wer"]) / len(data["wer"]),
                "mean_latency_ms": sum(data["lat"]) / len(data["lat"])
            }

        return BenchmarkReport(
            total_samples=total,
            mean_wer=avg_wer,
            entity_precision=avg_p,
            entity_recall=avg_r,
            entity_f1=avg_f1,
            tool_selection_accuracy=tool_correct / total if total > 0 else 0.0,
            procedure_compliance_rate=compliance_correct / total if total > 0 else 0.0,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            channel_stratification=stratification
        )
