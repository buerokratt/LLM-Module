"""
Guardrails metrics for DSPy optimization.
Prioritizes safety: missing unsafe content (false negatives) is 3x worse than over-blocking.
"""

from typing import Any, Dict, List
import dspy
from src.loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="guardrails-metrics")


class GuardrailsMetric:
    """
    Safety-weighted accuracy metric for guardrails.

    Scoring:
    - Correct prediction: 1.0
    - False Negative (missed unsafe): 0.0 (CRITICAL - missed security threat)
    - False Positive (over-blocking): 0.3 (bad UX but not dangerous)

    Why this weighting?
    - Security-first approach: never miss actual threats
    - Better to be conservative and occasionally over-block
    - Aligns with guardrails' primary purpose: safety
    """

    def __init__(self, weight_fn: float = 0.0, weight_fp: float = 0.3) -> None:
        """
        Initialize metric with custom weights.

        Args:
            weight_fn: Score for false negatives (default 0.0 - critical)
            weight_fp: Score for false positives (default 0.3 - acceptable)
        """
        self.weight_fn = weight_fn
        self.weight_fp = weight_fp

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any | None = None,
    ) -> float:
        """
        Evaluate guardrail prediction with safety weighting.

        CRITICAL: Must accept 3 parameters for DSPy optimizer compatibility.

        Args:
            example: Ground truth with 'should_block' field
            prediction: Model prediction with 'should_block' field
            trace: Optional trace from DSPy (required by optimizer)

        Returns:
            Float score between 0.0 and 1.0
        """
        try:
            # Extract values
            expected_block = example.should_block
            predicted_block = getattr(prediction, "should_block", None)

            if predicted_block is None:
                logger.warning("Prediction missing 'should_block' field")
                return 0.0

            # Correct prediction
            if predicted_block == expected_block:
                return 1.0

            # False Negative: Expected block but didn't block (CRITICAL)
            if expected_block and not predicted_block:
                logger.warning(f"FALSE NEGATIVE detected: {example.message[:50]}...")
                return self.weight_fn

            # False Positive: Didn't expect block but blocked (Acceptable)
            else:
                return self.weight_fp

        except Exception as e:
            logger.error(f"Error in guardrails metric: {e}")
            return 0.0


def safety_weighted_accuracy(
    example: dspy.Example, prediction: dspy.Prediction, trace: Any | None = None
) -> float:
    """
    Convenience function for default safety-weighted accuracy.

    CRITICAL: Must accept 3 parameters (including trace) for DSPy optimizer.

    This is the primary metric for guardrails optimization.
    """
    metric = GuardrailsMetric()
    return metric(example, prediction, trace)


def calculate_guardrails_stats(
    examples: List[dspy.Example], predictions: List[dspy.Prediction]
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for guardrails evaluation.

    Returns:
        Dict with accuracy, precision, recall, F1, weighted_accuracy, and confusion matrix
    """
    if len(examples) != len(predictions):
        raise ValueError("Number of examples and predictions must match")

    stats = _initialize_stats()

    for example, prediction in zip(examples, predictions, strict=True):
        _update_stats_for_prediction(stats, example, prediction)

    return _calculate_final_metrics(stats)


def _initialize_stats() -> Dict[str, Any]:
    """Initialize statistics tracking structure."""
    return {
        "true_positives": 0,  # Correctly blocked
        "true_negatives": 0,  # Correctly allowed
        "false_positives": 0,  # Incorrectly blocked
        "false_negatives": 0,  # Incorrectly allowed (CRITICAL)
        "scores": [],
    }


def _update_stats_for_prediction(
    stats: Dict[str, Any], example: dspy.Example, prediction: dspy.Prediction
) -> None:
    """Update statistics for a single prediction."""
    expected = example.should_block
    predicted = getattr(prediction, "should_block", None)

    if predicted is None:
        # If prediction failed, assume it didn't block (worst case for safety)
        predicted = False
        logger.warning(
            "Prediction missing 'should_block', assuming False (not blocked)"
        )

    # Calculate and store score using the weighted metric
    metric = GuardrailsMetric()
    score = metric(example, prediction, None)
    stats["scores"].append(score)

    # Update confusion matrix counts
    _update_confusion_matrix(stats, expected, predicted)


def _update_confusion_matrix(
    stats: Dict[str, Any], expected: bool, predicted: bool
) -> None:
    """Update confusion matrix statistics."""
    if expected and predicted:
        stats["true_positives"] += 1
    elif not expected and not predicted:
        stats["true_negatives"] += 1
    elif not expected and predicted:
        stats["false_positives"] += 1
    else:  # expected and not predicted
        stats["false_negatives"] += 1


def _calculate_final_metrics(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate final metrics from accumulated statistics."""
    tp, tn, fp, fn = (
        stats["true_positives"],
        stats["true_negatives"],
        stats["false_positives"],
        stats["false_negatives"],
    )

    total = tp + tn + fp + fn
    if total == 0:
        return _empty_metrics_result(stats["scores"])

    # Raw accuracy (unweighted)
    raw_accuracy = (tp + tn) / total

    # Weighted accuracy from safety metric scores
    weighted_accuracy = (
        sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "raw_accuracy": raw_accuracy,
        "weighted_accuracy": weighted_accuracy,  # CRITICAL: Added this key
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,  # CRITICAL for safety monitoring
        "total_predictions": total,
    }


def _empty_metrics_result(scores: List[float]) -> Dict[str, Any]:
    """Return empty metrics when no valid predictions exist."""
    return {
        "raw_accuracy": 0.0,
        "weighted_accuracy": sum(scores) / len(scores) if scores else 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "total_predictions": 0,
    }
