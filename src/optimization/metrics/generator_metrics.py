"""
Response Generator metrics for DSPy optimization.
Combines scope detection accuracy with answer quality using DSPy's SemanticF1.
"""

from typing import Any, Dict, List
import dspy
from dspy.evaluate import SemanticF1
from src.loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="generator-metrics")


class GeneratorMetric:
    """
    Combined metric for response generation.

    Evaluates two aspects:
    1. Scope detection: Does model correctly identify in-scope vs out-of-scope?
    2. Answer quality: For in-scope, how good is the answer? (using SemanticF1)

    Scope detection is critical - wrong scope = automatic failure (0.0 score)

    IMPORTANT: DSPy's SemanticF1 expects 'response' fields, not 'answer' fields.
    """

    def __init__(self, scope_weight: float = 0.5, quality_weight: float = 0.5) -> None:
        """
        Initialize metric with custom weights.

        Args:
            scope_weight: Weight for scope detection accuracy
            quality_weight: Weight for answer quality (SemanticF1)
        """
        self.scope_weight = scope_weight
        self.quality_weight = quality_weight

        # Initialize DSPy's native SemanticF1 with decompositional mode
        # This uses the configured LM to evaluate semantic similarity
        self.semantic_f1 = SemanticF1(decompositional=True)

        logger.info("Initialized GeneratorMetric with DSPy's native SemanticF1")

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any | None = None,
    ) -> float:
        """
        Evaluate generator prediction with combined metric.

        Args:
            example: Ground truth with 'should_be_in_scope' and 'expected_answer'
            prediction: Model prediction with 'questionOutOfLLMScope' and 'answer'
            trace: Optional trace information (ignored)

        Returns:
            Float score between 0.0 and 1.0
        """
        try:
            # Extract scope information
            expected_in_scope = example.should_be_in_scope
            predicted_out_of_scope = getattr(prediction, "questionOutOfLLMScope", None)

            if predicted_out_of_scope is None:
                logger.warning("Prediction missing 'questionOutOfLLMScope' field")
                return 0.0

            # Convert to consistent format
            predicted_in_scope = not predicted_out_of_scope

            # Check scope detection
            scope_correct = expected_in_scope == predicted_in_scope

            if not scope_correct:
                # Wrong scope = critical failure
                logger.debug(
                    f"Scope mismatch: expected={expected_in_scope}, predicted={predicted_in_scope}"
                )
                return 0.0

            # If out-of-scope and correctly detected, perfect score
            if not expected_in_scope:
                return 1.0

            # For in-scope questions, evaluate answer quality using SemanticF1
            expected_answer = example.expected_answer
            predicted_answer = getattr(prediction, "answer", "")

            if not predicted_answer:
                logger.warning("Prediction missing 'answer' field")
                return 0.5  # Correct scope but no answer

            try:
                question = getattr(example, "question", "")
                semantic_example = dspy.Example(
                    question=question,
                    response=expected_answer,
                ).with_inputs("question")

                semantic_prediction = dspy.Prediction(response=predicted_answer)

                quality_score = self.semantic_f1(semantic_example, semantic_prediction)

                # Ensure quality_score is a float (SemanticF1 returns float)
                quality_score = (
                    float(quality_score) if quality_score is not None else 0.0
                )

                logger.debug(f"SemanticF1 quality score: {quality_score:.3f}")

            except Exception as e:
                logger.warning(f"SemanticF1 evaluation failed: {e}, using fallback")
                # Fallback to simple string similarity
                quality_score = self._simple_similarity(
                    expected_answer, predicted_answer
                )

            # Combine scores (scope already correct at 1.0, so weight quality)
            final_score = self.scope_weight * 1.0 + self.quality_weight * quality_score

            return final_score

        except Exception as e:
            logger.error(f"Error in generator metric: {e}")
            return 0.0

    def _simple_similarity(self, expected: str, predicted: str) -> float:
        """
        Simple fallback similarity measure using Jaccard similarity.
        Only used if SemanticF1 fails.
        """
        expected_words = set(expected.lower().split())
        predicted_words = set(predicted.lower().split())

        if not expected_words or not predicted_words:
            return 0.0

        intersection = expected_words.intersection(predicted_words)
        union = expected_words.union(predicted_words)

        return len(intersection) / len(union) if union else 0.0


def combined_scope_and_quality_metric(
    example: dspy.Example, prediction: dspy.Prediction
) -> float:
    """
    Convenience function for combined scope and quality metric.

    This is the primary metric for generator optimization.
    Uses DSPy's native SemanticF1 for quality evaluation.
    """
    metric = GeneratorMetric()
    return metric(example, prediction)


class ScopeOnlyMetric:
    """
    Simplified metric that only evaluates scope detection.

    Useful for initial training phase or when answer quality is less critical.
    """

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
        """Evaluate only scope detection accuracy."""
        try:
            expected_in_scope = example.should_be_in_scope
            predicted_out_of_scope = getattr(prediction, "questionOutOfLLMScope", None)

            if predicted_out_of_scope is None:
                return 0.0

            predicted_in_scope = not predicted_out_of_scope

            return 1.0 if expected_in_scope == predicted_in_scope else 0.0

        except Exception as e:
            logger.error(f"Error in scope-only metric: {e}")
            return 0.0


def calculate_generator_stats(
    examples: List[dspy.Example], predictions: List[dspy.Prediction]
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for generator evaluation.

    Args:
        examples: Ground truth examples
        predictions: Model predictions

    Returns:
        Dictionary with evaluation statistics
    """
    try:
        if len(examples) != len(predictions):
            logger.error(
                f"Mismatch: {len(examples)} examples vs {len(predictions)} predictions"
            )
            return {
                "combined_score": 0.0,
                "scope_accuracy": 0.0,
                "in_scope_performance": 0.0,
                "out_scope_performance": 0.0,
                "error": "Length mismatch",
            }

        # Initialize counters
        total = len(examples)
        scope_correct = 0
        in_scope_correct = 0
        in_scope_total = 0
        out_scope_correct = 0
        out_scope_total = 0

        metric = GeneratorMetric()

        # Evaluate each example
        for example, prediction in zip(examples, predictions, strict=True):
            expected_in_scope = example.should_be_in_scope
            predicted_out_of_scope = getattr(prediction, "questionOutOfLLMScope", None)

            if predicted_out_of_scope is None:
                continue

            predicted_in_scope = not predicted_out_of_scope

            # Track scope detection
            if expected_in_scope == predicted_in_scope:
                scope_correct += 1

            # Track performance by category
            if expected_in_scope:
                in_scope_total += 1
                score = metric(example, prediction)
                if score > 0.5:  # Consider >0.5 as "correct"
                    in_scope_correct += 1
            else:
                out_scope_total += 1
                if (
                    predicted_in_scope == expected_in_scope
                ):  # Correctly identified as out-of-scope
                    out_scope_correct += 1

        # Calculate statistics
        scope_accuracy = scope_correct / total if total > 0 else 0.0
        in_scope_performance = (
            in_scope_correct / in_scope_total if in_scope_total > 0 else 0.0
        )
        out_scope_performance = (
            out_scope_correct / out_scope_total if out_scope_total > 0 else 0.0
        )

        # Combined score (weighted average)
        combined_score = (
            0.5 * scope_accuracy
            + 0.3 * in_scope_performance
            + 0.2 * out_scope_performance
        )

        stats = {
            "combined_score": combined_score,
            "scope_accuracy": scope_accuracy,
            "in_scope_performance": in_scope_performance,
            "out_scope_performance": out_scope_performance,
            "total_examples": total,
            "in_scope_examples": in_scope_total,
            "out_scope_examples": out_scope_total,
        }

        logger.debug(f"Generator stats: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Error calculating generator stats: {e}")
        return {
            "combined_score": 0.0,
            "scope_accuracy": 0.0,
            "in_scope_performance": 0.0,
            "out_scope_performance": 0.0,
            "error": str(e),
        }
