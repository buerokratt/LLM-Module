"""
Prompt Refiner metrics for DSPy optimization using LLM-as-Judge.
Uses DSPy's native LLM judge for semantic evaluation of refinements.
"""

from typing import Any, Dict, List
import dspy
from src.loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="refiner-metrics")


class RefinementJudge(dspy.Signature):
    """
    Judge if a refined question preserves intent and improves clarity.

    This signature defines how the LLM should evaluate refinement quality.
    The LLM will use its understanding to score multiple dimensions.
    """

    original_question: str = dspy.InputField(
        desc="The original user question that needs refinement"
    )
    conversation_history: str = dspy.InputField(
        desc="Recent conversation context for disambiguation"
    )
    refined_question: str = dspy.InputField(
        desc="The refined version of the question to evaluate"
    )
    expected_refinement: str = dspy.InputField(
        desc="A high-quality expected refinement for comparison"
    )

    # Output fields that the LLM will produce
    preserves_intent: bool = dspy.OutputField(
        desc="Does the refinement preserve the original intent and meaning?"
    )
    improves_clarity: bool = dspy.OutputField(
        desc="Is the refined version clearer, more explicit, and searchable?"
    )
    quality_score: float = dspy.OutputField(
        desc="Overall quality score from 0.0 to 1.0 (0.0=poor, 1.0=excellent)"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the evaluation (1-2 sentences)"
    )


class RefinerMetric:
    """
    LLM-as-Judge metric for prompt refinement quality.

    Uses a DSPy ChainOfThought module to evaluate refinements across
    multiple dimensions: intent preservation, clarity improvement, and quality.

    This is Option B from the recommendations - full LLM judge with reasoning.
    """

    def __init__(self) -> None:
        """
        Initialize the LLM judge metric.

        The judge uses whatever LM is configured in dspy.settings.lm
        """
        # Create a DSPy judge module with chain-of-thought reasoning
        self.judge = dspy.ChainOfThought(RefinementJudge)

        logger.info(
            "Initialized RefinerMetric with LLM-as-Judge (ChainOfThought reasoning)"
        )

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any | None = None,
    ) -> float:
        """
        Evaluate refinement quality using LLM judge.

        Args:
            example: Ground truth with 'expected_refinements', 'question', 'history'
            prediction: Model prediction with 'rewrites' field
            trace: Optional trace information

        Returns:
            Float score between 0.0 and 1.0
        """
        try:
            # Extract refinements
            expected = example.expected_refinements
            predicted = getattr(prediction, "rewrites", None)

            if predicted is None or len(predicted) == 0:
                logger.warning("Prediction missing 'rewrites' field or empty")
                return 0.0

            if not expected or len(expected) == 0:
                logger.warning("Example missing 'expected_refinements' or empty")
                return 0.0

            # Get original question and history
            original_question = getattr(
                example, "question", getattr(example, "original_question", "")
            )
            history = getattr(example, "history", "")

            # Evaluate top N predictions (default: top 3)
            num_to_evaluate = min(3, len(predicted))
            scores = []

            for i, pred_rewrite in enumerate(predicted[:num_to_evaluate]):
                # Use the first expected refinement as the gold standard
                # (or you could compare against all and take best match)
                best_expected = expected[0] if expected else pred_rewrite

                try:
                    # Call the LLM judge
                    judgment = self.judge(
                        original_question=original_question,
                        conversation_history=history,
                        refined_question=str(pred_rewrite),
                        expected_refinement=best_expected,
                    )

                    # Extract scores from judgment
                    intent_score = 1.0 if judgment.preserves_intent else 0.0
                    clarity_score = 1.0 if judgment.improves_clarity else 0.0
                    quality_score = float(judgment.quality_score)

                    # Ensure quality_score is in valid range
                    quality_score = max(0.0, min(1.0, quality_score))

                    # Combine scores with weights
                    # - Intent preservation is critical (30%)
                    # - Clarity improvement is important (30%)
                    # - Overall quality from LLM is most important (40%)
                    combined_score = (
                        0.3 * intent_score + 0.3 * clarity_score + 0.4 * quality_score
                    )

                    scores.append(combined_score)

                    logger.debug(
                        f"Refinement {i + 1}: intent={intent_score:.1f}, "
                        f"clarity={clarity_score:.1f}, quality={quality_score:.2f}, "
                        f"combined={combined_score:.3f}"
                    )
                    logger.debug(f"Judge reasoning: {judgment.reasoning}")

                except Exception as e:
                    logger.warning(f"Judge failed for refinement {i + 1}: {e}")
                    scores.append(0.0)

            # Return average score across evaluated refinements
            final_score = sum(scores) / len(scores) if scores else 0.0

            logger.debug(
                f"RefinerMetric final score: {final_score:.3f} "
                f"(avg of {len(scores)} refinements)"
            )

            return final_score

        except Exception as e:
            logger.error(f"Error in refiner LLM judge metric: {e}")
            return 0.0


def llm_judge_refinement_metric(
    example: dspy.Example, prediction: dspy.Prediction
) -> float:
    """
    Convenience function for LLM judge refinement metric.

    This is the primary metric for refiner optimization using LLM-as-Judge.
    """
    metric = RefinerMetric()
    return metric(example, prediction)


class SimpleLLMJudge(dspy.Signature):
    """
    Simplified LLM judge for faster evaluation.

    Only outputs a single quality score without detailed reasoning.
    Use this if you need faster optimization runs.
    """

    original_question: str = dspy.InputField()
    refined_question: str = dspy.InputField()
    expected_refinement: str = dspy.InputField()

    quality_score: float = dspy.OutputField(desc="Quality score from 0.0 to 1.0")


class FastRefinerMetric:
    """
    Faster LLM judge metric without chain-of-thought reasoning.

    Uses direct prediction instead of ChainOfThought for speed.
    Trade-off: faster but potentially less accurate.
    """

    def __init__(self) -> None:
        self.judge = dspy.Predict(SimpleLLMJudge)
        logger.info("Initialized FastRefinerMetric with simple LLM judge")

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any | None = None,
    ) -> float:
        """Evaluate using fast LLM judge."""
        try:
            expected = example.expected_refinements
            predicted = getattr(prediction, "rewrites", [])

            if not predicted or not expected:
                return 0.0

            original = getattr(
                example, "question", getattr(example, "original_question", "")
            )

            scores = []
            for pred in predicted[:2]:  # Evaluate only top 2 for speed
                try:
                    judgment = self.judge(
                        original_question=original,
                        refined_question=str(pred),
                        expected_refinement=expected[0],
                    )
                    score = max(0.0, min(1.0, float(judgment.quality_score)))
                    scores.append(score)
                except (ValueError, AttributeError, TypeError) as e:
                    logger.debug(f"Error evaluating prediction: {e}")
                    scores.append(0.0)

            return sum(scores) / len(scores) if scores else 0.0

        except Exception as e:
            logger.error(f"Error in fast refiner metric: {e}")
            return 0.0


def calculate_refiner_stats(
    examples: List[dspy.Example],
    predictions: List[dspy.Prediction],
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for refiner evaluation.

    Args:
        examples: Ground truth examples
        predictions: Model predictions
        use_llm_judge: Use LLM judge (True) or fast version (False)

    Returns:
        Dict with scores and statistics
    """
    if len(examples) != len(predictions):
        raise ValueError("Number of examples and predictions must match")

    # Choose metric based on flag
    if use_llm_judge:
        metric = RefinerMetric()
        metric_name = "LLM Judge (ChainOfThought)"
    else:
        metric = FastRefinerMetric()
        metric_name = "Fast LLM Judge"

    logger.info(f"Calculating refiner stats using: {metric_name}")

    scores = []
    refinement_counts = []

    for example, prediction in zip(examples, predictions, strict=True):
        score = metric(example, prediction)
        scores.append(score)

        # Track number of refinements generated
        predicted = getattr(prediction, "rewrites", [])
        refinement_counts.append(len(predicted) if predicted else 0)

    sorted_scores = sorted(scores)
    median_idx = len(sorted_scores) // 2

    return {
        "average_quality": sum(scores) / len(scores) if scores else 0.0,
        "median_quality": sorted_scores[median_idx] if scores else 0.0,
        "min_quality": min(scores) if scores else 0.0,
        "max_quality": max(scores) if scores else 0.0,
        "avg_refinements_per_question": sum(refinement_counts) / len(refinement_counts)
        if refinement_counts
        else 0.0,
        "total_examples": len(examples),
        "metric_type": metric_name,
        "scores": scores,
    }


# Optional: Fallback to simple similarity if LLM judge fails
class FallbackRefinerMetric:
    """
    Fallback metric using simple string matching.

    Only use this if LLM judge completely fails or for quick sanity checks.
    """

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
        """Simple matching metric for refinements."""
        try:
            expected = example.expected_refinements
            predicted = getattr(prediction, "rewrites", [])

            if not predicted or not expected:
                return 0.0

            # Extract key terms from expected (words longer than 3 chars)
            key_terms = set()
            for exp in expected:
                words = str(exp).split()
                key_terms.update([w.lower() for w in words if len(w) > 3])

            # Check how many key terms appear in predictions
            matches = 0
            for pred in predicted:
                pred_words = set(str(pred).lower().split())
                overlap = key_terms.intersection(pred_words)
                if len(overlap) > 0:
                    matches += len(overlap)

            # Normalize by number of key terms
            score = min(1.0, matches / len(key_terms)) if key_terms else 0.0

            return score

        except Exception as e:
            logger.error(f"Error in fallback refiner metric: {e}")
            return 0.0
