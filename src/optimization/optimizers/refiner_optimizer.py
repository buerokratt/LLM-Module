"""
Prompt Refiner optimizer using Bootstrap + MIPROv2 with LLM-as-Judge metric.
Uses DSPy's native LLM judge for semantic evaluation of refinement quality.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

import dspy
from src.loki_logger import LokiLogger
from optimization.metrics.refiner_metrics import (
    RefinerMetric,
    calculate_refiner_stats,
)

# Initialize Loki logger
logger = LokiLogger(service_name="refiner-optimizer")


class PromptRefinerSignature(dspy.Signature):
    """
    Produce N distinct, concise rewrites of the user's question using chat history.

    Constraints:
    - Preserve the original intent
    - Resolve pronouns with context when safe
    - Prefer explicit, searchable phrasing (entities, dates, units)
    - Make each rewrite meaningfully distinct
    - Return exactly N items as a list
    """

    history: str = dspy.InputField(desc="Recent conversation history")
    question: str = dspy.InputField(desc="The user's latest question to refine")
    n: int = dspy.InputField(desc="Number of rewrites to produce")

    rewrites: list[str] = dspy.OutputField(
        desc="Exactly N refined variations of the question"
    )


def load_refiner_data(split: str = "train") -> list[dspy.Example]:
    """Load refiner dataset."""
    data_path = Path(__file__).parent.parent / "optimization_data" / "refiner" / split

    file_map = {"train": "refiner_train.json", "val": "refiner_val.json"}

    filepath = data_path / file_map[split]

    logger.info(f"Loading refiner {split} data from {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        # Format conversation history
        history_str = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in item.get("conversation_history", [])
        )

        example = dspy.Example(
            history=history_str,
            question=item["original_question"],
            n=len(item["expected_refinements"]),
            expected_refinements=item["expected_refinements"],
            rewrites=item["expected_refinements"],  # For training
        ).with_inputs("history", "question", "n")
        examples.append(example)

    logger.info(f"Loaded {len(examples)} {split} examples")
    return examples


def optimize_refiner(
    lm: Optional[dspy.LM] = None,
    use_bootstrap: bool = True,
    bootstrap_demos: int = 8,
    labeled_demos: int = 4,
    num_candidates: int = 15,
    num_threads: int = 4,
    save_path: Optional[Path] = None,
    use_fast_judge: bool = False,
) -> tuple[dspy.Module, Dict[str, Any]]:
    """
    Optimize prompt refiner using Bootstrap + MIPROv2 with LLM-as-Judge.

    Two-phase approach:
    1. Bootstrap: Fast baseline (minutes)
    2. MIPROv2: Refinement with LLM judge (hours)

    Args:
        lm: Language model to use
        use_bootstrap: Whether to use bootstrap phase
        bootstrap_demos: Max bootstrapped examples
        labeled_demos: Max labeled examples
        num_candidates: Number of prompt variations
        num_threads: Parallel threads
        save_path: Path to save optimized module
        use_fast_judge: Use faster LLM judge (less accurate but quicker)

    Returns:
        Tuple of (optimized_module, results_dict)
    """
    logger.info("Starting Refiner Optimization (Bootstrap + MIPROv2 + LLM Judge)")

    if use_fast_judge:
        logger.info("Using Fast LLM Judge")
    else:
        logger.info("Using Full LLM Judge with ChainOfThought (accuracy over speed)")

    if lm is not None:
        dspy.settings.configure(lm=lm)

    # Load datasets
    trainset = load_refiner_data("train")
    valset = load_refiner_data("val")

    logger.info(f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}")

    # Create base module
    base_module = dspy.ChainOfThought(PromptRefinerSignature)

    # Initialize LLM-as-Judge metric
    metric = RefinerMetric()
    logger.info("Metric initialized: LLM-as-Judge for refinement quality")

    start_time = datetime.now()
    phase_times = {}

    # Phase 1: Bootstrap (optional but recommended)
    if use_bootstrap:
        logger.info("Phase 1: Bootstrap optimization")
        bootstrap_start = datetime.now()

        bootstrap = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=bootstrap_demos,
            max_labeled_demos=labeled_demos,
        )

        # Use subset of training data for bootstrap
        bootstrap_trainset = trainset[: min(50, len(trainset))]

        logger.info(f"Running bootstrap with {len(bootstrap_trainset)} examples...")

        try:
            bootstrap.compile(student=base_module, trainset=bootstrap_trainset)

            bootstrap_time = (datetime.now() - bootstrap_start).total_seconds()
            phase_times["bootstrap"] = bootstrap_time

            logger.success(f"Bootstrap complete in {bootstrap_time:.1f} seconds")

        except Exception as e:
            logger.warning(f"Bootstrap failed: {e}, continuing with base module")
            phase_times["bootstrap"] = 0
    else:
        phase_times["bootstrap"] = 0
    # Phase 2: MIPROv2
    logger.info("Phase 2: MIPROv2 optimization with LLM Judge")
    mipro_start = datetime.now()

    optimizer = dspy.MIPROv2(
        metric=metric,
        auto="medium",  # Balanced for quality
        init_temperature=0.7,  # Higher for diversity in refinements
        verbose=True,
        track_stats=True,
        num_threads=num_threads,
    )

    logger.info("Configured MIPROv2 (auto='medium', temp=0.7)")
    logger.info("Running MIPROv2 with LLM Judge.")
    logger.info("LLM judge will evaluate: intent preservation, clarity, quality")

    try:
        # Create a fresh uncompiled module for MIPROv2
        fresh_module = dspy.ChainOfThought(PromptRefinerSignature)

        optimized_module = optimizer.compile(
            student=fresh_module,
            trainset=trainset,
            valset=valset,
            max_bootstrapped_demos=bootstrap_demos,
            max_labeled_demos=labeled_demos,
            requires_permission_to_run=False,
        )

        mipro_time = (datetime.now() - mipro_start).total_seconds()
        phase_times["mipro"] = mipro_time

        logger.success(f"MIPROv2 complete in {mipro_time:.1f} seconds")

    except Exception as e:
        logger.error(f"MIPROv2 failed: {e}")
        raise

    total_time = (datetime.now() - start_time).total_seconds()

    # Evaluate
    logger.info("Evaluating optimized module with LLM Judge...")

    predictions = []
    for example in valset:
        try:
            pred = optimized_module(
                history=example.history, question=example.question, n=example.n
            )
            predictions.append(pred)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            predictions.append(dspy.Prediction(rewrites=[]))

    # Calculate statistics using LLM judge
    stats = calculate_refiner_stats(valset, predictions, use_llm_judge=True)

    logger.info("Validation Results:")
    logger.info(f"   Average Quality (LLM Judge): {stats['average_quality']:.3f}")
    logger.info(f"   Median Quality: {stats['median_quality']:.3f}")
    logger.info(
        f"   Avg Refinements/Question: {stats['avg_refinements_per_question']:.1f}"
    )

    # Prepare results
    results = {
        "component": "refiner",
        "optimizer": "Bootstrap+MIPROv2",
        "metric_type": "LLM-as-Judge (ChainOfThought)",
        "total_time_seconds": total_time,
        "phase_times": phase_times,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
        "num_candidates": num_candidates,
        "validation_stats": stats,
        "timestamp": datetime.now().isoformat(),
        "judge_config": {
            "evaluates": [
                "intent_preservation",
                "clarity_improvement",
                "quality_score",
            ],
            "uses_reasoning": not use_fast_judge,
        },
    }

    # Save
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        optimized_module.save(str(save_path))
        logger.info(f"Saved optimized module to {save_path}")

        results_path = save_path.parent / f"{save_path.stem}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_path}")

    return optimized_module, results
