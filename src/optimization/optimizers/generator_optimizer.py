"""
Response Generator optimizer using Bootstrap + MIPROv2.
Focuses on scope detection and answer quality using DSPy's native SemanticF1.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

import dspy
from src.loki_logger import LokiLogger
from optimization.metrics.generator_metrics import (
    GeneratorMetric,
    calculate_generator_stats,
)

# Initialize Loki logger
logger = LokiLogger(service_name="generator-optimizer")


class ResponseGeneratorSignature(dspy.Signature):
    """
    Produce a grounded answer from the provided context ONLY.

    CRITICAL LANGUAGE RULE:
    - The answer MUST be in the SAME language as the input question
    - Estonian question → Estonian answer
    - Russian question → Russian answer
    - English question → English answer
    - Maintain the natural language flow and grammar of the detected language

    Rules:
    - Use ONLY the provided context blocks; do not invent facts
    - If context is insufficient, set questionOutOfLLMScope=true
    - Do not include citations in the answer field
    - Be concise and direct
    """

    question: str = dspy.InputField(
        desc="User's question. Answer in the SAME language as this question."
    )
    context_blocks: list[str] = dspy.InputField(desc="Relevant context chunks")

    answer: str = dspy.OutputField(
        desc="Human-friendly answer in THE SAME LANGUAGE as the question, without citations"
    )
    questionOutOfLLMScope: bool = dspy.OutputField(
        desc="True if context is insufficient to answer"
    )


def load_generator_data(split: str = "train") -> list[dspy.Example]:
    """Load generator dataset."""
    data_path = Path(__file__).parent.parent / "optimization_data" / "generator" / split

    file_map = {"train": "generator_train.json", "val": "generator_val.json"}

    filepath = data_path / file_map[split]

    logger.info(f"Loading generator {split} data from {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        # Format context blocks
        context_blocks = []
        for i, chunk in enumerate(item.get("context_chunks", [])):
            text = chunk.get("text", "")
            if text:
                context_blocks.append(f"[Context {i + 1}]\n{text}")

        if not context_blocks:
            context_blocks = ["[Context 1]\n(No relevant context available.)"]

        example = dspy.Example(
            question=item["question"],
            context_blocks=context_blocks,
            should_be_in_scope=item["should_be_in_scope"],
            expected_answer=item["expected_answer"],
            answer=item["expected_answer"],  # For training
            questionOutOfLLMScope=not item["should_be_in_scope"],  # For training
        ).with_inputs("question", "context_blocks")
        examples.append(example)

    logger.info(f"Loaded {len(examples)} {split} examples")
    return examples


def optimize_generator(
    lm: Optional[dspy.LM] = None,
    use_bootstrap: bool = True,
    bootstrap_demos: int = 10,
    labeled_demos: int = 5,
    num_candidates: int = 20,
    num_threads: int = 4,
    save_path: Optional[Path] = None,
) -> tuple[dspy.Module, Dict[str, Any]]:
    """
    Optimize response generator using Bootstrap + MIPROv2.

    Uses FIXED GeneratorMetric with proper DSPy SemanticF1 integration.

    Args:
        lm: Language model to use
        use_bootstrap: Whether to use bootstrap phase
        bootstrap_demos: Max bootstrapped examples
        labeled_demos: Max labeled examples
        num_candidates: Number of prompt variations
        num_threads: Parallel threads
        save_path: Path to save optimized module

    Returns:
        Tuple of (optimized_module, results_dict)
    """
    logger.info("Starting Generator Optimization (Bootstrap + MIPROv2)")
    logger.info("Using FIXED GeneratorMetric with DSPy's native SemanticF1")

    if lm is not None:
        dspy.settings.configure(lm=lm)

    # Load datasets
    trainset = load_generator_data("train")
    valset = load_generator_data("val")

    logger.info(f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}")

    # Create base module
    base_module = dspy.ChainOfThought(ResponseGeneratorSignature)

    # Initialize metric with  SemanticF1
    metric = GeneratorMetric()
    logger.info("Metric initialized with DSPy's SemanticF1 for answer quality")

    start_time = datetime.now()
    phase_times = {}

    # Phase 1: Bootstrap
    if use_bootstrap:
        logger.info("Phase 1: Bootstrap optimization")
        bootstrap_start = datetime.now()

        bootstrap = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=bootstrap_demos,
            max_labeled_demos=labeled_demos,
        )

        # Use subset for bootstrap
        bootstrap_trainset = trainset[: min(100, len(trainset))]

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
    logger.info("🔧 Phase 2: MIPROv2 optimization")
    mipro_start = datetime.now()

    optimizer = dspy.MIPROv2(
        metric=metric,
        auto="medium",  # Balanced
        init_temperature=0.3,  # Low for factual accuracy
        verbose=True,
        track_stats=True,
        num_threads=num_threads,
    )

    logger.info("Configured MIPROv2")
    logger.info("Running MIPROv2.")

    try:
        # Create a fresh uncompiled module for MIPROv2
        fresh_module = dspy.ChainOfThought(ResponseGeneratorSignature)

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
    logger.info("Evaluating optimized module...")

    predictions = []
    for example in valset:
        try:
            pred = optimized_module(
                question=example.question, context_blocks=example.context_blocks
            )
            predictions.append(pred)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            predictions.append(dspy.Prediction(answer="", questionOutOfLLMScope=True))

    # Calculate statistics using FIXED metric
    stats = calculate_generator_stats(valset, predictions)

    logger.info("Validation Results:")
    logger.info(f"   Combined Score: {stats['combined_score']:.3f}")
    logger.info(f"   Scope Accuracy: {stats['scope_accuracy']:.3f}")
    logger.info(f"   In-Scope Performance: {stats['in_scope_performance']:.3f}")
    logger.info(f"   Out-Scope Performance: {stats['out_scope_performance']:.3f}")

    # Prepare results
    results = {
        "component": "generator",
        "optimizer": "Bootstrap+MIPROv2",
        "metric_type": "GeneratorMetric with DSPy SemanticF1",
        "total_time_seconds": total_time,
        "phase_times": phase_times,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
        "num_candidates": num_candidates,
        "validation_stats": stats,
        "timestamp": datetime.now().isoformat(),
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
