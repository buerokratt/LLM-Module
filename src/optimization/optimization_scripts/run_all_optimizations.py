"""
Master script to run all optimizations sequentially.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

import dspy
from src.loki_logger import LokiLogger
from llm_orchestrator_config import LLMManager
from optimizers.guardrails_optimizer import optimize_guardrails
from optimizers.refiner_optimizer import optimize_refiner
from optimizers.generator_optimizer import optimize_generator

# Initialize Loki logger
logger = LokiLogger(service_name="run-all-optimizations")


# Constants
TRACEBACK_MSG = "Full traceback:"
OPTIMIZED_MODULES_DIR = Path(__file__).parent.parent / "optimized_modules"
DEFAULT_ENVIRONMENT = "production"


def setup_logging(log_dir: Path) -> Path:
    """Setup comprehensive logging to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"optimization_{timestamp}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} - {message}",
        level="DEBUG",
    )

    logger.info(f"Logging to: {log_file}")
    return log_file


def initialize_llm_manager(
    environment: str, connection_id: Optional[str] = None
) -> LLMManager:
    """
    Initialize LLM Manager using the SAME pattern as orchestration service.

    Args:
        environment: Environment context (production/development/test)
        connection_id: Optional connection identifier for Vault integration

    Returns:
        Initialized LLMManager instance
    """
    try:
        logger.info(f"Initializing LLM Manager for environment: {environment}")

        llm_manager = LLMManager(environment=environment, connection_id=connection_id)

        llm_manager.ensure_global_config()

        logger.info("LLM Manager initialized successfully")
        return llm_manager

    except Exception as e:
        logger.error(f"Failed to initialize LLM Manager: {str(e)}")
        raise


def optimize_guardrails_component(
    lm: dspy.LM, base_save_dir: Path, timestamp: str
) -> Dict[str, Any]:
    """Run guardrails optimization."""
    logger.info("GUARDRAILS OPTIMIZATION")

    try:
        guardrails_save_path = (
            base_save_dir / "guardrails" / f"guardrails_optimized_{timestamp}.json"
        )

        _, guardrails_results = optimize_guardrails(
            lm=lm,
            num_candidates=10,
            max_bootstrapped_demos=4,
            max_labeled_demos=2,
            num_threads=4,
            save_path=guardrails_save_path,
        )
        # Extract validation stats
        validation_stats = guardrails_results["validation_stats"]

        result = {
            "status": "success",
            "weighted_accuracy": validation_stats["weighted_accuracy"],
            "raw_accuracy": validation_stats.get("raw_accuracy", 0.0),
            "precision": validation_stats.get("precision", 0.0),
            "recall": validation_stats.get("recall", 0.0),
            "false_negatives": validation_stats.get("false_negatives", 0),
            "time_seconds": guardrails_results["optimization_time_seconds"],
            "save_path": str(guardrails_save_path),
        }

        logger.success("Guardrails optimization complete!")
        logger.info(f"   Weighted Accuracy: {result['weighted_accuracy']:.3f}")
        logger.info(f"   Raw Accuracy: {result['raw_accuracy']:.3f}")
        logger.info(f"   False Negatives: {result['false_negatives']}")

        return result

    except Exception as e:
        logger.error(f"Guardrails optimization failed: {e}")
        logger.exception(TRACEBACK_MSG)
        return {"status": "failed", "error": str(e)}


def optimize_refiner_component(
    lm: dspy.LM, base_save_dir: Path, timestamp: str
) -> Dict[str, Any]:
    """Run refiner optimization."""
    logger.info("REFINER OPTIMIZATION")

    try:
        refiner_save_path = (
            base_save_dir / "refiner" / f"refiner_optimized_{timestamp}.json"
        )

        _, refiner_results = optimize_refiner(
            lm=lm,
            use_bootstrap=True,
            bootstrap_demos=8,
            labeled_demos=4,
            num_candidates=15,
            num_threads=4,
            save_path=refiner_save_path,
        )

        result = {
            "status": "success",
            "average_quality": refiner_results["validation_stats"]["average_quality"],
            "time_seconds": refiner_results["total_time_seconds"],
            "save_path": str(refiner_save_path),
        }

        logger.success("Refiner optimization complete!")
        logger.info(
            f"   Average Quality: "
            f"{refiner_results['validation_stats']['average_quality']:.3f}"
        )
        return result

    except Exception as e:
        logger.error(f"Refiner optimization failed: {e}")
        logger.exception(TRACEBACK_MSG)
        return {"status": "failed", "error": str(e)}


def optimize_generator_component(
    lm: dspy.LM, base_save_dir: Path, timestamp: str
) -> Dict[str, Any]:
    """Run generator optimization."""
    logger.info("GENERATOR OPTIMIZATION")

    try:
        generator_save_path = (
            base_save_dir / "generator" / f"generator_optimized_{timestamp}.json"
        )

        _, generator_results = optimize_generator(
            lm=lm,
            use_bootstrap=True,
            bootstrap_demos=10,
            labeled_demos=5,
            num_candidates=20,
            num_threads=4,
            save_path=generator_save_path,
        )

        result = {
            "status": "success",
            "combined_score": generator_results["validation_stats"]["combined_score"],
            "time_seconds": generator_results["total_time_seconds"],
            "save_path": str(generator_save_path),
        }

        logger.success("Generator optimization complete!")
        logger.info(
            f" Combined Score: "
            f"{generator_results['validation_stats']['combined_score']:.3f}"
        )
        return result

    except Exception as e:
        logger.error(f"Generator optimization failed: {e}")
        logger.exception(TRACEBACK_MSG)
        return {"status": "failed", "error": str(e)}


def print_optimization_summary(results_summary: Dict[str, Dict[str, Any]]) -> None:
    """Log optimization results summary."""
    logger.info("OPTIMIZATION SUMMARY")

    for component, result in results_summary.items():
        logger.info(f"\n{component.upper()}:")
        if result["status"] == "success":
            logger.info("   Status:  Success")

            # Guardrails metrics
            if "weighted_accuracy" in result:
                logger.info(f"   Weighted Accuracy: {result['weighted_accuracy']:.3f}")
                if "raw_accuracy" in result:
                    logger.info(f"   Raw Accuracy: {result['raw_accuracy']:.3f}")
                if "false_negatives" in result:
                    logger.info(f"   False Negatives: {result['false_negatives']}")
            # Refiner metrics
            if "average_quality" in result:
                logger.info(f"   Average Quality: {result['average_quality']:.3f}")

            # Generator metrics
            if "combined_score" in result:
                logger.info(f"   Combined Score: {result['combined_score']:.3f}")

            logger.info(f"   Time: {result['time_seconds']:.1f}s")
            logger.info(f"   Saved: {result['save_path']}")
        else:
            logger.error("   Status: Failed")
            logger.error(f"   Error: {result.get('error', 'Unknown')}")


def main() -> None:
    """Run all optimizations in sequence."""
    logger.info("MASTER OPTIMIZATION SCRIPT - Running All Components")

    # Setup logging
    log_dir = Path(__file__).parent.parent / "logs"
    log_file = setup_logging(log_dir)

    # Default to production (same as orchestration service default)
    environment = DEFAULT_ENVIRONMENT
    connection_id = None

    logger.info(
        f"Processing optimization with environment: {environment}, "
        f"connection_id: {connection_id}"
    )

    # Initialize LLM Manager
    initialize_llm_manager(environment=environment, connection_id=connection_id)

    # Verify DSPy LM is configured
    lm = dspy.settings.lm
    if lm is None:
        raise RuntimeError("DSPy LM not configured after LLMManager initialization")

    logger.info(f"Using LM: {lm}")

    # Base save directory - use the constant
    base_save_dir = OPTIMIZED_MODULES_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_summary: Dict[str, Dict[str, Any]] = {}

    # Run optimizations
    results_summary["guardrails"] = optimize_guardrails_component(
        lm, base_save_dir, timestamp
    )
    results_summary["refiner"] = optimize_refiner_component(
        lm, base_save_dir, timestamp
    )
    results_summary["generator"] = optimize_generator_component(
        lm, base_save_dir, timestamp
    )
    # Logging the results summary for debugging
    logger.info(f"Results Summary: {results_summary}")

    # Save summary
    summary_path = (
        Path(__file__).parent.parent
        / "optimization_results"
        / f"optimization_summary_{timestamp}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Full logs saved to: {log_file}")
    logger.success("ALL OPTIMIZATIONS COMPLETE!")


if __name__ == "__main__":
    main()
