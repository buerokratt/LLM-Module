"""
Diagnose why optimized guardrails config isn't loading.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.loki_logger import LokiLogger
from src.guardrails.optimized_guardrails_loader import OptimizedGuardrailsLoader

# Initialize Loki logger
logger = LokiLogger(service_name="diagnose-guardrails-loader")


def main() -> None:
    """Run diagnostics."""
    logger.info("GUARDRAILS LOADER DIAGNOSTICS")

    loader = OptimizedGuardrailsLoader()
    diagnosis = loader.diagnose()

    logger.info("\nDiagnostic Results:")

    for key, value in diagnosis.items():
        if isinstance(value, list):
            logger.info(f"{key}:")
            for item in value:
                logger.info(f"  - {item}")
        else:
            logger.info(f"{key}: {value}")

    # Try to get config path
    logger.info("\nAttempting to load optimized config:")
    config_path, metadata = loader.get_optimized_config_path()

    logger.info(f"Config path: {config_path}")
    logger.info(f"Metadata: {metadata}")

    if metadata.get("optimized"):
        logger.success("✓ Optimized config will be used!")
    else:
        logger.warning("✗ Base config will be used")
        logger.info("Reason: No optimized config file found")


if __name__ == "__main__":
    main()
