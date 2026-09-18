"""
Optimized Guardrails Loader for NeMo Guardrails.
Extracts optimized prompts from DSPy guardrails modules and generates updated config.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
from src.loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="optimized-guardrails-loader")


class OptimizedGuardrailsLoader:
    """
    Loads optimized guardrails prompts and creates updated NeMo config.

    Features:
    - Extracts optimized prompts from DSPy modules
    - Generates updated rails_config.yaml with optimized prompts
    - Falls back to base config if optimization not found
    """

    def __init__(self, optimized_modules_dir: Optional[Path] = None) -> None:
        """
        Initialize the guardrails loader.

        Args:
            optimized_modules_dir: Directory containing optimized modules.
                                  Defaults to src/optimization/optimized_modules
        """
        if optimized_modules_dir is None:
            # Path from src/guardrails/optimized_guardrails_loader.py
            # to src/optimization/optimized_modules
            current_file = Path(__file__).resolve()
            # Go up to src, then to optimization/optimized_modules
            src_dir = current_file.parent.parent
            optimized_modules_dir = src_dir / "optimization" / "optimized_modules"

        self.optimized_modules_dir = Path(optimized_modules_dir)
        self.base_config_path = Path(__file__).parent / "rails_config.yaml"

        logger.info(
            f"OptimizedGuardrailsLoader initialized "
            f"(modules: {self.optimized_modules_dir})"
        )

    def get_optimized_config_path(self) -> Tuple[Path, Dict[str, Any]]:
        """
        Get path to optimized guardrails config YAML file.

        Returns:
            Tuple of (config_path, metadata)
            If not found, returns (base_config_path, {'optimized': False})
        """
        try:
            # Find latest optimized module
            module_files = list(
                self.optimized_modules_dir.glob(
                    "guardrails/guardrails_optimized_*.json"
                )
            )
            module_files = [f for f in module_files if not f.stem.endswith("_results")]

            if not module_files:
                logger.info("No optimized guardrails modules found, using base config")
                return self.base_config_path, {"optimized": False, "version": "base"}

            # Get latest by timestamp in filename
            latest_module = max(module_files, key=lambda p: p.stem)
            module_stem = (
                latest_module.stem
            )  # e.g., "guardrails_optimized_20251022_104141"

            logger.debug(f"Latest module stem: {module_stem}")

            # Look for corresponding config file with exact same stem + _config.yaml
            config_file = latest_module.parent / f"{module_stem}_config.yaml"

            logger.debug(f"Looking for config at: {config_file}")
            logger.debug(f"Config exists: {config_file.exists()}")

            if config_file.exists():
                # Load results for metadata
                results_file = latest_module.parent / f"{module_stem}_results.json"
                metadata = {"optimized": True, "version": f"{module_stem}_results"}

                if results_file.exists():
                    try:
                        with open(results_file, "r") as f:
                            results_data = json.load(f)
                            metadata.update(
                                {
                                    "optimizer": results_data.get(
                                        "optimizer", "unknown"
                                    ),
                                    "metrics": results_data.get("validation_stats", {}),
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Could not load results file: {e}")

                logger.info(
                    f"✓ Using OPTIMIZED guardrails config (version: {metadata['version']})"
                )
                return config_file, metadata
            else:
                logger.warning(
                    f"Optimized module found but no extracted config at: {config_file}"
                )
                logger.info(
                    "Note: Run extract_guardrails_prompts.py to generate optimized config"
                )
                return self.base_config_path, {"optimized": False, "version": "base"}

        except Exception as e:
            logger.error(f"Error loading optimized config: {str(e)}")
            logger.exception("Full traceback:")
            return self.base_config_path, {
                "optimized": False,
                "version": "base",
                "error": str(e),
            }

    def diagnose(self) -> Dict[str, Any]:
        """
        Diagnose the state of optimized modules and configs.

        Returns:
            Dictionary with diagnostic information
        """
        try:
            guardrails_dir = self.optimized_modules_dir / "guardrails"

            if not guardrails_dir.exists():
                return {
                    "modules_dir": str(self.optimized_modules_dir),
                    "guardrails_dir": str(guardrails_dir),
                    "guardrails_dir_exists": False,
                    "base_config": str(self.base_config_path),
                    "base_config_exists": self.base_config_path.exists(),
                    "error": "Guardrails directory does not exist",
                }

            all_json = list(guardrails_dir.glob("guardrails_optimized_*.json"))
            module_files = [f for f in all_json if not f.stem.endswith("_results")]
            results_files = [f for f in all_json if f.stem.endswith("_results")]
            config_files = list(
                guardrails_dir.glob("guardrails_optimized_*_config.yaml")
            )

            diagnosis = {
                "modules_dir": str(self.optimized_modules_dir),
                "guardrails_dir": str(guardrails_dir),
                "guardrails_dir_exists": guardrails_dir.exists(),
                "base_config": str(self.base_config_path),
                "base_config_exists": self.base_config_path.exists(),
                "total_json_files": len(all_json),
                "module_files": [f.name for f in module_files],
                "results_files": [f.name for f in results_files],
                "config_files": [f.name for f in config_files],
            }

            if module_files:
                latest = max(module_files, key=lambda p: p.stem)
                expected_config = guardrails_dir / f"{latest.stem}_config.yaml"
                diagnosis["latest_module"] = latest.name
                diagnosis["expected_config"] = expected_config.name
                diagnosis["expected_config_exists"] = expected_config.exists()

            return diagnosis

        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_guardrails_loader_instance: Optional[OptimizedGuardrailsLoader] = None


def get_guardrails_loader(
    optimized_modules_dir: Optional[Path] = None,
) -> OptimizedGuardrailsLoader:
    """
    Get singleton instance of OptimizedGuardrailsLoader.

    Args:
        optimized_modules_dir: Optional custom directory

    Returns:
        OptimizedGuardrailsLoader instance
    """
    global _guardrails_loader_instance

    if _guardrails_loader_instance is None:
        _guardrails_loader_instance = OptimizedGuardrailsLoader(optimized_modules_dir)

    return _guardrails_loader_instance
