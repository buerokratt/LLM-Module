"""
Diagnostic script to verify all paths are correct.
"""

from pathlib import Path
from typing import Dict
from src.loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="check-paths")


def get_directory_structure() -> tuple[Path, Path]:
    """Get the directory structure based on script location."""
    script_path = Path(__file__).resolve()
    logger.info(f"This script: {script_path}")

    optimization_scripts_dir = script_path.parent
    optimization_dir = optimization_scripts_dir.parent
    src_dir = optimization_dir.parent

    logger.info("Directory structure:")
    logger.info(f"  optimization_scripts: {optimization_scripts_dir}")
    logger.info(f"  optimization:         {optimization_dir}")
    logger.info(f"  src:                  {src_dir}")

    return optimization_dir, src_dir


def check_key_paths(optimization_dir: Path, src_dir: Path) -> bool:
    """Check if key paths exist and return overall status."""
    paths_to_check: Dict[str, Path] = {
        "optimized_modules": optimization_dir / "optimized_modules",
        "guardrails (optimized)": optimization_dir / "optimized_modules" / "guardrails",
        "refiner (optimized)": optimization_dir / "optimized_modules" / "refiner",
        "generator (optimized)": optimization_dir / "optimized_modules" / "generator",
        "guardrails (code)": src_dir / "guardrails",
        "rails_config.yaml": src_dir / "guardrails" / "rails_config.yaml",
    }

    logger.info("Checking paths:")
    all_good = True
    for name, path in paths_to_check.items():
        exists = "✓" if path.exists() else "✗"
        logger.info(f"  {exists} {name:25s}: {path}")
        if not path.exists():
            all_good = False

    return all_good


def check_component_files(component_dir: Path, component: str) -> None:
    """Check files for a specific component."""
    json_files = list(component_dir.glob("*.json"))
    module_files = [f for f in json_files if not f.stem.endswith("_results")]
    config_files = list(component_dir.glob("*_config.yaml"))

    logger.info(f"\n  {component}:")
    logger.info(f"    Total JSON files: {len(json_files)}")
    logger.info(f"    Module files:     {len(module_files)}")
    logger.info(f"    Config files:     {len(config_files)}")

    if module_files:
        latest = max(module_files, key=lambda p: p.stem)
        logger.info(f"    Latest module:    {latest.name}")

    if config_files:
        for cfg in config_files:
            logger.info(f"    Config:           {cfg.name}")


def check_optimized_modules(optimization_dir: Path) -> None:
    """Check optimized module files for all components."""
    logger.info("Optimized module files:")
    for component in ["guardrails", "refiner", "generator"]:
        component_dir = optimization_dir / "optimized_modules" / component
        if component_dir.exists():
            check_component_files(component_dir, component)
        else:
            logger.warning(f"  {component}: Directory not found!")


def main() -> None:
    """Check all paths."""
    logger.info("PATH DIAGNOSTIC")

    optimization_dir, src_dir = get_directory_structure()
    all_good = check_key_paths(optimization_dir, src_dir)
    check_optimized_modules(optimization_dir)

    if all_good:
        logger.success("All paths look good!")
    else:
        logger.warning("Some paths are missing - check the output above")


if __name__ == "__main__":
    main()
