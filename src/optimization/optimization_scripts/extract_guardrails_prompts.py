"""
Extract optimized prompts from DSPy guardrails modules and inject into NeMo config.
This bridges DSPy optimization with NeMo Guardrails by extracting the optimized
instructions and few-shot examples.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from src.loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="extract-guardrails-prompts")

# Constants
FULL_TRACEBACK_MSG = "Full traceback:"
FEW_SHOT_EXAMPLES_HEADER = "\nFew-shot Examples (from optimization):"

# Output-rails streaming buffer, written into every generated config.
#
# OUTPUT_STREAMING_CONTEXT_SIZE MUST stay below OUTPUT_STREAMING_CHUNK_SIZE.
# NeMo's RollingBuffer drains with ``buffer = buffer[-context_size:]`` after each
# flush; if context_size >= chunk_size the buffer never falls back below the
# flush threshold, so every subsequent token flushes on its own and triggers a
# full self_check_output LLM round-trip. That makes streaming latency and cost
# scale linearly with answer length.
#
# Keep in sync with rails.output.streaming in src/guardrails/rails_config.yaml.
OUTPUT_STREAMING_CHUNK_SIZE = 200
OUTPUT_STREAMING_CONTEXT_SIZE = 50
OUTPUT_STREAMING_STREAM_FIRST = False

# Type aliases for better readability
JsonDict = Dict[str, Any]
PromptDict = Dict[str, Any]


def load_latest_guardrails_module() -> Optional[tuple[Path, Path]]:
    """
    Find the latest optimized guardrails module and its results.

    Returns:
        Tuple of (module_path, results_path) or None if not found
    """
    try:
        # Script is at: src/optimization/optimization_scripts/extract_guardrails_prompts.py
        # Modules are at: src/optimization/optimized_modules/guardrails/
        script_path = Path(__file__).resolve()
        optimization_dir = script_path.parent.parent
        guardrails_dir = optimization_dir / "optimized_modules" / "guardrails"

        logger.info(f"Looking for guardrails in: {guardrails_dir}")

        if not guardrails_dir.exists():
            logger.error(f"Guardrails directory not found: {guardrails_dir}")
            return None

        # Find all JSON files
        all_json = list(guardrails_dir.glob("guardrails_optimized_*.json"))
        logger.info(f"Found {len(all_json)} total JSON files")

        # Exclude _results.json files to get actual module files
        module_files = [f for f in all_json if not f.stem.endswith("_results")]

        logger.info(f"Found {len(module_files)} module files (excluding results)")

        if not module_files:
            logger.error("No optimized guardrails module files found")
            return None

        # Get latest by filename (timestamp in name)
        latest_module = max(module_files, key=lambda p: p.stem)
        results_file = latest_module.parent / f"{latest_module.stem}_results.json"

        logger.info(f"Latest module: {latest_module.name}")
        logger.info(
            f"Results file: {results_file.name} (exists: {results_file.exists()})"
        )

        return latest_module, results_file

    except Exception as e:
        logger.error(f"Error finding guardrails module: {str(e)}")
        logger.exception(FULL_TRACEBACK_MSG)
        return None


def _extract_signature_data(
    signature: Dict[str, Any], extracted: Dict[str, Any]
) -> None:
    """Extract instruction and fields from signature data."""
    logger.debug(f"Signature keys: {list(signature.keys())}")

    if "instructions" in signature:
        extracted["instruction"] = signature["instructions"]
        logger.info(f"Extracted instruction ({len(extracted['instruction'])} chars)")

    if "fields" in signature:
        extracted["signature_fields"] = signature["fields"]
        logger.info(f"Extracted {len(signature['fields'])} signature fields")


def _extract_demos_from_predict(
    predict_data: Dict[str, Any], extracted: Dict[str, Any]
) -> None:
    """Extract demonstrations from predict data."""
    if "demos" in predict_data:
        demos = predict_data["demos"]
        if isinstance(demos, list):
            extracted["demos"] = demos
            logger.info(f"Extracted {len(demos)} few-shot demonstrations")

            if demos:
                first_demo = demos[0]
                logger.debug(
                    f"First demo keys: {list(first_demo.keys()) if isinstance(first_demo, dict) else 'not a dict'}"
                )


def _extract_from_predict_structure(
    module_data: Dict[str, Any], extracted: Dict[str, Any]
) -> None:
    """Extract data from predict structure in module."""
    predict_data = module_data["predict"]
    logger.debug(f"Found 'predict' in module: {list(predict_data.keys())}")

    if "signature" in predict_data:
        _extract_signature_data(predict_data["signature"], extracted)

    _extract_demos_from_predict(predict_data, extracted)


def _log_extraction_summary(extracted: Dict[str, Any]) -> None:
    """Log summary of extraction results."""
    logger.info("Extraction complete:")
    logger.info(f"  - Instruction: {'Yes' if extracted['instruction'] else 'No'}")
    logger.info(f"  - Demos: {len(extracted['demos'])}")
    logger.info(f"  - Fields: {len(extracted['signature_fields'])}")


def extract_optimized_prompts(module_path: Path) -> Dict[str, Any]:
    """
    Extract optimized prompts from DSPy module JSON.

    DSPy MIPROv2 optimizes:
    1. Signature instructions (the docstring/description)
    2. Few-shot demonstrations (examples)

    Returns:
        Dict with 'instruction', 'demos', and 'signature_fields'
    """
    try:
        logger.info(f"Loading module from: {module_path}")

        with open(module_path, "r") as f:
            module_data = json.load(f)

        extracted = {
            "instruction": None,
            "demos": [],
            "signature_fields": {},
            "raw_data": {},
        }

        # DSPy ChainOfThought modules have a 'predict' attribute with the signature
        if "predict" in module_data:
            _extract_from_predict_structure(module_data, extracted)
        elif "demos" in module_data:
            # Also check top-level for demos (alternative structure)
            extracted["demos"] = module_data["demos"]
            logger.info(f"Extracted {len(extracted['demos'])} demos from top level")

        # Store raw data for debugging
        extracted["raw_data"] = {
            "top_level_keys": list(module_data.keys()),
            "has_predict": "predict" in module_data,
        }

        _log_extraction_summary(extracted)
        return extracted

    except Exception as e:
        logger.error(f"Error extracting prompts: {str(e)}")
        logger.exception(FULL_TRACEBACK_MSG)
        return {
            "instruction": None,
            "demos": [],
            "signature_fields": {},
            "error": str(e),
        }


def format_demos_for_nemo(demos: List[Dict[str, Any]]) -> str:
    """
    Format DSPy demonstrations as few-shot examples for NeMo prompts.

    Args:
        demos: List of demonstration dictionaries from DSPy

    Returns:
        Formatted string with examples for NeMo prompt
    """
    if not demos:
        return ""

    formatted_examples: List[str] = []

    for i, demo in enumerate(demos[:5]):  # Use top 5 demos
        try:
            # Extract message and should_block from demo
            message = demo.get("message", "")
            should_block = demo.get("should_block", False)

            if message:
                verdict = "unsafe" if should_block else "safe"
                formatted_examples.append(
                    f'Example {i + 1}:\nUser message: "{message}"\nAnswer: {verdict}\n'
                )
        except Exception as e:
            logger.warning(f"Could not format demo {i}: {e}")
            continue

    if formatted_examples:
        return "\n" + "\n".join(formatted_examples) + "\n"
    return ""


def _load_base_config(base_config_path: Path) -> Optional[JsonDict]:
    """Load base configuration from YAML file."""
    if not base_config_path.exists():
        logger.error(f"Base config not found: {base_config_path}")
        return None

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    logger.info("Loaded base config")
    return base_config


def _load_optimization_results(results_path: Path) -> JsonDict:
    """Load optimization results from JSON file."""
    results_data = {}
    if results_path.exists():
        with open(results_path, "r") as f:
            results_data = json.load(f)
        logger.info("Loaded optimization results")
    return results_data


def _extract_optimization_metrics(results_data: JsonDict) -> Tuple[str, str]:
    """Extract optimization metrics from results data."""
    validation_stats = results_data.get("validation_stats", {})
    weighted_acc = validation_stats.get("weighted_accuracy", "N/A")
    false_negatives = validation_stats.get("false_negatives", "N/A")
    return weighted_acc, false_negatives


def _enhance_prompt_with_demos(
    prompt: Dict[str, Any], demos_text: str, task_name: str
) -> bool:
    """Enhance a prompt with few-shot demonstrations."""
    if not demos_text:
        return False

    original_content = prompt["content"]
    lines = original_content.split("\n")
    insert_idx = -3  # Before the last few lines (User message, Answer)

    lines.insert(insert_idx, FEW_SHOT_EXAMPLES_HEADER)
    lines.insert(insert_idx + 1, demos_text.strip())

    prompt["content"] = "\n".join(lines)
    logger.info(f"Enhanced {task_name} with few-shot examples")
    return True


def _update_prompts_with_demos(
    base_config: Dict[str, Any], demos_text: str
) -> Tuple[bool, bool]:
    """Update prompts with few-shot demonstrations."""
    if "prompts" not in base_config:
        base_config["prompts"] = []

    updated_input = False
    updated_output = False

    for prompt in base_config["prompts"]:
        if prompt.get("task") == "self_check_input":
            updated_input = _enhance_prompt_with_demos(
                prompt, demos_text, "self_check_input"
            )
            if updated_input:
                break

    if not updated_input:
        logger.warning("Could not find self_check_input prompt to update")

    for prompt in base_config["prompts"]:
        if prompt.get("task") == "self_check_output":
            updated_output = _enhance_prompt_with_demos(
                prompt, demos_text, "self_check_output"
            )
            if updated_output:
                break

    if not updated_output:
        logger.warning("Could not find self_check_output prompt to update")

    return updated_input, updated_output


def _generate_metadata_comment(
    module_path: Path,
    weighted_acc: str,
    false_negatives: str,
    results_data: Dict[str, Any],
    optimized_prompts: Dict[str, Any],
    updated_input: bool,
    updated_output: bool,
) -> str:
    """Generate metadata comment for the optimized config."""
    import datetime

    return f"""
# OPTIMIZED NEMO GUARDRAILS CONFIG
# Generated from DSPy optimized module 
# Source module: {module_path.name}
# Optimization date: {datetime.datetime.fromtimestamp(module_path.stat().st_mtime).isoformat()}
# Generated: {datetime.datetime.now().isoformat()}
# Optimization Results:
#   - Weighted Accuracy: {weighted_acc}
#   - False Negatives: {false_negatives}
#   - Optimizer: {results_data.get("optimizer", "N/A")}
#   - Training set size: {results_data.get("trainset_size", "N/A")}
#   - Validation set size: {results_data.get("valset_size", "N/A")}
#
# Enhancements Applied:
#   - Few-shot demonstrations: {len(optimized_prompts["demos"])} examples
#   - Input prompt: {"Enhanced" if updated_input else "Not updated"}
#   - Output prompt: {"Enhanced" if updated_output else "Not updated"}
"""


def _ensure_required_config_structure(base_config: Dict[str, Any]) -> None:
    """
    Ensure the base config has the required rails and streaming structure.

    This function ensures the configuration includes:
    - Global streaming: True
    - rails.input.flows with self check input
    - rails.output.flows with self check output
    - rails.output.streaming with proper settings
    """
    # Ensure global streaming is enabled
    base_config["streaming"] = True

    # Ensure rails root and nested structure using setdefault()
    rails = base_config.setdefault("rails", {})

    # Configure input rails
    input_cfg = rails.setdefault("input", {})
    input_flows = input_cfg.setdefault("flows", [])

    if "self check input" not in input_flows:
        input_flows.append("self check input")

    # Configure output rails
    output_cfg = rails.setdefault("output", {})
    output_flows = output_cfg.setdefault("flows", [])
    output_streaming = output_cfg.setdefault("streaming", {})

    if "self check output" not in output_flows:
        output_flows.append("self check output")

    # Set required streaming parameters (override existing values to ensure consistency)
    output_streaming["enabled"] = True
    output_streaming["chunk_size"] = OUTPUT_STREAMING_CHUNK_SIZE
    output_streaming["context_size"] = OUTPUT_STREAMING_CONTEXT_SIZE
    output_streaming["stream_first"] = OUTPUT_STREAMING_STREAM_FIRST

    if OUTPUT_STREAMING_CONTEXT_SIZE >= OUTPUT_STREAMING_CHUNK_SIZE:
        raise ValueError(
            f"Invalid output-rails streaming constants: context_size="
            f"{OUTPUT_STREAMING_CONTEXT_SIZE} must be < chunk_size="
            f"{OUTPUT_STREAMING_CHUNK_SIZE}. Generating a config with this "
            f"setting would cause one guardrail LLM call per streamed token."
        )

    logger.info("✓ Ensured required rails and streaming configuration structure")


def _save_optimized_config(
    output_path: Path,
    metadata_comment: str,
    base_config: Dict[str, Any],
    optimized_prompts: Dict[str, Any],
    updated_input: bool,
    updated_output: bool,
) -> None:
    """Save the optimized configuration to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(metadata_comment)
        yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"  Saved optimized config to: {output_path}")
    logger.info(f"  Config size: {output_path.stat().st_size} bytes")
    logger.info(f"  Few-shot examples: {len(optimized_prompts['demos'])}")
    logger.info(f"  Prompts updated: Input={updated_input}, Output={updated_output}")


def generate_optimized_nemo_config(
    base_config_path: Path,
    optimized_prompts: Dict[str, Any],
    module_path: Path,
    results_path: Path,
    output_path: Path,
) -> bool:
    """
    Generate NeMo config with optimized prompts from DSPy.

    Args:
        base_config_path: Path to base rails_config.yaml
        optimized_prompts: Extracted prompts from DSPy module
        module_path: Path to the DSPy module JSON
        results_path: Path to optimization results JSON
        output_path: Path to save optimized config

    Returns:
        True if successful
    """
    try:
        logger.info("Generating optimized NeMo Guardrails config...")

        # Load base configuration
        base_config = _load_base_config(base_config_path)
        if base_config is None:
            return False

        # Load optimization results
        results_data = _load_optimization_results(results_path)

        # Extract metrics
        weighted_acc, false_negatives = _extract_optimization_metrics(results_data)

        # Format few-shot demonstrations
        demos_text = format_demos_for_nemo(optimized_prompts["demos"])

        # Update prompts with demonstrations
        updated_input, updated_output = _update_prompts_with_demos(
            base_config, demos_text
        )

        # Ensure required rails and streaming configuration structure
        _ensure_required_config_structure(base_config)

        # Generate metadata comment
        metadata_comment = _generate_metadata_comment(
            module_path,
            weighted_acc,
            false_negatives,
            results_data,
            optimized_prompts,
            updated_input,
            updated_output,
        )

        # Save optimized configuration
        _save_optimized_config(
            output_path,
            metadata_comment,
            base_config,
            optimized_prompts,
            updated_input,
            updated_output,
        )

        return True

    except Exception as e:
        logger.error(f"Error generating optimized config: {str(e)}")
        logger.exception(FULL_TRACEBACK_MSG)
        return False


def main() -> None:
    """Main execution."""
    logger.info("NEMO GUARDRAILS PROMPT EXTRACTION")
    logger.info("Extracting optimized prompts from DSPy module to NeMo YAML config")
    logger.info("")

    # Find latest module
    result = load_latest_guardrails_module()
    if result is None:
        logger.error("No guardrails module found, exiting")
        return

    module_path, results_path = result

    # Extract optimized prompts

    logger.info("Step 1: Extracting optimized prompts from DSPy module")

    optimized_prompts = extract_optimized_prompts(module_path)

    if optimized_prompts.get("error"):
        logger.error(f"Failed to extract prompts: {optimized_prompts['error']}")
        return

    if not optimized_prompts["demos"] and not optimized_prompts["instruction"]:
        logger.warning("No optimized prompts or demos found in module")
        logger.warning(
            "This might mean the module structure is different than expected"
        )
        logger.info(f"Raw data keys: {optimized_prompts['raw_data']}")

    # Determine paths
    logger.info("Step 2: Generating optimized NeMo config")

    script_path = Path(__file__).resolve()
    src_dir = (
        script_path.parent.parent.parent
    )  # optimization_scripts -> optimization -> src

    base_config_path = src_dir / "guardrails" / "rails_config.yaml"
    output_path = module_path.parent / f"{module_path.stem}_config.yaml"

    logger.info(f"Base config: {base_config_path}")
    logger.info(f"Output path: {output_path}")

    if not base_config_path.exists():
        logger.error(f"Base config not found: {base_config_path}")
        return

    # Generate optimized config
    success = generate_optimized_nemo_config(
        base_config_path=base_config_path,
        optimized_prompts=optimized_prompts,
        module_path=module_path,
        results_path=results_path,
        output_path=output_path,
    )

    if success:
        logger.success("EXTRACTION COMPLETE!")
        logger.info("Optimized NeMo config available at:")
        logger.info(f"  {output_path}")
        logger.info("The NeMo guardrails adapter will automatically use this")
        logger.info("optimized config on the next service restart or request.")
        logger.info("To verify it's being used, check the logs for:")
        logger.info('"Using OPTIMIZED guardrails config"')

    else:
        logger.error("EXTRACTION FAILED")
        logger.error("Check the error messages above for details")


if __name__ == "__main__":
    main()
