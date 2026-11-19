"""CI check: Dragonfly sizing is explicit and Dragonfly/Langfuse images are pinned.

Fails (exit 1) when:
- a compose file runs a ``dragonfly`` service without ``--maxmemory`` and
  ``--proactor_threads`` (Dragonfly would size itself from the host: risk R3);
- the Dragonfly Helm chart has an empty ``threads`` or ``maxmemory``;
- a Dragonfly or Langfuse image uses a floating tag (none, ``latest`` or
  major-only like ``3``), in compose files or Helm values (risk R2).

Usage: python .github/scripts/check_dragonfly_config.py [--root PATH]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

WATCHED_IMAGES = ("dragonfly", "langfuse")
# A pinned tag has at least major.minor, e.g. v2.0.2 or 3.225.2
PINNED_TAG = re.compile(r"^v?\d+\.\d+")
REQUIRED_DRAGONFLY_FLAGS = ("--maxmemory", "--proactor_threads")


def is_watched(image: str) -> bool:
    return any(name in image for name in WATCHED_IMAGES)


def is_pinned(tag: str) -> bool:
    return tag != "latest" and bool(PINNED_TAG.match(tag))


def split_image(image: str) -> tuple[str, str]:
    """Split ``registry/repo:tag`` into (name, tag); tag is '' when absent."""
    last = image.rsplit("/", 1)[-1]
    if ":" not in last:
        return image, ""
    name, tag = image.rsplit(":", 1)
    return name, tag


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def check_compose(path: Path) -> list[str]:
    errors: list[str] = []
    services = load_yaml(path).get("services") or {}
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        image = str(service.get("image", ""))
        if image and is_watched(image):
            _, tag = split_image(image)
            if not is_pinned(tag):
                errors.append(f"{path}: service '{name}' image '{image}' is not pinned")
        if "dragonfly" in image:
            command = service.get("command", "")
            text = " ".join(command) if isinstance(command, list) else str(command)
            errors.extend(
                f"{path}: service '{name}' is missing {flag}"
                for flag in REQUIRED_DRAGONFLY_FLAGS
                if flag not in text
            )
    return errors


def iter_image_blocks(node: object) -> list[dict[str, Any]]:
    """Find every mapping with 'repository' and 'tag' keys (Helm image blocks)."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "repository" in node and "tag" in node:
            found.append(node)
        for value in node.values():
            found.extend(iter_image_blocks(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(iter_image_blocks(value))
    return found


def check_helm_values(path: Path) -> list[str]:
    errors: list[str] = []
    values = load_yaml(path)
    for block in iter_image_blocks(values):
        repository = str(block["repository"])
        tag = str(block["tag"])
        if is_watched(repository) and not is_pinned(tag):
            errors.append(f"{path}: image '{repository}:{tag}' is not pinned")
    if path.parent.name == "Dragonfly":
        errors.extend(
            f"{path}: '{key}' must be set explicitly"
            for key in ("threads", "maxmemory")
            if not values.get(key)
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Dragonfly sizing and pinned Dragonfly/Langfuse image tags."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    root = parser.parse_args().root

    errors: list[str] = []
    for compose in sorted(root.glob("docker-compose*.yml")):
        errors.extend(check_compose(compose))
    for values in sorted(root.glob("kubernetes/charts/*/values.yaml")):
        errors.extend(check_helm_values(values))

    if errors:
        sys.stderr.write("Dragonfly config check FAILED:\n")
        sys.stderr.writelines(f"  - {error}\n" for error in errors)
        return 1
    sys.stdout.write("Dragonfly config check passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
