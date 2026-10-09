"""Shared loader for constants.ini — the service endpoint config Ruuter also reads.

Dependency-free on purpose (stdlib only): this module is mounted standalone into
the cron-manager container, which has no project dependencies installed.

Path resolution, in order:
  1. $RAG_SEARCH_CONSTANTS            — exported by DSL/CronManager/script/load_constants.sh
  2. /app/config/constants.ini        — cron-manager mount point, if it exists
  3. <project root>/constants.ini     — llm-orchestration (/app/constants.ini) and local dev

Candidate 2 matters because the cron-manager image bakes an unrelated
/app/constants.ini (Buerokratt training endpoints). Preferring the explicit
mount keeps a script that forgot to source load_constants.sh from silently
picking up the wrong file.
"""

import configparser
import os
from pathlib import Path
from typing import Dict

_CRON_MANAGER_PATH = Path("/app/config/constants.ini")
_PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1] / "constants.ini"


def _resolve_path() -> Path:
    explicit = os.environ.get("RAG_SEARCH_CONSTANTS")
    if explicit:
        return Path(explicit)
    if _CRON_MANAGER_PATH.is_file():
        return _CRON_MANAGER_PATH
    return _PROJECT_ROOT_PATH


CONSTANTS_INI_PATH = _resolve_path()


def _load_constants_ini() -> Dict[str, str]:
    # interpolation=None: treat '%' literally; optionxform=str: keep key case;
    # strict=False: a duplicated key takes the last value instead of failing.
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    if not parser.read(CONSTANTS_INI_PATH, encoding="utf-8"):
        raise RuntimeError(f"Configuration file not found: {CONSTANTS_INI_PATH}")
    if not parser.has_section("DSL"):
        raise RuntimeError(f"Missing [DSL] section in {CONSTANTS_INI_PATH}")
    return dict(parser.items("DSL"))


_CONSTANTS = _load_constants_ini()


def get_constant(name: str) -> str:
    """Return a required value from the [DSL] section of constants.ini.

    Raises:
        RuntimeError: If the key is missing or empty.
    """
    value = _CONSTANTS.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required key '{name}' is not set in {CONSTANTS_INI_PATH}")
    return value
