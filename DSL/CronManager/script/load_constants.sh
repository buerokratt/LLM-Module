#!/bin/bash
# Exports every key from the [DSL] section of the shared RAG constants file,
# so cron job scripts can reference service endpoints instead of hardcoding them.
#
# MUST BE SOURCED, not executed:
#     source /app/scripts/load_constants.sh
#
# Why a file and not environment variables:
# CronManager runs job scripts via Runtime.exec(command, envp, dir) with an
# explicit envp (ShellExecuteJob.java). A non-null envp means the child does NOT
# inherit the parent environment, so docker-compose `environment:` entries are
# invisible to these scripts -- a scheduled job sees exactly 3 bash-synthesized
# variables and nothing else. The config therefore has to be read from disk.
#
# Why /app/config and not /app:
# The cron-manager image bakes its own unrelated /app/constants.ini (Buerokratt
# training endpoints). That file is left untouched; ours is mounted alongside it.
#
# Keys are set as plain shell variables, NOT exported: the file also holds
# secrets (DB_PASSWORD), which must not leak into child processes (python,
# curl). Only RAG_SEARCH_CONSTANTS is exported, so Python processes started
# later in the same script read the same file themselves.

RAG_SEARCH_CONSTANTS="${RAG_SEARCH_CONSTANTS:-/app/config/constants.ini}"

if [ ! -f "$RAG_SEARCH_CONSTANTS" ]; then
    # stdout, not stderr: CronManager logs only the process stdout
    # (ShellExecuteJob reads getInputStream()), so stderr is silently discarded.
    echo "[ERROR] Constants file not found: $RAG_SEARCH_CONSTANTS"
    echo "[ERROR] Is constants.ini mounted into the cron-manager container?"
    # exit, not return: `return` from a sourced file only ends the sourcing and
    # lets the calling script continue on with unset endpoint variables.
    exit 1
fi

# Skip the [DSL] section header and comment lines; define everything else.
# Note: configparser does not strip end-of-line comments, so values in
# constants.ini must not carry trailing `# ...` text.
# shellcheck disable=SC1090
. <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$RAG_SEARCH_CONSTANTS")

export RAG_SEARCH_CONSTANTS
