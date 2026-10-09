#!/bin/bash

# PREVENT OVERLAPPING RUNS
# Non-blocking flock: if another run already holds the lock, skip this invocation instead of queuing.
LOCKFILE="/app/data/agency_data_resync.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    echo "An existing agency_data_resync run is already in progress - skipping this invocation."
    exit 0
fi

# DEFINING ENDPOINTS
source /app/scripts/load_constants.sh

CHECK_RESYNC_DATA_AVAILABILITY_ENDPOINT="${RAG_SEARCH_RUUTER_PUBLIC}/data/update"

# Construct payload to update training status using cat
payload=$(cat <<EOF
{}
EOF
)

echo "SENDING REQUEST TO CHECK_RESYNC_DATA_AVAILABILITY_ENDPOINT"
response=$(curl -s -X POST "$CHECK_RESYNC_DATA_AVAILABILITY_ENDPOINT" \
    -H "Content-Type: application/json" \
    -d "$payload")

echo "DATA RESYNC SUMMARY:"
  echo "$response"