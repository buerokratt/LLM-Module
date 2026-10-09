#!/bin/bash

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