#!/bin/bash

# DEFINING ENDPOINTS
source /app/scripts/load_constants.sh

BUDGET_RESET_ENDPOINT="${RAG_SEARCH_RUUTER_PUBLIC}/llm-connections/cost/reset"

payload=$(cat <<EOF
{}
EOF
)

echo "SENDING REQUEST TO RESET MONTHLY USED BUDGET TO 0"
response=$(curl -s -X POST "$BUDGET_RESET_ENDPOINT" \
    -H "Content-Type: application/json" \
    -d "$payload")

echo "BUDGET RESET SUMMARY:"
  echo "$response"
