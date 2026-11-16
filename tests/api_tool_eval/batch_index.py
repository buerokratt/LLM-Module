"""
Batch Indexer — sends all endpoints from test-endpoints.json to POST /api-tools/index.

Usage:
    python batch_index.py
    python batch_index.py --ruuter-url http://localhost:8088
"""

import argparse
import json
import time
from pathlib import Path

import requests

ENDPOINTS_FILE = Path(__file__).parent / "test-endpoints.json"
DEFAULT_RUUTER_URL = "http://localhost:8088"
INDEX_ENDPOINT = "/rag-search/api-tools/index"


def index_endpoint(ruuter_url: str, endpoint: dict) -> dict:
    """Send a single endpoint spec to the indexing API."""
    url = f"{ruuter_url}{INDEX_ENDPOINT}"
    try:
        response = requests.post(url, json=endpoint, timeout=60)
        return {
            "status_code": response.status_code,
            "body": response.json() if response.content else {},
            "ok": 200 <= response.status_code < 300,
        }
    except requests.exceptions.Timeout:
        return {"status_code": 408, "body": {"error": "Timeout"}, "ok": False}
    except requests.exceptions.ConnectionError as e:
        return {"status_code": 503, "body": {"error": str(e)}, "ok": False}
    except Exception as e:
        return {"status_code": 500, "body": {"error": str(e)}, "ok": False}


def main():
    parser = argparse.ArgumentParser(
        description="Batch index API endpoints into Qdrant"
    )
    parser.add_argument(
        "--ruuter-url",
        default=DEFAULT_RUUTER_URL,
        help=f"Base URL of Ruuter private (default: {DEFAULT_RUUTER_URL})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between indexing requests (default: 1.0)",
    )
    args = parser.parse_args()

    # Load endpoints
    if not ENDPOINTS_FILE.exists():
        print(f"ERROR: {ENDPOINTS_FILE} not found")
        return

    with open(ENDPOINTS_FILE, encoding="utf-8") as f:
        endpoints = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"Batch Indexer — {len(endpoints)} endpoints")
    print(f"Target: {args.ruuter_url}{INDEX_ENDPOINT}")
    print(f"{'=' * 60}\n")

    results = {"success": [], "failed": []}

    for i, endpoint in enumerate(endpoints, 1):
        name = endpoint.get("name", "unknown")
        endpoint_id = endpoint.get("endpointId", "?")
        print(f"[{i:02d}/{len(endpoints)}] Indexing: {name} ({endpoint_id[:8]}...)")

        result = index_endpoint(args.ruuter_url, endpoint)

        if result["ok"]:
            print(f"          Success (HTTP {result['status_code']})")
            results["success"].append(name)
        else:
            print(
                f"          Failed  (HTTP {result['status_code']}) — {result['body']}"
            )
            results["failed"].append(name)

        # Delay between requests to avoid overloading the embedding service
        if i < len(endpoints):
            time.sleep(args.delay)

    # Summary
    print(f"\n{'=' * 60}")
    print("INDEXING COMPLETE")
    print(f"{'=' * 60}")
    print(f" Success: {len(results['success'])}/{len(endpoints)}")
    print(f" Failed:  {len(results['failed'])}/{len(endpoints)}")
    if results["failed"]:
        print("\nFailed endpoints:")
        for name in results["failed"]:
            print(f"  - {name}")
    print("\nNext step: run  python eval_search.py  to test retrieval accuracy")


if __name__ == "__main__":
    main()
