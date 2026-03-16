#!/usr/bin/env bash
set -euo pipefail

echo "=== voice-rag smoke test ==="

# Standalone usage: uncomment to install from source
# pip install -e ".[all]" --quiet

# Ingest the sample document
echo "--- Ingesting sample doc..."
voice-rag ingest data/sample_docs/claude-code-changelog.md --recreate
echo "Ingest OK"

# Run a query and capture output
echo "--- Running query..."
OUTPUT=$(voice-rag query "What is the /loop command?")
echo "$OUTPUT"

# Assert at least one result with a positive score (score > 0.00)
# Pattern matches e.g. "score=0.82" or "score=1.00" but not "score=0.00"
if echo "$OUTPUT" | grep -qE "score=0\.[1-9]|score=[1-9]"; then
  echo "=== Smoke test PASSED ==="
else
  echo "=== Smoke test FAILED: no results with positive score ===" >&2
  exit 1
fi
