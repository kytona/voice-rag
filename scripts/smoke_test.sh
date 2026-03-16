#!/usr/bin/env bash
set -euo pipefail

echo "=== voice-rag smoke test ==="

CLI_CMD=(voice-rag)
if ! command -v "${CLI_CMD[0]}" >/dev/null 2>&1; then
  CLI_CMD=(python3 -m voice_rag.cli)
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "--- Checking CLI entrypoint..."
"${CLI_CMD[@]}" --help >/dev/null

echo "--- Initializing config..."
"${CLI_CMD[@]}" init --dir "$TMP_DIR"
test -f "$TMP_DIR/voice-rag.yaml"

echo "--- Running doctor..."
"${CLI_CMD[@]}" doctor --config "$TMP_DIR/voice-rag.yaml" >/dev/null

echo "=== Smoke test PASSED ==="
