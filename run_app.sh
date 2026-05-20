#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Coffee Quant — Interactive Research Tool Launcher
# Usage: ./run_app.sh [port]
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

PORT=${1:-8501}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "────────────────────────────────────────────"
echo "  ☕  Coffee Quant v0.2"
echo "  http://localhost:${PORT}"
echo "────────────────────────────────────────────"

# Check Python ≥ 3.10
python_cmd=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    ver=$("$cmd" -c "import sys; print(sys.version_info >= (3,10))" 2>/dev/null || echo "False")
    if [ "$ver" = "True" ]; then
      python_cmd="$cmd"
      break
    fi
  fi
done

if [ -z "$python_cmd" ]; then
  echo "ERROR: Python 3.10+ required"
  exit 1
fi

# Install / upgrade deps silently if needed
"$python_cmd" -c "import streamlit" 2>/dev/null || {
  echo "Installing dependencies …"
  "$python_cmd" -m pip install -r "${SCRIPT_DIR}/requirements.txt" -q
}

# Launch
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"
exec "$python_cmd" -m streamlit run app.py \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false \
  --theme.base dark \
  --theme.primaryColor "#6f4e37"
