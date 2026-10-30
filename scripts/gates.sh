#!/usr/bin/env bash
set -euo pipefail
bash scripts/guard.sh
python -m compileall -q solution scripts
python -m pytest solution -q
echo "GATES G0+G1 OK"
