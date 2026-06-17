#!/bin/bash
set -e

# Dynamically find the project root directory, so this script can be run from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# All python scripts should be run from the project root to ensure that
# Hydra can find the `config` directory and imports work as expected
cd "$PROJECT_ROOT"

echo "[Data] Running download_data.py..."
python "data/download_data.py" "$@"

echo "[Data] Running preprocessing.py..."
python "data/preprocessing.py" "$@"

echo "[Data] Data generation complete."