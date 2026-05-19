#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "========================================"
echo "  ДОКТОР ХАУС — Чумной Доктор"
echo "========================================"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install deps if missing
if [ ! -f "venv/.installed" ]; then
    echo "[*] Installing dependencies..."
    pip install -q -r requirements.txt
    touch venv/.installed
fi

# Train model if not present
if [ ! -f "doctor_house_model_v8.pkl" ]; then
    echo "[*] Training model..."
    python step1_train_v8_final.py
fi

echo "[*] Starting server..."
echo "    Open http://127.0.0.1:5000 in browser"
echo "========================================"
python server.py
