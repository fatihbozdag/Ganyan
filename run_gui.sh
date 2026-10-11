#!/bin/bash
# GanyanGUI launcher (macOS).
# Run: bash run_gui.sh
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/GanyanGUI:$PWD/src:$PYTHONPATH"
exec uv run python GanyanGUI/main.py
