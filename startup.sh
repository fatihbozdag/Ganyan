#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run predictions
python3 predict_race.py

# Keep terminal window open
echo ""
echo "Press Enter to close..."
read 