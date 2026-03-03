#!/bin/bash

# 1. Load variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found!"
    exit 1
fi

# 2. Check if the key was actually loaded
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY is not set in the .env file!"
    exit 1
fi

# 3. Activate Virtual Environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found. Please run: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# 4. Final check for model file
if [ ! -f "resnet18_emotion.pth" ]; then
    echo "Warning: resnet18_emotion.pth not found. Vision model might fail."
fi

echo "Starting AI Driver Assistant on http://127.0.0.1:8000"

# 5. Launch the server
python3 -m uvicorn app:app --reload