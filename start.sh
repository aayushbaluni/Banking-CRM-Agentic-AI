#!/bin/bash
set -e

# Seed DB and train model at startup (uses runtime env vars for config)
python app/db/seed.py
python app/ml/train_propensity.py

# Start FastAPI backend (internal, port 8000)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 3

# Start Streamlit frontend (exposed on port 7860 for HF Spaces)
exec python -m streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0
