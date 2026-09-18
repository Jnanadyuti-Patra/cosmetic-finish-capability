#!/usr/bin/env bash
# Launch the interactive dashboard.
cd "$(dirname "$0")"
python -m streamlit run streamlit_app.py
