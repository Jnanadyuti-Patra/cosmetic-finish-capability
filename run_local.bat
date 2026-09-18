@echo off
REM Launch the interactive dashboard.
cd /d "%~dp0"
python -m streamlit run streamlit_app.py
