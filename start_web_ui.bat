@echo off
REM Launch Streamlit Web UI for Windows Command Prompt

cd /d D:\ChatBot
echo Starting AI Chatbot Web Interface...
echo Opening browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

env\Scripts\streamlit.exe run app.py
pause
