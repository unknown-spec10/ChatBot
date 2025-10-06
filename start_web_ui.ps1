# Launch Streamlit Web UI
# Double-click this file to start the web interface

$env:Path = "D:\ChatBot\env\Scripts;" + $env:Path
Set-Location "D:\ChatBot"

Write-Host "Starting AI Chatbot Web Interface..." -ForegroundColor Green
Write-Host "Opening browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

.\env\Scripts\streamlit.exe run app.py
