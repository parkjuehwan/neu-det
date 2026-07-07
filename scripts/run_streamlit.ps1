Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path ".logs" | Out-Null
try {
    "Starting Streamlit at $(Get-Date -Format o)" | Out-File -FilePath ".logs\streamlit-script.log" -Encoding utf8
    & ".\.venv\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless true *> ".logs\streamlit-script.log"
}
catch {
    $_ | Out-File -FilePath ".logs\streamlit-script-error.log" -Encoding utf8
}
