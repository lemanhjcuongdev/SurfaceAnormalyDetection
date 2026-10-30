<#
Register the project Python environment as a local Jupyter kernel.
Run this script from the SOURCE\PhatHienLoi project directory.
#>

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project .venv was not found. Set up the project environment first."
}

& $python -m ipykernel install --user --name 'phathienloi-w1-local' --display-name 'Python 3.11 (PhatHienLoi W1 - local)'

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output 'Local kernel registered. In Antigravity choose: Python 3.11 (PhatHienLoi W1 - local).'
