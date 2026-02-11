# Audio Transcriber WhisperX Setup Script (Windows)
# Run this script with: .\setup.ps1

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  Audio Transcriber WhisperX - Setup Script (Windows)" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/8] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.9+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Check for CUDA/GPU
Write-Host ""
Write-Host "[2/8] Checking for NVIDIA GPU..." -ForegroundColor Yellow
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host "NVIDIA GPU detected! CUDA support will be enabled." -ForegroundColor Green
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} else {
    Write-Host "No NVIDIA GPU detected. CPU mode will be used." -ForegroundColor Yellow
}

# Create virtual environment
Write-Host ""
Write-Host "[3/8] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
    $response = Read-Host "Do you want to recreate it? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Remove-Item -Recurse -Force .venv
        python -m venv .venv
        Write-Host "Virtual environment recreated." -ForegroundColor Green
    }
} else {
    python -m venv .venv
    Write-Host "Virtual environment created." -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "[4/8] Activating virtual environment..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1
Write-Host "Virtual environment activated." -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "[5/8] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel | Out-Null
Write-Host "pip upgraded successfully." -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "[6/8] Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Dependencies installed successfully." -ForegroundColor Green
    } else {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "ERROR: requirements.txt not found" -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host ""
Write-Host "[7/8] Creating directories..." -ForegroundColor Yellow
$directories = @("models", "output", "input", "static", "templates")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created: $dir/" -ForegroundColor Green
    } else {
        Write-Host "Exists: $dir/" -ForegroundColor Yellow
    }
}

# Setup environment file
Write-Host ""
Write-Host "[8/8] Setting up environment file..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host ".env file created from .env.example" -ForegroundColor Green
        Write-Host ""
        Write-Host "IMPORTANT: Edit .env and add your Hugging Face token!" -ForegroundColor Red
        Write-Host "Get your token from: https://huggingface.co/settings/tokens" -ForegroundColor Yellow
    } else {
        Write-Host "WARNING: .env.example not found" -ForegroundColor Yellow
    }
} else {
    Write-Host ".env file already exists." -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Edit .env and add your HF_TOKEN" -ForegroundColor White
Write-Host "2. Run the Flask app:" -ForegroundColor White
Write-Host "   python app.py" -ForegroundColor Cyan
Write-Host "3. Or run the CLI tool:" -ForegroundColor White
Write-Host "   python transcribe.py -i input/your-audio.mp3" -ForegroundColor Cyan
Write-Host ""
Write-Host "For Docker deployment, see DOCKER.md" -ForegroundColor Yellow
Write-Host ""
