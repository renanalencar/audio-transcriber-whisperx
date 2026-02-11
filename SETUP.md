# Setup Guide

This guide will help you set up the Audio Transcriber WhisperX application.

## Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (recommended for faster transcription)
- Hugging Face account and API token

## Quick Setup

### Automated Setup

We provide setup scripts for both Windows and Linux/Mac:

#### Windows (PowerShell)

```powershell
.\setup.ps1
```

#### Linux/Mac (Bash)

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. ✅ Check Python version
2. ✅ Detect NVIDIA GPU/CUDA
3. ✅ Create virtual environment
4. ✅ Install all dependencies
5. ✅ Create necessary directories
6. ✅ Set up environment file

### Manual Setup

If you prefer to set up manually:

#### 1. Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Create Directories

```bash
mkdir models output input
```

#### 5. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Hugging Face token:
```env
HF_TOKEN=your_actual_token_here
```

## Getting Your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "WhisperX")
4. Select "Read" access
5. Click "Generate"
6. Copy the token to your `.env` file

## Verify Installation

### Check Python Packages

```bash
pip list | grep -E "whisperx|torch|flask"
```

### Check GPU Support

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Test the Installation

**Flask Web App:**
```bash
python app.py
```
Then open http://localhost:5000

**CLI Tool:**
```bash
python transcribe.py -i input/your-audio.mp3
```

## Troubleshooting

### Python Version Issues

Make sure you're using Python 3.9+:
```bash
python --version
```

### CUDA/GPU Issues

If GPU is not detected:

1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```

2. Reinstall PyTorch with CUDA support:
   ```bash
   pip uninstall torch torchaudio
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Import Errors

If you get import errors, try:
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

### Permission Errors (Linux/Mac)

If you get permission errors when running setup.sh:
```bash
chmod +x setup.sh
```

### Virtual Environment Not Found

Make sure you've activated the virtual environment:

**Windows:**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### Disk Space Issues

WhisperX models can be large (several GB). Ensure you have at least 10GB free space.

## Directory Structure

After setup, your project should look like:

```
audio-transcriber-whisperx/
├── .venv/                  # Virtual environment (created by setup)
├── models/                 # Downloaded models (auto-created)
├── output/                 # Transcription outputs
├── input/                  # Audio input files
├── templates/              # Flask HTML templates
├── static/                 # Static files (CSS, JS)
├── app.py                  # Flask web application
├── transcribe.py           # CLI tool
├── main.py                 # Original script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from .env.example)
├── .env.example           # Example env file
├── setup.ps1              # Windows setup script
├── setup.sh               # Linux/Mac setup script
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
└── README.md              # Project documentation
```

## GPU Memory Requirements

Recommended GPU memory based on model and batch size:

| Model      | Batch Size | GPU Memory |
|------------|-----------|------------|
| large-v2   | 16        | 12GB+      |
| large-v2   | 8         | 8GB+       |
| large-v2   | 4         | 6GB+       |
| medium     | 16        | 8GB+       |
| medium     | 8         | 6GB+       |

Adjust `batch_size` in `app.py` or `transcribe.py` based on your GPU memory.

## Next Steps

After successful setup:

1. **Web Interface**: Run `python app.py` and visit http://localhost:5000
2. **CLI Tool**: Run `python transcribe.py -i path/to/audio.mp3`
3. **Docker**: See [DOCKER.md](DOCKER.md) for containerized deployment

## Update Dependencies

To update dependencies:

```bash
pip install --upgrade -r requirements.txt
```

## Clean Installation

To start fresh:

**Windows:**
```powershell
Remove-Item -Recurse -Force .venv
.\setup.ps1
```

**Linux/Mac:**
```bash
rm -rf .venv
./setup.sh
```

## Getting Help

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting) above
2. Review the logs and error messages
3. Ensure all prerequisites are met
4. Check GitHub issues for similar problems

## Security Notes

- Never commit `.env` file to version control
- Keep your HF_TOKEN private
- Use `.gitignore` to exclude sensitive files
- Consider using secrets management in production
