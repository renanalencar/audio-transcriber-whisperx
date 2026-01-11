# AGENTS.md - Audio Transcriber WhisperX Development Guide

## Project Overview
This is a Python-based AI/ML project for audio transcription using WhisperX with speaker diarization capabilities. The project processes audio files to generate timestamped transcriptions with speaker identification, optimized for GPU acceleration.

## Build, Lint, and Test Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies (using uv - recommended)
uv pip install -e .

# Alternative using pip
pip install -r requirements.txt
```

### Running the Application
```bash
# Run the main transcription script
python main.py

# For different audio files, modify the audio_file variable in main.py
# or create environment-specific configurations
```

### Testing
**Note: No test suite currently exists.** When adding tests:
```bash
# Future test commands (when implemented)
pytest                           # Run all tests
pytest tests/test_transcription.py  # Run single test file
pytest -v                       # Verbose output
pytest -k "test_speaker"        # Run tests matching pattern
```

### Code Quality Tools
**Note: No linting/formatting tools currently configured.** Recommended additions:
```bash
# Install and run code quality tools (recommended)
pip install black flake8 mypy isort
black .                          # Format code
flake8 .                        # Lint code
mypy .                          # Type checking
isort .                         # Sort imports
```

## Code Style Guidelines

### Import Organization
Based on `main.py`, follow this import order:
1. Standard library imports
2. Third-party library imports (torch, warnings)
3. Local application imports (whisperx, dotenv)
4. Specific component imports

```python
import os
import torch
import warnings
from datetime import datetime

import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline
```

### Formatting Standards
- **Encoding**: Use UTF-8 encoding for all files
- **Line length**: No explicit limit observed, but keep reasonable (≤100 chars recommended)
- **Indentation**: 4 spaces (Python standard)
- **Quotes**: Use double quotes for strings consistently
- **Comments**: Use `#` for single-line comments, include docstrings for functions

### Naming Conventions
- **Variables**: Use snake_case (`audio_file`, `batch_size`, `compute_type`)
- **Functions**: Use snake_case with descriptive names (`log`, `load_model`)
- **Constants**: Use UPPER_SNAKE_CASE for constants
- **Files**: Use kebab-case for filenames (`audio-en_US.mp3`)
- **Directories**: Use lowercase (`input`, `output`, `models`)

### Type Annotations
**Current status**: No type hints in existing code. **Recommended** for new code:
```python
def log(message: str) -> None:
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
```

### Error Handling
- Use warning filters for known third-party library warnings:
```python
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
```
- Include proper exception handling for file operations and model loading
- Use descriptive error messages with context

### File Structure Conventions
```
project_root/
├── main.py              # Main application entry point
├── pyproject.toml       # Project configuration and dependencies
├── requirements.txt     # Pip-compatible dependencies
├── .env                 # Environment variables (DO NOT commit)
├── .env_example         # Example environment file (commit this)
├── input/              # Audio input files
├── output/             # Generated transcription files
├── models/             # Downloaded ML models (auto-generated)
└── .venv/              # Virtual environment (DO NOT commit)
```

## Environment Variables
Required environment variables (see `.env_example`):
- `HF_TOKEN`: Hugging Face API token for speaker diarization models

## Dependencies and Package Management

### Core Dependencies
- `whisperx`: Audio transcription with speaker diarization
- `torch`, `torchaudio`: PyTorch for GPU acceleration
- `pyannote-audio`: Speaker diarization pipeline
- `python-dotenv`: Environment variable management
- `tqdm`: Progress bars

### Python Version Requirements
- **Minimum**: Python ≥3.8 (specified in pyproject.toml)
- **Recommended**: Python 3.9+ for better compatibility

### GPU Requirements
- CUDA-compatible GPU recommended for optimal performance
- Configure `compute_type` and `batch_size` based on available GPU memory:
  - High-end GPU: `float16`, `batch_size=16`
  - Lower memory: `int8`, reduce `batch_size`

## Project Architecture

### Main Components
1. **Audio Loading**: Load and preprocess audio files
2. **Transcription**: Generate text from speech using WhisperX
3. **Alignment**: Align transcribed text with precise timing
4. **Diarization**: Identify and separate different speakers
5. **Output Generation**: Create formatted transcription files

### Configuration Parameters
Key configurable parameters in `main.py`:
- `device`: Computing device ("cuda" for GPU, "cpu" for CPU)
- `batch_size`: Processing batch size (adjust for GPU memory)
- `compute_type`: Precision ("float16" or "int8")
- `model_dir`: Local model storage directory

### Output Format
Generated transcriptions follow this format:
```
[start_time - end_time] SPEAKER_ID: transcribed_text
```

## Development Best Practices

### Code Organization
- Keep the main script focused on orchestration
- Extract reusable functions for common operations
- Consider modularizing when the codebase grows beyond single script

### Performance Considerations
- Use GPU acceleration when available
- Implement memory management for large audio files
- Cache downloaded models locally (`model_dir` parameter)
- Monitor GPU memory usage and adjust batch sizes accordingly

### Security
- Never commit `.env` files or API tokens
- Use environment variables for sensitive configuration
- Validate input file paths and types

### Logging and Progress
- Use structured logging with timestamps
- Implement progress bars for long-running operations
- Include meaningful status messages for debugging

## Future Improvements
- Add comprehensive test suite
- Implement CLI argument parsing for configuration
- Add code formatting and linting configuration (black, flake8, mypy)
- Create proper module structure for larger codebase
- Add README.md documentation
- Implement proper error handling and recovery mechanisms