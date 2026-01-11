# Audio Transcriber WhisperX

A powerful Python-based audio transcription tool that combines state-of-the-art speech recognition with speaker diarization capabilities. Built on WhisperX and pyannote-audio, this tool provides accurate, timestamped transcriptions with speaker identification, optimized for GPU acceleration.

## Features

- **High-Quality Transcription**: Uses OpenAI's Whisper large-v2 model via WhisperX for superior accuracy
- **Speaker Diarization**: Automatically identifies and separates different speakers in audio
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs with configurable precision settings
- **Multi-language Support**: Supports various languages with automatic language detection
- **Timestamped Output**: Provides precise start and end times for each speech segment
- **Batch Processing**: Efficient processing with configurable batch sizes
- **Word-level Alignment**: Accurate word-level timestamps for detailed analysis
- **Progress Tracking**: Real-time progress indication during processing

## Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Minimum 8GB GPU memory for `float16` precision
- Minimum 4GB GPU memory for `int8` precision (reduced accuracy)

### Tested Configuration
This project has been developed and tested on the following setup:

| Component | Specification |
|---|---|
| **Operating System** | Microsoft Windows 11 Pro (Build 26200) |
| **Motherboard** | Gigabyte B760M AORUS ELITE |
| **CPU** | Intel Core i7-13700K (13th Gen) |
| **GPU** | NVIDIA GeForce RTX 3060 (12GB VRAM) |
| **RAM** | 128GB (130,840 MB total) |
| **Storage** | • Kingston SNV2S1000G (1TB NVMe SSD)<br>• Kingston SNV2S2000G (2TB NVMe SSD) |
| **CUDA** | Version 13.1 |
| **GPU Driver** | NVIDIA 591.44 |

**Optimal Settings for this Configuration:**
```python
device = "cuda"
batch_size = 8          # Works well with RTX 3060 12GB
compute_type = "float16" # Full precision with available VRAM
```

### Dependencies
- `whisperx`: Advanced speech-to-text with speaker diarization
- `torch` & `torchaudio`: PyTorch for deep learning acceleration
- `pyannote-audio`: Speaker diarization pipeline
- `python-dotenv`: Environment variable management
- `tqdm`: Progress visualization

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd audio-transcriber-whisperx
```

### 2. Create Virtual Environment
```bash
# Create and activate virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies

**Option A: Using uv (recommended)**
```bash
pip install uv
uv pip install -e .
```

**Option B: Using pip**
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
```bash
# Copy the example environment file
cp .env_example .env

# Edit .env and add your Hugging Face token
# HF_TOKEN="your_huggingface_token_here"
```

### 5. Get Hugging Face Token
1. Visit [Hugging Face](https://huggingface.co/) and create an account
2. Go to Settings → Access Tokens
3. Create a new token with read permissions
4. Add the token to your `.env` file

## Usage

### Basic Usage
1. Place your audio file in the `input/` directory
2. Update the `audio_file` variable in `demo.py` with your file path
3. Run the transcription:

```bash
python demo.py
```

### Configuration Options

Edit the configuration variables in `demo.py`:

```python
# GPU/CPU selection
device = "cuda"  # Use "cpu" if no GPU available

# Audio file path
audio_file = "./input/your-audio-file.mp3"

# Performance settings
batch_size = 16          # Reduce if low on GPU memory
compute_type = "float16"  # Use "int8" for lower memory/accuracy

# Model storage
model_dir = "./models"   # Local model cache directory
```

### Performance Tuning

**High-end GPU (16GB+ VRAM):**
```python
batch_size = 16
compute_type = "float16"
```

**Mid-range GPU (8-16GB VRAM):**
```python
batch_size = 8
compute_type = "float16"
```

**Low memory GPU (4-8GB VRAM):**
```python
batch_size = 4
compute_type = "int8"
```

## Output Format

The transcription is saved to the `output/` directory with the following format:

```
[start_time - end_time] SPEAKER_ID: transcribed_text
```

**Example output:**
```
[0.03s - 8.52s] SPEAKER_04: Hello and welcome to The English We Speak, where we explain phrases used by fluent English speakers so that you can use them too.
[8.58s - 10.97s] SPEAKER_04: I'm Feifei and I'm here with Beth.
[11.21s - 11.91s] SPEAKER_03: Hi, Beth.
[12.03s - 12.81s] SPEAKER_03: Hi, Feifei.
```

## Supported Audio Formats

- MP3
- WAV
- FLAC
- M4A
- Any format supported by `torchaudio`

## Project Structure

```
audio-transcriber-whisperx/
├── demo.py              # Main application script
├── pyproject.toml       # Project configuration
├── requirements.txt     # Dependencies
├── LICENSE              # MIT license
├── README.md            # Project documentation
├── .env_example         # Environment template
├── input/              # Place audio files here
├── output/             # Transcription results
├── models/             # Downloaded models (auto-created)
└── .venv/              # Virtual environment
```

## Troubleshooting

### Common Issues

**GPU Memory Errors:**
- Reduce `batch_size` (try 8, 4, or 2)
- Change `compute_type` to `"int8"`
- Close other GPU-intensive applications

**Missing Hugging Face Token:**
```
Error: Authentication required for pyannote-audio models
```
- Ensure `.env` file contains valid `HF_TOKEN`
- Verify token has appropriate permissions

**CUDA Not Available:**
- Install CUDA toolkit and cuDNN
- Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Set `device = "cpu"` for CPU-only processing (slower)

**Model Download Issues:**
- Ensure stable internet connection
- Models are cached in `./models/` directory
- Clear model cache if corruption suspected

### Performance Tips

1. **First Run**: Models will be downloaded automatically (requires internet)
2. **GPU Memory**: Monitor usage with `nvidia-smi`
3. **Long Audio**: Consider splitting files >30 minutes for better memory management
4. **CPU Fallback**: Set `device = "cpu"` if GPU unavailable (significantly slower)

## Development

### Code Quality
```bash
# Install development tools
pip install black flake8 mypy isort

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Testing
Currently no test suite exists. Contributions welcome!

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure code follows the style guidelines in `AGENTS.md`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**MIT License Summary:**
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❌ No warranty provided
- ❌ No liability assumed

## Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) for the enhanced Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) for the base speech recognition model
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) for speaker diarization capabilities
- [PyTorch](https://pytorch.org/) for the deep learning framework

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a new issue with detailed description and error logs