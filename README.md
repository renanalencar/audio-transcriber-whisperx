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
- **CLI Interface**: Command-line interface with flexible argument parsing
- **Custom Speaker Names**: Map speaker IDs to custom names (e.g., "John", "Mary")
- **Speaker Constraints**: Configure minimum and maximum number of speakers
- **Language Override**: Manual language specification to override auto-detection

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

**Option A: Using the CLI script (recommended)**
```bash
# Basic usage with default settings
python transcribe.py

# Specify input and output files
python transcribe.py -i ./input/your-audio-file.mp3 -o ./output/your-transcription.txt

# Specify language (optional, auto-detects if not provided)
python transcribe.py -i ./input/audio.mp3 -l pt  # Portuguese
python transcribe.py -i ./input/audio.mp3 -l en  # English

# Set speaker constraints
python transcribe.py -i ./input/audio.mp3 --min-speakers 2 --max-speakers 4

# Use custom speaker names
python transcribe.py -i ./input/audio.mp3 --speaker-names "SPEAKER_00:John,SPEAKER_01:Mary"
# Or using JSON format
python transcribe.py -i ./input/audio.mp3 --speaker-names '{"SPEAKER_00": "John", "SPEAKER_01": "Mary"}'
```

**Option B: Using the main script**
1. Place your audio file in the `input/` directory
2. Update the `audio_file` variable in `main.py` with your file path
3. Run the transcription:

```bash
python main.py
```

### Configuration Options

**CLI Arguments (for transcribe.py):**
```bash
# View all available options
python transcribe.py --help

# Key options:
-i, --input          Input audio file path (default: ./input/audio-pt_BR.mp3)
-o, --output         Output transcription file path (default: ./output/transcription-pt_BR.txt)
-l, --language       Language code (e.g., 'pt', 'en', 'es'). Auto-detect if not specified.
--min-speakers       Minimum number of speakers (optional)
--max-speakers       Maximum number of speakers (optional)
--speaker-names      Speaker name mapping (optional)
```

**Direct Configuration (for main.py):**

Edit the configuration variables in `main.py`:

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

### CLI Usage Examples

**Basic transcription:**
```bash
# Transcribe with default settings
python transcribe.py -i ./input/meeting.mp3

# Transcribe with custom output location
python transcribe.py -i ./input/meeting.mp3 -o ./output/meeting_notes.txt
```

**Multi-language support:**
```bash
# Auto-detect language (default)
python transcribe.py -i ./input/international_call.mp3

# Force specific language
python transcribe.py -i ./input/portuguese_audio.mp3 -l pt
python transcribe.py -i ./input/english_audio.mp3 -l en
python transcribe.py -i ./input/spanish_audio.mp3 -l es
```

**Speaker management:**
```bash
# Set speaker constraints
python transcribe.py -i ./input/interview.mp3 --min-speakers 2 --max-speakers 3

# Use custom speaker names (comma-separated format)
python transcribe.py -i ./input/meeting.mp3 --speaker-names "SPEAKER_00:Alice,SPEAKER_01:Bob,SPEAKER_02:Carol"

# Use custom speaker names (JSON format)
python transcribe.py -i ./input/meeting.mp3 --speaker-names '{"SPEAKER_00": "CEO", "SPEAKER_01": "CTO", "SPEAKER_02": "CFO"}'
```

**Complete example with all options:**
```bash
python transcribe.py \
  -i ./input/board_meeting.mp3 \
  -o ./output/board_meeting_transcript.txt \
  -l en \
  --min-speakers 3 \
  --max-speakers 5 \
  --speaker-names "SPEAKER_00:Chairman,SPEAKER_01:CEO,SPEAKER_02:Board Member"
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
├── main.py              # Main application script (basic usage)
├── transcribe.py        # CLI script with advanced options
├── pyproject.toml       # Project configuration
├── requirements.txt     # Dependencies
├── LICENSE              # MIT license
├── README.md            # Project documentation
├── AGENTS.md            # Development guide
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

### Development Guidelines
For detailed development guidelines, code style conventions, and project architecture information, see the [AGENTS.md](AGENTS.md) file. This guide includes:

- Code style guidelines and formatting standards
- Import organization and naming conventions
- Error handling best practices
- Project architecture overview
- Performance considerations
- Security guidelines

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure code follows the style guidelines in [AGENTS.md](AGENTS.md)
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