import os
import torch
import warnings
import typing
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline
from pyannote.audio.core.task import Specifications, Problem, Resolution
import omegaconf

# Suppress deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", message=".*Model was trained with.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*TensorFloat-32.*")


def log(message):
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} ̶  {message}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using WhisperX with speaker diarization"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="./input/audio-pt_BR.mp3",
        help="Input audio file path (default: ./input/audio-pt_BR.mp3)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output/transcription-pt_BR.txt",
        help="Output transcription file path (default: ./output/transcription-pt_BR.txt)"
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'pt', 'en', 'es'). If not specified, auto-detect."
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers (optional)"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers (optional)"
    )
    parser.add_argument(
        "--speaker-names",
        type=str,
        default=None,
        help='Speaker name mapping in format "SPEAKER_00:John,SPEAKER_01:Mary" or JSON format {"SPEAKER_00": "John", "SPEAKER_01": "Mary"}'
    )
    return parser.parse_args()


def parse_speaker_names(speaker_names_str):
    """Parse speaker names from command-line argument."""
    if not speaker_names_str:
        return {}
    
    # Try parsing as JSON first
    if speaker_names_str.startswith("{"):
        try:
            return json.loads(speaker_names_str)
        except json.JSONDecodeError:
            log("Warning: Invalid JSON format for speaker names, trying comma-separated format")
    
    # Parse comma-separated format: SPEAKER_00:John,SPEAKER_01:Mary
    speaker_map = {}
    for pair in speaker_names_str.split(","):
        if ":" in pair:
            speaker_id, name = pair.split(":", 1)
            speaker_map[speaker_id.strip()] = name.strip()
    
    return speaker_map


# Load environment variables from .env file
load_dotenv()

# Parse command-line arguments
args = parse_arguments()
speaker_names = parse_speaker_names(args.speaker_names)

if speaker_names:
    log(f"Using speaker name mapping: {speaker_names}")

# Fix for PyTorch 2.8+ compatibility with pyannote-audio
# Temporarily patch torch.load to use weights_only=False for model loading
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    # Force weights_only=False for all torch.load calls
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

torch.serialization.add_safe_globals(
    [
        torch.torch_version.TorchVersion,
        Specifications,
        Problem,
        Resolution,
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
        omegaconf.base.ContainerMetadata,
        omegaconf._utils.ValueKind,
        typing.Any,
    ]
)

device = "cuda"
audio_file = args.input
batch_size = 8  # optimized for RTX 3060 12GB VRAM
compute_type = "float16"  # GPU-optimized precision

# Progress tracking
steps = [
    "Loading model",
    "Loading audio",
    "Transcribing",
    "Aligning output",
    "Loading diarization",
    "Diarizing speakers",
    "Assigning speakers",
    "Saving transcription",
]
progress_bar = tqdm(total=len(steps), desc="Processing", unit="step")

# 1. Transcribe with original whisper (batched)
log("Loading Whisper model...")
progress_bar.set_description(steps[0])
progress_bar.update(1)

# save model to local path (optional)
model_dir = "./models"
model = whisperx.load_model(
    "large-v2", device, compute_type=compute_type, download_root=model_dir
)

log("Loading audio file...")
progress_bar.set_description(steps[1])
audio = whisperx.load_audio(audio_file)
progress_bar.update(1)

log("Transcribing audio...")
progress_bar.set_description(steps[2])
transcribe_options = {"batch_size": batch_size}
if args.language:
    transcribe_options["language"] = args.language
result = model.transcribe(audio, **transcribe_options)
progress_bar.update(1)
# print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
log("Aligning whisper output...")
progress_bar.set_description(steps[3])
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)
result = whisperx.align(
    result["segments"], model_a, metadata, audio, device, return_char_alignments=False
)
progress_bar.update(1)

# print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
log("Loading diarization model...")
progress_bar.set_description(steps[4])
diarize_model = DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)
progress_bar.update(1)

# add min/max number of speakers if known
log("Performing speaker diarization...")
progress_bar.set_description(steps[5])
diarize_options = {}
if args.min_speakers:
    diarize_options["min_speakers"] = args.min_speakers
if args.max_speakers:
    diarize_options["max_speakers"] = args.max_speakers
diarize_segments = diarize_model(audio, **diarize_options)
progress_bar.update(1)

log("Assigning speakers to words...")
progress_bar.set_description(steps[6])
result = whisperx.assign_word_speakers(diarize_segments, result)
progress_bar.update(1)
# print(diarize_segments)
# print(result["segments"]) # segments are now assigned speaker IDs

# Save transcription to text file
log("Saving transcription to file...")
progress_bar.set_description(steps[7])
output_file = args.output
# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    for segment in result["segments"]:
        speaker_id = segment.get("speaker", "UNKNOWN")
        # Map speaker ID to custom name if provided
        speaker = speaker_names.get(speaker_id, speaker_id)
        text = segment["text"]
        start = segment["start"]
        end = segment["end"]
        f.write(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}\n")
progress_bar.update(1)
progress_bar.close()

log(f"Transcription saved to {output_file}")
log("Processing complete!")
