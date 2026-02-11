# app.py
from flask import Flask, request, jsonify, send_file, render_template
import os
import torch
import warnings
import typing
import json
import whisperx
from whisperx.diarize import DiarizationPipeline
from pyannote.audio.core.task import Specifications, Problem, Resolution
import omegaconf
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import tempfile
import threading
import uuid
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import shutil

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", message=".*Model was trained with.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*TensorFloat-32.*")

# Configure logging
def setup_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s (%(funcName)s:%(lineno)d): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler for all logs
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # File handler for errors only
    error_handler = RotatingFileHandler(
        'logs/error.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger

# Setup logging
logger = setup_logging()

app = Flask(__name__)

# Store transcription jobs and their progress
jobs = {}

# Load environment variables
load_dotenv()
logger.info("Application starting...")
logger.info("Environment variables loaded")

# Fix for PyTorch 2.8+ compatibility with pyannote-audio
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load
torch.serialization.add_safe_globals([
    torch.torch_version.TorchVersion,
    Problem,
    Specifications,
    Resolution,
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    omegaconf._utils.ValueKind,
    typing.Any,
])

# Model config
model_dir = "./models"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
batch_size = 16

logger.info(f"Device: {device}")
logger.info(f"Compute type: {compute_type}")
logger.info(f"Batch size: {batch_size}")
logger.info(f"Model directory: {model_dir}")

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
else:
    logger.warning("Running on CPU - transcription will be slower")

# Check if ffmpeg is available
ffmpeg_available = shutil.which("ffmpeg") is not None
if ffmpeg_available:
    logger.info("FFmpeg detected - video file support enabled")
else:
    logger.warning("FFmpeg not found - video file support disabled")

# Supported file extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.wma', '.aac'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg'}


def is_video_file(filename):
    """Check if file is a video based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in VIDEO_EXTENSIONS


def extract_audio_from_video(video_path, output_path):
    """Extract audio from video file using ffmpeg."""
    try:
        logger.debug(f"Extracting audio from video: {video_path}")
        # Use ffmpeg to extract audio
        # -i: input file
        # -vn: disable video
        # -acodec pcm_s16le: convert to WAV PCM format
        # -ar 16000: resample to 16kHz (WhisperX default)
        # -ac 1: convert to mono
        # -y: overwrite output file
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # no video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # mono
            '-y',  # overwrite
            output_path
        ]
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        logger.info(f"Audio extracted successfully to: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return False


@app.route("/")
def home():
    logger.debug("Home page accessed")
    return render_template("upload.html")


def update_progress(job_id, step, message):
    """Update job progress."""
    jobs[job_id]["progress"] = step
    jobs[job_id]["message"] = message
    jobs[job_id]["updated_at"] = datetime.now().isoformat()
    logger.info(f"Job {job_id[:8]}: {step}% - {message}")


def transcribe_audio(job_id, temp_path, filename, speaker_names=None):
    """Background transcription task."""
    logger.info(f"Starting transcription for job {job_id[:8]} - File: {filename}")
    start_time = datetime.now()
    
    try:
        # 1. Load Whisper model
        update_progress(job_id, 12, "Loading Whisper model...")
        logger.debug(f"Job {job_id[:8]}: Loading Whisper model from {model_dir}")
        model = whisperx.load_model(
            "large-v2", device, compute_type=compute_type, download_root=model_dir
        )
        logger.debug(f"Job {job_id[:8]}: Model loaded successfully")

        # 2. Load audio
        update_progress(job_id, 25, "Loading audio file...")
        logger.debug(f"Job {job_id[:8]}: Loading audio from {temp_path}")
        audio = whisperx.load_audio(temp_path)
        audio_duration = len(audio) / 16000  # Assuming 16kHz sample rate
        logger.info(f"Job {job_id[:8]}: Audio loaded - Duration: {audio_duration:.2f}s")

        # 3. Transcribe
        update_progress(job_id, 37, "Transcribing audio...")
        logger.debug(f"Job {job_id[:8]}: Starting transcription with batch_size={batch_size}")
        result = model.transcribe(audio, batch_size=batch_size)

        # Check if transcription has required fields
        if "language" not in result:
            logger.error(f"Job {job_id[:8]}: Transcription failed - no language detected")
            raise ValueError("Transcription failed: no language detected")
        if "segments" not in result or not result["segments"]:
            logger.error(f"Job {job_id[:8]}: Transcription failed - no segments generated")
            raise ValueError("Transcription failed: no segments generated")
        
        detected_language = result["language"]
        segment_count = len(result["segments"])
        logger.info(f"Job {job_id[:8]}: Transcription complete - Language: {detected_language}, Segments: {segment_count}")

        # 4. Align whisper output
        update_progress(job_id, 50, "Aligning transcription...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False
        )

        # 5. Assign speaker labels
        update_progress(job_id, 62, "Loading diarization model...")
        logger.debug(f"Job {job_id[:8]}: Loading diarization model")
        diarize_model = DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)
        
        update_progress(job_id, 75, "Identifying speakers...")
        logger.debug(f"Job {job_id[:8]}: Running speaker diarization")
        diarize_segments = diarize_model(audio)
        
        update_progress(job_id, 87, "Assigning speakers to segments...")
        logger.debug(f"Job {job_id[:8]}: Assigning speakers to words")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Count unique speakers
        speakers = set()
        for segment in result.get("segments", []):
            if "speaker" in segment:
                speakers.add(segment["speaker"])
        logger.info(f"Job {job_id[:8]}: Identified {len(speakers)} speakers")

        # Format output
        update_progress(job_id, 95, "Formatting transcription...")
        transcription_text = []
        
        # Create speaker name mapping
        speaker_mapping = {}
        if speaker_names and isinstance(speaker_names, list):
            for i, name in enumerate(speaker_names):
                speaker_mapping[f"SPEAKER_{i:02d}"] = name
            logger.debug(f"Job {job_id[:8]}: Using speaker mapping: {speaker_mapping}")
        
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "UNKNOWN")
            # Replace speaker ID with custom name if available
            if speaker in speaker_mapping:
                speaker_display = speaker_mapping[speaker]
            else:
                speaker_display = speaker
            
            text = segment.get("text", "")
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            transcription_text.append(f"[{start:.2f}s - {end:.2f}s] {speaker_display}: {text}")

        # Clean up temp file
        logger.debug(f"Job {job_id[:8]}: Cleaning up temp file")
        os.remove(temp_path)

        # Mark as complete
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Transcription complete!"
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "transcription": "\n".join(transcription_text),
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
            "file": filename
        }
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"Job {job_id[:8]}: Completed successfully in {processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Job {job_id[:8]}: Failed with error: {str(e)}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 0
        jobs[job_id]["message"] = f"Error: {str(e)}"
        # Try to clean up temp file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"Job {job_id[:8]}: Temp file cleaned up after error")
        except Exception as cleanup_error:
            logger.warning(f"Job {job_id[:8]}: Failed to clean up temp file: {cleanup_error}")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        logger.info(f"Transcription request received from {request.remote_addr}")
        
        if "audio_file" not in request.files:
            logger.warning("No file uploaded in request")
            return jsonify({"error": "No file uploaded"}), 400

        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            logger.warning("Empty filename in request")
            return jsonify({"error": "No file selected"}), 400
        
        # Get speaker names from request (optional)
        speaker_names = None
        if "speaker_names" in request.form:
            try:
                speaker_names_json = request.form["speaker_names"]
                speaker_names = json.loads(speaker_names_json) if speaker_names_json else None
                if speaker_names:
                    logger.info(f"Speaker names provided: {speaker_names}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse speaker_names JSON: {e}")

        # Save to temp
        filename = secure_filename(audio_file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        logger.debug(f"Saving uploaded file to: {temp_path}")
        audio_file.save(temp_path)
        
        file_size = os.path.getsize(temp_path) / (1024 * 1024)  # MB
        logger.info(f"File uploaded: {filename} ({file_size:.2f} MB)")
        
        # Check if it's a video file and extract audio
        audio_path = temp_path
        extracted_audio = False
        if is_video_file(filename):
            if not ffmpeg_available:
                logger.error(f"Video file uploaded but FFmpeg not available: {filename}")
                os.remove(temp_path)
                return jsonify({"error": "Video files not supported (FFmpeg not installed)"}), 400
            
            logger.info(f"Video file detected: {filename}, extracting audio...")
            audio_path = os.path.join(tempfile.gettempdir(), f"extracted_{uuid.uuid4().hex[:8]}.wav")
            
            if not extract_audio_from_video(temp_path, audio_path):
                logger.error(f"Failed to extract audio from video: {filename}")
                os.remove(temp_path)
                return jsonify({"error": "Failed to extract audio from video"}), 400
            
            # Remove original video file
            os.remove(temp_path)
            extracted_audio = True
            logger.info(f"Audio extracted successfully, will process: {audio_path}")

        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting transcription...",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "speaker_names": speaker_names
        }
        logger.info(f"Created job {job_id[:8]} for file: {filename}")

        # Start background transcription
        thread = threading.Thread(target=transcribe_audio, args=(job_id, audio_path, filename, speaker_names))
        thread.daemon = True
        thread.start()
        logger.debug(f"Started background thread for job {job_id[:8]}")
        
        message = "Transcription started"
        if extracted_audio:
            message = "Audio extracted from video, transcription started"

        return jsonify({
            "job_id": job_id,
            "status": "processing",
            "message": message
        })

    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route("/progress/<job_id>", methods=["GET"])
def get_progress(job_id):
    """Get progress of a transcription job."""
    logger.debug(f"Progress check for job {job_id[:8]}")
    
    if job_id not in jobs:
        logger.warning(f"Progress requested for unknown job: {job_id[:8]}")
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    response = {
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"]
    }
    
    if job["status"] == "completed":
        response["result"] = job["result"]
        logger.debug(f"Job {job_id[:8]}: Returning completed result")
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")
        logger.debug(f"Job {job_id[:8]}: Returning error status")
    
    return jsonify(response)


@app.route("/download/<job_id>", methods=["GET"])
def download_transcription(job_id):
    """Download transcription as a text file."""
    logger.info(f"Download requested for job {job_id[:8]} from {request.remote_addr}")
    
    if job_id not in jobs:
        logger.warning(f"Download requested for unknown job: {job_id[:8]}")
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        logger.warning(f"Download requested for incomplete job {job_id[:8]}: {job['status']}")
        return jsonify({"error": "Transcription not completed yet"}), 400
    
    result = job["result"]
    transcription = result["transcription"]
    filename = result["file"]
    
    # Create a temporary file with the transcription
    temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt')
    try:
        # Write header
        temp_file.write(f"Transcription of: {filename}\n")
        temp_file.write(f"Language: {result['language']}\n")
        temp_file.write(f"Generated: {job['created_at']}\n")
        temp_file.write("=" * 80 + "\n\n")
        
        # Write transcription
        temp_file.write(transcription)
        temp_file.close()
        
        # Generate download filename
        base_name = os.path.splitext(filename)[0]
        download_name = f"transcription_{base_name}.txt"
        
        logger.info(f"Sending download file for job {job_id[:8]}: {download_name}")
        
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/plain'
        )
    except Exception as e:
        logger.error(f"Error creating download for job {job_id[:8]}: {str(e)}", exc_info=True)
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Starting Flask server on http://localhost:5000")
    logger.info("="*60)
    app.run(debug=True, host="0.0.0.0", port=5000)


