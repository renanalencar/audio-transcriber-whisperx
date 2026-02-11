# Docker Deployment Guide

## Prerequisites

1. **Docker** and **Docker Compose** installed
2. **NVIDIA Docker runtime** (for GPU support)
   - Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
3. **NVIDIA GPU** with CUDA support
4. **Hugging Face account** and API token

## Quick Start

### 1. Setup Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Hugging Face token
# HF_TOKEN=your_actual_token_here
```

### 2. Build and Run with Docker Compose

```bash
# Build the Docker image
docker-compose build

# Start the container
docker-compose up -d

# View logs
docker-compose logs -f
```

The app will be available at: http://localhost:5000

### 3. Stop the Container

```bash
docker-compose down
```

## Manual Docker Commands

### Build Image

```bash
docker build -t audio-transcriber-whisperx .
```

### Run Container

```bash
docker run -d \
  --name whisperx-app \
  --gpus all \
  -p 5000:5000 \
  -e HF_TOKEN=your_token_here \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  audio-transcriber-whisperx
```

### View Logs

```bash
docker logs -f whisperx-app
```

### Stop Container

```bash
docker stop whisperx-app
docker rm whisperx-app
```

## GPU Support

### Verify GPU Access

```bash
# Check if GPU is accessible in the container
docker run --rm --gpus all audio-transcriber-whisperx nvidia-smi
```

### Troubleshooting GPU Issues

If GPU is not detected:

1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```

2. Install NVIDIA Container Toolkit:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. Test GPU access:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

## Volume Mounts

The Docker Compose configuration mounts these directories:

- `./models:/app/models` - Persists downloaded WhisperX models
- `./output:/app/output` - Stores transcription outputs
- `./input:/app/input` - Optional input audio files

This ensures models are downloaded once and reused across container restarts.

## Production Deployment

### Using Gunicorn (Recommended)

Modify the Dockerfile CMD:

```dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
```

Install Gunicorn:

```bash
pip install gunicorn
```

Add to requirements.txt:
```
gunicorn==21.2.0
```

### Environment Variables

Production environment variables can be set in `docker-compose.yml`:

```yaml
environment:
  - HF_TOKEN=${HF_TOKEN}
  - FLASK_ENV=production
  - WORKERS=1
  - CUDA_VISIBLE_DEVICES=0
```

### Resource Limits

Adjust GPU memory and CPU limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 16G
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Monitoring

### Check Container Health

```bash
docker ps
docker inspect whisperx-app | grep -A 10 Health
```

### Monitor Resource Usage

```bash
docker stats whisperx-app
```

### View GPU Usage

```bash
docker exec whisperx-app nvidia-smi
```

## Common Issues

### Port Already in Use

```bash
# Change port in docker-compose.yml
ports:
  - "8080:5000"  # Use port 8080 instead
```

### Out of Memory

Reduce batch size in app.py:
```python
batch_size = 4  # Reduce from 16 if needed
```

### Models Not Persisting

Ensure the models directory has correct permissions:
```bash
chmod -R 755 ./models
```

## Security Notes

- Never commit `.env` file to version control
- Use secrets management in production (e.g., Docker Secrets, Kubernetes Secrets)
- Consider adding authentication to the Flask app
- Use HTTPS in production with a reverse proxy (Nginx, Traefik)
