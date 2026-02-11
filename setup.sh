#!/bin/bash
# Audio Transcriber WhisperX Setup Script (Linux/Mac)
# Run this script with: bash setup.sh or ./setup.sh

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}===========================================================${NC}"
echo -e "${CYAN}  Audio Transcriber WhisperX - Setup Script (Linux/Mac)${NC}"
echo -e "${CYAN}===========================================================${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}[1/8] Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed${NC}"
    echo -e "${RED}Please install Python 3.9+ first${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}Found: $PYTHON_VERSION${NC}"

# Check for CUDA/GPU
echo ""
echo -e "${YELLOW}[2/8] Checking for NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected! CUDA support will be enabled.${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}No NVIDIA GPU detected. CPU mode will be used.${NC}"
fi

# Create virtual environment
echo ""
echo -e "${YELLOW}[3/8] Creating virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists.${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        python3 -m venv .venv
        echo -e "${GREEN}Virtual environment recreated.${NC}"
    fi
else
    python3 -m venv .venv
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}[4/8] Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}Virtual environment activated.${NC}"

# Upgrade pip
echo ""
echo -e "${YELLOW}[5/8] Upgrading pip...${NC}"
python -m pip install --upgrade pip setuptools wheel > /dev/null
echo -e "${GREEN}pip upgraded successfully.${NC}"

# Install dependencies
echo ""
echo -e "${YELLOW}[6/8] Installing dependencies...${NC}"
echo -e "${YELLOW}This may take several minutes...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed successfully.${NC}"
else
    echo -e "${RED}ERROR: requirements.txt not found${NC}"
    exit 1
fi

# Create necessary directories
echo ""
echo -e "${YELLOW}[7/8] Creating directories...${NC}"
directories=("models" "output" "input" "static" "templates")
for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}Created: $dir/${NC}"
    else
        echo -e "${YELLOW}Exists: $dir/${NC}"
    fi
done

# Setup environment file
echo ""
echo -e "${YELLOW}[8/8] Setting up environment file...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}.env file created from .env.example${NC}"
        echo ""
        echo -e "${RED}IMPORTANT: Edit .env and add your Hugging Face token!${NC}"
        echo -e "${YELLOW}Get your token from: https://huggingface.co/settings/tokens${NC}"
    else
        echo -e "${YELLOW}WARNING: .env.example not found${NC}"
    fi
else
    echo -e "${YELLOW}.env file already exists.${NC}"
fi

# Summary
echo ""
echo -e "${CYAN}===========================================================${NC}"
echo -e "${CYAN}  Setup Complete!${NC}"
echo -e "${CYAN}===========================================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "${NC}1. Edit .env and add your HF_TOKEN${NC}"
echo -e "${NC}2. Run the Flask app:${NC}"
echo -e "${CYAN}   python app.py${NC}"
echo -e "${NC}3. Or run the CLI tool:${NC}"
echo -e "${CYAN}   python transcribe.py -i input/your-audio.mp3${NC}"
echo ""
echo -e "${YELLOW}For Docker deployment, see DOCKER.md${NC}"
echo ""
echo -e "${YELLOW}Remember to activate the virtual environment:${NC}"
echo -e "${CYAN}   source .venv/bin/activate${NC}"
echo ""
