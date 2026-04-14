import os
import shutil
import yaml
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

if not token:
    print("Please set HF_TOKEN in your .env file to download the models initially.")
    exit(1)

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "pyannote"))
os.makedirs(model_dir, exist_ok=True)

# List of repositories and files needed for diarization locally
repos = [
    ("pyannote/speaker-diarization-3.1", "config.yaml", "config.yaml"),
    ("pyannote/segmentation-3.0", "pytorch_model.bin", "segmentation-3.0_pytorch_model.bin"),
    ("pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin", "wespeaker-voxceleb-resnet34-LM_pytorch_model.bin"),
]

paths = {}
for repo, filename, local_filename in repos:
    print(f"Downloading {filename} from {repo}...")
    try:
        downloaded_path = hf_hub_download(repo_id=repo, filename=filename, token=token)
        local_path = os.path.join(model_dir, local_filename)
        shutil.copy2(downloaded_path, local_path)
        paths[repo] = local_path
        print(f"Saved to {local_path}")
    except Exception as e:
        print(f"Failed to download {repo}/{filename}: {e}")
        exit(1)

# Now rewrite config.yaml to use the local files
config_path = paths["pyannote/speaker-diarization-3.1"]

print(f"Updating {config_path} with local paths...")

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# The keys in the config.yaml that need to be updated
config['pipeline']['params']['segmentation'] = paths["pyannote/segmentation-3.0"]
config['pipeline']['params']['embedding'] = paths["pyannote/wespeaker-voxceleb-resnet34-LM"]

with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False)

print("All models downloaded and config.yaml updated successfully.")
print(f"Your local config is located at: {config_path}")
print("You can now safely run your diarization offline using this configuration path.")
