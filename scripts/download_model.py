#!/usr/bin/env python3
"""
video.cpp Model Downloader and Converter

Downloads LTX-2.3-GGUF model files from HuggingFace and converts them to video.cpp GGUF-VID format.

Usage:
    python scripts/download_model.py [--output-dir ./models]
"""

import argparse
import os
import sys
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HAS_HF = True
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download, list_repo_files
    HAS_HF = True

# Model repository
MODEL_REPO = "unsloth/LTX-2.3-GGUF"
TEXT_ENCODER_REPO = "unsloth/gemma-3-12b-it-qat-GGUF"

# Quantization variants
QUANT_VARIANTS = {
    "Q4_K_M": {
        "size_gb": 15.1,
        "quality": "Balanced",
        "recommended": True,
    },
    "Q5_K_S": {
        "size_gb": 15.8,
        "quality": "Better",
        "recommended": False,
    },
    "Q8_0": {
        "size_gb": 22.8,
        "quality": "Best",
        "recommended": False,
    },
    "Q4_K_S": {
        "size_gb": 13.7,
        "quality": "Smaller",
        "recommended": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download LTX-2.3-GGUF model files for video.cpp"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./models"),
        help="Output directory for model files"
    )
    parser.add_argument(
        "--quant", "-q",
        choices=list(QUANT_VARIANTS.keys()),
        default="Q4_K_M",
        help="Quantization variant to download"
    )
    parser.add_argument(
        "--include-text-encoder", "-t",
        action="store_true",
        default=True,
        help="Download text encoder model"
    )
    parser.add_argument(
        "--include-vae", "-v",
        action="store_true",
        default=True,
        help="Download VAE models"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist"
    )
    parser.add_argument(
        "--list-only", "-l",
        action="store_true",
        help="Only list available files"
    )
    return parser.parse_args()


def list_available_files():
    """List all files available in the model repository."""
    print(f"\nFetching file list from {MODEL_REPO}...")
    try:
        files = list_repo_files(MODEL_REPO)
        print(f"\nAvailable files ({len(files)}):")
        for f in sorted(files):
            size = "?"
            if f.endswith(".gguf"):
                size = "model"
            print(f"  - {f}")
        return files
    except Exception as e:
        print(f"Error fetching file list: {e}")
        return []


def download_file(repo_id: str, filename: str, output_dir: Path, 
                  skip_existing: bool = True) -> Optional[Path]:
    """Download a single file from HuggingFace Hub."""
    output_path = output_dir / filename
    
    if skip_existing and output_path.exists():
        print(f"  Skipping {filename} (already exists)")
        return output_path
    
    print(f"  Downloading {filename}...")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        return Path(path)
    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
        return None


def download_model(args) -> bool:
    """Download the main model files."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    quant = args.quant
    print(f"\n{'='*60}")
    print(f"video.cpp Model Downloader")
    print(f"{'='*60}")
    print(f"\nModel: {MODEL_REPO}")
    print(f"Quantization: {quant}")
    print(f"Output: {output_dir}")
    print(f"\nQuantization variants:")
    for name, info in QUANT_VARIANTS.items():
        marker = " (recommended)" if info["recommended"] else ""
        print(f"  - {name}: {info['size_gb']}GB ({info['quality']}){marker}")
    print()
    
    # Files to download
    unet_model = f"ltx-2.3-22b-dev-{quant}.gguf"
    vae_video = "vae/ltx-2.3-22b-dev_video_vae.safetensors"
    vae_audio = "vae/ltx-2.3-22b-dev_audio_vae.safetensors"
    connectors = "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors"
    
    # Text encoder files
    gemma_model = f"gemma-3-12b-it-qat-{quant}.gguf"
    mmproj = "mmproj-BF16.gguf"
    
    print(f"\nDownloading DiT model...")
    result = download_file(MODEL_REPO, unet_model, output_dir, args.skip_existing)
    if not result:
        print("Failed to download DiT model!")
        return False
    
    if args.include_vae:
        print(f"\nDownloading VAE models...")
        download_file(MODEL_REPO, vae_video, output_dir / "vae", args.skip_existing)
        download_file(MODEL_REPO, vae_audio, output_dir / "vae", args.skip_existing)
    
    if args.include_text_encoder:
        print(f"\nDownloading text encoder...")
        download_file(TEXT_ENCODER_REPO, gemma_model, output_dir / "text_encoders", args.skip_existing)
        download_file(TEXT_ENCODER_REPO, mmproj, output_dir / "text_encoders", args.skip_existing)
        download_file(MODEL_REPO, connectors, output_dir / "text_encoders", args.skip_existing)
    
    # Create config file
    config = {
        "model_type": "ltx-video",
        "architecture": "ltxv",
        "quantization": quant,
        "model_repo": MODEL_REPO,
        "version": "2.3",
        "parameters": "22B",
        "latent_channels": 16,
        "latent_height": 64,
        "latent_width": 64,
        "latent_frames": 9,
        "frame_rate": 24,
        "video_vae": "vae/ltx-2.3-22b-dev_video_vae.safetensors",
        "audio_vae": "vae/ltx-2.3-22b-dev_audio_vae.safetensors",
        "text_encoder": f"text_encoders/{gemma_model}",
        "text_encoder_mmproj": "text_encoders/mmproj-BF16.gguf",
        "text_encoder_connectors": "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors",
        "requirements": {
            "min_vram_gb": 8 if quant == "Q4_K_S" else 12,
            "recommended_vram_gb": 16,
            "min_steps": 20,
            "resolution_divisible_by": 32,
            "frames_divisible_by": 9,
        }
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: {config_path}")
    
    # Create .env file
    env_content = f'''# video.cpp Configuration
# Generated by download_model.py

VIDEO_MODEL_PATH={unet_model}
VIDEO_VAE_PATH=vae/ltx-2.3-22b-dev_video_vae.safetensors
VIDEO_AUDIO_VAE_PATH=vae/ltx-2.3-22b-dev_audio_vae.safetensors
VIDEO_TEXT_ENCODER_PATH=text_encoders/{gemma_model}
VIDEO_TEXT_ENCODER_MMPROJ=text_encoders/mmproj-BF16.gguf

# Backend
VIDEO_BACKEND=cpu
VIDEO_USE_GPU=true
VIDEO_VRAM_SIZE_MB=16384

# Generation
VIDEO_STEPS=30
VIDEO_GUIDANCE_SCALE=7.5
VIDEO_SAMPLER=euler
VIDEO_FPS=24

# Quantization
VIDEO_QUANTIZATION={quant}
'''
    
    env_path = output_dir / ".env"
    with open(env_path, "w") as f:
        f.write(env_content)
    print(f".env saved to: {env_path}")
    
    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")
    print(f"\nFiles downloaded to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. cd {output_dir}")
    print(f"  2. Build video.cpp core: cd core && cargo build --release")
    print(f"  3. Run: ../bin/video -m {unet_model} -p 'your prompt'")
    
    return True


def main():
    args = parse_args()
    
    if args.list_only:
        list_available_files()
        return
    
    success = download_model(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
