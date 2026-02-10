"""Bundle all cached models into a zip for transfer to an airgapped PC.

Run on the machine that has internet access (after a successful transcription):
    python bundle_models.py

This creates 'notetaker_models.zip' containing:
  - Whisper medium model (faster-whisper)
  - wav2vec2 alignment model
  - pyannote diarization models (segmentation, speaker-diarization, wespeaker)

Then upload as a GitHub Release:
    gh release create v1.0-models notetaker_models.zip --title "Model bundle"

On the airgapped PC, run install_models.py to unpack.
"""

import os
import zipfile
import sys

HOME = os.path.expanduser("~")

# (archive_path_prefix, source_directory)
MODEL_DIRS = [
    ("huggingface/hub/models--Systran--faster-whisper-medium",
     os.path.join(HOME, ".cache", "huggingface", "hub", "models--Systran--faster-whisper-medium")),
    ("torch/pyannote",
     os.path.join(HOME, ".cache", "torch", "pyannote")),
    ("torch/hub/checkpoints",
     os.path.join(HOME, ".cache", "torch", "hub", "checkpoints")),
]

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notetaker_models.zip")


def main():
    total_size = 0
    file_count = 0

    print("Bundling models into", OUTPUT)
    print()

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for prefix, src_dir in MODEL_DIRS:
            if not os.path.isdir(src_dir):
                print(f"  SKIP (not found): {src_dir}")
                continue
            print(f"  Adding: {src_dir}")
            for root, _, files in os.walk(src_dir):
                for fname in files:
                    full_path = os.path.join(root, fname)
                    rel = os.path.relpath(full_path, src_dir)
                    arc_name = os.path.join(prefix, rel).replace("\\", "/")
                    zf.write(full_path, arc_name)
                    size = os.path.getsize(full_path)
                    total_size += size
                    file_count += 1

    zip_size = os.path.getsize(OUTPUT)
    print()
    print(f"Done! {file_count} files, {total_size / 1024 / 1024:.0f} MB uncompressed")
    print(f"Zip size: {zip_size / 1024 / 1024:.0f} MB")
    print()
    print("Next steps:")
    print(f"  1. gh release create v1.0-models \"{OUTPUT}\" --title \"Model bundle\"")
    print("  2. On the airgapped PC:  gh release download v1.0-models")
    print("  3. Then run:  python install_models.py")


if __name__ == "__main__":
    main()
