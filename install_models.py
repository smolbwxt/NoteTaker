"""Unpack bundled models on an airgapped PC.

Usage (after downloading the release asset):
    python install_models.py notetaker_models.zip

Or, if 'notetaker_models.zip' is in the same directory as this script:
    python install_models.py

This places model files into the standard cache directories under ~/.cache
so that WhisperX, pyannote, and wav2vec2 find them automatically.
"""

import os
import sys
import zipfile

HOME = os.path.expanduser("~")

# Maps archive prefix → local destination
DEST_MAP = {
    "huggingface/hub/models--Systran--faster-whisper-medium":
        os.path.join(HOME, ".cache", "huggingface", "hub", "models--Systran--faster-whisper-medium"),
    "torch/pyannote":
        os.path.join(HOME, ".cache", "torch", "pyannote"),
    "torch/hub/checkpoints":
        os.path.join(HOME, ".cache", "torch", "hub", "checkpoints"),
}


def main():
    # Find the zip file
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    else:
        zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notetaker_models.zip")

    if not os.path.isfile(zip_path):
        print(f"ERROR: Cannot find '{zip_path}'")
        print()
        print("Usage:  python install_models.py [path/to/notetaker_models.zip]")
        sys.exit(1)

    print(f"Unpacking: {zip_path}")
    print(f"Target:    {os.path.join(HOME, '.cache')}")
    print()

    extracted = 0
    skipped = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            # Figure out which prefix this file belongs to
            dest_dir = None
            rel_path = None
            for prefix, target in DEST_MAP.items():
                if member.filename.startswith(prefix + "/"):
                    dest_dir = target
                    rel_path = member.filename[len(prefix) + 1:]
                    break

            if dest_dir is None:
                print(f"  SKIP (unknown prefix): {member.filename}")
                skipped += 1
                continue

            out_path = os.path.join(dest_dir, rel_path.replace("/", os.sep))
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)

            # Extract the file
            with zf.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

            extracted += 1

    print()
    print(f"Done! Extracted {extracted} files.")
    if skipped:
        print(f"  ({skipped} files skipped — unknown prefix)")
    print()
    print("Models are now installed in ~/.cache/ and ready to use.")
    print("You can delete the zip file to save disk space.")


if __name__ == "__main__":
    main()
