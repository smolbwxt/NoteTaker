"""Unpack bundled models on an airgapped PC.

Usage (after downloading the release asset):
    python install_models.py notetaker_models.zip

Or, if 'notetaker_models.zip' is in the same directory as this script:
    python install_models.py

Optionally specify a custom cache directory (instead of ~/.cache):
    python install_models.py notetaker_models.zip --cache-dir /path/to/cache

Then set "model_cache_dir": "/path/to/cache" in .notetaker_config.json so
the application finds the models there.

This places model files into the standard cache directories so that
WhisperX, pyannote, and wav2vec2 find them automatically.
"""

import os
import sys
import zipfile


def _build_dest_map(cache_root: str) -> dict:
    """Build archive-prefix → local-destination mapping for the given cache root."""
    return {
        "huggingface/hub/models--Systran--faster-whisper-medium":
            os.path.join(cache_root, "huggingface", "hub", "models--Systran--faster-whisper-medium"),
        "torch/pyannote":
            os.path.join(cache_root, "torch", "pyannote"),
        "torch/hub/checkpoints":
            os.path.join(cache_root, "torch", "hub", "checkpoints"),
    }


def main():
    # Parse args
    args = sys.argv[1:]
    cache_dir = None
    zip_path = None

    i = 0
    while i < len(args):
        if args[i] == "--cache-dir" and i + 1 < len(args):
            cache_dir = args[i + 1]
            i += 2
        elif zip_path is None:
            zip_path = args[i]
            i += 1
        else:
            i += 1

    if zip_path is None:
        zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notetaker_models.zip")

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache")

    DEST_MAP = _build_dest_map(cache_dir)

    if not os.path.isfile(zip_path):
        print(f"ERROR: Cannot find '{zip_path}'")
        print()
        print("Usage:  python install_models.py [path/to/notetaker_models.zip]")
        sys.exit(1)

    print(f"Unpacking: {zip_path}")
    print(f"Target:    {cache_dir}")
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
    print(f"Models are now installed in {cache_dir} and ready to use.")
    if cache_dir != os.path.join(os.path.expanduser("~"), ".cache"):
        print(f'Set "model_cache_dir": "{cache_dir}" in .notetaker_config.json')
    print("You can delete the zip file to save disk space.")


if __name__ == "__main__":
    main()
