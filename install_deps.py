#!/usr/bin/env python3
"""Install pip dependencies from a local directory of .whl files.

Setup:
    1. On a PC with internet, download the wheels:
           pip download -r requirements-cli.txt -d ./notetaker_deps/
       (Also download torch/torchaudio separately if needed)

    2. Copy the wheel directory to the target PC

    3. Run this script:
           python install_deps.py /path/to/notetaker_deps

    For Linux/HPC (no recording, no PyAudioWPatch):
        python install_deps.py /path/to/notetaker_deps --cli

    For Windows (full GUI + recording):
        python install_deps.py C:\\path\\to\\notetaker_deps

Or set the path in .notetaker_config.json:
    { "deps_dir": "/path/to/notetaker_deps" }
"""

import os
import sys
import subprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FULL = os.path.join(_SCRIPT_DIR, "requirements.txt")
REQUIREMENTS_CLI = os.path.join(_SCRIPT_DIR, "requirements-cli.txt")


def main():
    # Parse args
    deps_dir = None
    cli_only = False

    for arg in sys.argv[1:]:
        if arg == "--cli":
            cli_only = True
        elif not arg.startswith("-"):
            deps_dir = arg

    # Fall back to config
    if deps_dir is None:
        from config import ConfigManager
        cfg = ConfigManager()
        deps_dir = cfg.get("deps_dir", "")

    if not deps_dir:
        deps_dir = os.path.join(_SCRIPT_DIR, "notetaker_deps")

    if not os.path.isdir(deps_dir):
        print(f"ERROR: Dependencies directory not found: {deps_dir}")
        print()
        print("Usage:")
        print(f"  python install_deps.py /path/to/wheels         # full (Windows)")
        print(f"  python install_deps.py /path/to/wheels --cli   # CLI only (Linux/HPC)")
        print()
        print("Or set \"deps_dir\" in .notetaker_config.json")
        sys.exit(1)

    wheels = [f for f in os.listdir(deps_dir)
              if f.endswith((".whl", ".tar.gz", ".zip"))]

    if not wheels:
        print(f"ERROR: No .whl files found in {deps_dir}")
        sys.exit(1)

    req_file = REQUIREMENTS_CLI if cli_only else REQUIREMENTS_FULL
    mode = "CLI only (no PyAudioWPatch)" if cli_only else "Full (GUI + recording)"

    print(f"Installing from: {deps_dir}")
    print(f"Mode: {mode}")
    print(f"Requirements: {os.path.basename(req_file)}")
    print(f"Found {len(wheels)} package(s)")
    print()

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-index",
        "--find-links", deps_dir,
        "-r", req_file,
    ]

    print(f"$ {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\nAll dependencies installed successfully!")
    else:
        print(f"\npip exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
