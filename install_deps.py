#!/usr/bin/env python3
"""Install pip dependencies from a local directory of .whl files.

Setup:
    1. On a PC with internet, download the wheels:
           pip download -r requirements.txt -d ./notetaker_deps/
       (Also download torch/torchaudio separately if needed)

    2. Copy the wheel directory to the airgapped PC

    3. Set the path in .notetaker_config.json:
           { "deps_dir": "C:\\path\\to\\notetaker_deps" }

    4. Run this script:
           python install_deps.py

Or pass the path directly:
    python install_deps.py C:\\path\\to\\notetaker_deps
"""

import os
import sys
import subprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(_SCRIPT_DIR, "requirements.txt")


def main():
    # Resolve deps directory: CLI arg > config > ./notetaker_deps/
    deps_dir = None

    if len(sys.argv) > 1:
        deps_dir = sys.argv[1]
    else:
        from config import ConfigManager
        cfg = ConfigManager()
        deps_dir = cfg.get("deps_dir", "")

    if not deps_dir:
        deps_dir = os.path.join(_SCRIPT_DIR, "notetaker_deps")

    if not os.path.isdir(deps_dir):
        print(f"ERROR: Dependencies directory not found: {deps_dir}")
        print()
        print("Options:")
        print(f"  1. Set \"deps_dir\" in .notetaker_config.json")
        print(f"  2. Pass the path:  python install_deps.py C:\\path\\to\\wheels")
        print(f"  3. Place wheels in: {os.path.join(_SCRIPT_DIR, 'notetaker_deps')}")
        sys.exit(1)

    wheels = [f for f in os.listdir(deps_dir)
              if f.endswith((".whl", ".tar.gz", ".zip"))]

    if not wheels:
        print(f"ERROR: No .whl files found in {deps_dir}")
        sys.exit(1)

    print(f"Installing from: {deps_dir}")
    print(f"Found {len(wheels)} package(s)")
    print()

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-index",
        "--find-links", deps_dir,
        "-r", REQUIREMENTS,
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
