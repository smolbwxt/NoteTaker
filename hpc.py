"""HPC job submission via SSH/SCP (uses Windows built-in OpenSSH)."""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class HPCConfig:
    host: str
    user: str
    remote_dir: str          # e.g. /scratch/user/notetaker
    partition: str = "gpu"
    nodes: int = 1
    time: str = "01:00:00"
    modules: str = ""        # space-separated module names
    python: str = "python3"
    diarize: bool = True
    model: str = "medium"

    @classmethod
    def from_config(cls, cfg) -> "HPCConfig":
        return cls(
            host=cfg.get("hpc_host", ""),
            user=cfg.get("hpc_user", ""),
            remote_dir=cfg.get("hpc_remote_dir", ""),
            partition=cfg.get("hpc_partition", "gpu"),
            nodes=cfg.get("hpc_nodes", 1),
            time=cfg.get("hpc_time", "01:00:00"),
            modules=cfg.get("hpc_modules", ""),
            python=cfg.get("hpc_python", "python3"),
            diarize=cfg.get("hpc_diarize", True),
            model=cfg.get("hpc_model", "medium"),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.user and self.remote_dir)


def _ssh_target(hpc: HPCConfig) -> str:
    return f"{hpc.user}@{hpc.host}"


def _run_ssh(hpc: HPCConfig, command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command on the cluster via SSH."""
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         _ssh_target(hpc), command],
        capture_output=True, text=True, timeout=30,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"SSH command failed (exit {result.returncode}):\n"
            f"  command: {command}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result


def _run_scp(src: str, dst: str, timeout: int = 300) -> None:
    """Copy a file via SCP."""
    result = subprocess.run(
        ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", src, dst],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SCP failed (exit {result.returncode}):\n"
            f"  {result.stderr.strip()}"
        )


def check_connection(hpc: HPCConfig) -> bool:
    """Test SSH connectivity to the cluster."""
    try:
        result = _run_ssh(hpc, "echo ok", check=False)
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def submit_job(
    hpc: HPCConfig,
    audio_path: str,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    """Upload audio and submit an sbatch job. Returns the SLURM job ID."""

    def _status(msg: str):
        if progress:
            progress(msg)

    remote_uploads = f"{hpc.remote_dir}/uploads"
    remote_results = f"{hpc.remote_dir}/results"
    filename = os.path.basename(audio_path)
    remote_audio = f"{remote_uploads}/{filename}"

    # Ensure remote directories exist
    _status("Preparing remote directories...")
    _run_ssh(hpc, f"mkdir -p {remote_uploads} {remote_results}")

    # Upload audio file
    _status(f"Uploading {filename}...")
    _run_scp(
        audio_path,
        f"{_ssh_target(hpc)}:{remote_audio}",
        timeout=600,  # large files may take a while
    )

    # Build sbatch script
    _status("Generating SLURM job...")
    script = _build_sbatch_script(hpc, remote_audio, remote_results)

    # Upload the sbatch script
    remote_script = f"{hpc.remote_dir}/job_{os.path.splitext(filename)[0]}.sh"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, newline="\n"
    ) as f:
        f.write(script)
        local_script = f.name

    try:
        _run_scp(local_script, f"{_ssh_target(hpc)}:{remote_script}")
    finally:
        os.unlink(local_script)

    # Submit the job
    _status("Submitting job to SLURM...")
    result = _run_ssh(hpc, f"sbatch {remote_script}")

    # Parse job ID from "Submitted batch job 12345"
    job_id = ""
    for word in result.stdout.strip().split():
        if word.isdigit():
            job_id = word
            break

    if not job_id:
        raise RuntimeError(
            f"Could not parse job ID from sbatch output:\n{result.stdout}"
        )

    _status(f"Job submitted: {job_id}")
    return job_id


def check_job_status(hpc: HPCConfig, job_id: str) -> str:
    """Check SLURM job status. Returns state like PENDING, RUNNING, COMPLETED, FAILED."""
    result = _run_ssh(
        hpc,
        f"sacct -j {job_id} --format=State --noheader --parsable2 | head -1",
        check=False,
    )
    state = result.stdout.strip().split("\n")[0].strip() if result.stdout else ""
    return state or "UNKNOWN"


def list_results(hpc: HPCConfig) -> list[str]:
    """List completed result files on the cluster."""
    result = _run_ssh(
        hpc,
        f"ls -1 {hpc.remote_dir}/results/ 2>/dev/null",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def download_results(
    hpc: HPCConfig,
    local_dir: str,
    progress: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Download all result files from the cluster."""
    if progress:
        progress("Checking for results...")

    files = list_results(hpc)
    if not files:
        return []

    os.makedirs(local_dir, exist_ok=True)
    downloaded = []

    for f in files:
        if progress:
            progress(f"Downloading {f}...")
        remote = f"{_ssh_target(hpc)}:{hpc.remote_dir}/results/{f}"
        local = os.path.join(local_dir, f)
        try:
            _run_scp(remote, local)
            downloaded.append(local)
        except RuntimeError:
            pass  # skip files that fail to download

    return downloaded


def _build_sbatch_script(hpc: HPCConfig, remote_audio: str, remote_results: str) -> str:
    """Generate an sbatch script for transcription."""
    module_lines = ""
    if hpc.modules:
        module_lines = "\n".join(
            f"module load {m}" for m in hpc.modules.split()
        )

    cli_args = f'"{remote_audio}" --model {hpc.model} -o "{remote_results}"'
    if hpc.diarize:
        cli_args += " --diarize"

    return f"""#!/bin/bash
#SBATCH --job-name=notetaker
#SBATCH --partition={hpc.partition}
#SBATCH -N {hpc.nodes}
#SBATCH --time={hpc.time}
#SBATCH --output={hpc.remote_dir}/slurm_%j.log

source /cm/local/apps/environment-modules/current/init/bash 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
{module_lines}

cd "{hpc.remote_dir}"
export PATH="{hpc.remote_dir}:$PATH"
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi
# Add pip-installed cuDNN to library path
CUDNN_LIB=$({hpc.python} -c "import nvidia.cudnn.lib as _l,os;print(os.path.dirname(_l.__file__))" 2>/dev/null)
if [ -n "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
fi
{hpc.python} cli.py {cli_args}
"""
