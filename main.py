"""
NoteTaker — One-button meeting recorder with AI-powered notes.

Records system audio + microphone, transcribes with WhisperX,
summarizes via local Ollama, and exports structured .docx meeting notes.

Setup:
    pip install -r requirements.txt

    External requirements:
    - ffmpeg (on PATH or next to this script)
    - Ollama (https://ollama.com) with a pulled model:
        ollama pull llama3.2:3b
        ollama serve
"""

# PyTorch 2.6+ defaults weights_only=True in torch.load, which breaks
# whisperx/pyannote checkpoint loading.  This env var restores the old default.
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Tell HuggingFace / transformers to use only locally-cached models and never
# reach out to the internet.  Models must be pre-installed via install_models.py.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Allow a custom model cache directory (set in .notetaker_config.json as
# "model_cache_dir").  When set, this redirects HuggingFace and PyTorch model
# lookups so models can live on a different drive or shared location.
from config import ConfigManager as _BootCfg
_boot_cache = _BootCfg().get("model_cache_dir", "")
if _boot_cache and os.path.isdir(_boot_cache):
    os.environ.setdefault("HF_HOME", os.path.join(_boot_cache, "huggingface"))
    os.environ.setdefault("TORCH_HOME", os.path.join(_boot_cache, "torch"))

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import time
from datetime import datetime

from config import ConfigManager
from recorder import DualAudioRecorder
from transcriber import WhisperXTranscriber, TranscriptionResult
from summarizer import OllamaSummarizer, MeetingSummary, build_clipboard_prompt
from exporter import DocxExporter
from speaker_db import SpeakerDB
from hpc import HPCConfig, submit_job, check_connection, check_job_status, download_results

# Ensure ffmpeg next to script is on PATH
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _script_dir + os.pathsep + os.environ.get("PATH", "")

RECORDINGS_DIR = os.path.join(_script_dir, "recordings")
EXPORTS_DIR = os.path.join(_script_dir, "exports")


class NoteTakerApp:
    SUPPORTED_EXTENSIONS = (".mp3", ".mp4", ".wav", ".m4a", ".webm", ".flac", ".ogg")
    MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NoteTaker")
        self.root.geometry("900x800")
        self.root.minsize(750, 650)

        self.cfg = ConfigManager()

        # -- Tk variables --
        self.file_path = tk.StringVar()
        self.model_choice = tk.StringVar(value=self.cfg.get("model", "base"))
        self.status_text = tk.StringVar(value="Ready")
        self.diarize_enabled = tk.BooleanVar(value=False)
        self.hf_token = tk.StringVar(value=self.cfg.get("hf_token", ""))
        self.min_speakers = tk.StringVar(value="")
        self.max_speakers = tk.StringVar(value="")
        self.loopback_device_var = tk.StringVar()
        self.mic_device_var = tk.StringVar()
        self.process_on_hpc = tk.BooleanVar(value=self.cfg.get("process_on_hpc", False))

        # -- State --
        self._is_recording = False
        self._is_processing = False
        self._recording_start: float = 0
        self._timer_id = None
        self._recorder: DualAudioRecorder | None = None
        self._current_result: TranscriptionResult | None = None
        self._current_summary: MeetingSummary | None = None

        # Device maps: display name -> device index
        self._loopback_map: dict[str, int] = {}
        self._mic_map: dict[str, int] = {}

        # Speaker voice database for cross-meeting recognition
        self._speaker_db = SpeakerDB(
            os.path.join(_script_dir, ".notetaker_speakers.json")
        )

        self._build_ui()
        self._refresh_audio_devices()

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_ui(self):
        # --- Recording Controls ---
        rec_frame = ttk.LabelFrame(self.root, text="Recording", padding=10)
        rec_frame.pack(fill="x", padx=15, pady=(10, 5))

        dev_row = ttk.Frame(rec_frame)
        dev_row.pack(fill="x", pady=(0, 8))

        ttk.Label(dev_row, text="System Audio:").pack(side="left")
        self.loopback_combo = ttk.Combobox(
            dev_row, textvariable=self.loopback_device_var, state="readonly", width=28
        )
        self.loopback_combo.pack(side="left", padx=(5, 15))

        ttk.Label(dev_row, text="Microphone:").pack(side="left")
        self.mic_combo = ttk.Combobox(
            dev_row, textvariable=self.mic_device_var, state="readonly", width=28
        )
        self.mic_combo.pack(side="left", padx=(5, 10))

        ttk.Button(dev_row, text="Refresh", command=self._refresh_audio_devices).pack(
            side="left"
        )

        btn_row = ttk.Frame(rec_frame)
        btn_row.pack(fill="x")

        self.record_btn = ttk.Button(
            btn_row, text="Start Recording", command=self._toggle_recording
        )
        self.record_btn.pack(side="left")

        self.recording_time_label = ttk.Label(btn_row, text="00:00", foreground="gray")
        self.recording_time_label.pack(side="left", padx=(10, 0))

        ttk.Label(
            btn_row,
            text="(Stop recording to auto-transcribe and summarize)",
            foreground="gray",
        ).pack(side="left", padx=(15, 0))

        # --- Or Upload File ---
        file_frame = ttk.LabelFrame(self.root, text="Or Upload Audio File", padding=10)
        file_frame.pack(fill="x", padx=15, pady=5)

        ttk.Entry(file_frame, textvariable=self.file_path, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(0, 10)
        )
        ttk.Button(file_frame, text="Browse...", command=self._browse_file).pack(
            side="right"
        )

        # --- Options ---
        opts_frame = ttk.LabelFrame(self.root, text="Options", padding=10)
        opts_frame.pack(fill="x", padx=15, pady=5)

        row1 = ttk.Frame(opts_frame)
        row1.pack(fill="x")

        ttk.Label(row1, text="Whisper Model:").pack(side="left")
        ttk.Combobox(
            row1,
            textvariable=self.model_choice,
            values=self.MODEL_OPTIONS,
            state="readonly",
            width=12,
        ).pack(side="left", padx=(5, 15))

        ttk.Label(
            row1,
            text="(tiny=fastest, large-v3=most accurate)",
            foreground="gray",
        ).pack(side="left")

        # --- Diarization ---
        diar_frame = ttk.LabelFrame(
            self.root, text="Speaker Diarization (optional)", padding=10
        )
        diar_frame.pack(fill="x", padx=15, pady=5)

        diar_row1 = ttk.Frame(diar_frame)
        diar_row1.pack(fill="x", pady=(0, 5))

        ttk.Checkbutton(
            diar_row1,
            text="Enable speaker detection",
            variable=self.diarize_enabled,
            command=self._toggle_diarize_fields,
        ).pack(side="left")

        ttk.Label(
            diar_row1,
            text="(requires free Hugging Face token)",
            foreground="gray",
        ).pack(side="left", padx=(10, 0))

        self.diar_fields_frame = ttk.Frame(diar_frame)
        self.diar_fields_frame.pack(fill="x", pady=2)

        ttk.Label(self.diar_fields_frame, text="HF Token:").grid(
            row=0, column=0, sticky="w"
        )
        self.token_entry = ttk.Entry(
            self.diar_fields_frame, textvariable=self.hf_token, show="*", width=45
        )
        self.token_entry.grid(row=0, column=1, padx=(5, 10), sticky="w")

        self.show_token_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.diar_fields_frame,
            text="Show",
            variable=self.show_token_var,
            command=self._toggle_token_visibility,
        ).grid(row=0, column=2, sticky="w")

        spk_row = ttk.Frame(self.diar_fields_frame)
        spk_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))
        ttk.Label(spk_row, text="Min speakers:").pack(side="left")
        ttk.Entry(spk_row, textvariable=self.min_speakers, width=5).pack(
            side="left", padx=(5, 15)
        )
        ttk.Label(spk_row, text="Max speakers:").pack(side="left")
        ttk.Entry(spk_row, textvariable=self.max_speakers, width=5).pack(
            side="left", padx=(5, 10)
        )
        ttk.Label(spk_row, text="(blank = auto-detect)", foreground="gray").pack(
            side="left"
        )

        links_frame = ttk.Frame(diar_frame)
        links_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(links_frame, text="Setup:", foreground="gray").pack(side="left")
        for label, url in [
            ("HF Account", "https://huggingface.co/join"),
            ("Diarization Terms", "https://huggingface.co/pyannote/speaker-diarization-3.1"),
            ("Segmentation Terms", "https://huggingface.co/pyannote/segmentation-3.0"),
            ("Create Token", "https://huggingface.co/settings/tokens"),
        ]:
            link = ttk.Label(
                links_frame, text=label, foreground="dodgerblue", cursor="hand2"
            )
            link.pack(side="left", padx=(10, 0))
            link.bind("<Button-1>", lambda e, u=url: _open_url(u))

        self._toggle_diarize_fields()

        # --- Action Buttons ---
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=15, pady=5)

        self.transcribe_btn = ttk.Button(
            btn_frame, text="Transcribe Only", command=self._start_transcription_only
        )
        self.transcribe_btn.pack(side="left")

        self.summarize_btn = ttk.Button(
            btn_frame, text="Summarize", command=self._start_summarization, state="disabled"
        )
        self.summarize_btn.pack(side="left", padx=(10, 0))

        self.export_btn = ttk.Button(
            btn_frame, text="Export to Word", command=self._export_notes, state="disabled"
        )
        self.export_btn.pack(side="left", padx=(10, 0))

        self.save_btn = ttk.Button(
            btn_frame, text="Save Transcript", command=self._save_transcript, state="disabled"
        )
        self.save_btn.pack(side="left", padx=(10, 0))

        self.copy_btn = ttk.Button(
            btn_frame, text="Copy to Clipboard", command=self._copy_to_clipboard, state="disabled"
        )
        self.copy_btn.pack(side="left", padx=(10, 0))

        self.copy_prompt_btn = ttk.Button(
            btn_frame, text="Copy Summary Prompt", command=self._copy_summary_prompt, state="disabled"
        )
        self.copy_prompt_btn.pack(side="left", padx=(10, 0))

        self.name_speakers_btn = ttk.Button(
            btn_frame, text="Name Speakers", command=self._show_speaker_naming, state="disabled"
        )
        self.name_speakers_btn.pack(side="left", padx=(10, 0))

        # --- HPC ---
        hpc_frame = ttk.LabelFrame(self.root, text="HPC (GPU Cluster)", padding=10)
        hpc_frame.pack(fill="x", padx=15, pady=5)

        ttk.Checkbutton(
            hpc_frame, text="Process on HPC",
            variable=self.process_on_hpc,
            command=self._save_hpc_toggle,
        ).pack(side="left", padx=(0, 10))

        self.hpc_submit_btn = ttk.Button(
            hpc_frame, text="Submit to HPC", command=self._hpc_submit
        )
        self.hpc_submit_btn.pack(side="left")

        self.hpc_status_btn = ttk.Button(
            hpc_frame, text="Check Job", command=self._hpc_check_job, state="disabled"
        )
        self.hpc_status_btn.pack(side="left", padx=(10, 0))

        self.hpc_download_btn = ttk.Button(
            hpc_frame, text="Download Results", command=self._hpc_download_results
        )
        self.hpc_download_btn.pack(side="left", padx=(10, 0))

        ttk.Button(
            hpc_frame, text="HPC Settings...", command=self._hpc_settings
        ).pack(side="right")

        self.hpc_status_label = ttk.Label(hpc_frame, text="", foreground="gray")
        self.hpc_status_label.pack(side="left", padx=(15, 0))

        # Track last submitted job
        self._hpc_job_id: str = ""

        # --- Status ---
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=15, pady=(5, 0))

        self.progress = ttk.Progressbar(
            status_frame, mode="determinate", length=200, maximum=100
        )
        self.progress.pack(side="left")

        ttk.Label(
            status_frame, textvariable=self.status_text, foreground="gray"
        ).pack(side="left", padx=(10, 0))

        # --- Tabbed Output ---
        output_frame = ttk.LabelFrame(self.root, text="Output", padding=10)
        output_frame.pack(fill="both", expand=True, padx=15, pady=(5, 10))

        self.notebook = ttk.Notebook(output_frame)
        self.notebook.pack(fill="both", expand=True)

        # Transcript tab
        transcript_tab = ttk.Frame(self.notebook)
        self.transcript_box = scrolledtext.ScrolledText(
            transcript_tab, wrap="word", font=("Consolas", 10), state="disabled"
        )
        self.transcript_box.pack(fill="both", expand=True)
        self.notebook.add(transcript_tab, text="Transcript")

        # Summary tab
        summary_tab = ttk.Frame(self.notebook)
        self.summary_box = scrolledtext.ScrolledText(
            summary_tab, wrap="word", font=("Segoe UI", 11), state="disabled"
        )
        self.summary_box.pack(fill="both", expand=True)
        self.notebook.add(summary_tab, text="Summary")

    # ==================================================================
    # Audio Devices
    # ==================================================================

    def _refresh_audio_devices(self):
        try:
            loopback_devs, mic_devs = DualAudioRecorder.get_audio_devices()
        except Exception:
            loopback_devs, mic_devs = [], []

        self._loopback_map = {d["name"]: d["index"] for d in loopback_devs}
        self._mic_map = {d["name"]: d["index"] for d in mic_devs}

        lb_names = list(self._loopback_map.keys())
        mic_names = list(self._mic_map.keys())

        self.loopback_combo["values"] = lb_names or ["(no loopback devices)"]
        self.mic_combo["values"] = mic_names or ["(no microphones)"]

        # Restore saved selection or pick first
        saved_lb = self.cfg.get("audio_device_loopback", "")
        saved_mic = self.cfg.get("audio_device_mic", "")
        self.loopback_device_var.set(saved_lb if saved_lb in lb_names else (lb_names[0] if lb_names else ""))
        self.mic_device_var.set(saved_mic if saved_mic in mic_names else (mic_names[0] if mic_names else ""))

    # ==================================================================
    # Recording
    # ==================================================================

    def _toggle_recording(self):
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        lb_name = self.loopback_device_var.get()
        mic_name = self.mic_device_var.get()

        lb_idx = self._loopback_map.get(lb_name)
        mic_idx = self._mic_map.get(mic_name)

        if lb_idx is None and mic_idx is None:
            messagebox.showwarning(
                "No devices",
                "No audio devices selected.\n\n"
                "Click 'Refresh' and select at least one device.",
            )
            return

        try:
            self._recorder = DualAudioRecorder()
            self._recorder.start_recording(
                loopback_device_index=lb_idx, mic_device_index=mic_idx
            )
        except Exception as e:
            messagebox.showerror("Recording Error", str(e))
            return

        # Save device choices
        self.cfg.set_many({
            "audio_device_loopback": lb_name,
            "audio_device_mic": mic_name,
        })

        self._is_recording = True
        self._recording_start = time.time()
        self.record_btn.config(text="Stop Recording")
        self.recording_time_label.config(foreground="red")
        self._set_controls_enabled(False)
        self._tick_timer()

    def _stop_recording(self):
        if not self._is_recording:
            return

        self._is_recording = False
        if self._timer_id:
            self.root.after_cancel(self._timer_id)
            self._timer_id = None

        self.record_btn.config(text="Start Recording")
        self.recording_time_label.config(foreground="gray")

        if self._recorder:
            self._recorder.stop_recording()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_path = os.path.join(RECORDINGS_DIR, f"meeting_{timestamp}.wav")

            self.status_text.set("Saving recording...")
            try:
                self._recorder.save_mixed_wav(wav_path)
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save recording:\n{e}")
                self._set_controls_enabled(True)
                return

            self.file_path.set(wav_path)
            if self.process_on_hpc.get():
                self._hpc_submit()
            else:
                self._start_full_pipeline(wav_path)

    def _tick_timer(self):
        if not self._is_recording:
            return
        elapsed = int(time.time() - self._recording_start)
        m, s = divmod(elapsed, 60)
        self.recording_time_label.config(text=f"{m:02d}:{s:02d}")
        self._timer_id = self.root.after(1000, self._tick_timer)

    # ==================================================================
    # Pipeline: Transcribe -> Summarize -> Export
    # ==================================================================

    def _start_full_pipeline(self, audio_path: str):
        self._is_processing = True
        self._set_controls_enabled(False)
        self._set_text(self.transcript_box, "")
        self._set_text(self.summary_box, "")
        self.progress["value"] = 0
        self.status_text.set("Starting pipeline...")

        thread = threading.Thread(
            target=self._pipeline_worker, args=(audio_path,), daemon=True
        )
        thread.start()

    def _pipeline_worker(self, audio_path: str):
        self._save_config()

        # --- Step 1: Transcribe ---
        try:
            transcriber = WhisperXTranscriber(
                model_name=self.model_choice.get(),
                download_root=self._whisper_download_root(),
            )
            result = transcriber.transcribe(
                audio_path=audio_path,
                enable_diarization=self.diarize_enabled.get(),
                hf_token=self.hf_token.get().strip() or None,
                min_speakers=_parse_int(self.min_speakers.get()),
                max_speakers=_parse_int(self.max_speakers.get()),
                progress_callback=self._update_status,
            )
            self._current_result = result
            transcript_text = result.format_as_text()
            self.root.after(0, lambda: self._set_text(self.transcript_box, transcript_text))
        except ImportError as e:
            self._on_error(f"Import error:\n\n{e}\n\n(pip install whisperx)")
            return
        except Exception as e:
            self._on_error(f"Transcription failed:\n\n{e}")
            return

        # --- Step 2: Summarize (Ollama if available, otherwise clipboard prompt) ---
        summarizer = OllamaSummarizer(
            ollama_url=self.cfg.get("ollama_url", "http://localhost:11434"),
            model=self.cfg.get("ollama_model", "llama3.2:3b"),
        )

        if summarizer.check_available():
            try:
                summary = summarizer.summarize(
                    transcript=transcript_text,
                    progress_callback=self._update_status,
                )
                self._current_summary = summary
                self.root.after(0, lambda: self._set_text(self.summary_box, summary.raw_summary))
            except Exception as e:
                self._current_summary = None
                self._update_status("Ollama failed — copying prompt to clipboard instead...")
                self._clipboard_prompt_fallback(transcript_text)
        else:
            # No Ollama — use clipboard prompt as primary path
            self._current_summary = None
            self._update_status("Ollama not found — copying summary prompt to clipboard...")
            self._clipboard_prompt_fallback(transcript_text)

        # --- Step 3: Export ---
        self._auto_export(audio_path)

    def _auto_export(self, audio_path: str):
        """Export .docx + .txt to the exports directory."""
        self._update_status("Exporting meeting notes...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(audio_path))[0]

        docx_path = os.path.join(EXPORTS_DIR, f"{base}_notes.docx")
        txt_path = os.path.join(EXPORTS_DIR, f"{base}_transcript.txt")

        try:
            exporter = DocxExporter()
            transcript_text = self._current_result.format_as_text() if self._current_result else ""

            if self._current_summary:
                exporter.export_meeting_notes(
                    summary=self._current_summary,
                    transcript=transcript_text,
                    output_path=docx_path,
                )
            DocxExporter.save_transcript_txt(transcript_text, txt_path)
        except Exception as e:
            self._on_error(f"Export failed:\n{e}")
            return

        def _done():
            self.progress["value"] = 100
            self._is_processing = False
            self._set_controls_enabled(True)
            self.notebook.select(1)  # Switch to Summary tab

            saved = f"Transcript: {txt_path}"
            if self._current_summary:
                saved = f"Notes: {docx_path}\n{saved}"
                self.status_text.set("Pipeline complete — notes exported!")
            else:
                saved += "\n\nSummary prompt copied to clipboard — paste into ChatGPT."
                self.status_text.set("Pipeline complete — paste prompt into ChatGPT for summary.")

            messagebox.showinfo("Export Complete", f"Files saved:\n\n{saved}")

            # Enable speaker naming if voice prints were extracted
            if self._current_result and self._current_result.speaker_embeddings:
                self.name_speakers_btn.config(state="normal")
                self._show_speaker_naming()

        self.root.after(0, _done)

    def _clipboard_prompt_fallback(self, transcript_text: str):
        """Copy the summary prompt to clipboard so user can paste into ChatGPT."""
        prompt = build_clipboard_prompt(transcript_text)

        def _do():
            self.root.clipboard_clear()
            self.root.clipboard_append(prompt)
            self._set_text(
                self.summary_box,
                "(Summary prompt copied to clipboard — paste into ChatGPT)\n\n"
                "---\n\n" + prompt,
            )

        self.root.after(0, _do)

    # ==================================================================
    # Transcribe Only (file upload path)
    # ==================================================================

    def _start_transcription_only(self):
        path = self.file_path.get()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("No file", "Please select or record an audio file first.")
            return

        if self.process_on_hpc.get():
            self._hpc_submit()
            return

        offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        if self.diarize_enabled.get() and not offline and not self.hf_token.get().strip():
            messagebox.showwarning(
                "Token required",
                "Speaker diarization requires a Hugging Face token.\n\n"
                "Enter your token or uncheck 'Enable speaker detection'.",
            )
            return

        self._save_config()
        self._is_processing = True
        self._set_controls_enabled(False)
        self._set_text(self.transcript_box, "")
        self.progress["value"] = 0
        self.status_text.set("Loading model...")

        thread = threading.Thread(
            target=self._transcribe_only_worker, daemon=True
        )
        thread.start()

    def _transcribe_only_worker(self):
        try:
            transcriber = WhisperXTranscriber(
                model_name=self.model_choice.get(),
                download_root=self._whisper_download_root(),
            )
            result = transcriber.transcribe(
                audio_path=self.file_path.get(),
                enable_diarization=self.diarize_enabled.get(),
                hf_token=self.hf_token.get().strip() or None,
                min_speakers=_parse_int(self.min_speakers.get()),
                max_speakers=_parse_int(self.max_speakers.get()),
                progress_callback=self._update_status,
            )
            self._current_result = result
            transcript_text = result.format_as_text()

            def _done():
                self._set_text(self.transcript_box, transcript_text)
                self.progress["value"] = 100
                self.status_text.set(f"Transcribed in {result.elapsed_time:.1f}s")
                self._is_processing = False
                self._set_controls_enabled(True)
                self.summarize_btn.config(state="normal")
                self.save_btn.config(state="normal")
                self.copy_btn.config(state="normal")
                self.copy_prompt_btn.config(state="normal")

                # Enable speaker naming if voice prints were extracted
                if result.speaker_embeddings:
                    self.name_speakers_btn.config(state="normal")
                    self._show_speaker_naming()

            self.root.after(0, _done)

        except ImportError as e:
            self._on_error(f"Import error:\n\n{e}\n\n(pip install whisperx)")
        except Exception as e:
            self._on_error(f"Transcription failed:\n\n{e}")

    # ==================================================================
    # Summarize (standalone button)
    # ==================================================================

    def _start_summarization(self):
        if not self._current_result:
            messagebox.showwarning("No transcript", "Run transcription first.")
            return

        self._is_processing = True
        self._set_controls_enabled(False)
        self.progress["value"] = 0
        self.status_text.set("Starting summarization...")

        thread = threading.Thread(target=self._summarize_worker, daemon=True)
        thread.start()

    def _summarize_worker(self):
        try:
            summarizer = OllamaSummarizer(
                ollama_url=self.cfg.get("ollama_url", "http://localhost:11434"),
                model=self.cfg.get("ollama_model", "llama3.2:3b"),
            )
            transcript_text = self._current_result.format_as_text()
            summary = summarizer.summarize(
                transcript=transcript_text,
                progress_callback=self._update_status,
            )
            self._current_summary = summary

            def _done():
                self._set_text(self.summary_box, summary.raw_summary)
                self.progress["value"] = 100
                self.status_text.set("Summarization complete!")
                self._is_processing = False
                self._set_controls_enabled(True)
                self.export_btn.config(state="normal")
                self.notebook.select(1)

            self.root.after(0, _done)

        except Exception as e:
            self._on_error(f"Summarization failed:\n\n{e}")

    # ==================================================================
    # Export (standalone button)
    # ==================================================================

    def _export_notes(self):
        if not self._current_result:
            messagebox.showwarning("Nothing to export", "Run transcription first.")
            return

        base = os.path.splitext(os.path.basename(self.file_path.get() or "meeting"))[0]
        docx_default = f"{base}_notes.docx"

        path = filedialog.asksaveasfilename(
            defaultextension=".docx",
            initialfile=docx_default,
            initialdir=EXPORTS_DIR,
            filetypes=[("Word Document", "*.docx"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            exporter = DocxExporter()
            transcript_text = self._current_result.format_as_text()

            if self._current_summary:
                exporter.export_meeting_notes(
                    summary=self._current_summary,
                    transcript=transcript_text,
                    output_path=path,
                )

            txt_path = os.path.splitext(path)[0] + "_transcript.txt"
            DocxExporter.save_transcript_txt(transcript_text, txt_path)

            self.status_text.set(f"Exported to {os.path.basename(path)}")
            messagebox.showinfo(
                "Export Complete",
                f"Files saved:\n\n{path}\n{txt_path}",
            )
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # ==================================================================
    # UI Helpers
    # ==================================================================

    def _toggle_diarize_fields(self):
        state = "normal" if self.diarize_enabled.get() else "disabled"
        for child in self.diar_fields_frame.winfo_children():
            try:
                child.config(state=state)
            except tk.TclError:
                pass
            if hasattr(child, "winfo_children"):
                for grandchild in child.winfo_children():
                    try:
                        grandchild.config(state=state)
                    except tk.TclError:
                        pass

    def _toggle_token_visibility(self):
        self.token_entry.config(show="" if self.show_token_var.get() else "*")

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[
                ("Audio / Video", " ".join(f"*{ext}" for ext in self.SUPPORTED_EXTENSIONS)),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.file_path.set(path)

    def _set_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for btn in (
            self.transcribe_btn,
            self.summarize_btn,
            self.export_btn,
            self.save_btn,
            self.copy_btn,
            self.copy_prompt_btn,
            self.name_speakers_btn,
        ):
            btn.config(state=state)
        # Record button always available (to stop)
        self.record_btn.config(state="normal")

    # Progress step mapping — maps status message prefixes to bar percentages
    _PROGRESS_STEPS = {
        "Loading":       5,
        "Loading audio": 10,
        "Transcribing":  50,
        "Aligning":      65,
        "Running speaker": 80,
        "Extracting":    95,
        "Checking Ollama": 70,
        "Summarizing":   80,
        "Parsing":       95,
        "Exporting":     90,
        "Starting":      0,
    }

    def _update_status(self, msg: str):
        def _do():
            self.status_text.set(msg)
            # Advance progress bar based on known pipeline stages
            for prefix, pct in self._PROGRESS_STEPS.items():
                if msg.startswith(prefix):
                    self.progress["value"] = pct
                    break
        self.root.after(0, _do)

    @staticmethod
    def _set_text(widget, text: str):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        if text:
            widget.insert("1.0", text)
        widget.config(state="disabled")

    def _on_error(self, msg: str):
        def _do():
            self.progress["value"] = 0
            self.status_text.set("Error")
            self._is_processing = False
            self._set_controls_enabled(True)
            messagebox.showerror("Error", msg)

        self.root.after(0, _do)

    def _save_transcript(self):
        text = self.transcript_box.get("1.0", "end").strip()
        if not text:
            return
        base = os.path.splitext(os.path.basename(self.file_path.get() or "meeting"))[0]
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"{base}_transcript.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.status_text.set(f"Saved to {os.path.basename(path)}")

    def _copy_to_clipboard(self):
        # Copy from whichever tab is active
        active_tab = self.notebook.index(self.notebook.select())
        widget = self.transcript_box if active_tab == 0 else self.summary_box
        text = widget.get("1.0", "end").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_text.set("Copied to clipboard!")

    def _copy_summary_prompt(self):
        """Build and copy a ready-to-paste summary prompt from the current transcript."""
        if not self._current_result:
            messagebox.showwarning("No transcript", "Run transcription first.")
            return
        transcript_text = self._current_result.format_as_text()
        prompt = build_clipboard_prompt(transcript_text)
        self.root.clipboard_clear()
        self.root.clipboard_append(prompt)
        self._set_text(
            self.summary_box,
            "(Summary prompt copied to clipboard — paste into ChatGPT)\n\n"
            "---\n\n" + prompt,
        )
        self.notebook.select(1)
        self.status_text.set("Summary prompt copied to clipboard!")

    def _whisper_download_root(self) -> str | None:
        """Return the HF hub cache path for faster-whisper, or None for default."""
        cache_dir = self.cfg.get("model_cache_dir", "")
        if cache_dir:
            return os.path.join(cache_dir, "huggingface", "hub")
        return None

    # ==================================================================
    # Speaker Recognition
    # ==================================================================

    def _show_speaker_naming(self):
        """Match speaker embeddings against DB and show naming dialog."""
        if not self._current_result or not self._current_result.speaker_embeddings:
            return

        embeddings = self._current_result.speaker_embeddings

        # Pre-match against stored voices
        suggestions = {}
        for label, emb in embeddings.items():
            suggestions[label] = self._speaker_db.match(emb) or ""

        dialog = SpeakerNameDialog(self.root, suggestions)
        self.root.wait_window(dialog)

        if not dialog.result:
            return

        # Replace speaker labels in segments
        for seg in self._current_result.segments:
            old = seg.get("speaker", "")
            new_name = dialog.result.get(old, "")
            if new_name:
                seg["speaker"] = new_name

        # Save embeddings to DB
        for label, name in dialog.result.items():
            if name and label in embeddings:
                self._speaker_db.add_or_update(name, embeddings[label])

        # Refresh transcript display with real names
        self._set_text(
            self.transcript_box, self._current_result.format_as_text()
        )
        self.status_text.set("Speaker names updated!")

    # ==================================================================
    # HPC Submission
    # ==================================================================

    def _get_hpc_config(self) -> HPCConfig:
        return HPCConfig.from_config(self.cfg)

    def _hpc_submit(self):
        """Upload audio and submit a SLURM job to the HPC cluster."""
        # Determine audio file to submit
        audio_path = self.file_path.get()
        if not audio_path or not os.path.isfile(audio_path):
            messagebox.showwarning(
                "No audio file",
                "Record or select an audio file first.",
            )
            return

        hpc = self._get_hpc_config()
        if not hpc.is_configured:
            messagebox.showwarning(
                "HPC not configured",
                "Set your HPC connection details first.\n\n"
                'Click "HPC Settings..." to configure.',
            )
            self._hpc_settings()
            return

        # Test connection first
        self.hpc_status_label.config(text="Connecting...")
        self.root.update_idletasks()

        self._set_controls_enabled(False)
        self.hpc_submit_btn.config(state="disabled")

        def _worker():
            try:
                if not check_connection(hpc):
                    self.root.after(0, lambda: self._hpc_error(
                        f"Cannot connect to {hpc.user}@{hpc.host}\n\n"
                        "Check your SSH key setup and HPC settings."
                    ))
                    return

                job_id = submit_job(
                    hpc, audio_path,
                    progress=lambda msg: self.root.after(
                        0, lambda m=msg: self.hpc_status_label.config(text=m)
                    ),
                )
                self._hpc_job_id = job_id

                def _done():
                    self._set_controls_enabled(True)
                    self.hpc_submit_btn.config(state="normal")
                    self.hpc_status_btn.config(state="normal")
                    self.hpc_status_label.config(
                        text=f"Job {job_id} submitted!", foreground="green"
                    )
                    messagebox.showinfo(
                        "Job Submitted",
                        f"SLURM job {job_id} submitted to {hpc.host}\n\n"
                        f"Audio: {os.path.basename(audio_path)}\n"
                        f"Model: {hpc.model}\n"
                        f'Diarize: {"yes" if hpc.diarize else "no"}\n\n'
                        'Click "Check Job" to monitor progress,\n'
                        '"Download Results" when complete.',
                    )
                self.root.after(0, _done)

            except Exception as e:
                self.root.after(0, lambda: self._hpc_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _hpc_check_job(self):
        """Check the status of the last submitted job."""
        if not self._hpc_job_id:
            return

        hpc = self._get_hpc_config()
        self.hpc_status_label.config(text="Checking...", foreground="gray")
        self.root.update_idletasks()

        def _worker():
            try:
                state = check_job_status(hpc, self._hpc_job_id)
                color = {
                    "COMPLETED": "green",
                    "RUNNING": "dodgerblue",
                    "PENDING": "orange",
                    "FAILED": "red",
                }.get(state, "gray")

                def _done():
                    self.hpc_status_label.config(
                        text=f"Job {self._hpc_job_id}: {state}", foreground=color
                    )
                    if state == "COMPLETED":
                        if messagebox.askyesno(
                            "Job Complete",
                            f"Job {self._hpc_job_id} finished!\n\n"
                            "Download results now?",
                        ):
                            self._hpc_download_results()
                self.root.after(0, _done)

            except Exception as e:
                self.root.after(0, lambda: self.hpc_status_label.config(
                    text=f"Check failed: {e}", foreground="red"
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _hpc_download_results(self):
        """Download result files from the cluster."""
        hpc = self._get_hpc_config()
        if not hpc.is_configured:
            return

        self.hpc_status_label.config(text="Downloading...", foreground="gray")
        self.root.update_idletasks()

        def _worker():
            try:
                local_dir = self.cfg.get("default_export_dir", EXPORTS_DIR)
                files = download_results(
                    hpc, local_dir,
                    progress=lambda msg: self.root.after(
                        0, lambda m=msg: self.hpc_status_label.config(text=m)
                    ),
                )

                def _done():
                    if files:
                        self.hpc_status_label.config(
                            text=f"Downloaded {len(files)} file(s)",
                            foreground="green",
                        )
                        # Load the first transcript into the UI
                        transcript_file = None
                        for f in files:
                            if f.endswith("_transcript.txt"):
                                transcript_file = f
                                break
                        if transcript_file and os.path.isfile(transcript_file):
                            with open(transcript_file, "r", encoding="utf-8") as fh:
                                text = fh.read()
                            self._set_text(self.transcript_box, text)
                            self.save_btn.config(state="normal")
                            self.copy_btn.config(state="normal")
                            self.copy_prompt_btn.config(state="normal")

                        # Load summary prompt if present
                        prompt_file = None
                        for f in files:
                            if f.endswith("_summary_prompt.txt"):
                                prompt_file = f
                                break
                        if prompt_file and os.path.isfile(prompt_file):
                            with open(prompt_file, "r", encoding="utf-8") as fh:
                                prompt = fh.read()
                            self._set_text(self.summary_box, prompt)

                        messagebox.showinfo(
                            "Results Downloaded",
                            f"Downloaded {len(files)} file(s) to:\n{local_dir}",
                        )
                    else:
                        self.hpc_status_label.config(
                            text="No results found", foreground="orange"
                        )
                self.root.after(0, _done)

            except Exception as e:
                self.root.after(0, lambda: self._hpc_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _hpc_settings(self):
        """Open the HPC settings dialog."""
        dialog = HPCSettingsDialog(self.root, self.cfg)
        self.root.wait_window(dialog)

    def _hpc_error(self, msg: str):
        self._set_controls_enabled(True)
        self.hpc_submit_btn.config(state="normal")
        self.hpc_status_label.config(text="Error", foreground="red")
        messagebox.showerror("HPC Error", msg)

    def _save_hpc_toggle(self):
        self.cfg.set("process_on_hpc", self.process_on_hpc.get())

    def _save_config(self):
        self.cfg.set_many({
            "hf_token": self.hf_token.get().strip(),
            "model": self.model_choice.get(),
        })


# ==================================================================
# HPC Settings Dialog
# ==================================================================

class HPCSettingsDialog(tk.Toplevel):
    """Modal dialog for configuring HPC / SLURM connection settings."""

    _FIELDS = [
        ("hpc_host",      "Hostname:",      "e.g. hpc.university.edu"),
        ("hpc_user",      "Username:",      "SSH username"),
        ("hpc_remote_dir","Remote dir:",     "e.g. /scratch/user/notetaker"),
        ("hpc_partition",  "Partition:",     "e.g. gpu"),
        ("hpc_nodes",     "Nodes (-N):",    "e.g. 1"),
        ("hpc_time",      "Time limit:",    "e.g. 01:00:00"),
        ("hpc_modules",   "Modules:",       "space-separated, e.g. cuda python"),
        ("hpc_python",    "Python cmd:",    "e.g. python3"),
        ("hpc_model",     "Whisper model:", "tiny/base/small/medium/large-v2"),
    ]

    def __init__(self, parent, cfg):
        super().__init__(parent)
        self.title("HPC Settings")
        self.cfg = cfg
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)

        frame = ttk.Frame(self, padding=15)
        frame.pack(fill="both")

        self._vars: dict[str, tk.StringVar] = {}
        for i, (key, label, hint) in enumerate(self._FIELDS):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(cfg.get(key, "")))
            entry = ttk.Entry(frame, textvariable=var, width=35)
            entry.grid(row=i, column=1, padx=(10, 5), pady=2)
            ttk.Label(frame, text=hint, foreground="gray").grid(
                row=i, column=2, sticky="w", padx=(0, 5)
            )
            self._vars[key] = var

        # Diarize checkbox
        row = len(self._FIELDS)
        self._diarize_var = tk.BooleanVar(value=cfg.get("hpc_diarize", True))
        ttk.Checkbutton(
            frame, text="Enable diarization on HPC", variable=self._diarize_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 5))

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Test Connection", command=self._test).pack(
            side="left", padx=5
        )
        ttk.Button(btn_frame, text="Save", command=self._save).pack(
            side="left", padx=5
        )
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(
            side="left", padx=5
        )

        self._status = ttk.Label(self, text="", foreground="gray")
        self._status.pack(pady=(0, 10))

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _test(self):
        self._status.config(text="Testing connection...", foreground="gray")
        self.update_idletasks()
        hpc = HPCConfig(
            host=self._vars["hpc_host"].get().strip(),
            user=self._vars["hpc_user"].get().strip(),
            remote_dir=self._vars["hpc_remote_dir"].get().strip(),
        )
        if not hpc.host or not hpc.user:
            self._status.config(text="Enter hostname and username first", foreground="red")
            return

        def _worker():
            ok = check_connection(hpc)
            self.after(0, lambda: self._status.config(
                text="Connected!" if ok else f"Connection failed to {hpc.host}",
                foreground="green" if ok else "red",
            ))
        threading.Thread(target=_worker, daemon=True).start()

    def _save(self):
        updates = {key: var.get().strip() for key, var in self._vars.items()}
        updates["hpc_diarize"] = self._diarize_var.get()
        # Store nodes as int
        try:
            updates["hpc_nodes"] = int(updates.get("hpc_nodes", 1))
        except (ValueError, TypeError):
            updates["hpc_nodes"] = 1
        self.cfg.set_many(updates)
        self.destroy()


# ==================================================================
# Speaker Name Dialog
# ==================================================================

class SpeakerNameDialog(tk.Toplevel):
    """Modal dialog for assigning real names to detected speakers."""

    def __init__(self, parent, suggestions: dict):
        """
        suggestions: {speaker_label: suggested_name_or_empty_string}
        """
        super().__init__(parent)
        self.title("Name Speakers")
        self.result: dict = {}
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)

        ttk.Label(
            self,
            text="Assign names to detected speakers.\n"
            "Leave blank to keep the original label.",
        ).pack(padx=15, pady=(15, 5))

        frame = ttk.Frame(self)
        frame.pack(padx=15, pady=5, fill="x")

        self._entries: dict[str, tk.StringVar] = {}
        for i, (label, suggestion) in enumerate(sorted(suggestions.items())):
            ttk.Label(frame, text=f"{label}:").grid(
                row=i, column=0, sticky="w", pady=3
            )
            var = tk.StringVar(value=suggestion)
            entry = ttk.Entry(frame, textvariable=var, width=30)
            entry.grid(row=i, column=1, padx=(10, 0), pady=3)
            if suggestion:
                ttk.Label(frame, text="(matched)", foreground="green").grid(
                    row=i, column=2, padx=(5, 0)
                )
            self._entries[label] = var

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="Save", command=self._save).pack(
            side="left", padx=5
        )
        ttk.Button(btn_frame, text="Skip", command=self.destroy).pack(
            side="left", padx=5
        )

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _save(self):
        self.result = {
            label: var.get().strip()
            for label, var in self._entries.items()
        }
        self.destroy()


# ==================================================================
# Utilities
# ==================================================================

def _open_url(url: str):
    import webbrowser
    webbrowser.open(url)


def _parse_int(val: str) -> int | None:
    val = val.strip()
    return int(val) if val.isdigit() else None


def main():
    root = tk.Tk()
    NoteTakerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
