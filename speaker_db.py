"""Persistent speaker voice database for cross-meeting recognition."""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


class SpeakerDB:
    """Stores named speaker embeddings and matches new voices against them."""

    def __init__(self, db_path: str):
        self._path = db_path
        self._speakers: list[dict] = []
        self._threshold = 0.65
        self.load()

    def load(self):
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            self._speakers = data.get("speakers", [])
            self._threshold = data.get("threshold", 0.65)
        except (FileNotFoundError, json.JSONDecodeError):
            self._speakers = []

    def save(self):
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w") as f:
            json.dump({
                "threshold": self._threshold,
                "speakers": self._speakers,
            }, f, indent=2)

    def match(self, embedding) -> Optional[str]:
        """Return the name of the best matching speaker above threshold, or None."""
        if not self._speakers or embedding is None:
            return None

        emb = np.array(embedding, dtype=np.float64)
        best_name = None
        best_score = -1.0

        for speaker in self._speakers:
            stored = np.array(speaker["embedding"], dtype=np.float64)
            score = _cosine_similarity(emb, stored)
            if score > best_score:
                best_score = score
                best_name = speaker["name"]

        return best_name if best_score >= self._threshold else None

    def add_or_update(self, name: str, embedding) -> None:
        """Add a new speaker or update an existing one with a running average."""
        emb_list = embedding if isinstance(embedding, list) else list(embedding)

        for speaker in self._speakers:
            if speaker["name"].lower() == name.lower():
                n = speaker.get("num_samples", 1)
                old = np.array(speaker["embedding"], dtype=np.float64)
                new = np.array(emb_list, dtype=np.float64)
                speaker["embedding"] = ((old * n + new) / (n + 1)).tolist()
                speaker["num_samples"] = n + 1
                speaker["name"] = name
                self.save()
                return

        self._speakers.append({
            "name": name,
            "embedding": emb_list,
            "num_samples": 1,
        })
        self.save()

    def list_speakers(self) -> list[str]:
        return [s["name"] for s in self._speakers]

    def remove(self, name: str) -> bool:
        before = len(self._speakers)
        self._speakers = [s for s in self._speakers if s["name"].lower() != name.lower()]
        if len(self._speakers) < before:
            self.save()
            return True
        return False


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
