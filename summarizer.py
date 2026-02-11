"""Meeting transcript summarization — clipboard prompt or local Ollama."""
from __future__ import annotations

import re
import requests
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class MeetingSummary:
    speakers: list = field(default_factory=list)
    key_points: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    action_items: list = field(default_factory=list)  # list of dicts
    insights: list = field(default_factory=list)
    follow_up: list = field(default_factory=list)
    raw_summary: str = ""


SYSTEM_PROMPT = (
    "You are an expert meeting notes assistant for an engineering team. "
    "Analyze meeting transcripts and extract structured, actionable information. "
    "Be concise and focus on substance."
)

SUMMARY_TEMPLATE = """Analyze the following meeting transcript and provide a comprehensive summary.

Structure your response with EXACTLY these section headers:

## SPEAKERS
List each speaker who participated (one per line, prefixed with -).

## KEY DISCUSSION POINTS
- Main topics discussed, organized by importance
- Include who raised each topic when clear

## DECISIONS MADE
- Concrete decisions reached during the meeting

## ACTION ITEMS
For each action item use this format:
- [Owner] Task description (Due: deadline if mentioned) [Priority: High/Medium/Low]

## KEY INSIGHTS
- Important observations, realizations, or technical insights worth highlighting

## FOLLOW-UP
- Questions left unanswered
- Topics needing further discussion

---
TRANSCRIPT:
{transcript}"""


class OllamaSummarizer:
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout: int = 300,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_available(self) -> bool:
        """Return True if Ollama is running and the configured model is pulled."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Match with or without tag suffix (e.g. "llama3.2:3b" matches "llama3.2:3b")
            return any(self.model in m or m.startswith(self.model) for m in models)
        except (requests.ConnectionError, requests.Timeout):
            return False

    def summarize(
        self,
        transcript: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> MeetingSummary:
        def _status(msg: str):
            if progress_callback:
                progress_callback(msg)

        _status("Checking Ollama connection...")
        if not self._is_server_reachable():
            raise ConnectionError(
                f"Ollama is not running at {self.ollama_url}.\n\n"
                "Start it with:  ollama serve"
            )

        _status(f"Summarizing with {self.model} (this may take a few minutes on CPU)...")

        prompt = SUMMARY_TEMPLATE.format(transcript=transcript)

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {"temperature": 0.7, "top_p": 0.9},
                },
                timeout=self.timeout,
            )
        except requests.Timeout:
            raise RuntimeError(
                f"Summarization timed out after {self.timeout}s.\n"
                "Try a smaller model or a shorter transcript."
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama returned status {resp.status_code}: {resp.text[:300]}"
            )

        raw_text = resp.json().get("response", "")
        if not raw_text.strip():
            raise RuntimeError("Ollama returned an empty response.")

        _status("Parsing summary...")
        return self._parse_response(raw_text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_server_reachable(self) -> bool:
        try:
            requests.get(self.ollama_url, timeout=5)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False

    @staticmethod
    def _parse_response(text: str) -> MeetingSummary:
        """Parse the structured LLM response into a MeetingSummary."""
        summary = MeetingSummary(raw_summary=text)

        sections = re.split(r"^##\s+", text, flags=re.MULTILINE)

        for section in sections:
            if not section.strip():
                continue

            header_end = section.find("\n")
            if header_end == -1:
                continue
            header = section[:header_end].strip().upper()
            body = section[header_end:].strip()

            bullets = _extract_bullets(body)

            if "SPEAKER" in header:
                summary.speakers = bullets
            elif "KEY DISCUSSION" in header or "DISCUSSION POINT" in header:
                summary.key_points = bullets
            elif "DECISION" in header:
                summary.decisions = bullets
            elif "ACTION" in header:
                summary.action_items = [_parse_action_item(b) for b in bullets]
            elif "INSIGHT" in header:
                summary.insights = bullets
            elif "FOLLOW" in header:
                summary.follow_up = bullets

        return summary


def _extract_bullets(text: str) -> list[str]:
    """Pull bullet-pointed lines from a section body."""
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "• ")):
            items.append(stripped[2:].strip())
        elif stripped.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            items.append(re.sub(r"^\d+\.\s*", "", stripped))
    return items


def _parse_action_item(text: str) -> dict:
    """Parse '[Owner] task (Due: date) [Priority: X]' into a dict."""
    item = {"task": text, "owner": "TBD", "deadline": "TBD", "priority": "Medium"}

    owner_match = re.match(r"\[([^\]]+)\]\s*(.*)", text)
    if owner_match:
        item["owner"] = owner_match.group(1).strip()
        text = owner_match.group(2).strip()

    due_match = re.search(r"\(Due:\s*([^)]+)\)", text, re.IGNORECASE)
    if due_match:
        item["deadline"] = due_match.group(1).strip()
        text = text[: due_match.start()] + text[due_match.end() :]

    priority_match = re.search(r"\[Priority:\s*([^\]]+)\]", text, re.IGNORECASE)
    if priority_match:
        item["priority"] = priority_match.group(1).strip()
        text = text[: priority_match.start()] + text[priority_match.end() :]

    item["task"] = text.strip()
    return item


# ==================================================================
# Clipboard prompt (primary path — no local LLM required)
# ==================================================================

CLIPBOARD_PROMPT = """{system}

{summary_template}"""


def build_clipboard_prompt(transcript: str) -> str:
    """Return a ready-to-paste prompt combining instructions + transcript.

    The user copies this into their airgapped ChatGPT to get structured
    meeting notes without needing a local LLM.
    """
    return CLIPBOARD_PROMPT.format(
        system=SYSTEM_PROMPT,
        summary_template=SUMMARY_TEMPLATE.format(transcript=transcript),
    )
