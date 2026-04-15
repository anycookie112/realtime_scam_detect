"""Shared bank knowledge base: index builder, lookup, and bank name detection."""

from __future__ import annotations

import re
from pathlib import Path

BANKS_DIR = Path(__file__).parent / "data"


def _build_bank_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for md_file in BANKS_DIR.glob("*.md"):
        if md_file.stem == "GENERAL":
            continue
        index[md_file.stem.lower()] = md_file
        for line in md_file.read_text().splitlines():
            if line.strip().startswith("- aliases:"):
                for alias in line.split(":", 1)[1].split(","):
                    alias = alias.strip().lower()
                    if alias:
                        index[alias] = md_file
    return index


_BANK_INDEX = _build_bank_index()


def load_bank_data(bank_name: str) -> str:
    """Look up bank guidelines by name/alias. Falls back to GENERAL.md."""
    key = bank_name.strip().lower().replace(" ", "")
    match = _BANK_INDEX.get(key)
    if not match:
        for index_key, path in _BANK_INDEX.items():
            if key in index_key or index_key in key:
                match = path
                break
    if match:
        return match.read_text()
    general = BANKS_DIR / "GENERAL.md"
    if general.exists():
        return general.read_text()
    return "No bank-specific or general guidelines found."


def detect_bank_name(transcript: str) -> str:
    """Scan a transcript for known bank names/aliases. Returns first match or 'general'."""
    text_lower = transcript.lower()
    for alias in sorted(_BANK_INDEX.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
            return alias
    return "general"


def load_all_bank_context() -> str:
    """Load all bank guidelines concatenated (for pre-loading into system prompts)."""
    parts = []
    for md_file in sorted(BANKS_DIR.glob("*.md")):
        parts.append(f"## {md_file.stem}\n{md_file.read_text()}")
    return "\n\n".join(parts)
