"""Bank knowledge base — loaded from config/banks/*.yaml files.

Each YAML file describes one bank: aliases (for detection), hotlines,
official numbers, SMS senders, apps, and what the bank would never vs
legitimately do during a call.
"""

from __future__ import annotations

import yaml
from pathlib import Path


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "banks"


def load_banks(config_dir: Path = _CONFIG_DIR) -> dict[str, dict]:
    """Load all bank YAMLs into a dict keyed by bank key.

    Normalizes the structure to match what server._bank_context() expects:
    official_numbers becomes a flat list of strings (not dicts).
    """
    banks: dict[str, dict] = {}
    if not config_dir.exists():
        return banks

    for yaml_path in sorted(config_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_path.read_text())
        except yaml.YAMLError as exc:
            print(f"  ! Failed to parse {yaml_path.name}: {exc}")
            continue
        if not isinstance(data, dict):
            continue

        key = data.get("key") or yaml_path.stem

        # Normalize official_numbers — can be list of dicts or list of strings.
        raw_numbers = data.get("official_numbers", [])
        flat_numbers: list[str] = []
        for n in raw_numbers:
            if isinstance(n, dict):
                flat_numbers.append(n.get("number", ""))
            elif isinstance(n, str):
                flat_numbers.append(n)
        data["official_numbers"] = [n for n in flat_numbers if n]

        # Clean up multi-line strings from YAML block scalars.
        for k in ("never", "legit"):
            if k in data and isinstance(data[k], str):
                data[k] = " ".join(data[k].split())

        banks[key] = data

    return banks


def detect_bank(text: str, banks: dict[str, dict]) -> str | None:
    """Return bank key if any bank alias is mentioned in *text*, else None."""
    lower = text.lower()
    for bank_key, info in banks.items():
        for alias in info.get("aliases", []):
            if alias in lower:
                return bank_key
    return None


def bank_context(bank_key: str, banks: dict[str, dict]) -> str:
    """Return a compact bank-specific context string for prompt injection."""
    b = banks.get(bank_key, {})
    name = b.get("name", bank_key.upper())
    return (
        f"== Bank detected: {name} ==\n"
        f"Hotline: {b.get('hotline', 'N/A')} | Fraud: {b.get('fraud_line', 'N/A')}\n"
        f"App: {b.get('app', 'N/A')} | Auth: {b.get('auth', 'N/A')} | Online: {b.get('online', 'N/A')}\n"
        f"NEVER does: {b.get('never', 'N/A')}\n"
        f"May legitimately do: {b.get('legit', 'N/A')}\n"
        f"Use these facts to improve your risk assessment. "
        f"Recommend calling {b.get('hotline', 'the official hotline')} if suspicious."
    )
