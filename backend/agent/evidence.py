"""Per-run evidence ledger: every tool call gets an id (E1, E2, ...) briefs can cite."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Literal

Source = Literal["real", "mock"]


class Ledger:
    """Thread-safe record of one run's tool calls; create one Ledger per run."""

    def __init__(self) -> None:
        """Start an empty ledger whose first entry will be E1."""
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, tool: str, args: dict[str, Any], result: dict[str, Any], source: Source,
            cached: bool = False) -> dict[str, Any]:
        """Record one tool call and return the new entry (with its evidence id).

        cached=True adds "cached": true, marking a result reused from data/cache/.
        """
        if source not in ("real", "mock"):
            raise ValueError("source must be 'real' or 'mock'.")
        # Round-trip through JSON so stored evidence is JSON-safe and detached from callers.
        safe_args = json.loads(json.dumps(args, default=str))
        safe_result = json.loads(json.dumps(result, default=str))
        with self._lock:
            entry = {
                "id": f"E{len(self._entries) + 1}",
                "tool": tool,
                "args": safe_args,
                "result": safe_result,
                "source": source,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            if cached:
                entry["cached"] = True
            self._entries.append(entry)
        return entry

    def get(self, id: str) -> dict[str, Any] | None:
        """Return the entry with this evidence id (e.g. "E3"), or None."""
        key = id.strip().strip("[]").upper()
        with self._lock:
            return next((e for e in self._entries if e["id"] == key), None)

    def all(self) -> list[dict[str, Any]]:
        """Return all entries in call order."""
        with self._lock:
            return list(self._entries)

    @classmethod
    def from_entries(cls, entries: list[dict[str, Any]]) -> "Ledger":
        """Rebuild a ledger from saved entries (ids are kept, so new ids continue the sequence)."""
        ledger = cls()
        ledger._entries = json.loads(json.dumps(list(entries)))
        return ledger

    def to_json(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable copy of every entry."""
        return json.loads(json.dumps(self.all()))
