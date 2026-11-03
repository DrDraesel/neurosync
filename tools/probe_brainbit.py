"""Headless BrainBit probe through the app's own engine (no GUI).

Scans, connects to the saved/supported device, runs the 5 s contact check,
collects a few seconds of EEG, prints what came back, disconnects.

Run:  .venv/Scripts/python.exe tools/probe_brainbit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402

import engine  # noqa: E402

app = QCoreApplication([])

status: list[str] = []
info: dict = {}
contact: dict = {}
batches: list[dict] = []

client = engine.EEGClient(scan_seconds=12, auto_contact=True, contact_seconds=5)
client.status_changed.connect(lambda s: status.append(s))
client.device_info.connect(lambda d: info.update(d))
client.contact_received.connect(lambda d: contact.update(d))
client.batch_received.connect(lambda b: batches.append(b))

loop = QEventLoop()
deadline = QTimer()
deadline.setSingleShot(True)
deadline.timeout.connect(loop.quit)
deadline.start(60000)          # hard stop after 60 s

def _maybe_done() -> None:
    rows = sum(len(b.get("samples_uv") or []) for b in batches)
    if contact and rows >= 1000:      # ~4 s at 250 Hz
        loop.quit()

poll = QTimer()
poll.timeout.connect(_maybe_done)
poll.start(500)

client.start()
loop.exec()
client.stop()
client.wait(5000)

rows = sum(len(b.get("samples_uv") or []) for b in batches)
print("STATUS:", " | ".join(dict.fromkeys(status))[:600])
print("INFO:", json.dumps(info, default=str)[:400])
print("CONTACT:", json.dumps(contact.get("ohms", {}), default=str)[:400])
print("BATCHES:", len(batches), "ROWS:", rows)
if batches:
    b = batches[0]
    print("PAYLOAD KEYS:", sorted(b.keys()))
    s = b.get("samples_uv") or []
    print("FIRST ROW:", s[0] if s else None, "| CHANNELS:", b.get("channels"))
print("RESULT:", "PASS" if rows > 0 else "FAIL")
