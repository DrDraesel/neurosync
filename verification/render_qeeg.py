"""Readable screenshot of the qEEG 5x5 tab, driven by the labelled simulator.

Renders on the REAL Qt platform (offscreen text is tofu in this environment) and
uses the built-in DragonEEG simulator, so no headset is touched. Writes:

    verification/out_qeeg/qeeg_window.png
    verification/out_qeeg/qeeg_sheet.png

Run:  .venv/Scripts/python.exe verification/render_qeeg.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from PyQt6.QtWidgets import QApplication          # noqa: E402

import main as app_main                           # noqa: E402

OUT = Path(__file__).resolve().parent / 'out_qeeg'


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.05)


def main():
    OUT.mkdir(exist_ok=True)
    app = QApplication(sys.argv[:1])
    window = app_main.NeuroSyncApp(autostart=False, simulated='dragon')
    window.resize(1600, 1000)
    window.show()
    window.start_connection()
    pump(app, 12.0)                      # enough for several Welch segments

    window.tabs.setCurrentWidget(window.qeeg_panel)
    pump(app, 2.0)
    window.qeeg_panel.refresh(force=True)
    pump(app, 1.0)

    for name, widget in (('qeeg_window.png', window),
                         ('qeeg_sheet.png', window.qeeg_panel.sheet)):
        pixmap = widget.grab()
        target = OUT / name
        pixmap.save(str(target))
        print(f'{name}: {pixmap.width()}x{pixmap.height()} -> {target}')

    exported = window.qeeg_panel.export_sheet_png(OUT / 'qeeg_sheet_export.png')
    print('exported:', exported, exported.stat().st_size, 'bytes')
    print('reference:', window.qeeg_panel.reference_label.text())
    print('cells:', len(window.qeeg_panel.sheet.cells))
    window.close()
    app.processEvents()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
