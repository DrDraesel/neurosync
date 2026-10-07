"""Readable screenshot of BOTH NeuroSync windows with the simulator running.

Per skill reference qt-ui-verification.md: offscreen text is tofu boxes, so
this renders on the REAL platform (the windows flash briefly on the desktop)
and drives the labelled simulator for both devices so every panel is populated.

Run:  python verification/render_readable.py
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
import second_device                              # noqa: E402

OUT = Path(__file__).resolve().parent / 'out_dual_sim'
PUMP_SECONDS = 18.0


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.05)


def main():
    OUT.mkdir(exist_ok=True)
    app = QApplication(sys.argv[:1])

    window = app_main.NeuroSyncApp(autostart=False, simulated='brainbit')
    window.resize(1560, 980)
    window.show()
    window.start_connection()

    second = second_device.SecondDeviceWindow(parent=None, simulated='dragon')
    second.resize(1180, 860)
    second.show()
    second.connect_device()

    pump(app, 6.0)          # maps first render
    pump(app, PUMP_SECONDS)  # let stable estimate + predominant settle

    window.analyze_stable_now()
    second.analyze_now()
    pump(app, 1.0)

    for name, widget in (('main_window_readable.png', window),
                         ('second_window_readable.png', second),
                         ('main_map_panel.png', window.map_plot),
                         ('main_average_wave_panel.png', window.avg_plot),
                         ('second_map_panel.png', second.map_plot)):
        pixmap = widget.grab()
        target = OUT / name
        pixmap.save(str(target))
        print(f'{name}: {pixmap.width()}x{pixmap.height()} -> {target}')

    print('main  predominant:', window.predominant.text())
    print('second predominant:', second.predominant.text())
    print('main  map rendered:', window.map_image.image is not None)
    print('second map rendered:', second.map_image.image is not None)
    window.close()
    second.close()
    app.processEvents()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
