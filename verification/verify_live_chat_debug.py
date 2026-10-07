"""DEBUG variant: short interval, verbose poll prints, quits after first matches."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from PyQt6.QtCore import QTimer                       # noqa: E402
from PyQt6.QtWidgets import QApplication              # noqa: E402

import ai_doctor                                      # noqa: E402
import main as app_main                               # noqa: E402
import storage                                        # noqa: E402


class _StubBox:
    information = staticmethod(lambda *a, **k: None)
    warning = staticmethod(lambda *a, **k: None)
    critical = staticmethod(lambda *a, **k: None)


def main():
    work = Path(tempfile.mkdtemp(prefix='neurosync_chatdbg_'))
    storage.DEFAULT_ROOT = work / 'subjects'
    storage.DEFAULT_ROOT.mkdir(parents=True, exist_ok=True)
    app_main.SUBJECT_ROOT = storage.DEFAULT_ROOT
    app_main.QMessageBox = _StubBox

    app = QApplication(sys.argv[:1])
    window = app_main.NeuroSyncApp(autostart=False, simulated='brainbit')
    window.show()
    window.start_connection()

    window.chat_interval_s = 2.0
    window.chat_enabled = True
    window.chat_toggle.setText('⏹ Stop live monitor')
    threading.Thread(target=lambda: ai_doctor.warm(ai_doctor.DEFAULT_MODEL),
                     daemon=True).start()
    window.maybe_chat_update()

    state = {'ticks': 0, 'hit': False}

    def poll():
        state['ticks'] += 1
        worker = window.chat_worker
        busy = worker is not None and worker.isRunning()
        text = window.chat_log.toPlainText()
        count = text.count('LOCAL MODEL')
        cond = (not busy and window.chat_updates >= 2 and count >= 2)
        print(f"[poll {state['ticks']}] busy={busy} updates={window.chat_updates} "
              f"count={count} cond={cond} worker_id={id(worker)} "
              f"alive={app_main.widget_alive(window.chat_log)}", flush=True)
        if state['ticks'] % 10 == 0:
            print('---- transcript so far ----', flush=True)
            print(text[-600:], flush=True)
        if cond and not state['hit']:
            state['hit'] = True
            print('CONDITION HIT', flush=True)
            QTimer.singleShot(200, app.quit)

    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(1000)
    QTimer.singleShot(60000, app.quit)
    app.exec()
    print('DEBUG DONE hit=', state['hit'], 'ticks=', state['ticks'], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
