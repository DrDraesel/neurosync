"""Offscreen end-to-end check of the NeuroSync LIVE CHAT monitor (real local model).

Simulated BrainBit stream -> the live chat posts updates while the recording is
"on" -> the script waits for two streamed updates, grabs screenshots and writes
result.json. Requires Ollama with qwen3.8:latest on 127.0.0.1:11434.

Run (functional, tofu screenshots):  QT_QPA_PLATFORM=offscreen <venv-python> verification/verify_live_chat.py
Run (readable screenshots, window flashes on the desktop):  <venv-python> verification/verify_live_chat.py
Writes verification/out_live_chat/{result.json, window.png, chat_panel.png}.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from PyQt6.QtCore import QTimer                       # noqa: E402
from PyQt6.QtWidgets import QApplication              # noqa: E402

import ai_doctor                                      # noqa: E402
import main as app_main                               # noqa: E402
import storage                                        # noqa: E402

OUT = Path(__file__).resolve().parent / 'out_live_chat'
TARGET_UPDATES = 2
DEADLINE_S = 300.0


class _StubBox:
    @staticmethod
    def information(*args, **kwargs):
        return None

    @staticmethod
    def warning(*args, **kwargs):
        return None

    @staticmethod
    def critical(*args, **kwargs):
        return None


def main():
    OUT.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='neurosync_livechat_'))
    storage.DEFAULT_ROOT = work / 'subjects'
    storage.DEFAULT_ROOT.mkdir(parents=True, exist_ok=True)
    app_main.SUBJECT_ROOT = storage.DEFAULT_ROOT
    app_main.QMessageBox = _StubBox

    app = QApplication(sys.argv[:1])
    window = app_main.NeuroSyncApp(autostart=False, simulated='brainbit')
    window.resize(1560, 980)
    window.show()
    window.start_connection()

    # enable the live-chat monitor exactly the way the toggle button does
    window.chat_interval_s = 6.0
    window.chat_enabled = True
    window.chat_toggle.setText('⏹ Stop live monitor')
    window._set_chat_status('On — verification run (simulated BrainBit)…')
    threading.Thread(target=lambda: ai_doctor.warm(ai_doctor.DEFAULT_MODEL),
                     daemon=True).start()
    window.maybe_chat_update()

    result = {'when': datetime.now(timezone.utc).isoformat(),
              'platform': app.platformName(), 'device': None, 'updates': 0,
              'status': '', 'chat_text': '', 'warm_model': ai_doctor.DEFAULT_MODEL}
    finished = {'done': False, 'draining': False}

    def poll():
        if finished['done']:
            return
        worker_busy = window.chat_worker is not None and window.chat_worker.isRunning()
        result['device'] = window.device_metadata.get('device_label')
        result['updates'] = window.chat_updates
        result['status'] = window.chat_status.text()
        text = (window.chat_log.toPlainText()
                if app_main.widget_alive(window.chat_log) else '')
        if (not finished['draining'] and window.chat_updates >= TARGET_UPDATES
                and text.count('LOCAL MODEL') >= TARGET_UPDATES):
            # Fewer than two updates may still be streaming; stop scheduling new
            # ones and let the in-flight reply drain before grabbing evidence.
            window.chat_enabled = False
            window.chat_toggle.setText('▶ Start live monitor')
            try:
                window.right_scroller.ensureWidgetVisible(window.chat_log)
            except Exception:
                pass
            finished['draining'] = True
            return
        if finished['draining'] and not worker_busy:
            finished['done'] = True
            result['chat_text'] = text
            result['simulated_stated'] = 'simulated' in text.lower()
            try:
                window.grab().save(str(OUT / 'window.png'))
                window.chat_log.grab().save(str(OUT / 'chat_panel.png'))
                result['screenshots'] = True
            except Exception as exc:                       # pragma: no cover
                result['screenshots'] = f'failed: {exc}'
            (OUT / 'result.json').write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            print('LIVE CHAT VERIFIED ' + json.dumps(
                {k: result[k] for k in ('platform', 'device', 'updates', 'screenshots')}),
                flush=True)
            print('---- chat transcript ----', flush=True)
            print(text, flush=True)
            QTimer.singleShot(300, app.quit)

    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(2000)

    def give_up():
        if not finished['done']:
            result['chat_text'] = window.chat_log.toPlainText()
            (OUT / 'result.json').write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            print('LIVE CHAT TIMEOUT ' + json.dumps(result)[:900], flush=True)
            app.quit()

    QTimer.singleShot(int(DEADLINE_S * 1000), give_up)
    app.exec()
    return 0 if finished['done'] else 1


if __name__ == '__main__':
    sys.exit(main())
