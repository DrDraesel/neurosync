"""End-to-end study-flow verification (offscreen, simulated signal).

Proves the full-duration study path: Start study -> 92,500 rows fed over ~74 s
wall time (the old 60 s deque would have kept only 15,000) -> Stop study ->
full-span analysis + save -> AI doctor note -> patient report with the
clinician reading-method section. TEST DATA ONLY (sine waves, never real EEG).
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # repo root, so `import main` resolves

import numpy as np
from PyQt6.QtWidgets import QApplication

import main
import storage
import patient_report

SUBJECT = ('StudyE2E', '')
FS = 250
BATCH = 500          # rows per feed
INTERVAL = 2.0       # seconds per feed -> 250 Hz
FEEDS = 37           # 37 * 500 = 18,500 rows... need 92,500 -> 185 feeds? see note
# 500 rows / 2 s = 250 Hz. 37 feeds = 18,500 rows in 74 s wall.
# To exceed the 15,000-row deque we need >15,000 rows -> 31 feeds (62 s).
# Use 37 feeds: rows = 18,500 (>15,000 deque), wall span ~74 s (>60 s).


class StubClient:
    mode = 'signal'

    def isRunning(self):
        return True

    def stop(self):
        return True


def main_run():
    app = QApplication.instance() or QApplication([])
    window = main.NeuroSyncApp(autostart=False)
    window.show()
    window.subject = {'name': SUBJECT[0], 'phone': SUBJECT[1]}
    window.client = StubClient()
    window.fs = FS
    window.channels = list(main.devices.CLASSIC.channels) if hasattr(main, 'devices') else ['O1', 'O2', 'T3', 'T4']
    window.device_metadata = {'device_key': 'brainbit-classic', 'device_label': 'SIMULATED BrainBit',
                              'channels': window.channels, 'family': 'classic',
                              'scalp_channels': window.channels, 'poly_channels': []}
    window.simulated = True

    window.start_study()
    assert window.study_started_monotonic is not None, 'study did not start'
    print('study started', flush=True)

    t = np.arange(BATCH) / FS
    sine = 18.0 * np.sin(2 * np.pi * 10.0 * t)
    rows = np.tile(sine[:, None], (1, len(window.channels)))
    total = 0
    started = time.monotonic()
    for i in range(FEEDS):
        batch = {'channels': tuple(window.channels),
                 'samples_uv': rows.tolist(),
                 'packnums': [total // 2 + j for j in range(BATCH)],
                 'arrived_monotonic': time.monotonic(),
                 'sample_count': total + BATCH}
        window.handle_batch(batch)
        total += BATCH
        app.processEvents()
        time.sleep(INTERVAL)
    wall = time.monotonic() - started
    print(f'fed {total} rows over {wall:.1f} s; study rows now: {len(window.study_rows)}', flush=True)

    window.stop_study(reason='user')
    app.processEvents()

    subject_dir = storage.subject_dir(storage.DEFAULT_ROOT, SUBJECT[0], SUBJECT[1])
    sessions = sorted(subject_dir.glob('*'), key=lambda p: p.name)
    folder = sessions[-1]
    summary_path = folder / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    print('saved folder:', folder, flush=True)
    print('summary recorded_span_s:', summary.get('recorded_span_s'),
          '| raw_rows:', summary.get('raw_rows'), '| duration_s:', summary.get('duration_s'), flush=True)
    csv_rows = sum(1 for _ in open(folder / 'raw.csv', encoding='utf-8')) - 1
    print('csv data rows:', csv_rows, flush=True)

    ok = (summary.get('raw_rows', 0) > 15000
          and summary.get('recorded_span_s', 0) > 60
          and csv_rows == summary['raw_rows']
          and summary.get('study', {}).get('kind') == 'study'
          and summary.get('limits') is not None)
    print('SPAN_CHECK', 'PASS' if ok else 'FAIL', flush=True)

    # Wait for the AI doctor note to land in summary.json (async worker).
    ai_deadline = time.monotonic() + 240
    while time.monotonic() < ai_deadline:
        app.processEvents()
        cur = json.loads(summary_path.read_text(encoding='utf-8'))
        if cur.get('ai'):
            summary = cur
            break
        time.sleep(2)
    ai = summary.get('ai') or {}
    print('AI note:', 'present' if ai.get('text') else 'absent/empty',
          '| model:', ai.get('model'), '| provenance:', ai.get('provenance'), flush=True)

    out = ROOT / 'verification' / 'out_study_e2e'
    out.mkdir(parents=True, exist_ok=True)
    report_path = patient_report.build(storage.DEFAULT_ROOT, SUBJECT[0], SUBJECT[1], out_path=out / 'report.html')
    html = report_path.read_text(encoding='utf-8')
    has_method = 'Reading method (for the clinician)' in html
    has_span = 'recording span' in html
    has_notadevice = 'Not a medical device' in html
    print('REPORT:', 'method_section=' + str(has_method),
          '| span_wording=' + str(has_span),
          '| not_a_device=' + str(has_notadevice), flush=True)

    window.close()
    app.processEvents()
    # Remove the synthetic subject so real data stays clean.
    import shutil
    shutil.rmtree(subject_dir, ignore_errors=True)
    final = ok and has_method and has_span and has_notadevice
    print('E2E', 'PASS' if final else 'FAIL', flush=True)
    return 0 if final else 1


if __name__ == '__main__':
    sys.exit(main_run())
