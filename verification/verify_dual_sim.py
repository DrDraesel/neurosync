"""Offscreen end-to-end check: BOTH device windows streaming at once (simulated).

Main window  -> simulated BrainBit Classic (4 ch)
Second window-> simulated DragonEEG (24 ch)
Both run simultaneously for ~30 s; then this script saves one session per
subject, builds the downloadable patient report and grabs window screenshots.

Run:  QT_QPA_PLATFORM=offscreen python verification/verify_dual_sim.py
Writes evidence to verification/out_dual_sim/ (result.json + PNGs).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from PyQt6.QtCore import QTimer                       # noqa: E402
from PyQt6.QtWidgets import QApplication              # noqa: E402

import main as app_main                               # noqa: E402
import patient_report                                 # noqa: E402
import second_device                                  # noqa: E402
import storage                                        # noqa: E402

OUT = Path(__file__).resolve().parent / 'out_dual_sim'
RUN_SECONDS = 30.0


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


def curve_points(curve):
    data = curve.getData()
    return 0 if data is None or data[0] is None else len(data[0])


def main():
    OUT.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='neurosync_dualsim_'))
    subject_root = work / 'subjects'
    subject_root.mkdir(parents=True, exist_ok=True)
    storage.DEFAULT_ROOT = subject_root
    app_main.SUBJECT_ROOT = subject_root
    app_main.QMessageBox = _StubBox
    second_device.QMessageBox = _StubBox

    app = QApplication(sys.argv[:1])
    result = {'when': datetime.now(timezone.utc).isoformat(), 'workdir': str(work)}

    window = app_main.NeuroSyncApp(autostart=False, simulated='brainbit')
    window.show()
    window.start_connection()

    second = second_device.SecondDeviceWindow(parent=window, simulated='dragon')
    second.show()
    second.connect_device()

    def snapshot():
        report_a = window.last_report or {}
        report_b = second.last_report or {}
        result['main'] = {
            'device': window.device_metadata.get('device_label'),
            'channels': list(window.channels),
            'samples': len(window.samples),
            'report_valid': bool(report_a.get('valid')),
            'dominant': report_a.get('dominant'),
            'dominant_pct': report_a.get('dominant_pct'),
            'peak_hz': report_a.get('peak_hz'),
            'predominant_label': window.predominant.text(),
            'avg_curve_points': curve_points(window.avg_curve),
            'map_rendered': window.map_image.image is not None,
            'channel_rows': window.channel_table.rowCount(),
        }
        result['second'] = {
            'device': second.device_metadata.get('device_label'),
            'channels': list(second.channels),
            'samples': len(second.samples),
            'report_valid': bool(report_b.get('valid')),
            'dominant': report_b.get('dominant'),
            'dominant_pct': report_b.get('dominant_pct'),
            'predominant_label': second.predominant.text(),
            'map_rendered': second.map_image.image is not None,
        }
        try:
            window.grab().save(str(OUT / 'main_window.png'))
            second.grab().save(str(OUT / 'second_window.png'))
            result['screenshots'] = True
        except Exception as exc:                       # pragma: no cover
            result['screenshots'] = f'failed: {exc}'

    def finish():
        window.subject = {'name': 'Sim Verify A', 'phone': '0001'}
        second.subject = {'name': 'Sim Verify B', 'phone': '0002'}
        window.save_session_clicked()
        second.save_session()
        sessions_a = storage.list_sessions(subject_root, 'Sim Verify A', '0001')
        sessions_b = storage.list_sessions(subject_root, 'Sim Verify B', '0002')
        result['saved_sessions'] = {'main': len(sessions_a), 'second': len(sessions_b)}
        try:
            report_path = patient_report.build(
                subject_root, 'Sim Verify A', '0001',
                out_path=OUT / 'patient_report_sim_verify.html')
            text = report_path.read_text(encoding='utf-8')
            result['patient_report'] = {
                'path': str(report_path), 'bytes': report_path.stat().st_size,
                'has_simulated_banner': 'SIMULATED DATA' in text,
                'has_embedded_charts': text.count('data:image/png;base64,') > 0,
                'embedded_chart_count': text.count('data:image/png;base64,'),
                'has_interpretation_helper': 'How to read these numbers' in text,
                'has_history_table': 'History — all saved sessions' in text,
            }
        except Exception as exc:
            result['patient_report'] = {'error': repr(exc)}
        (OUT / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))
        print('EVIDENCE:', OUT)
        shutil.copy(report_path if 'patient_report' in result and 'path' in result['patient_report']
                    else OUT / 'result.json', OUT / 'patient_report_copy.html') \
            if result.get('patient_report', {}).get('path') else None

    def has_map():
        return window.map_image.image is not None and second.map_image.image is not None

    def early():
        if has_map():
            window.analyze_stable_now(); second.analyze_now()
            result['both_maps_ready_after_s'] = (datetime.now(timezone.utc)).isoformat()
            early_timer.stop()
            return
        # keep pumping
    early_timer = QTimer(); early_timer.timeout.connect(early); early_timer.start(500)

    def arm():
        early_timer.stop()
        QTimer.singleShot(int(RUN_SECONDS * 1000), snapshot)
        QTimer.singleShot(int((RUN_SECONDS + 2) * 1000), finish)
        QTimer.singleShot(int((RUN_SECONDS + 4) * 1000), app.quit)

    QTimer.singleShot(300, arm)
    QTimer.singleShot(int((RUN_SECONDS + 30) * 1000), app.quit)   # hard stop
    code = app.exec()
    pending = OUT / 'result.json'
    if not pending.exists():
        print('FAILED: no result.json produced (window never reached snapshot)')
        return 2
    payload = json.loads(pending.read_text(encoding='utf-8'))
    ok = (payload.get('main', {}).get('report_valid')
          and payload.get('second', {}).get('report_valid')
          and payload.get('main', {}).get('map_rendered')
          and payload.get('second', {}).get('map_rendered')
          and payload.get('main', {}).get('avg_curve_points', 0) > 0
          and payload.get('saved_sessions', {}).get('main') == 1
          and payload.get('saved_sessions', {}).get('second') == 1)
    print('VERDICT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
