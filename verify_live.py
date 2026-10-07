"""Launch the real GUI and capture one evidence frame; NEVER generate EEG."""
import sys
import json
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from main import NeuroSyncApp
app=QApplication(sys.argv)
window=NeuroSyncApp()
window.show()
output=Path(__file__).resolve().parent/'verification'
output.mkdir(exist_ok=True)
timer=QTimer()
def verify():
    if len(window.samples)<1250 or not window.last_report or not window.last_report['ready']:
        return
    window.update_plots()
    curves={ch:len(curve.getData()[0]) for ch,curve in window.raw_curves.items()}
    bands={name:len(curve.getData()[0]) for name,curve in window.band_curves.items()}
    colors={name:curve.opts['pen'].color().name() for name,curve in window.band_curves.items()}
    result={'source':'LIVE SDK; no generated samples','total_samples':window.total_received,
            'raw_curve_points':curves,'band_curve_points':bands,'band_colors':colors,
            'contact':window.contact,'analysis_valid':window.last_report['valid'],
            'analysis_reason':window.last_report['reason'],'dominant':window.last_report['dominant'],
            'transport':window.transport_metrics}
    assert set(curves)=={'O1','O2','T3','T4'} and all(n==1250 for n in curves.values())
    assert len(bands)==5 and all(n==1250 for n in bands.values()) and len(set(colors.values()))==5
    window.grab().save(str(output/'live-desktop.png'))
    (output/'live-verification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('LIVE_UI_VERIFIED '+json.dumps(result),flush=True)
    timer.stop()
timer.timeout.connect(verify)
timer.start(1000)
sys.exit(app.exec())
