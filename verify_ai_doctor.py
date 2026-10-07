"""Live check of the AI-doctor path against the local Ollama model.

Uses TEST ONLY synthetic fixtures (a 10 Hz sine), never real EEG, and prints
the raw reply so the wiring, the model and the non-diagnostic framing can be
verified in one command:
    python verify_ai_doctor.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import ai_doctor as ai
import eeg_analysis as eeg


def main():
    t = np.arange(15000) / 250
    samples = np.tile((12 * np.sin(2 * np.pi * 10 * t))[:, None], (1, 4))  # TEST ONLY
    contact = dict.fromkeys(eeg.CHANNELS, 500000.)
    report = eeg.analyze_window(samples[-1250:], fs=250, notch_hz=60,
                                contact_ohms=contact, contact_age_s=0.)
    stable = eeg.analyze_stable(samples, fs=250, notch_hz=60,
                                contact_ohms=contact, contact_age_s=0.)
    payload = ai.build_payload(report=report, stable=stable,
                               transport={'samples': 15000, 'rate_hz': 250.1, 'gap_events': 0,
                                          'battery': None, 'mode': 'signal'},
                               device={'name': 'BrainBit', 'serial': 'TESTONLY-FIXTURE',
                                       'fs': 250, 'battery': None, 'address': ''},
                               contact=contact)
    print('installed models:', ai.available_models())
    print('fixture five-second valid:', report['valid'],
          '| stable valid:', stable['valid'],
          f"| clean epochs: {stable['epochs_used']}/{stable['epochs_total']}")
    start = time.time()
    result = ai.ask(payload, model=ai.DEFAULT_MODEL, timeout=540.)
    result['elapsed_s'] = round(time.time() - start, 1)
    print(json.dumps(result, indent=2))
    Path(__file__).with_name('verification').mkdir(exist_ok=True)
    (Path(__file__).parent / 'verification' / 'ai-doctor-last-reply.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    return 0 if result.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
