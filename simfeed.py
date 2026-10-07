"""Clearly-labelled simulated EEG feed for NeuroSync.

WHY THIS EXISTS
---------------
The app's real-device path is BLE-only, which makes it impossible to rehearse
or debug the live analysis pipeline (band powers, topomaps, predominant-rhythm
read-out, stable estimate, session saving, AI payload) without physically
wearing a headset.  This module provides a *simulator* that emits exactly the
same signal/batch/metrics dictionaries as ``engine.EEGClient`` so every live
analysis can be exercised and verified end-to-end on this machine.

HONESTY RULES (do not weaken)
-----------------------------
* Simulated data is ALWAYS labelled: the device info dict carries
  ``simulated: True`` and a device label that starts with ``SIMULATED``,
  the batch dicts carry ``simulated: True``, and saved summaries/payloads
  state it too.  Nothing that leaves this module may look like a measurement.
* The simulation is a fixed mathematical signal (a dominant 10 Hz alpha rhythm
  with harmonics, 1/f noise, line noise and per-channel attenuation), not a
  render of any person's brain.  It exists to prove the pipeline RUNS in real
  time — never to support a biological claim.
"""
from __future__ import annotations

import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

import devices

SIM_NOTE = ('SIMULATED ELECTRODE DATA — no headset is connected. The signal is a '
            'fixed mathematical model (alpha rhythm + noise) generated on this PC; '
            'use it to rehearse the live analysis pipeline only.')

DEFAULT_FAMILY = 'brainbit'
# For the DragonEEG layout the simulator uses the documented 21 scalp sites
# (devices.DRAGON_DOCUMENTED_EEG_CHANNELS); channel order is preserved so the
# simulated batches exercise the same 21-channel paths as a real headset.
FS = 250
BLOCK_MS = 40.0          # 10 samples per block, like a BLE notification burst
ALPHA_HZ = 10.0
ALPHA_UV = 21.0          # occipital alpha amplitude in microvolts


def _channel_weight(name: str) -> float:
    """Fixed per-electrode weight of the 10 Hz rhythm (0..1).

    Posterior sites carry the strongest alpha rhythm in this model, frontal
    sites the weakest — the same posterior-dominance convention used in
    methodological EEG teaching, NOT a claim about any real recording.
    """
    name = name.upper()
    if name.startswith('O'):
        return 1.00
    if name.startswith('P'):
        return 0.78
    if name.startswith('T'):
        return 0.62
    if name.startswith('C'):
        return 0.55
    if name.startswith('Fp'):
        return 0.28
    if name.startswith('F'):
        return 0.38
    if name in ('D1', 'D2', 'D3'):
        return 0.20
    return 0.5


class Simulator:
    """Pure signal generator (no Qt) — unit-testable without a window."""

    def __init__(self, channels, fs: int = FS, seed: int = 20260929):
        self.channels = tuple(channels)
        self.fs = int(fs)
        self._rng = np.random.default_rng(seed)
        self._n = 0                    # samples generated so far
        self._weights = np.array([_channel_weight(name) for name in self.channels])

    def block(self, count: int):
        """(count, n_channels) microvolt block continuing the signal."""
        t = (np.arange(self._n, self._n + count) + 0.5) / self.fs
        self._n += count
        base = np.sin(2 * np.pi * ALPHA_HZ * t)[:, None]
        harmonic = 0.18 * np.sin(2 * np.pi * 2 * ALPHA_HZ * t)[:, None]
        slow = 0.35 * np.sin(2 * np.pi * 4.4 * t + 0.7)[:, None]      # theta-ish
        signal = (ALPHA_UV * self._weights)[None, :] * (base + harmonic) \
            + (6.0 * self._weights)[None, :] * slow
        # 1/f-ish noise + a clearly larger line-noise component on the wet-ish
        # channels, so the notch path also has something to remove.
        noise = self._rng.normal(0.0, 4.5, size=(count, len(self.channels)))
        line = 2.2 * np.sin(2 * np.pi * 60.0 * t)[:, None]
        return signal + noise + line

    @property
    def generated(self) -> int:
        return self._n


def channels_for(family: str) -> tuple:
    family = (family or DEFAULT_FAMILY).lower()
    if family.startswith('dragon'):
        return tuple(devices.DRAGON_DOCUMENTED_EEG_CHANNELS)
    return tuple(devices.CLASSIC_CHANNELS)


def device_info_for(family: str) -> dict:
    family = (family or DEFAULT_FAMILY).lower()
    if family.startswith('dragon'):
        channels = channels_for(family)
        key, label = 'dragon_eeg', 'SIMULATED DragonEEG / NeuroEEG (21 scalp sites, 250 Hz)'
    else:
        channels = channels_for(family)
        key, label = 'brainbit_classic', 'SIMULATED BrainBit Classic / Black (4 channels, 250 Hz)'
    return {
        'name': 'Simulator', 'serial': 'SIM-0000', 'address': 'SIM:00:00:00:00:00',
        'supported': True,
        'battery': 100, 'sampling_frequency': FS, 'fs': FS,
        'family': 'simulated', 'device_key': key, 'device_label': label,
        'channels': list(channels),
        'scalp_channels': list(devices.scalp_channels(channels)),
        'poly_channels': list(devices.poly_channels(channels)),
        'channel_source': 'built-in simulator (not a device)',
        'pairing_required': False, 'rssi': None,
        'simulated': True, 'simulation_note': SIM_NOTE,
        'configuration': {'transport': 'simulated', 'fs_hz': FS,
                          'amplifier': 'not applicable (simulator)',
                          'channels': list(channels)},
    }


def contact_for(channels) -> dict:
    """A passing resistance set — every value comfortably below the 1 MOhm gate."""
    channels = tuple(channels)
    rng = np.random.default_rng(7)
    ohms = {name: float(round(min(120_000 + i * 18_000 + rng.integers(0, 40_000), 880_000)))
            for i, name in enumerate(channels)}
    aux = {'A1': 240_000.0, 'A2': 260_000.0, 'Bias': 280_000.0}
    return {'ohms': ohms, 'aux': aux, 'channels': list(channels),
            'device_key': 'simulator', 'simulated': True,
            'measured_monotonic': time.monotonic()}


class SimulatedEEGClient(QThread):
    """Drop-in stand-in for ``engine.EEGClient`` (same signals + methods)."""

    data_received = pyqtSignal(list)
    batch_received = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    device_info = pyqtSignal(dict)
    devices_found = pyqtSignal(list)
    contact_received = pyqtSignal(dict)
    stream_reset = pyqtSignal(str)
    metrics_received = pyqtSignal(dict)

    def __init__(self, family: str = DEFAULT_FAMILY, block_ms: float = BLOCK_MS,
                 auto_contact: bool = True, **kwargs):
        # Unknown engine kwargs (scanner_factory, target_address, ...) are
        # accepted and ignored so call sites need no special-casing.
        super().__init__(kwargs.get('parent'))
        self.family = (family or DEFAULT_FAMILY).lower()
        self.channels = tuple(channels_for(self.family))
        self.fs = FS
        self.block_ms = float(block_ms)
        self.auto_contact = bool(auto_contact)
        self.mode = 'signal'
        self.simulator = Simulator(self.channels, fs=self.fs)
        self._running = False
        self._contact_done = False
        self.contact_seconds = 3.0
        self.sample_count = 0
        self.packet_number = 0
        self.started_at = None

    # ------------------------------------------------------------- interface
    def isRunning(self) -> bool:                     # noqa: N802 (Qt name)
        return self._running

    def stop(self, *args, **kwargs):
        self._running = False
        self.wait(2000)
        return True

    def select_device(self, address):                # scan flow is not used
        self._status(f'Simulator ignores device selection ({address}).')

    def check_contact(self):
        if not self._running:
            return
        self.mode = 'contact'
        self._reset_stream('Electrode resistance check (simulated); '
                           'collecting a fresh five-second window')
        self.contact_received.emit(contact_for(self.channels))
        self._contact_done = True
        self.mode = 'signal'
        self._status('Simulated contact check complete: all channels below 1 MOhm.')

    # -------------------------------------------------------------- internal
    def _status(self, text):
        self.status_changed.emit(text)

    def _reset_stream(self, reason):
        self.stream_reset.emit(reason)

    def _metrics(self):
        elapsed = max(time.monotonic() - (self.started_at or time.monotonic()), 1e-3)
        rate = self.sample_count / elapsed
        self.metrics_received.emit({
            'samples': self.sample_count, 'rate_hz': round(rate, 1),
            'gap_events': 0, 'counter_anomalies': 0, 'battery': 100,
            'mode': self.mode, 'channels': list(self.channels),
            'last_sample_age_s': 0.0, 'simulated': True,
        })

    def run(self):                                   # pragma: no cover - timing loop
        self._running = True
        self.mode = 'connecting'
        self._status('Simulator: preparing a ' + ('DragonEEG' if self.family.startswith('dragon')
                                                  else 'BrainBit') + ' layout…')
        time.sleep(0.4)
        self.device_info.emit(device_info_for(self.family))
        self.devices_found.emit([device_info_for(self.family)])
        self.mode = 'signal'
        self._status('SIMULATED stream running — no headset is connected. ' + SIM_NOTE)
        self.started_at = time.monotonic()
        per_block = max(1, int(round(self.fs * self.block_ms / 1000.0)))
        next_tick = time.monotonic()
        last_metrics = time.monotonic()
        while self._running:
            values = self.simulator.block(per_block)
            self.packet_number += 1
            self.sample_count += values.shape[0]
            batch = {
                'samples_uv': [list(map(float, row)) for row in values],
                'packnums': [self.packet_number] * values.shape[0],
                'arrived_monotonic': time.monotonic(),
                'sample_count': self.sample_count,
                'gap_events': 0, 'counter_anomalies': 0,
                'channels': list(self.channels), 'fs': self.fs,
                'device_key': 'simulator', 'simulated': True,
            }
            self.batch_received.emit(batch)
            for row in batch['samples_uv']:
                self.data_received.emit(row)
            if self.auto_contact and not self._contact_done \
                    and self.started_at is not None \
                    and time.monotonic() - self.started_at > 1.5:
                self.check_contact()
            now = time.monotonic()
            if now - last_metrics >= 1.0:
                self._metrics()
                last_metrics = now
            next_tick += self.block_ms / 1000.0
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()     # keep real-time pacing after a stall
