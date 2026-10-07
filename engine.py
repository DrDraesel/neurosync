"""Unified BLE transport for BrainBit Classic/Black, BrainBit 2/Pro/Flex and
BrainBit DragonEEG (NeuroEEG), driven only by the manufacturer SDK.

https://sdk.brainbit.com/sdk2_bb/ (Classic: 250 Hz, volts, packet counter wraps
2048, two samples per packet number, signal and resistance are exclusive)
https://pypi.org/project/pyneurosdk2/ (Python API, Scanner/create_sensor,
callbacks, per-family amplifier parameters)

Rules preserved from the original verified four-channel transport:

* one SDK callback stays tiny - samples and a monotonic arrival time go into a
  bounded queue and are decoded on the acquisition thread;
* a discontinuity (packet anomaly, callback gap, queue overflow) restarts the
  window instead of bridging the hole, and missing samples are never invented;
* EEG and resistance acquisition are never run at the same time;
* everything is torn down with StopSignal/StopResist, detached callbacks and
  disconnect in ``finally``.

Device-specific work added for the DragonEEG (LENeuroEEG):

* the channel table is read from the device (``supported_channels``, ordered by
  the device's ``Num``) instead of being assumed;
* Frequency, ChannelGain and ChannelMode are written AFTER connecting and
  verified by read-back BEFORE StartSignal - an unconfigured Dragon streams
  flat zeros;
* per-channel resistance arrives as ResistChannelsData with A1/A2/Bias
  reference values in ohms;
* first pairing requires the physical device to be in pairing mode; the app
  only reports the SDK's ``PairingRequired`` flag and asks the user to do that
  step, it never fakes it.

The Dragon path is written against the installed SDK and its shipped sample
(neurosdk/sample.py) and has NOT been exercised on physical Dragon hardware by
this project; see METHODS.md for the exact list of open verification items.
"""
import math
import time
import threading
from collections import deque
from queue import Queue, Empty, Full

from PyQt6.QtCore import QThread, pyqtSignal
from neurosdk.scanner import Scanner
from neurosdk.cmn_types import SensorCommand, SensorState

import devices
from devices import CLASSIC, SCAN_FAMILIES

CHANNELS = devices.CLASSIC_CHANNELS      # backward-compatible name
GENERIC_SCALE_UV = 1e6                   # volts -> microvolts (see devices.py)


class PacketTracker:
    """Classic hardware delivers two samples with each packet number.

    Repeated pair numbers are not dropped samples. Discontinuities clear the
    pre-gap part of a batch; derived windows must never bridge an interruption.
    """
    def __init__(self):
        self.last = None
        self.repeats = 0
        self.gap_events = 0

    def reset(self):
        self.last = None
        self.repeats = 0

    def decode(self, packets):
        rows, numbers = [], []
        gap = False
        for packet in packets:
            n = int(packet.PackNum)
            values = [float(getattr(packet, ch)) * GENERIC_SCALE_UV for ch in CHANNELS]
            bad = not 0 <= n <= 2047 or not all(math.isfinite(x) for x in values)
            if bad:
                gap = True
                self.gap_events += 1
                rows.clear(); numbers.clear(); self.reset()
                continue
            delta = None if self.last is None else (n - self.last) % 2048
            self.repeats = self.repeats + 1 if delta == 0 else 1
            if (delta is not None and delta not in (0, 1)) or self.repeats > 2:
                gap = True
                self.gap_events += 1
                rows.clear(); numbers.clear()
                self.repeats = 1
            self.last = n
            rows.append(values); numbers.append(n)
        return {'samples_uv': rows, 'packnums': numbers, 'gap': gap}


class ChannelPacketTracker:
    """Generic multi-channel packets (DragonEEG / BrainBit 2 families).

    Each packet carries a flat ``Samples`` list; the app splits it into
    consecutive frames of ``channel_count`` values (one value per channel per
    frame) and converts volts to microvolts exactly once. The packet-number
    semantics of these families are not documented for the app, so a counter
    that goes backwards or repeats is COUNTED as an anomaly rather than
    silently trusted, and only an excessive run of identical counters is
    treated as a discontinuity. Timing gaps and queue overflow, which are
    verifiable host-side facts, still invalidate the window.
    """
    def __init__(self, channel_count, max_repeats=4):
        if not isinstance(channel_count, int) or channel_count <= 0:
            raise ValueError('channel_count must be a positive integer')
        self.channel_count = channel_count
        self.max_repeats = int(max_repeats)
        self.last = None
        self.repeats = 0
        self.gap_events = 0
        self.counter_anomalies = 0

    def reset(self):
        self.last = None
        self.repeats = 0

    def decode(self, packets):
        rows, numbers = [], []
        gap = False
        for packet in packets:
            samples = [float(value) for value in getattr(packet, 'Samples', [])]
            number = int(getattr(packet, 'PackNum', 0))
            if (not samples or len(samples) % self.channel_count
                    or not all(math.isfinite(value) for value in samples)):
                gap = True
                self.gap_events += 1
                rows.clear(); numbers.clear(); self.reset()
                continue
            delta = None if self.last is None else number - self.last
            if delta is not None and delta <= 0:
                self.counter_anomalies += 1
                self.repeats = self.repeats + 1 if delta == 0 else 1
            else:
                self.repeats = 1
            if self.repeats > self.max_repeats:
                gap = True
                self.gap_events += 1
                rows.clear(); numbers.clear(); self.repeats = 1
            self.last = number
            for start in range(0, len(samples), self.channel_count):
                frame = samples[start:start + self.channel_count]
                rows.append([value * GENERIC_SCALE_UV for value in frame])
                numbers.append(number)
        return {'samples_uv': rows, 'packnums': numbers, 'gap': gap}


class EEGClient(QThread):
    """One scan-and-stream worker for any supported headset.

    Behaviour depends on the constructor arguments:

    * ``target_address`` set (default: the saved BrainBit Classic address) - the
      scan looks for that exact address and connects to it; nothing else is
      ever selected silently.
    * ``selection_required=True`` - the scan covers every supported family,
      emits ``devices_found`` with a JSON-safe description of each device and
      waits for ``select_device(address)`` (or, after ``selection_timeout``
      seconds, falls back to the saved address if it was among the results).
    """
    data_received = pyqtSignal(list)          # one row in uV (all channels)
    batch_received = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    device_info = pyqtSignal(dict)
    devices_found = pyqtSignal(list)
    contact_received = pyqtSignal(dict)
    stream_reset = pyqtSignal(str)
    metrics_received = pyqtSignal(dict)

    def __init__(self, scanner_factory=Scanner, scan_seconds=15,
                 target_address='DA:BB:A3:A0:73:4E', auto_contact=False,
                 contact_seconds=5, selection_required=False,
                 selection_timeout=120., device_spec=None):
        super().__init__()
        self.scanner_factory = scanner_factory
        self.scan_seconds = scan_seconds
        self.target_address = (target_address or '').upper()
        self.auto_contact = auto_contact
        self.contact_seconds = contact_seconds
        self.selection_required = bool(selection_required)
        self.selection_timeout = float(selection_timeout)
        self.device_spec = device_spec
        self._running = True
        self.device = None
        self.scanner = None
        self.infos = {}
        self.known_devices = []
        self.sensor_info = None
        self.sample_count = 0
        self.last_sample_at = None
        self.contact_requested = threading.Event()
        self.queue_overflow = threading.Event()
        self.samples_queue = Queue(maxsize=128)
        self.tracker = PacketTracker()
        self.rate_history = deque(maxlen=6)
        self.mode = 'disconnected'
        self.battery = None
        self.latest_contact = None
        self.channels = CLASSIC.channels
        self.fs = CLASSIC.fs
        self.role = 'classic'
        self.channel_source = 'device specification default'
        self.amplifier_report = {}
        self._last_callback = None
        self._selection = threading.Event()
        self._selected_address = None

    # ---------------------------------------------------------------- reports
    def _status(self, text):
        print(text, flush=True)
        self.status_changed.emit(text)

    def _clear_queue(self):
        while True:
            try: self.samples_queue.get_nowait()
            except Empty: break

    def _reset_stream(self, reason):
        self._clear_queue()
        self.tracker.reset()
        self.last_sample_at = self._last_callback = None
        self.rate_history.clear()
        self.stream_reset.emit(reason)

    def check_contact(self):
        self.contact_requested.set()

    def select_device(self, address):
        """Called from the GUI thread once the user picked a scanned device."""
        self._selected_address = (address or '').upper()
        self._selection.set()

    # ------------------------------------------------------------- callbacks
    def _on_signal(self, sensor, packets):
        # Native callbacks stay small: no filtering, file access or widget work.
        try:
            self.samples_queue.put_nowait((time.monotonic(), packets))
        except Full:
            self.queue_overflow.set()

    def _resist_callback(self):
        """Resistance decoder matching the connected device family."""
        role = self.role
        channels = tuple(self.channels)
        if role == 'classic':
            def received(sender, data):
                values = {name: _resistance(float(getattr(data, name, float('nan'))))
                          for name in CLASSIC.channels}
                self.latest_contact = {'ohms': values, 'aux': {},
                                       'channels': list(CLASSIC.channels),
                                       'device_key': CLASSIC.key,
                                       'measured_monotonic': time.monotonic()}
        else:
            def received(sender, data):
                packets = data if isinstance(data, (list, tuple)) else [data]
                values, aux = {}, {}
                for packet in packets:
                    samples = [float(value) for value in getattr(packet, 'Samples',
                                                                 getattr(packet, 'Values', []) or [])]
                    for index, name in enumerate(channels):
                        if index < len(samples):
                            values[name] = _resistance(samples[index])
                    for label in ('A1', 'A2', 'Bias'):
                        if hasattr(packet, label):
                            aux[label] = _resistance(float(getattr(packet, label)))
                self.latest_contact = {'ohms': values, 'aux': aux,
                                       'channels': list(channels),
                                       'device_key': self.device_spec.key if self.device_spec else 'unknown',
                                       'measured_monotonic': time.monotonic()}
        return received

    def _drain_samples(self):
        if self.queue_overflow.is_set():
            self.queue_overflow.clear()
            self.tracker.gap_events += 1
            self._reset_stream('Host queue overflow; collecting a new five-second window')
        for _ in range(128):
            try: arrived, packets = self.samples_queue.get_nowait()
            except Empty: break
            result = self.tracker.decode(packets)
            if self._last_callback is not None and arrived - self._last_callback > .5:
                result['gap'] = True
                self.tracker.gap_events += 1
            self._last_callback = arrived
            if result['gap']:
                self.stream_reset.emit('Packet/timing discontinuity; five-second window restarted')
            if not result['samples_uv']:
                continue
            self.sample_count += len(result['samples_uv'])
            self.last_sample_at = arrived
            result.update(arrived_monotonic=arrived, sample_count=self.sample_count,
                          gap_events=self.tracker.gap_events,
                          counter_anomalies=getattr(self.tracker, 'counter_anomalies', 0),
                          channels=list(self.channels), fs=self.fs,
                          device_key=self.device_spec.key if self.device_spec else 'unknown')
            self.batch_received.emit(result)
            for row in result['samples_uv']:
                self.data_received.emit(row)

    def _measure_contact(self, resume=False):
        if resume:
            self.device.exec_command(SensorCommand.StopSignal)
        self.mode = 'contact'
        self._reset_stream('Electrode resistance check; EEG temporarily paused')
        self._status('Checking electrode contact — keep the headset still '
                     f'({self.contact_seconds:g} seconds)')
        self.latest_contact = None
        self.device.resistDataReceived = self._resist_callback()
        started = False
        try:
            self.device.exec_command(SensorCommand.StartResist)
            started = True
            deadline = time.monotonic() + self.contact_seconds
            while self._running and time.monotonic() < deadline:
                time.sleep(.05)
        finally:
            if started:
                self.device.exec_command(SensorCommand.StopResist)
            self.device.resistDataReceived = None
        contact = self.latest_contact or {'ohms': {name: None for name in self.channels},
                                          'aux': {}, 'channels': list(self.channels),
                                          'device_key': self.device_spec.key if self.device_spec else 'unknown',
                                          'measured_monotonic': time.monotonic()}
        self.contact_received.emit(contact)
        self._status('Contact check complete: '
                     + str({name: _round_ohm(value) for name, value in contact['ohms'].items()})
                     + ' ohms')
        self._reset_stream('Contact check complete; collecting a fresh five-second EEG window')
        if resume and self._running:
            self.device.exec_command(SensorCommand.StartSignal)
            self.mode = 'signal'

    # --------------------------------------------------------- scan and setup
    def _default_spec(self):
        return self.device_spec or CLASSIC

    def _scan(self):
        if self.scanner is None:
            self.scanner = self.scanner_factory(list(SCAN_FAMILIES))
        try:
            self.scanner.start()
        except Exception as exc:
            self._status(f'Scanner restart failed ({exc}); creating a new scanner')
            self.scanner = self.scanner_factory(list(SCAN_FAMILIES))
            self.scanner.start()
        found = {}
        try:
            deadline = time.monotonic() + max(0., float(self.scan_seconds))
            while self._running:
                for info in self.scanner.sensors():
                    address = str(getattr(info, 'Address', '') or '').upper()
                    if address:
                        found[address] = info
                if time.monotonic() >= deadline:
                    break
                time.sleep(.1)
        finally:
            try:
                self.scanner.stop()
            except Exception as exc:
                self._status(f'Scan stop warning: {exc}')
        return found

    def _describe(self, infos):
        target = self.target_address
        described = []
        for address, info in infos.items():
            fallback = CLASSIC if address == target else None
            described.append(devices.describe_sensor_info(info, fallback))
        described.sort(key=lambda item: (not item['supported'], item['device_label'], item['address']))
        return described

    def _await_selection(self, described):
        if self._selection.is_set():          # the picker answered before we waited
            return self._selected_address
        deadline = time.monotonic() + max(0., self.selection_timeout)
        while self._running and time.monotonic() < deadline:
            if self._selection.wait(.1):
                return self._selected_address
        return None

    def _choose(self, infos, described):
        target = self.target_address
        if self.selection_required and described:
            self.mode = 'awaiting_selection'
            self._status(f"{len(described)} device(s) found — select one in the device list "
                         "and press Connect")
            selected = self._await_selection(described)
            if selected and selected in infos:
                return selected
            if selected:
                self._status(f'Selected device {selected} disappeared from the scan results')
            if target and target in infos:
                self._status('No new selection; using the saved BrainBit address')
                return target
            if not self._running:
                return None
            raise RuntimeError('No device selected. Press Scan, choose a device, then Connect.')
        if target and target in infos:
            return target
        if not infos:
            raise RuntimeError('No devices found. Power the headset on (in pairing mode the first '
                               'time), keep it near the PC, close other BrainBit apps, then scan again.')
        raise RuntimeError('Saved BrainBit was not found during the scan. Power-cycle it, bring it '
                           'nearby, close other BrainBit apps, then scan again.')

    def _supported_channels(self, sensor):
        try:
            supported = sensor.supported_channels
        except Exception as exc:
            self._status(f'Channel table unavailable ({exc}); using the documented list')
            return None
        return supported

    def _prepare_device(self, spec):
        """Device-type-specific configuration. Returns a JSON-safe report."""
        sensor = self.device
        if spec.transport == 'classic':
            self.role = 'classic'
            self.channels = tuple(CLASSIC.channels)
            self.channel_source = 'verified Classic contract (O1/O2/T3/T4)'
            frequency = sensor.sampling_frequency.name
            if frequency != 'FrequencyHz250':
                raise RuntimeError(f'Unsupported sample rate {frequency}; analysis requires confirmed 250Hz')
            self.fs = 250
            return {'transport': 'classic', 'fs_hz': 250,
                    'amplifier': 'not applicable (Classic exposes no amplifier block)',
                    'channels': list(self.channels)}
        if spec.transport == 'neuroeeg':
            self.role = 'neuroeeg'
            supported = self._supported_channels(sensor)
            names = devices.channel_names(supported)
            self.channels = tuple(names)
            self.channel_source = ('read from NeuroEEGSensor.supported_channels'
                                   if supported else 'documented fallback: device reported no channel table')
            # Generic multi-channel packets: one value per channel per frame.
            self.tracker = ChannelPacketTracker(len(self.channels))
            template = sensor.amplifier_param
            param = devices.neuroeeg_amplifier_param(template, len(self.channels), fs=spec.fs)
            sensor.amplifier_param = param
            check = sensor.amplifier_param
            problems = devices.neuroeeg_amplifier_problems(check, len(self.channels), spec.fs)
            if problems:
                raise RuntimeError('DragonEEG amplifier configuration failed: ' + '; '.join(problems))
            self.fs = spec.fs
            return {'transport': 'neuroeeg', 'fs_hz': spec.fs,
                    'amplifier': 'written and verified after connect, before StartSignal',
                    'referent_resist_measure_allow': bool(getattr(check, 'ReferentResistMesureAllow', False)),
                    'referent_mode': devices.family_name(getattr(check, 'ReferentMode', None)),
                    'channel_modes': [devices.family_name(mode) for mode in getattr(check, 'ChannelMode', [])],
                    'channel_gains': [devices.family_name(gain) for gain in getattr(check, 'ChannelGain', [])],
                    'channels': list(self.channels)}
        # BrainBit 2 / Pro / Flex
        self.role = 'brainbit2'
        supported = self._supported_channels(sensor)
        names = devices.channel_names(supported) if supported else spec.channels
        self.channels = tuple(names)
        self.channel_source = ('read from supported_channels' if supported
                               else 'documented fallback: device reported no channel table')
        self.tracker = ChannelPacketTracker(len(self.channels))
        template = sensor.amplifier_param
        param = devices.brainbit2_amplifier_param(template, len(self.channels))
        sensor.amplifier_param = param
        self.fs = spec.fs
        return {'transport': 'brainbit2', 'fs_hz': spec.fs,
                'amplifier': 'BrainBit2 parameter block written after connect',
                'channels': list(self.channels)}

    # -------------------------------------------------------------------- run
    def run(self):
        scanner = None
        streaming = False
        try:
            self._running = True
            infos = {}
            if self.sensor_info is None:
                self.mode = 'scanning'
                self._status(f'Scanning for {len(SCAN_FAMILIES)} device families: '
                             'BrainBit Classic/Black/2 and DragonEEG (NeuroEEG)...')
                infos = self._scan()
                scanner = self.scanner
                self.infos = infos
                described = self._describe(infos)
                self.known_devices = described
                self.devices_found.emit(described)
                if not self._running:
                    return
                address = self._choose(infos, described)
                if address is None:
                    return
                self.target_address = address
                self.sensor_info = infos[address]
            info = self.sensor_info
            address = str(getattr(info, 'Address', self.target_address) or self.target_address).upper()
            family = getattr(info, 'SensFamily', None)
            if self.device_spec is not None:
                spec = self.device_spec
            elif family is None:
                # A scanner fixture / legacy entry without a family may still be
                # the user's saved BrainBit address; anything else is refused.
                spec = self._default_spec() if address == self.target_address else None
            else:
                spec = devices.spec_for_family(family)
            if spec is None:
                raise RuntimeError(
                    f'This device family ({devices.family_name(family) or "unknown"}) is not '
                    'supported by the app; nothing was started.')
            self.device_spec = spec
            if self.scanner is None:
                raise RuntimeError('No scanner available; press Scan again.')
            self.mode = 'connecting'
            self._status(f'Connecting to {spec.label} via BLE...')
            self.device = self.scanner.create_sensor(info)
            report = self._prepare_device(spec)
            self.amplifier_report = report
            self.battery = self.device.batt_power
            info_out = {'name': self.device.name, 'serial': self.device.serial_number,
                        'address': address, 'battery': self.battery,
                        'sampling_frequency': report.get('sampling_frequency'),
                        'fs': self.fs, 'family': devices.family_name(family),
                        'device_key': spec.key, 'device_label': spec.label,
                        'channels': list(self.channels),
                        'scalp_channels': list(devices.scalp_channels(self.channels)),
                        'poly_channels': list(devices.poly_channels(self.channels)),
                        'channel_source': self.channel_source,
                        'pairing_required': bool(getattr(info, 'PairingRequired', False)),
                        'rssi': getattr(info, 'RSSI', None),
                        'configuration': report}
            self.device_info.emit(info_out)
            self._status(f"Connected {spec.label} | serial {info_out['serial']} | "
                         f"battery {self.battery}% | {self.fs} Hz | {len(self.channels)} channels")
            self.device.signalDataReceived = self._on_signal
            if self.auto_contact:
                self._measure_contact(resume=False)
            if not self._running:
                return
            self._reset_stream('Connected; collecting five seconds')
            self.device.exec_command(SensorCommand.StartSignal)
            streaming = True
            self.mode = 'signal'
            start = last_report = time.monotonic()
            self._drain_samples()
            while self._running:
                if self.contact_requested.is_set():
                    self.contact_requested.clear()
                    self._measure_contact(resume=True)
                    start = time.monotonic()
                self._drain_samples()
                now = time.monotonic()
                if now - (self.last_sample_at or start) > 8:
                    raise RuntimeError('No EEG for 8s. Check power/range, then reconnect.')
                if now - last_report >= 1:
                    if self.device.state != SensorState.StateInRange:
                        raise RuntimeError('Headset disconnected or out of range')
                    self.battery = self.device.batt_power
                    self.rate_history.append((now, self.sample_count))
                    rate = None
                    if len(self.rate_history) > 1:
                        t0, n0 = self.rate_history[0]
                        rate = (self.sample_count - n0) / (now - t0)
                    metrics = {'samples': self.sample_count, 'rate_hz': rate,
                               'gap_events': self.tracker.gap_events,
                               'counter_anomalies': getattr(self.tracker, 'counter_anomalies', 0),
                               'battery': self.battery, 'mode': self.mode,
                               'channels': list(self.channels),
                               'last_sample_age_s': None if self.last_sample_at is None
                               else now - self.last_sample_at}
                    self.metrics_received.emit(metrics)
                    previous = self.sample_count - int(rate or 0)
                    if len(self.rate_history) == 1 or self.sample_count // (7500 * max(1, len(self.channels) // 4)) \
                            != previous // (7500 * max(1, len(self.channels) // 4)):
                        self._status(f'Live EEG | {self.sample_count} samples | '
                                     f'{len(self.channels)} channels | gaps {self.tracker.gap_events}')
                    last_report = now
                time.sleep(.01)
        except Exception as exc:
            self.mode = 'error'
            self._status(f'Connection error: {exc}')
            self.stream_reset.emit('Connection lost; previous window invalidated')
        finally:
            if self.device is not None:
                try:
                    if streaming:
                        self.device.exec_command(SensorCommand.StopSignal)
                except Exception as exc:
                    self._status(f'Stop warning: {exc}')
                try:
                    self.device.signalDataReceived = None
                except Exception:
                    pass
                try:
                    self.device.resistDataReceived = None
                except Exception:
                    pass
                try:
                    self.device.disconnect()
                except Exception as exc:
                    self._status(f'Disconnect warning: {exc}')
                self.device = None
            if scanner is not None:
                try:
                    scanner.sensorsChanged = None
                except Exception:
                    pass
                try:
                    scanner.stop()
                except Exception as exc:
                    self._status(f'Scanner cleanup warning: {exc}')
            if not self._running:
                self.mode = 'disconnected'
                self._status('Disconnected')

    def stop(self):
        self._running = False
        self._selection.set()
        return self.wait(3000)


def _resistance(value):
    """Finite positive ohm reading, else None (never converted to zero)."""
    return value if math.isfinite(value) and value > 0 else None


def _round_ohm(value):
    return None if value is None else round(value)


def channel_count_of(row):
    """Helper for the UI: number of channels in one decoded row."""
    try:
        return len(row)
    except TypeError:
        return 0
