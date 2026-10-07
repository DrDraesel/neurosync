"""Device catalogue: which headsets this app knows, and with what contracts.

Two device families are supported by the same application:

* ``classic``   - BrainBit Classic / BrainBit Black (SensorFamily.LEBrainBit,
                  LEBrainBitBlack). Four monopolar channels O1, O2, T3, T4 at
                  250 Hz, named fields in the SDK callback, values in volts,
                  one resistance value per channel. This is the path the app
                  was originally verified with.
* ``neuroeeg``  - BrainBit DragonEEG (SensorFamily.LENeuroEEG). A 24-piece
                  channel set: the scalp EEG set plus auxiliary poly channels
                  that the SDK reports as D1/D2/D3. Channel names and order come
                  from the device itself (``NeuroEEGSensor.supported_channels``,
                  ordered by ``Num``); the tuple below is only the offline
                  fallback. The amplifier parameters (Frequency, ChannelGain,
                  ChannelMode) MUST be written after connecting and before
                  StartSignal, otherwise this device streams zeros.
* ``brainbit2`` - BrainBit 2 / Pro / Flex (LEBrainBit2 / LEBrainBitPro /
                  LEBrainBitFlex). Generic channel packets, amplifier
                  parameters in the BrainBit2 shape.

Honesty notes
-------------
* The exact channel order and the packet layout of a real Dragon are taken from
  the device at runtime; the fallback list is documentation, not a measurement.
  Anything the app cannot read back from the device is labelled as such.
* Volt-to-microvolt conversion (x1e6) is the contract of the BrainBit SDK's
  Classic callback. The same conversion is applied to the generic channel
  packets because every member of this device family reports volts in the SDK;
  this assumption is listed in METHODS.md as a hardware-verification item.
* The auxiliary channels D1/D2/D3 are NOT scalp sites: the manufacturer
  describes three poly channels (ECG/EMG/EOG), but which identifier is which
  cannot be established from the SDK enum alone, so the app keeps the SDK
  identifiers and never labels them as a body signal.
* Nothing here is a diagnosis, a clinical electrode map or a mental-state
  classifier. Positions used for the head map are schematic (see topomap.py).
"""
from dataclasses import dataclass, field

from neurosdk.cmn_types import (EEGChannelMode, EEGChannelType, EEGChannelId,
                               SensorFamily, SensorGain, SensorSamplingFrequency)

# --------------------------------------------------------------------------- #
# Channel naming
# --------------------------------------------------------------------------- #

# BrainBit Classic / Black channel contract (unchanged, verified path).
CLASSIC_CHANNELS = ('O1', 'O2', 'T3', 'T4')

# Canonical display names for the SDK's channel identifiers. The SDK enum uses
# 'EEGChIdOZ' style names; the app always displays standard 10-20 spelling.
SDK_CHANNEL_NAMES = {
    EEGChannelId.EEGChIdO1: 'O1',
    EEGChannelId.EEGChIdP3: 'P3',
    EEGChannelId.EEGChIdC3: 'C3',
    EEGChannelId.EEGChIdF3: 'F3',
    EEGChannelId.EEGChIdFp1: 'Fp1',
    EEGChannelId.EEGChIdT5: 'T5',
    EEGChannelId.EEGChIdT3: 'T3',
    EEGChannelId.EEGChIdF7: 'F7',
    EEGChannelId.EEGChIdF8: 'F8',
    EEGChannelId.EEGChIdT4: 'T4',
    EEGChannelId.EEGChIdT6: 'T6',
    EEGChannelId.EEGChIdFp2: 'Fp2',
    EEGChannelId.EEGChIdF4: 'F4',
    EEGChannelId.EEGChIdC4: 'C4',
    EEGChannelId.EEGChIdP4: 'P4',
    EEGChannelId.EEGChIdO2: 'O2',
    EEGChannelId.EEGChIdOZ: 'Oz',
    EEGChannelId.EEGChIdPZ: 'Pz',
    EEGChannelId.EEGChIdCZ: 'Cz',
    EEGChannelId.EEGChIdFZ: 'Fz',
    EEGChannelId.EEGChIdFpZ: 'Fpz',
    EEGChannelId.EEGChIdD1: 'D1',
    EEGChannelId.EEGChIdD2: 'D2',
    EEGChannelId.EEGChIdD3: 'D3',
}

# Auxiliary/differential identifiers reported alongside the scalp sites.
POLY_CHANNELS = ('D1', 'D2', 'D3')

# Documented DragonEEG scalp list supplied with the device notes, used ONLY as
# the offline fallback when the connected device reports no channel table.
#
# Count note, stated instead of smoothed over: the notes list 23 names, but two
# of them (Fp7, Fp8) cannot be reported by this SDK at all - EEGChannelId has no
# such identifiers - and the remaining 21 are exactly the standard 10-20 set
# (Fp1/Fp2, F7/F3/Fz/F4/F8, T3/C3/Cz/C4/T4, T5/P3/Pz/P4/T6, O1/Oz/O2) that the
# SDK does define. The fallback therefore carries those 21 sites in the
# documented order. If a future device reports an unexpected name, the app takes
# the device's own text for it (see channel_names) and only the head map drops
# names it has no schematic position for.
DRAGON_DOCUMENTED_EEG_CHANNELS = (
    'Fp1', 'F7', 'F3', 'Fz', 'Fpz',
    'Fp2', 'F8', 'F4',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'Oz', 'O2',
)

# Sampling rates this app is willing to analyse (analysis needs fs >= 200 Hz and
# an integral 5 s sample count; see eeg_analysis._parameters).
SUPPORTED_FS = (250, 500, 1000)
DEFAULT_FS = 250
FS_TO_FREQUENCY = {
    value: getattr(SensorSamplingFrequency, f'FrequencyHz{value}')
    for value in SUPPORTED_FS
}
FREQUENCY_NAMES = {FS_TO_FREQUENCY[value].name: value for value in SUPPORTED_FS}


def channel_name(channel_id):
    """Display name for an EEGChannelId, or None for unknown identifiers."""
    try:
        return SDK_CHANNEL_NAMES.get(EEGChannelId(channel_id))
    except (ValueError, KeyError, TypeError):
        return None


def channel_names(supported):
    """Ordered display names from ``NeuroEEGSensor.supported_channels``.

    ``supported`` is the SDK list of EEGChannelInfo (Id, ChType, Name, Num).
    Entries are ordered by Num (the device's own ordering); unknown identifiers
    are skipped rather than guessed. An empty or unreadable table falls back to
    the documented Dragon list - never to a partially guessed mix.
    """
    if not supported:
        return tuple(DRAGON_DOCUMENTED_EEG_CHANNELS)
    try:
        ordered = sorted(supported, key=lambda item: int(getattr(item, 'Num', 0)))
    except (TypeError, ValueError, AttributeError):
        return tuple(DRAGON_DOCUMENTED_EEG_CHANNELS)
    names = []
    for item in ordered:
        name = channel_name(getattr(item, 'Id', None))
        if name is None:
            # Prefer the device's own text when the identifier is unrecognised.
            text = str(getattr(item, 'Name', '') or '').strip()
            name = text or None
        if name and name not in names:
            names.append(name)
    return tuple(names) if names else tuple(DRAGON_DOCUMENTED_EEG_CHANNELS)


def scalp_channels(channels):
    """Subset of ``channels`` that has a schematic head position (EEG sites)."""
    from topomap import ELECTRODE_POSITIONS
    return tuple(name for name in channels if name in ELECTRODE_POSITIONS)


def poly_channels(channels):
    """Subset of ``channels`` without a scalp position (auxiliary channels)."""
    return tuple(name for name in channels if name not in scalp_channels(channels))


# --------------------------------------------------------------------------- #
# Device specifications
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DeviceSpec:
    key: str                       # stable identifier stored in every session
    label: str                     # human label for the picker
    families: tuple                # SensorFamily members served by this spec
    transport: str                 # 'classic' | 'neuroeeg' | 'brainbit2'
    channels: tuple                # default channel names when the device is quiet
    fs: int                        # analysis sampling rate the app requests
    resistance: bool               # per-channel resistance available
    needs_amplifier_setup: bool    # MUST write amplifier params before streaming
    notes: str
    extras: tuple = field(default=())

    def matches(self, family):
        return family in self.families

    @property
    def channel_count(self):
        return len(self.channels)


CLASSIC = DeviceSpec(
    key='brainbit_classic',
    label='BrainBit Classic / Black (4 channels, 250 Hz)',
    families=(SensorFamily.LEBrainBit, SensorFamily.LEBrainBitBlack),
    transport='classic',
    channels=CLASSIC_CHANNELS,
    fs=250,
    resistance=True,
    needs_amplifier_setup=False,
    notes='Verified app path: O1/O2/T3/T4 monopolar channels, volts x1e6 = uV, '
          'resistance in ohms per channel, EEG and resistance never at once.',
)

DRAGON = DeviceSpec(
    key='dragon_eeg',
    label='DragonEEG / NeuroEEG (EEG + auxiliary channels, 250/500/1000 Hz)',
    families=(SensorFamily.LENeuroEEG,),
    transport='neuroeeg',
    channels=DRAGON_DOCUMENTED_EEG_CHANNELS,
    fs=DEFAULT_FS,
    resistance=True,
    needs_amplifier_setup=True,
    notes='Amplifier parameters (Frequency, ChannelGain, ChannelMode) are written '
          'after connect and verified by read-back before StartSignal; first '
          'pairing requires the device to be in pairing mode. The channel set and '
          'its order come from the device; the documented set is 21 scalp sites '
          '(10-20) plus auxiliary D1/D2/D3, and resistance is reported per channel '
          'in ohms together with A1/A2/Bias reference values.',
)

BRAINBIT2 = DeviceSpec(
    key='brainbit2',
    label='BrainBit 2 / Pro / Flex (channel set, 250 Hz)',
    families=(SensorFamily.LEBrainBit2, SensorFamily.LEBrainBitPro,
              SensorFamily.LEBrainBitFlex),
    transport='brainbit2',
    channels=CLASSIC_CHANNELS,
    fs=250,
    resistance=True,
    needs_amplifier_setup=True,
    notes='Configured through the BrainBit2 amplifier parameter shape '
          '(ChGain/ChSignalMode/ChResistUse). Implemented from the SDK sample; '
          'NOT verified on physical hardware by this project.',
)

SPECS = (CLASSIC, DRAGON, BRAINBIT2)

# One scan covers every supported family, so a single device picker can find
# both the Classic headset and the Dragon.
SCAN_FAMILIES = tuple(family for spec in SPECS for family in spec.families)


def spec_for_family(family, default=None):
    """DeviceSpec for a SensorFamily, else ``default`` (None when unknown)."""
    for spec in SPECS:
        if spec.matches(family):
            return spec
    return default


def spec_for_key(key, default=None):
    """DeviceSpec by stable key, else ``default``."""
    for spec in SPECS:
        if spec.key == key:
            return spec
    return default


def family_name(family):
    """SensorFamily name for display, else None."""
    return getattr(family, 'name', None) or (str(family) if family is not None else None)


def describe_sensor_info(info, default_spec=None):
    """JSON-safe description of one scanner SensorInfo for the picker.

    The device's own SensorFamily decides the specification. ``default_spec`` is
    used only when the entry carries no family at all (a scanner fixture or a
    legacy record); a KNOWN family that this app has no specification for is
    listed as unsupported rather than quietly driven as a BrainBit.
    """
    family = getattr(info, 'SensFamily', None)
    spec = spec_for_family(family) if family is not None else default_spec
    address = str(getattr(info, 'Address', '') or '').upper()
    return {
        'address': address,
        'name': str(getattr(info, 'Name', '') or ''),
        'serial': str(getattr(info, 'SerialNumber', '') or ''),
        'rssi': getattr(info, 'RSSI', None),
        'pairing_required': bool(getattr(info, 'PairingRequired', False)),
        'family': family_name(family),
        'device_key': spec.key if spec else 'unknown',
        'device_label': spec.label if spec else 'Unrecognised BLE device',
        'channels': list(spec.channels) if spec else [],
        'channels_source': 'documented fallback (read from the device after connect)',
        'sampling_rate_hz': spec.fs if spec else None,
        'supports_resistance': bool(spec.resistance) if spec else False,
        'notes': spec.notes if spec else 'Family not supported: not connected.',
        'supported': spec is not None,
    }


# --------------------------------------------------------------------------- #
# Amplifier configuration (Dragon / BrainBit2)
# --------------------------------------------------------------------------- #

def neuroeeg_amplifier_param(template, channel_count, fs=DEFAULT_FS,
                             gain=SensorGain.Gain6,
                             mode=EEGChannelMode.EEGChModeSignalResist):
    """Fill a NeuroEEGAmplifierParam for ``channel_count`` channels.

    ``template`` is the parameter block read back from the device: the app keeps
    the device's reference mode and respiration flag rather than inventing them,
    and sets the sampling frequency, per-channel gain, per-channel mode and the
    reference-resistance permission. ``ReferentResistMesureAllow`` is forced True
    because that is what the shipped SDK sample does before streaming, and the
    resistance path needs it. Returns a new parameter object; the template is
    never mutated.
    """
    if not isinstance(channel_count, int) or channel_count <= 0:
        raise ValueError('channel_count must be a positive integer')
    if fs not in FS_TO_FREQUENCY:
        raise ValueError(f'fs must be one of {SUPPORTED_FS} Hz')
    if not isinstance(gain, SensorGain):
        raise ValueError('gain must be a SensorGain member')
    if not isinstance(mode, EEGChannelMode):
        raise ValueError('mode must be an EEGChannelMode member')
    from neurosdk.cmn_types import NeuroEEGAmplifierParam
    template = template if template is not None else NeuroEEGAmplifierParam(
        ReferentResistMesureAllow=True,
        Frequency=FS_TO_FREQUENCY[DEFAULT_FS],
        ReferentMode=None,
        ChannelMode=[],
        ChannelGain=[],
        RespirationOn=False,
    )
    return NeuroEEGAmplifierParam(
        ReferentResistMesureAllow=True,
        Frequency=FS_TO_FREQUENCY[fs],
        ReferentMode=getattr(template, 'ReferentMode', None),
        ChannelMode=[mode] * channel_count,
        ChannelGain=[gain] * channel_count,
        RespirationOn=bool(getattr(template, 'RespirationOn', False)),
    )


def neuroeeg_amplifier_problems(param, channel_count, fs=DEFAULT_FS):
    """Read-back check of a written amplifier block; returns problem strings.

    An empty list means the device echoes the requested frequency, has the
    requested number of channels configured and no channel left switched off.
    A non-empty list must abort acquisition instead of streaming zeros.
    """
    problems = []
    if param is None:
        return ['amplifier parameters unreadable after write']
    frequency = getattr(param, 'Frequency', None)
    if getattr(frequency, 'name', None) != FS_TO_FREQUENCY[fs].name:
        problems.append(f'frequency read-back {family_name(frequency)!r} != {fs} Hz')
    modes = list(getattr(param, 'ChannelMode', []) or [])
    gains = list(getattr(param, 'ChannelGain', []) or [])
    if len(modes) < channel_count or len(gains) < channel_count:
        problems.append(f'channel block shorter than {channel_count} channels')
    off = [index for index, mode in enumerate(modes[:channel_count])
           if getattr(mode, 'name', None) in ('EEGChModeOff', None)]
    if off:
        problems.append(f'channels switched off after configuration: {off}')
    return problems


def brainbit2_amplifier_param(template, channel_count, gain=SensorGain.Gain6):
    """Fill a BrainBit2AmplifierParam (ChGain / ChSignalMode / ChResistUse)."""
    if not isinstance(channel_count, int) or channel_count <= 0:
        raise ValueError('channel_count must be a positive integer')
    from neurosdk.cmn_types import (BrainBit2AmplifierParam, BrainBit2ChannelMode,
                                    GenCurrent)
    return BrainBit2AmplifierParam(
        ChSignalMode=[BrainBit2ChannelMode.ChModeNormal] * channel_count,
        ChResistUse=[True] * channel_count,
        ChGain=[gain] * channel_count,
        Current=getattr(template, 'Current', GenCurrent.GenCurr6nA),
    )


def channel_type_name(channel_type):
    """Readable name of an EEGChannelType member (for logging, not for claims)."""
    try:
        return EEGChannelType(channel_type).name
    except (ValueError, TypeError):
        return None
