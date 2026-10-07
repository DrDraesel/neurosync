"""AI doctor: descriptive, non-diagnostic EEG analysis assistance.

Sends ONLY derived numbers (band powers, relative shares, quality flags,
transport counters, device metadata) — never raw EEG samples or personal
identifiers. By default the request goes to the LOCAL Ollama server on
127.0.0.1:11434 and nothing leaves the PC. Optionally, setting AI_API_URL
(+ AI_API_KEY, AI_API_MODEL) uses any OpenAI-compatible API instead — a cloud
provider or a local server — in which case the same derived numbers are sent
to that provider. The reply is an assistive description of the measured
numbers: it is not a medical interpretation, not a diagnosis and must not be
used for treatment decisions. If the backend is unreachable the caller
receives an explicit error dict, never invented analysis.

Ollama chat API: https://github.com/ollama/ollama/blob/main/docs/api.md
OpenAI-compatible chat API: POST {AI_API_URL}/chat/completions
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

# Optional per-computer AI settings file (gitignored — never committed):
# KEY=VALUE lines in ai_settings.env next to this file.  Real environment
# variables always win over values from the file.
def _load_ai_settings(path):
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


_load_ai_settings(Path(__file__).resolve().parent / 'ai_settings.env')

DEFAULT_ENDPOINT = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434')
DEFAULT_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen3.8:latest')
# Optional: any OpenAI-compatible API (cloud or local server) instead of the
# local Ollama model.  When AI_API_URL is set, the AI doctor and the live chat
# use that endpoint with AI_API_KEY (optional) and AI_API_MODEL.
# Examples: https://api.openai.com/v1 · https://openrouter.ai/api/v1 ·
# https://api.anthropic.com/v1 (OpenAI-compat) · http://127.0.0.1:1234/v1 (LM Studio)
AI_API_URL = os.environ.get('AI_API_URL', '').strip().rstrip('/')
AI_API_KEY = os.environ.get('AI_API_KEY', '').strip()
AI_API_MODEL = os.environ.get('AI_API_MODEL', '').strip()
DEFAULT_TIMEOUT_S = 300.
DISCLAIMER = 'Assistive description only - not a medical interpretation.'


def api_mode():
    """True when an OpenAI-compatible API endpoint is configured."""
    return bool(AI_API_URL)


def effective_model():
    """Model name for the active backend (shown in the UI, sent to it)."""
    return AI_API_MODEL if api_mode() else DEFAULT_MODEL


def endpoint_label():
    """Endpoint label for the active backend (shown in the UI)."""
    return AI_API_URL if api_mode() else DEFAULT_ENDPOINT


def _api_headers():
    headers = {'Content-Type': 'application/json'}
    if AI_API_KEY:
        headers['Authorization'] = 'Bearer ' + AI_API_KEY
    return headers


def _http_detail(exc):
    """Short, safe snippet of an HTTP error body for the user-facing message."""
    try:
        raw = exc.read().decode('utf-8', 'replace').strip()
    except Exception:
        return ''
    return (' — ' + raw[:200]) if raw else ''


def _api_ask(messages, timeout, opener):
    """One non-streaming chat completion on the OpenAI-compatible endpoint."""
    body = json.dumps({'model': AI_API_MODEL, 'messages': messages,
                       'stream': False, 'temperature': 0.2},
                      allow_nan=False).encode('utf-8')
    request = urllib.request.Request(AI_API_URL + '/chat/completions', data=body,
                                     headers=_api_headers(), method='POST')
    try:
        with opener(request, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        return {'ok': False, 'model': AI_API_MODEL,
                'error': f'API HTTP {exc.code}{_http_detail(exc)} — '
                         'check AI_API_URL / AI_API_KEY / AI_API_MODEL'}
    except OSError as exc:
        return {'ok': False, 'model': AI_API_MODEL,
                'error': f'API unreachable: {exc}'}
    except (ValueError, UnicodeDecodeError) as exc:
        return {'ok': False, 'model': AI_API_MODEL,
                'error': f'Unreadable reply from the API: {exc}'}
    try:
        content = str(data['choices'][0]['message'].get('content') or '').strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        content = ''
    if not content:
        return {'ok': False, 'model': AI_API_MODEL,
                'error': 'API returned an empty message'}
    return {'ok': True, 'model': AI_API_MODEL, 'text': content[:4000]}


def _api_stream(messages, timeout, opener):
    """Stream one chat completion from the OpenAI-compatible endpoint."""
    body = json.dumps({'model': AI_API_MODEL, 'messages': messages,
                       'stream': True, 'temperature': 0.2},
                      allow_nan=False).encode('utf-8')
    request = urllib.request.Request(AI_API_URL + '/chat/completions', data=body,
                                     headers=_api_headers(), method='POST')
    try:
        response = opener(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        yield {'type': 'error',
               'error': f'API HTTP {exc.code}{_http_detail(exc)} — '
                        'check AI_API_URL / AI_API_KEY / AI_API_MODEL'}
        return
    except OSError as exc:
        yield {'type': 'error', 'error': f'API unreachable: {exc}'}
        return
    got_content = False
    try:
        with response:
            for raw in response:
                line = (raw.decode('utf-8', 'replace')
                        if isinstance(raw, (bytes, bytearray)) else str(raw)).strip()
                if not line or not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    break
                try:
                    event = json.loads(payload)
                except ValueError:
                    continue                       # ignore malformed stream lines
                try:
                    delta = event['choices'][0].get('delta') or {}
                    content = delta.get('content')
                except (KeyError, IndexError, TypeError, AttributeError):
                    content = None
                if content:
                    got_content = True
                    yield {'type': 'chunk', 'text': str(content)}
    except (OSError, ValueError) as exc:
        yield {'type': 'error', 'error': f'API stream failed: {exc}'}
        return
    if got_content:
        yield {'type': 'done'}
    else:
        yield {'type': 'error',
               'error': 'API returned no answer text (check AI_API_MODEL).'}

# Shared hard rules for every local-model prompt (single source; the honesty
# contract of the app is encoded here and must not be weakened).
HARD_RULES = (
    'Hard rules:\n'
    '- Never diagnose; never suggest or exclude diseases or disorders; never recommend '
    'treatment, medication or supplements.\n'
    '- Never infer emotions, attention, drowsiness, sleep stages, personality or any '
    'other mental state from these numbers.\n'
    '- Use only the measurements and quality flags provided in the JSON. Never '
    'invent, extrapolate or round-trip values that are missing or null.\n'
    '- If data is invalid, stale, discontinuous or flagged, say plainly that no '
    'reliable description is possible and name which checks failed.\n'
    '- If the JSON says simulated: true, the numbers came from the built-in '
    'simulator (no headset is connected): state that first and describe it as a '
    'pipeline demonstration only.\n'
    '- Contact resistance context: this app investigates contact above 1 MOhm '
    '(1,000,000 ohms); do not describe lower resistances as "extremely high" or '
    '"poor", but never claim resistance alone proves signal quality.\n'
)

SYSTEM_PROMPT = (
    'You are an EEG analysis assistant inside a local research viewer for BrainBit '
    'headsets (4 channels O1 O2 T3 T4, one shared reference — Classic/Black/2/Pro/Flex) '
    'and the DragonEEG / NeuroEEG headset (up to 21 scalp electrodes plus auxiliary '
    'channels). Nominal sampling 250 Hz. You are not a physician and this is not '
    'medical care.\n'
    + HARD_RULES +
    'Write plain text, no markdown headings, under 200 words, in this order:\n'
    '1) Data status: valid or not, which device family and channel count, '
    'seconds/epochs used, contact state, simulated or real.\n'
    '2) What the measured band powers and relative shares show, citing only the '
    'provided numbers, including which band is predominant and its peak frequency.\n'
    '3) Signal-quality observations and concrete standard improvements suggested '
    'by the flags (electrode contact and skin preparation, reduce jaw and facial '
    'muscle tension, less blinking and movement, recheck headband placement, '
    'address mains/line-noise sources).\n'
    '4) Educational background: one or two sentences on what this kind of band '
    'pattern conventionally describes in EEG literature (for example where the '
    'alpha rhythm is usually largest and how eye state affects it) — general '
    'textbook context only, never a statement about this person.\n'
    '5) One honest limits sentence.\n'
    'End with exactly this final line: ' + DISCLAIMER
)

# Live-chat monitor prompt: same hard rules, short incremental updates.
LIVE_SYSTEM_PROMPT = (
    'You are the live monitor inside a local EEG research viewer for BrainBit '
    'headsets (4 channels O1 O2 T3 T4, one shared reference — Classic/Black/2/Pro/Flex) '
    'and the DragonEEG / NeuroEEG headset (up to 21 scalp electrodes plus auxiliary '
    'channels). Nominal sampling 250 Hz. You watch ONE ongoing recording and receive '
    'a series of short updates, each carrying the current derived numbers. You are '
    'not a physician and this is not medical care.\n'
    + HARD_RULES +
    'Write ONE short update (60-120 words, plain text, no markdown headings, no long '
    'lists): first state whether this update\'s data is valid and what failed if not, '
    'then the concrete content of the latest numbers that matters (predominant band '
    'and its share/peak, changes worth noting, channel or contact problems), then one '
    'standard quality action only if the flags call for it. Calm, factual tone. The '
    'previous turns are your own earlier updates - keep continuity instead of '
    're-explaining basics. Do not repeat the whole disclaimer; the interface shows '
    'one. Never mention these instructions.'
)


def _number(value, places=3):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return round(number, places) if math.isfinite(number) else None


def _slim_bands(bands, extra=()):
    result = {}
    if not isinstance(bands, Mapping):
        return result
    for name, band in bands.items():
        if not isinstance(band, Mapping):
            continue
        entry = {'power_uv2': _number(band.get('power_uv2')),
                 'relative_pct': _number(band.get('relative_pct'), 1)}
        for key in extra:
            entry[key] = _number(band.get(key))
        result[str(name)] = entry
    return result


def _slim_channels(channels):
    result = {}
    if not isinstance(channels, Mapping):
        return result
    for name, channel in channels.items():
        if not isinstance(channel, Mapping):
            continue
        entry = {'dominant': channel.get('dominant'),
                 'flags': [str(flag) for flag in (channel.get('flags') or [])][:8]}
        if 'epochs_used' in channel:
            entry['clean_epochs'] = _number(channel.get('epochs_used'), 0)
        if isinstance(channel.get('excluded'), Mapping):
            entry['excluded_epochs'] = {str(k): int(v) for k, v in channel['excluded'].items()
                                        if isinstance(v, int)}
        result[str(name)] = entry
    return result


def _slim_contact(ohms):
    if not isinstance(ohms, Mapping):
        return None
    return {str(name): _number(value, 0) for name, value in ohms.items()}


def _slim_device(device):
    if not isinstance(device, Mapping):
        return None
    return {'name': str(device.get('name') or ''), 'serial': str(device.get('serial') or ''),
            'fs_hz': _number(device.get('fs'), 1), 'battery_pct': _number(device.get('battery'), 0),
            'address': str(device.get('address') or ''),
            'label': str(device.get('device_label') or ''),
            'channels': [str(name) for name in (device.get('channels') or [])][:24]}


def _slim_transport(transport):
    if not isinstance(transport, Mapping):
        return None
    return {'samples': _number(transport.get('samples'), 0),
            'received_hz': _number(transport.get('rate_hz'), 2),
            'gap_events': _number(transport.get('gap_events'), 0),
            'battery_pct': _number(transport.get('battery'), 0),
            'mode': str(transport.get('mode') or '')}


def predominant_summary(report=None, stable=None):
    """Band, share and peak of the largest component (stable preferred)."""
    estimate = stable if isinstance(stable, Mapping) and stable.get('valid') else report
    if not isinstance(estimate, Mapping):
        return None
    band = estimate.get('dominant')
    if not band:
        return None
    return {'band': str(band), 'relative_pct': _number(estimate.get('dominant_pct'), 1),
            'peak_hz': _number(estimate.get('peak_hz'), 2),
            'source': 'stable multi-epoch estimate' if estimate is stable else '5 s window'}


def build_payload(report=None, stable=None, transport=None, device=None, contact=None):
    """Bounded, JSON-safe payload of derived numbers only - never raw samples."""
    payload = {'device': _slim_device(device), 'transport': _slim_transport(transport),
               'contact_ohms': _slim_contact(contact),
               'simulated': bool(isinstance(device, Mapping) and device.get('simulated')),
               'predominant': predominant_summary(report, stable)}
    if isinstance(report, Mapping):
        payload['five_second'] = {
            'valid': bool(report.get('valid')), 'ready': bool(report.get('ready')),
            'reason': str(report.get('reason') or '')[:400],
            'dominant': report.get('dominant'),
            'dominant_pct': _number(report.get('dominant_pct'), 1),
            'peak_hz': _number(report.get('peak_hz'), 2),
            'samples': _number(report.get('samples'), 0),
            'bands': _slim_bands(report.get('bands')),
            'channels': _slim_channels(report.get('channels')),
        }
    else:
        payload['five_second'] = None
    if isinstance(stable, Mapping):
        payload['stable_estimate'] = {
            'valid': bool(stable.get('valid')),
            'reason': str(stable.get('reason') or '')[:400],
            'dominant': stable.get('dominant'),
            'dominant_pct': _number(stable.get('dominant_pct'), 1),
            'seconds_used': _number(stable.get('seconds_used'), 1),
            'seconds_total': _number(stable.get('seconds_total'), 1),
            'epochs_used': _number(stable.get('epochs_used'), 0),
            'epochs_total': _number(stable.get('epochs_total'), 0),
            'bands': _slim_bands(stable.get('bands'), extra=('sem_uv2',)),
            'channels': _slim_channels(stable.get('channels')),
        }
    else:
        payload['stable_estimate'] = None
    return payload


def build_messages(payload):
    user = ('Measurements JSON (only these values are real; null means not measured):\n'
            + json.dumps(payload, separators=(',', ':'), allow_nan=False)
            + '\n\nDescribe and assist as instructed.')
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user}]


def ask(payload, model=DEFAULT_MODEL, endpoint=DEFAULT_ENDPOINT,
        timeout=DEFAULT_TIMEOUT_S, opener=None):
    """Query the configured model backend once. Never raises."""
    opener = opener or urllib.request.urlopen
    if api_mode():
        return _api_ask(build_messages(payload), timeout, opener)
    body = json.dumps({'model': model, 'messages': build_messages(payload),
                       'stream': False,
                       # Thinking models would otherwise burn the whole token
                       # budget on hidden reasoning and return empty content.
                       'think': False,
                       'options': {'temperature': 0.2, 'num_predict': 800}},
                      allow_nan=False).encode('utf-8')
    request = urllib.request.Request(endpoint.rstrip('/') + '/api/chat', data=body,
                                     headers={'Content-Type': 'application/json'},
                                     method='POST')
    try:
        with opener(request, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        return {'ok': False, 'model': model,
                'error': f'Ollama HTTP {exc.code}; is the model "{model}" installed?'}
    except OSError as exc:
        return {'ok': False, 'model': model, 'error': f'Local Ollama unreachable: {exc}'}
    except (ValueError, UnicodeDecodeError) as exc:
        return {'ok': False, 'model': model, 'error': f'Unreadable reply from Ollama: {exc}'}
    message = data.get('message') if isinstance(data, dict) else None
    content = str(message.get('content') or '').strip() if isinstance(message, Mapping) else ''
    if not content:
        thinking = str(message.get('thinking') or '').strip() if isinstance(message, Mapping) else ''
        if thinking:
            return {'ok': False, 'model': model,
                    'error': 'Ollama returned reasoning without an answer; retry or raise num_predict'}
        return {'ok': False, 'model': model, 'error': 'Ollama returned an empty message'}
    return {'ok': True, 'model': model, 'text': content[:4000]}


def available_models(endpoint=DEFAULT_ENDPOINT, timeout=5., opener=None):
    """Names of locally installed Ollama models; [] when unreachable.

    With an API backend configured, reports the one configured model name.
    """
    if api_mode():
        return [AI_API_MODEL] if AI_API_MODEL else []
    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(endpoint.rstrip('/') + '/api/tags', method='GET')
    try:
        with opener(request, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
    except (OSError, ValueError, UnicodeDecodeError):
        return []
    models = data.get('models') if isinstance(data, dict) else None
    return [str(entry.get('name')) for entry in models or [] if isinstance(entry, Mapping)]


# ---------------------------------------------------------------------------
# Live chat monitor — streaming, bounded, local-model-only.
# The operator sees short updates appear while a recording runs; only the same
# derived numbers as build_payload() are ever sent, and only to 127.0.0.1.
# ---------------------------------------------------------------------------

def _clean_history(history, limit=6, cap=700):
    """Previous live-chat turns echoed back for continuity (bounded, role-checked)."""
    cleaned = []
    for item in list(history or [])[-limit:]:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get('role') or '')
        content = str(item.get('content') or '').strip()
        if role not in ('user', 'assistant') or not content:
            continue
        cleaned.append({'role': role, 'content': content[:cap]})
    return cleaned


def build_live_messages(payload, history=None, question=None, update_number=1):
    """Chat messages for one live-monitor update (or an operator question)."""
    messages = [{'role': 'system', 'content': LIVE_SYSTEM_PROMPT}]
    messages.extend(_clean_history(history))
    measurements = json.dumps(payload, separators=(',', ':'), allow_nan=False)
    if question:
        user = ('Live update #%d — the operator asks about the ongoing recording. '
                'Measurements JSON (only these values are real; null means not measured):\n%s'
                '\n\nOperator question: %s\nAnswer from the measured numbers only, '
                'briefly (under 120 words).'
                % (int(update_number), measurements, str(question).strip()[:400]))
    else:
        user = ('Live update #%d — the recording is ongoing. Measurements JSON (only '
                'these values are real; null means not measured):\n%s'
                '\n\nWrite the short update now.' % (int(update_number), measurements))
    messages.append({'role': 'user', 'content': user})
    return messages


def stream_ask(messages, model=DEFAULT_MODEL, endpoint=DEFAULT_ENDPOINT,
               timeout=DEFAULT_TIMEOUT_S, opener=None, keep_alive='30m',
               num_predict=320):
    """Stream one chat completion; yields event dicts and never raises.

    Events: {'type': 'chunk', 'text': str} per content fragment, then exactly
    one terminal {'type': 'done'} or {'type': 'error', 'error': str}.
    """
    opener = opener or urllib.request.urlopen
    if api_mode():
        yield from _api_stream(messages, timeout, opener)
        return
    body = json.dumps({'model': model, 'messages': messages, 'stream': True,
                       # A thinking model would burn the latency budget on hidden
                       # reasoning before any visible update arrives.
                       'think': False, 'keep_alive': keep_alive,
                       'options': {'temperature': 0.2, 'num_predict': num_predict}},
                      allow_nan=False).encode('utf-8')
    request = urllib.request.Request(endpoint.rstrip('/') + '/api/chat', data=body,
                                     headers={'Content-Type': 'application/json'},
                                     method='POST')
    try:
        response = opener(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        yield {'type': 'error',
               'error': f'Ollama HTTP {exc.code}; is the model "{model}" installed?'}
        return
    except OSError as exc:
        yield {'type': 'error', 'error': f'Local Ollama unreachable: {exc}'}
        return
    got_content = False
    try:
        with response:
            for raw in response:
                line = (raw.decode('utf-8', 'replace')
                        if isinstance(raw, (bytes, bytearray)) else str(raw)).strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue                       # ignore malformed stream lines
                if not isinstance(event, dict):
                    continue
                message = event.get('message')
                content = message.get('content') if isinstance(message, Mapping) else None
                if content:
                    got_content = True
                    yield {'type': 'chunk', 'text': str(content)}
                if event.get('done'):
                    break
    except (OSError, ValueError) as exc:
        yield {'type': 'error', 'error': f'Local Ollama stream failed: {exc}'}
        return
    if got_content:
        yield {'type': 'done'}
    else:
        yield {'type': 'error',
               'error': 'Ollama returned no answer text (empty or reasoning-only reply).'}


def warm(model=DEFAULT_MODEL, endpoint=DEFAULT_ENDPOINT, timeout=300., opener=None,
         keep_alive='30m'):
    """Best-effort model preload so the first live update is not delayed by the load."""
    if api_mode():
        return False                      # nothing to preload for an API backend
    opener = opener or urllib.request.urlopen
    body = json.dumps({'model': model, 'prompt': '', 'stream': False,
                       'keep_alive': keep_alive,
                       'options': {'num_predict': 1}}).encode('utf-8')
    request = urllib.request.Request(endpoint.rstrip('/') + '/api/generate', data=body,
                                     headers={'Content-Type': 'application/json'},
                                     method='POST')
    try:
        with opener(request, timeout=timeout) as response:
            response.read()
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Offline interpretation helper — "how to read these numbers".
#
# Plain-language textbook context for the measured bands, generated locally
# (no model required), so the numbers can be understood without reading code
# or methods docs.  It describes conventions from EEG literature and compares
# measured values WITHIN this recording only.  It never states anything about
# the person: no disorder, no state, no trait.
# ---------------------------------------------------------------------------
BAND_CONTEXT = {
    'Delta': ('1–4 Hz slow waves. Conventionally the largest component during deep sleep; '
              'in a short awake recording a large delta share is more often a sign of slow '
              'artifact (movement, sweat, drift) than of brain activity — check quality first.'),
    'Theta': ('4–8 Hz. Conventionally prominent in drowsiness and in children, and linked in '
              'research to memory-related tasks. Jaw clenching, tongue position and eye '
              'movements also produce theta-range activity, so read it together with the '
              'artifact flags for the same window.'),
    'Alpha': ('8–13 Hz. The classic posterior resting rhythm: usually largest over O1/O2 and '
              'neighbouring posterior sites, and clearly reduced when the eyes open. Alpha '
              'shares are only comparable between recordings made in the same eye-state '
              '(eyes closed vs open).'),
    'Beta': ('13–30 Hz. Low-amplitude fast activity, often relatively larger over frontal '
             'sites. Muscle activity (jaw, neck, forehead) is a common contaminant in the '
             'beta range; in short recordings beta changes often track EMG rather than EEG.'),
    'Gamma': ('30–45 Hz. At scalp electrodes with a 250 Hz sampling rate this band is '
              'dominated by muscle (EMG) and line-noise leakage far more than by cortical '
              'rhythms; treat any share here as quality-dependent.'),
}
SYMMETRIC_PAIRS = (('Fp1', 'Fp2'), ('F1', 'F2'), ('F3', 'F4'), ('F7', 'F8'), ('C1', 'C2'),
                   ('C3', 'C4'), ('T3', 'T4'), ('T7', 'T8'), ('T5', 'T6'), ('P3', 'P4'),
                   ('P7', 'P8'), ('O1', 'O2'))


def local_interpretation(report=None, stable=None, device=None):
    """Locally generated plain-language reading guide for the current numbers.

    Returns {'sections': [{'title', 'text'}, ...], 'predominant': {...}|None,
    'disclaimer': str}.  Descriptive engineering context only — not a
    diagnosis and never a statement about the person.
    """
    sections = []
    estimate = stable if isinstance(stable, Mapping) and stable.get('valid') else report
    estimate_name = 'stable multi-epoch estimate' if estimate is stable else 'latest 5 s window'
    device = device if isinstance(device, Mapping) else {}
    channels = [str(name) for name in (device.get('channels') or [])]
    sections.append({
        'title': 'What was measured',
        'text': (f"Device: {device.get('device_label') or 'unknown'}"
                 + (f" ({len(channels)} channels: {', '.join(channels[:24])})" if channels else '')
                 + ('. SIMULATED data — pipeline demonstration, not a measurement.'
                    if device.get('simulated') else '.')
                 + f" Numbers below come from the {estimate_name}. The live window does not "
                   'record the eye state, so the same recording can contain eyes-open and '
                   'eyes-closed stretches; only your stored baseline carries a condition label.'),
    })
    if isinstance(estimate, Mapping) and estimate.get('dominant'):
        band = str(estimate.get('dominant'))
        share = _number(estimate.get('dominant_pct'), 1)
        peak = _number(estimate.get('peak_hz'), 2)
        sections.append({
            'title': 'Predominant rhythm',
            'text': (f"{band} holds the largest share"
                     + (f" ({share:.1f}% of 1–45 Hz)" if share is not None else '')
                     + (f" with peak frequency {peak:.1f} Hz" if peak is not None else '')
                     + f" in the {estimate_name}. " + BAND_CONTEXT.get(band, '')
                     + ' The share describes this recording only, not a lasting characteristic.'),
        })
        band_lines = []
        for name in ('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'):
            entry = (estimate.get('bands') or {}).get(name) or {}
            share = _number(entry.get('relative_pct'), 1)
            if share is None:
                continue
            band_lines.append(f"— {name}: {share:.1f}% of 1–45 Hz. {BAND_CONTEXT.get(name, '')}")
        if band_lines:
            sections.append({'title': 'Band-by-band context', 'text': '\n'.join(band_lines)})
    if isinstance(estimate, Mapping) and estimate.get('channels'):
        pair_lines = []
        for left, right in SYMMETRIC_PAIRS:
            left_entry = (estimate.get('channels') or {}).get(left) or {}
            right_entry = (estimate.get('channels') or {}).get(right) or {}
            left_share, right_share = left_entry.get('relative_pct') or {}, right_entry.get('relative_pct') or {}
            if not left_share or not right_share:
                continue
            left_band = left_entry.get('dominant'); right_band = right_entry.get('dominant')
            try:
                left_value = float(left_share.get(left_band)); right_value = float(right_share.get(right_band))
            except (TypeError, ValueError):
                continue
            difference = left_value - right_value
            pair_lines.append(f"— {left} vs {right}: {difference:+.1f} percentage points "
                              f"({left_band} {left_value:.1f}% vs {right_band} {right_value:.1f}%)")
        if pair_lines:
            sections.append({
                'title': 'Left / right electrode comparison',
                'text': ('Descriptive within-recording comparison of the dominant-band shares '
                         'between symmetric positions (larger differences can also come from '
                         'electrode placement, contact and hair, not only from the signal):\n'
                         + '\n'.join(pair_lines)),
            })
    reason = (report or {}).get('reason') if isinstance(report, Mapping) else None
    flagged = 0
    if isinstance(estimate, Mapping):
        for entry in (estimate.get('channels') or {}).values():
            if isinstance(entry, Mapping) and entry.get('valid') is False:
                flagged += 1
    sections.append({
        'title': 'Signal quality for this window',
        'text': ((str(reason)[:300] + ' ') if reason else '')
                + (f"{flagged} channel(s) were flagged as unreliable in this window. " if flagged else '')
                + 'Contact, movement, jaw tension and eye blinks dominate short recordings; '
                  'clean contact (this app investigates above 1 MOhm) and stillness are what '
                  'make the band numbers reproducible.',
    })
    sections.append({
        'title': 'What cannot be concluded from this',
        'text': ('A short single-condition spectral summary cannot support any clinical or '
                 'psychiatric conclusion: no disorder, no treatment indication, no mental '
                 'state. Clinical EEG interpretation uses engineered montages, provocation '
                 '(eyes open/closed, hyperventilation, photic stimulation), artifact review '
                 'by a trained reader, and comparison against normative databases — the '
                 '"average brain" reference (for example qEEG z-scores, where a value at '
                 '|z| ≥ about 2 lies outside the 95% band). This app\'s only references are '
                 'the subject\'s own stored baseline and the numbers shown. For anything '
                 'that matters medically, record a proper study and take it to a qualified '
                 'clinician.'),
    })
    return {'sections': sections, 'predominant': predominant_summary(report, stable),
            'disclaimer': DISCLAIMER}
