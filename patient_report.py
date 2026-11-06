"""Downloadable per-subject patient report (single self-contained HTML file).

Bundles everything stored for one subject into ONE portable file the user can
open anywhere or print to PDF:

* subject header + honesty banner,
* one section per saved session (newest first): device, quality gates, band
  tables (5 s window and stable estimate), predominant rhythm, per-channel
  dominant band, the saved charts (embedded as PNG data URIs) and the AI doctor
  text that was stored with that session,
* a cross-session history table (relative power % per band, dominant band),
* an offline "how to read these numbers" interpretation helper
  (``ai_doctor.local_interpretation``),
* a "Reading method (for the clinician)" section condensing
  ``docs/QEEG-RESEARCH.md`` (the doctor-channel reading method, quoted as its
  own source): what the five sheet rows, the z colours and the reference block
  kinds mean here, what each device can and cannot support, and the rule that
  interpretation is clinician-led,

Everything is generated locally from files on this PC and the file itself is
never uploaded. If an AI API is configured, only the derived numbers behind the
stored AI text were sent to it — the provenance line on that section says so.

Honesty: the report states that it is not a medical device, describes measured
signal-quality numbers, and prints the electrode limitation of the headset that
recorded each session. It is not a diagnosis, not a clinical measurement, and
carries no mental-state claims.
"""
from __future__ import annotations

import base64
import html
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

import ai_doctor
import storage

BAND_ORDER = ('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
MAX_CHART_MB = 40.0
QEEG_NOTE = 'Not a normative clinical database; educational display only.'
RAW_TRACE_REMINDER = 'Check the raw traces before trusting a map — artifact looks like a region.'

# Reading-method section (for the clinician). The source line is quoted
# verbatim from docs/QEEG-RESEARCH.md; the method itself is condensed from the
# same document and from docs/QEEG-GRID-SPEC.md. Educational only: it invents
# no norms and adds no diagnostic claim.
READING_METHOD_HEADING = 'Reading method (for the clinician)'
READING_METHOD_SOURCE = ('Source: @supernormalbydrtrishleigh '
                         '(Dr. Trish Leigh, Leigh Brain & Spine), reviewed 2026-11-03.')

READING_ROWS = (
    ('Absolute power', 'how much energy (µV²) that band carries at that site — how "charged" it is.'),
    ('Relative power', 'that band\'s share of the total at that site — which of the five speeds dominates.'),
    ('Amplitude asymmetry', 'left vs right difference at homologous sites.'),
    ('Coherence', 'how similarly two sites behave in that band (shared information; '
                  'too much = rigidity, too little = poor communication).'),
    ('Phase lag', 'the timing offset between sites — how late or early the signal arrives.'),
)

# Reference kinds with the exact wording the grid spec allows (never reworded
# into a claim of a clinical database).
REFERENCE_KINDS = (
    ('session', 'self-referenced vs this session\'s channel mean'),
    ('baseline', 'vs this subject\'s saved baseline (date)'),
    ('norm', 'vs local norm n=…, age … (a local file, only usable once 20 or more '
             'subjects exist and the age band is recorded)'),
    ('none', 'raw values — no reference selected'),
)

# Electrode limitation of each supported headset, printed next to the numbers it
# produced: a four-electrode BrainBit is not a 19-channel qEEG, and the
# DragonEEG is a research headset, not a medical device.
BRAINBIT_ELECTRODES = 'BrainBit: 4 electrodes, shared reference — not a 19-channel qEEG'
DRAGON_ELECTRODES = 'DragonEEG: 21 scalp channels, not a medical device'
NOT_A_DEVICE = ('Not a medical device: NeuroSync is a research viewer for measured '
                'signal-quality numbers. It does not diagnose, treat or monitor any '
                'condition, and nothing in this document is a clinical result.')


def _esc(value) -> str:
    return html.escape(str(value if value is not None else '—'))


def _fmt(value, places=2, suffix='') -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '—'
    text = f'{number:,.{places}f}'
    return text + suffix


def _bands_table(report: dict, title: str) -> str:
    if not isinstance(report, dict) or not report.get('bands'):
        return ''
    rows = []
    for name in BAND_ORDER:
        band = (report.get('bands') or {}).get(name) or {}
        power = band.get('power_uv2')
        relative = band.get('relative_pct')
        sem = band.get('sem_uv2')
        power_text = _fmt(power, 2)
        if sem is not None:
            power_text += ' ±' + _fmt(sem, 2)
        rows.append(f'<tr><td>{_esc(name)}</td><td>{power_text}</td>'
                    f'<td>{_fmt(relative, 1, "%") if relative is not None else "—"}</td></tr>')
    return (f'<h3>{_esc(title)}</h3>'
            '<table><tr><th>Band</th><th>Power µV²</th><th>Relative % of 1–45 Hz</th></tr>'
            + ''.join(rows) + '</table>')


def _channel_table(report: dict, limit: int = 24) -> str:
    channels = (report or {}).get('channels') or {}
    if not channels:
        return ''
    rows = []
    for name, entry in list(channels.items())[:limit]:
        relative = entry.get('relative_pct') or {}
        dominant = entry.get('dominant') or ('mixed' if relative else 'unreliable')
        share = relative.get(dominant) if dominant in relative else None
        flags = ', '.join(str(flag) for flag in (entry.get('flags') or [])[:4]) or '—'
        rows.append(f'<tr><td>{_esc(name)}</td><td>{_esc(dominant)}</td>'
                    f'<td>{_fmt(share, 1, "%") if share is not None else "—"}</td>'
                    f'<td>{_esc(flags)}</td></tr>')
    return ('<h3>Per-channel dominant band</h3>'
            '<table><tr><th>Channel</th><th>Dominant</th><th>Share</th><th>Flags</th></tr>'
            + ''.join(rows) + '</table>')


def _image_data_uri(path: Path, budget: dict) -> str:
    try:
        size_mb = path.stat().st_size / 1e6
        if budget['remaining_mb'] - size_mb < 0:
            return ''
        data = path.read_bytes()
    except OSError:
        return ''
    budget['remaining_mb'] -= len(data) / 1e6
    mime = mimetypes.guess_type(path.name)[0] or 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')


def _charts_html(charts: list, budget: dict) -> str:
    parts = []
    for path in charts:
        if Path(path).name == 'qeeg_grid.png':
            continue                      # printed by its own reference-labelled section
        uri = _image_data_uri(Path(path), budget)
        if not uri:
            continue
        parts.append(f'<figure><img src="{uri}" alt="{_esc(Path(path).name)}">'
                     f'<figcaption>{_esc(Path(path).name)}</figcaption></figure>')
    return '<div class="figs">' + ''.join(parts) + '</div>' if parts else ''


def _tbr_html(record: dict) -> str:
    """Theta/beta card as a table: per-channel ratio, channel mean, teaching note."""
    data = record.get('theta_beta')
    data = data if isinstance(data, dict) else {}
    channels = data.get('channels') or {}
    if not channels and data.get('mean') is None:
        return ''
    cells = list(channels.items())
    rows = []
    for start in range(0, len(cells), 4):
        chunk = cells[start:start + 4]
        tds = ''.join(f'<td>{_esc(name)}</td><td>{_fmt(value, 2)}</td>'
                      for name, value in chunk)
        rows.append('<tr>' + tds + '<td></td><td></td>' * (4 - len(chunk)) + '</tr>')
    head = ''.join('<th>Channel</th><th>θ/β</th>' for _ in range(4))
    mean = data.get('mean')
    flag = bool(data.get('flag'))
    mean_line = (f"Channel mean θ/β {_fmt(mean, 2)}" if mean is not None
                 else 'Channel mean θ/β —')
    if flag:
        mean_line += (' — the mean is at or above 3.0 and both bands are usable; this is the '
                      'pattern Dr. Leigh describes, not a diagnosis.')
    elif mean is not None:
        mean_line += ' (below the 3:1 flag; nothing is inferred from one number).'
    note = data.get('note') or ''
    return (f'<h3>Theta / Beta</h3>'
            f"<p class='meta'>{_esc(mean_line)}</p>"
            f'<table class="tbr"><tr>{head}</tr>{"".join(rows)}</table>'
            f"<p class='meta'>{_esc(note)}</p>")


def _region_html(record: dict) -> str:
    """Region roll-up: band rows × region columns of mean z, flagged cells marked."""
    data = record.get('regions')
    data = data if isinstance(data, dict) else {}
    if not data.get('available'):
        return ''
    bands = [str(band) for band in data.get('bands') or []]
    regions = [str(region) for region in data.get('regions') or []]
    values = data.get('values') or {}
    if not bands or not regions:
        return ''
    flagged = {(item.get('band'), item.get('region')): item
               for item in data.get('flags') or [] if isinstance(item, dict)}
    head = '<tr><th>Band</th>' + ''.join(f'<th>{_esc(region.replace("_", " "))}</th>'
                                         for region in regions) + '</tr>'
    rows = []
    for band in bands:
        cells = []
        for region in regions:
            value = (values.get(band) or {}).get(region)
            is_flag = (band, region) in flagged
            text = '—' if value is None else f'{float(value):+.2f}'
            if is_flag:
                text += ' ▲'
            cells.append(f"<td class='{'flagcell' if is_flag else ''}'>{text}</td>")
        rows.append(f'<tr><td>{_esc(band)}</td>{"".join(cells)}</tr>')
    caption = ('Flagged: ' + '; '.join(
        f"{_esc(item.get('band'))} · {_esc(str(item.get('region')).replace('_', ' '))} "
        f"z {_fmt(item.get('z'), 2)}"
        + (f" ({_esc(item.get('n_sites'))} sites)" if item.get('n_sites') else '')
        for item in flagged.values())) if flagged else 'No region flagged (|z| ≥ 2 with ≥ 2 sites).'
    rule = data.get('rule') or ('mean z per region; flag when |z| >= 2 with >= 2 sites')
    row = data.get('row')
    row_text = f'row: {str(row).replace("_", " ").title()} · ' if row else ''
    return (f'<h3>Region roll-up (mean z)</h3>'
            f'<table class="regions">{head}{"".join(rows)}</table>'
            f"<p class='meta'>{_esc(caption)} · {_esc(row_text + rule)}. Regions with fewer than "
            f"two available sites are omitted.</p>")


def _qeeg_html(session: dict, budget: dict) -> str:
    """The qEEG 5x5 sheet section, with its reference printed above the image.

    The reference block is whatever the panel recorded with the sheet
    (kind/label/n). When a PNG exists without that block, the section still
    prints "reference not recorded" and the fixed disclaimer — it never
    invents a normative reference.
    """
    summary = session.get('summary') or {}
    record = summary.get('qeeg_grid')
    record = record if isinstance(record, dict) else {}
    image = next((Path(path) for path in (session.get('charts') or [])
                  if Path(path).name == 'qeeg_grid.png'), None)
    if not record and image is None:
        return ''
    label = record.get('reference_label') or record.get('reference_kind') or 'not recorded'
    reference_line = (record.get('reference_line')
                      or 'Reference — not recorded with this session, so the sheet cannot be '
                         'traced to a comparison. ' + QEEG_NOTE)
    meta = []
    if record.get('condition'):
        meta.append(f"condition: {record.get('condition')}")
    if record.get('reference_n') not in (None, ''):
        meta.append(f"reference n: {record.get('reference_n')}")
    if record.get('channels'):
        meta.append(f"channels: {len(record.get('channels') or [])}")
    if record.get('bands'):
        meta.append('bands: ' + ', '.join(str(band) for band in record.get('bands') or []))
    if record.get('reference_source'):
        meta.append(f"reference source: {record.get('reference_source')}")
    if record.get('reference_age_band'):
        meta.append(f"age band: {record.get('reference_age_band')}")
    if record.get('norm_scope_warning'):
        meta.append(str(record.get('norm_scope_warning')))
    meta_html = f"<p class='meta'>{_esc(' · '.join(meta))}</p>" if meta else ''
    tbr_html = _tbr_html(record)
    region_html = _region_html(record)
    figure = ''
    if image is not None:
        uri = _image_data_uri(image, budget)
        if uri:
            figure = (f'<figure class="qeeg"><img src="{uri}" alt="qEEG 5x5 grid">'
                      f'<figcaption>{_esc(image.name)}</figcaption></figure>')
    return (f'<h3>qEEG grid (reference: {_esc(label)})</h3>'
            f"<p class='meta'>{_esc(reference_line)}</p>{meta_html}{figure}"
            f"{tbr_html}{region_html}"
            f"<p class='meta'>{_esc(RAW_TRACE_REMINDER)}</p>"
            f"<p class='meta'><b>{_esc(QEEG_NOTE)}</b></p>")


def _reading_method_html() -> str:
    """Condensed qEEG reading method for the clinician, sourced from the docs.

    Explains what the five sheet rows, the z colours and the reference block
    kinds mean in THIS app, states per device what the numbers can and cannot
    support, and closes with the rule that interpretation is clinician-led.
    It is an educational description of the reading method only: no invented
    norms, no diagnostic claim, and every honesty banner elsewhere stays.
    """
    row_rows = ''.join(
        f'<tr><td><b>{_esc(name)}</b></td><td>{_esc(meaning)}</td></tr>'
        for name, meaning in READING_ROWS)
    ref_rows = ''.join(
        f'<tr><td>{_esc(kind)}</td><td>{_esc(wording)}</td></tr>'
        for kind, wording in REFERENCE_KINDS)
    return f"""
<h2>{_esc(READING_METHOD_HEADING)}</h2>
<div class="note"><b>Method source.</b> The qEEG sheet and these notes follow the
reading method taught on the source channel below; the app re-implements the sheet
layout and colour rule, not a licensed normative database.<br>
<b>{_esc(READING_METHOD_SOURCE)}</b></div>
<div class="note"><b>The five rows of the sheet.</b>
<table>
<tr><th>Row</th><th>What it means here</th></tr>{row_rows}
</table></div>
<div class="note"><b>Z-scores and colours.</b> The sheet legend is a z-scale from −3 to
+3: green = at the reference, blue = below, yellow → orange → red = above. In this app
the z is always computed against the reference block printed with the sheet — never
against an age-matched clinical database, because none is used here.</div>
<div class="note"><b>Reference block kind.</b> Every sheet prints which reference its
z-scores were computed against; a reference is never invented.
<table>
<tr><th>Kind</th><th>Meaning (printed with the sheet)</th></tr>{ref_rows}
</table>
A <code>norm</code> is a local file that only becomes usable once 20 or more subjects
exist with the age band recorded; until then the app does not call it normative.</div>
<div class="note"><b>What the devices can and cannot support.</b>
<ul>
<li><b>BrainBit (4 channels, O1/O2/T3/T4, shared reference)</b> — screening-grade view at
four sites: band powers, relative shares, two asymmetry pairs (O1/O2, T3/T4) and at most
six coherence/phase pairs. Its four-site map is mostly extrapolation and is NOT a
19-channel qEEG. It can support a coarse spectral look at four sites; it cannot support
a regional or whole-brain qEEG reading.</li>
<li><b>DragonEEG (21 scalp channels + 3 auxiliary)</b> — qEEG-grade recording: the device
the 5×5 sheet is designed for (21 sites, nine homologous asymmetry pairs, about 200
coherence/phase pairs). The headset is not a medical device and this app is not a
medical device, so its output stays a descriptive research recording, not a clinical
measurement.</li>
</ul>
<b>These numbers can support:</b> a descriptive spectral summary of the recorded sites;
a within-subject comparison against that subject's own saved baseline of the same
condition; documentation of recording conditions and measured signal quality.<br>
<b>They cannot support:</b> a diagnosis, a screening or treatment decision, a
mental-state claim, a comparison with an age-matched normative database, any conclusion
beyond the recorded sites, or any conclusion from a single number or a single head —
read clusters across rows and bands, and check the raw traces first (artifact looks
like a region).</div>
<div class="note"><b>Interpretation is clinician-led.</b> Nothing in this document
interprets the numbers medically. Whether any number is clinically meaningful is for
the clinician reviewing the original recording to decide; this section only explains
the method the sheet uses.</div>
"""


def _device_electrodes(summary: dict) -> str:
    """Electrode limitation of the headset that recorded this session."""
    device = summary.get('device') or {}
    key = str(summary.get('device_type') or device.get('device_key') or '').lower()
    channels = summary.get('channels') or []
    if key.startswith('dragon') or len(channels) > 4:
        return DRAGON_ELECTRODES
    return BRAINBIT_ELECTRODES


def _ai_html(summary: dict) -> str:
    """Stored AI text with the model, timestamp and provenance it came with."""
    record = summary.get('ai_doctor') or {}
    text = record.get('text') or ''
    if not text:
        return ''
    bits = []
    if record.get('model'):
        bits.append(f'model {_esc(record["model"])}')
    if record.get('generated_at'):
        bits.append(f'generated {_esc(record["generated_at"])}')
    if record.get('provenance'):
        bits.append(_esc(record['provenance']))
    meta = f"<p class='meta'>{' · '.join(bits)}</p>" if bits else ''
    return f"<h3>Stored AI doctor description</h3>{meta}<div class='ai'>{_esc(text)}</div>"


def _session_section(session: dict, index: int, budget: dict) -> str:
    summary = session.get('summary') or {}
    report = summary.get('analysis') or {}
    stable = summary.get('stable_estimate') or {}
    device = summary.get('device') or {}
    simulated = bool(summary.get('simulated') or device.get('simulated'))
    title = f"Session {summary.get('started_at') or session.get('path').name}"
    quality = ('state: ' + ('VALID' if report.get('valid') else 'not valid')
               + (f" · {report.get('reason')}" if report.get('reason') else ''))
    recorded_span_s = summary.get('recorded_span_s')
    if recorded_span_s is not None:
        # Full-duration study: raw_rows/duration_s are the true recording, and
        # recorded_span_s is the authoritative span — print it as such.
        span_bit = (f"recording span: {_fmt(recorded_span_s, 1, ' s')} "
                    f"({_esc(summary.get('raw_rows'))} rows) — full-duration study")
    else:
        # Legacy/quick save: the raw CSV holds the rolling analysis buffer, not
        # the whole session; keep the honest buffer wording.
        span_note = (summary.get('raw_span_note')
                     or 'the rolling analysis buffer stored with the session, not the '
                        'study length')
        span_bit = (f"raw buffer span: {_fmt(summary.get('duration_s'), 1, ' s')} "
                    f"({_esc(summary.get('raw_rows'))} rows) — {_esc(span_note)}")
    clean_channels = (f"{report.get('channels_valid', '—')}/"
                      f"{report.get('channels_total', len(summary.get('channels') or []) or '—')}")
    meta_bits = [
        f"device: {_esc(summary.get('device_label'))}",
        f"channels: {_esc(', '.join(summary.get('channels') or []))}",
        f"electrodes: {_esc(_device_electrodes(summary))}",
        f"sampling: {_esc(summary.get('fs_hz'))} Hz · notch {_esc(summary.get('notch_hz'))} Hz",
        span_bit,
        f"simulated: {'YES — pipeline demonstration, not measured EEG' if simulated else 'no'}",
        f"clean channels: {_esc(clean_channels)}",
        quality,
    ]
    predominant = ''
    estimate = stable if stable.get('valid') else report
    if estimate.get('dominant'):
        predominant = (f"<p class='pred'><b>Predominant rhythm:</b> {_esc(estimate.get('dominant'))} "
                       f"— {_fmt(estimate.get('dominant_pct'), 1, '%')} of 1–45 Hz"
                       + (f", peak {_fmt(estimate.get('peak_hz'), 2, ' Hz')}"
                          if estimate.get('peak_hz') is not None else '')
                       + f" ({'stable estimate' if stable.get('valid') else '5 s window'}).</p>")
    ai_html = _ai_html(summary)
    qeeg_html = _qeeg_html(session, budget)
    baseline = summary.get('vs_baseline') or {}
    baseline_html = ''
    if baseline:
        delta_rows = []
        for name, entry in (baseline.get('bands') or {}).items():
            delta_rows.append(f"<tr><td>{_esc(name)}</td>"
                              f"<td>{_fmt(entry.get('baseline_pct'), 1, '%')}</td>"
                              f"<td>{_fmt(entry.get('current_pct'), 1, '%')}</td>"
                              f"<td>{_fmt(entry.get('delta_pp'), 1, ' pp')}</td></tr>")
        baseline_html = ('<h3>vs this subject’s baseline '
                         f"({_esc(baseline.get('condition'))})</h3>"
                         '<table><tr><th>Band</th><th>Baseline</th><th>Now</th><th>Δ</th></tr>'
                         + ''.join(delta_rows) + '</table>')
    return f"""
<section class="session">
  <h2>{index}. {_esc(title)}</h2>
  <p class="meta">{' · '.join(meta_bits)}</p>
  {predominant}
  {_bands_table(report, 'Latest 5 s window')}
  {_bands_table(stable, 'Stable multi-epoch estimate')}
  {_channel_table(report)}
  {baseline_html}
  {qeeg_html}
  {_charts_html(session.get('charts') or [], budget)}
  {ai_html}
</section>"""


def _span_text(summary: dict) -> str:
    """True recorded span for study sessions; buffer span for legacy/quick saves."""
    recorded = summary.get('recorded_span_s')
    if recorded is not None:
        return f'{_fmt(recorded, 1)} s'
    duration = summary.get('duration_s')
    if duration is None:
        return '—'
    return f'{_fmt(duration, 1)} s (buffer)'


def _history_table(sessions: list) -> str:
    rows = []
    for session in sessions:
        summary = session.get('summary') or {}
        report = summary.get('analysis') or {}
        stable = summary.get('stable_estimate') or {}
        estimate = stable if stable.get('valid') else report
        shares = {name: (estimate.get('bands') or {}).get(name, {}).get('relative_pct')
                  for name in BAND_ORDER}
        cells = ''.join(f'<td>{_fmt(shares[name], 1)}</td>' for name in BAND_ORDER)
        rows.append(
            f"<tr><td>{_esc(summary.get('started_at'))}</td>"
            f"<td>{_esc(summary.get('device_label'))}</td>"
            f"<td>{_span_text(summary)}</td>"
            f"<td>{_esc(estimate.get('dominant'))}</td>"
            f"<td>{_fmt(estimate.get('dominant_pct'), 1)}</td>"
            f"{cells}</tr>")
    return ('<table><tr><th>Session</th><th>Device</th><th>Span</th><th>Dominant</th>'
            '<th>Dominant %</th>'
            + ''.join(f'<th>{name} %</th>' for name in BAND_ORDER) + '</tr>'
            + ''.join(rows) + '</table>')


def build(root, name: str, phone: str = '', out_path=None) -> Path:
    """Compose the report for one subject. Returns the written HTML path."""
    root = Path(root)
    sessions = []
    for entry in storage.list_sessions(root, name, phone):
        try:
            loaded = storage.load_session(entry['path'])
        except (OSError, ValueError):
            continue
        loaded['path'] = entry['path']
        sessions.append(loaded)
    if not sessions:
        raise ValueError(f'No saved sessions for subject {name!r} under {root}')
    budget = {'remaining_mb': MAX_CHART_MB}
    sections = ''.join(_session_section(session, number, budget)
                       for number, session in enumerate(sessions, start=1))
    interpretation = ai_doctor.local_interpretation(
        report=(sessions[0]['summary'] or {}).get('analysis'),
        stable=(sessions[0]['summary'] or {}).get('stable_estimate'),
        device=(sessions[0]['summary'] or {}).get('device'))
    helper_html = ''.join(
        f"<div class='note'><b>{_esc(item['title'])}</b><br>{_esc(item['text'])}</div>"
        for item in interpretation.get('sections', []))
    subject_key = storage.subject_key(name, phone)
    generated = datetime.now(timezone.utc).isoformat()
    simulated_any = any((s.get('summary') or {}).get('simulated') for s in sessions)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>NeuroSync patient report — {_esc(name)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body {{ font-family: system-ui, 'Segoe UI', sans-serif; background:#0f1a22; color:#e3edf4;
        margin:0; padding:26px; max-width:1040px; }}
 h1 {{ font-size:21px; margin:0 0 4px; color:#8be6ce; }}
 h2 {{ font-size:16px; margin:26px 0 6px; color:#a8dcff; border-bottom:1px solid #2b4152; padding-bottom:4px; }}
 h3 {{ font-size:13px; margin:16px 0 5px; color:#cde6f2; }}
 .meta {{ color:#9fb6c3; font-size:12px; margin:4px 0 8px; }}
 .pred {{ background:#14262f; border:1px solid #2b4152; padding:8px 10px; font-size:13px; }}
 table {{ border-collapse:collapse; width:100%; font-size:12.5px; margin:4px 0 10px; }}
 th, td {{ text-align:left; padding:5px 8px; border-bottom:1px solid #22343f; }}
 th {{ color:#a8dcff; font-size:11.5px; }}
 td.flagcell {{ background:#3a2c14; color:#ffd9a0; font-weight:700; }}
 table.tbr td, table.tbr th {{ padding:3px 6px; font-size:12px; }}
 table.regions td {{ font-size:11.5px; padding:3px 5px; }}
 .ai {{ background:#17242e; border:1px solid #2b4152; padding:10px; font-size:12.5px;
        white-space:pre-wrap; }}
 .note {{ background:#17242e; border-left:3px solid #4a7f9c; padding:8px 10px; margin:6px 0;
          font-size:12.5px; }}
 .figs {{ display:flex; flex-wrap:wrap; gap:12px; }}
 figure {{ margin:0; width:320px; }}
 figure img {{ width:100%; border:1px solid #2b4152; border-radius:6px; }}
 figure.qeeg {{ width:100%; }}
 figure.qeeg img {{ width:100%; }}
 figcaption {{ color:#9fb6c3; font-size:11px; padding-top:3px; }}
 .banner {{ background:#3a2c14; border:1px solid #8a6a2a; color:#ffd9a0; padding:10px;
            font-size:12.5px; margin:12px 0; }}
 .honesty {{ color:#c9a227; font-size:12px; margin-top:18px; }}
 @page {{ size: A4; margin: 12mm; }}
 @media print {{ body {{ background:#fff; color:#111; max-width:none; }}
   h1,h2,h3,th {{ color:#0b4a75; }} .meta, figcaption {{ color:#555; }}
   .pred, .ai, .note {{ background:#f4f6f8; border-color:#ccc; color:#111; }}
   .banner {{ background:#fff6e0; color:#7a5a00; }} .honesty {{ color:#7a5a00; }} }}
</style></head><body>
<h1>NeuroSync patient report</h1>
<p class="meta">Subject: <b>{_esc(name)}</b> · phone {_esc(phone or '—')} · key {_esc(subject_key)} ·
 generated {_esc(generated)} · source folder {_esc(root)}</p>
{'<div class="banner"><b>SIMULATED DATA:</b> at least one included session was recorded with the '
 'built-in simulator (no headset). Those sections are pipeline demonstrations, not measurements.</div>'
 if simulated_any else ''}
<div class="banner"><b>What this document is:</b> a local engineering record of measured EEG
 numbers (band powers, relative shares, signal-quality flags) with pictures of those numbers.
 It is <b>not</b> a diagnosis, not a clinical measurement, and carries no mental-state or
 disease conclusion. Show it to a qualified clinician together with a properly recorded
 study (e.g. a qEEG with eyes open/closed, eyes-closed resting EEG) if a real assessment
 is wanted.</div>
<div class="banner"><b>{_esc(NOT_A_DEVICE)}</b> The electrode limitation of each headset is
 printed with the session it recorded (a four-electrode BrainBit covers four sites with one
 shared reference; the DragonEEG reports 21 scalp sites).</div>
<h2>History — all saved sessions ({len(sessions)})</h2>
{_history_table(sessions)}
<h2>How to read these numbers (offline helper)</h2>
{helper_html}
{_reading_method_html()}
{sections}
<p class="honesty">{_esc(ai_doctor.DISCLAIMER)} Not a medical device. Descriptive,
 engineering-grade spectral summary; the electrode limitation of the recording headset is
 stated with each session, and so is the provenance (model, timestamp, backend) of any
 stored AI text. Recordings and this report are written on this PC and are never uploaded.
 Methods: METHODS.md.</p>
</body></html>"""
    target = Path(out_path) if out_path else (Path(sessions[0]['path']).parent /
                                              f'patient_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding='utf-8')
    return target
