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

Everything is generated locally from files on this PC; nothing is uploaded.

Honesty: the report describes measured signal-quality numbers.  It is not a
diagnosis, not a clinical measurement, and carries no mental-state claims.
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
        uri = _image_data_uri(Path(path), budget)
        if not uri:
            continue
        parts.append(f'<figure><img src="{uri}" alt="{_esc(Path(path).name)}">'
                     f'<figcaption>{_esc(Path(path).name)}</figcaption></figure>')
    return '<div class="figs">' + ''.join(parts) + '</div>' if parts else ''


def _session_section(session: dict, index: int, budget: dict) -> str:
    summary = session.get('summary') or {}
    report = summary.get('analysis') or {}
    stable = summary.get('stable_estimate') or {}
    device = summary.get('device') or {}
    simulated = bool(summary.get('simulated') or device.get('simulated'))
    title = f"Session {summary.get('started_at') or session.get('path').name}"
    quality = ('state: ' + ('VALID' if report.get('valid') else 'not valid')
               + (f" · {report.get('reason')}" if report.get('reason') else ''))
    meta_bits = [
        f"device: {_esc(summary.get('device_label'))}",
        f"channels: {_esc(', '.join(summary.get('channels') or []))}",
        f"sampling: {_esc(summary.get('fs_hz'))} Hz · notch {_esc(summary.get('notch_hz'))} Hz",
        f"duration: {_fmt(summary.get('duration_s'), 1, ' s')} "
        f"({_esc(summary.get('raw_rows'))} rows)",
        f"simulated: {'YES — pipeline demonstration, not measured EEG' if simulated else 'no'}",
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
    ai = (summary.get('ai_doctor') or {}).get('text') or ''
    ai_html = (f"<h3>Stored AI doctor description</h3><div class='ai'>{_esc(ai)}</div>"
               if ai else '')
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
  {_charts_html(session.get('charts') or [], budget)}
  {ai_html}
</section>"""


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
            f"<td>{_esc(estimate.get('dominant'))}</td>"
            f"<td>{_fmt(estimate.get('dominant_pct'), 1)}</td>"
            f"{cells}</tr>")
    return ('<table><tr><th>Session</th><th>Device</th><th>Dominant</th><th>Dominant %</th>'
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
 .ai {{ background:#17242e; border:1px solid #2b4152; padding:10px; font-size:12.5px;
        white-space:pre-wrap; }}
 .note {{ background:#17242e; border-left:3px solid #4a7f9c; padding:8px 10px; margin:6px 0;
          font-size:12.5px; }}
 .figs {{ display:flex; flex-wrap:wrap; gap:12px; }}
 figure {{ margin:0; width:320px; }}
 figure img {{ width:100%; border:1px solid #2b4152; border-radius:6px; }}
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
<h2>History — all saved sessions ({len(sessions)})</h2>
{_history_table(sessions)}
<h2>How to read these numbers (offline helper)</h2>
{helper_html}
{sections}
<p class="honesty">{_esc(ai_doctor.DISCLAIMER)} Descriptive, engineering-grade spectral summary.
 Recordings and this report never leave this PC. Methods: METHODS.md.</p>
</body></html>"""
    target = Path(out_path) if out_path else (Path(sessions[0]['path']).parent /
                                              f'patient_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding='utf-8')
    return target
