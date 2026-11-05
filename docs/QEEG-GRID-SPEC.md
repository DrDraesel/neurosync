# qEEG grid — build contract (NeuroSync)

Target: reproduce the **Z-Scored FFT Summary** sheet (Leigh Brain & Spine "Optimal Brain Map") in NeuroSync:
5 rows × 5 bands of topographic heads, z-coloured, with an honest reference label.

Reference sheet rows (top → bottom): **Absolute Power, Relative Power, Amplitude Asymmetry, Coherence, Phase Lag**.
Bands (left → right): **Delta, Theta, Alpha, Beta, High Beta**.
Legend: z −3 … +3, **blue = below**, **green ≈ 0**, **yellow/orange/red = above**.

## Non-negotiable honesty rule

We do **not** have a normative database. A z-score against nothing is a lie.
Therefore every grid must carry a `reference` block:

| kind | meaning | allowed wording |
|---|---|---|
| `session` | mean/SD across channels **within this recording** | "self-referenced vs this session's channel mean" |
| `baseline` | a saved baseline session of the same subject+condition | "vs this subject's saved baseline (date)" |
| `norm` | a local norm file with ≥20 subjects, age band recorded | "vs local norm n=…, age …" |
| `none` | raw values, no colour z | "raw values — no reference selected" |

Never call any of these "NeuroGuide normative z". The report and the panel must print the reference line.

## Module: `qeeg.py`

```python
BANDS = OrderedDict([("Delta",(1,4)),("Theta",(4,8)),("Alpha",(8,13)),("Beta",(13,30)),("HighBeta",(30,45))])
ROWS  = ("absolute","relative","asymmetry","coherence","phase")

compute_metrics(samples, fs, channels, bands=BANDS, pair_max=200) -> Metrics
    samples: (n_samples, n_channels) float array in µV, scalp channels only
    Metrics = {
      "bands": {band: {"absolute": {ch: µV²}, "relative": {ch: %}, "peak_hz": {ch: Hz}}},
      "asymmetry": {band: {pair: pct}},          # pairs only where both channels exist
      "coherence": {band: {ch: 0..1}},           # per-electrode mean over usable pairs
      "phase": {band: {ch: degrees}},            # per-electrode mean |phase|, signed mean
      "pairs": {band: {pair: {"coherence":..,"phase_deg":..}}},
      "n_samples": int, "n_epochs": int, "channels": [...], "fs": float,
      "window_s": float, "usable_channels": [...]
    }

grid(metrics, reference=None) -> Grid
    Grid = {"rows": {row: {band: {ch: {"value": float|None, "z": float|None}}}},
            "reference": {"kind","label","n"}, "limits": {"z":[-3,3]},
            "notes": [str], "channels": [...], "bands": [...], "rows": [...]}

zscore(v, mean, sd) -> float|None            # sd<=0 or None -> None
reference_from_session(metrics) -> Reference  # mean/SD across channels per (row,band)
reference_from_baseline(metrics, label, n) -> Reference
save_reference(ref, path) / load_reference(path)   # JSON, explicit kind/norm metadata
color_for_z(z, limits=(-3,3)) -> "#rrggbb"     # blue→green→red ramp, green at 0
ratios(metrics) -> {"theta_beta": float, "alpha_theta": float, "slow_fast": float}
```

Rules:
- Compute on **scalp channels only** (`devices.scalp_channels`).
- Asymmetry pairs (homologous): Fp1/Fp2, F3/F4, C3/C4, P3/P4, O1/O2, F7/F8, T3/T4, T5/T6, T4/T6? — use the 10-20 homolog list; value = `(L−R)/((L+R)/2)*100`, sign = left minus right.
- Coherence = magnitude-squared coherence per pair per band (Welch cross-spectrum, Hann, 50 % overlap), then **electrode value = mean of that electrode's pair coherences** (document it).
- Phase = mean of `angle(cross_spectrum)` in degrees over the same pairs, electrode value = mean of signed pair phases (document it).
- Below 3 channels → `grid()` returns notes and no heads (never interpolate 2 points).
- `pair_max` bounds O(n²) work on 21 channels.

## UI (main.py)

New tab **"qEEG grid"** (next to the existing map), containing:
1. Toolbar: band chips (all five by default), row checkboxes (5 rows), **Reference** selector (`This session` / `Saved baseline` / `Load norm file…` / `Raw`) and a **Condition** label (eyes closed / eyes open) — required, same rule as baselines.
2. The grid: 5×5 miniature heads (≈150 px), row labels left, band labels on top, a single z colour bar in the middle column labelled `−3 … +3`, each head with its own min/max on hover (`setToolTip`).
3. A footer line that always states the reference kind + n + "not a normative clinical database".
4. **"How to read this"** collapsible panel with the guide text below (from Dr. Leigh's teaching, attributed, educational only).
5. **Export sheet PNG** (the whole grid, white background, titles + legend + reference line) and **Add to patient report** (writes the PNG into the session folder and includes it in `patient_report.py`).

Performance: compute once per open/refresh, not on the 1 s timer. Cache by `(session_id, band_set, reference)`.

## Guide text (put verbatim in the panel)

```
Five bands = five "speeds". Delta (1–4 Hz) restorative sleep; Theta (4–8) drowsy / shifting
to sleep — the classic ADHD marker when excessive; Alpha (8–13) calm and relaxed;
Beta (13–30) focused thinking; High Beta (30–45) stress response.
Colours: blue = below the reference, green = at it, yellow→red = above it.
Absolute power = how much energy is in that band at that site.
Relative power = that band's share of the total (which speed dominates).
Amplitude asymmetry = left vs right difference at homologous sites.
Coherence = how similarly two sites behave in that band (shared information).
Phase lag = the timing offset between those sites (how late or early the signal arrives).
Read clusters, not single heads: a single coloured head is weak evidence, a region that
repeats across rows and bands is the pattern worth discussing.
This app compares against a reference you select (session, your saved baseline, or a local
norm file). It is NOT an age-matched clinical normative database, and nothing here is a diagnosis.
```

## Tests

- `test_qeeg.py`: synthetic sine per band → absolute power peaks in the right band; relative sums to ~100 %; asymmetry sign for a louder left channel; coherence = 1.0 for two identical channels, ≈0 for independent noise; phase ≈ known injected delay; z-score None when SD = 0; grid returns None values (not zeros) when a channel is missing; <3 channels → notes and no heads.
- `test_qeeg_ui.py`: panel constructs offscreen, grid has 25 cells for 21 channels, reference label text present, export writes a PNG.

## Do not

- Do not invent a normative database, do not label anything "NeuroGuide", do not add diagnosis text.
- Do not run the computation on the 1 s UI timer.
- Do not include aux (D1/D2/D3) channels.
