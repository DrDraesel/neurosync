# NeuroSync

Live EEG study viewer for **BrainBit** headsets (BrainBit Classic/Black, BrainBit 2/Pro/Flex) and the **DragonEEG** 21-channel headset, driven by the manufacturer's `pyneurosdk2` BLE SDK. Everything runs and stays on this PC — recordings are never uploaded.

> **Not a medical device.** Engineering signal-quality numbers and descriptive EEG pattern displays only. No diagnosis, no mental-state claim, no emotion or sleep inference. Limits are documented in [`METHODS.md`](METHODS.md).

---

## Supported hardware

| Device | Scalp sites | Rate | What it can show |
|---|---|---|---|
| BrainBit Classic / Black / 2 / Pro / Flex | 4 (O1, O2, T3, T4) | 250 Hz | bands, power, asymmetry on 2 pairs, coherence on 6 pairs — **screening view only**; maps are mostly extrapolation |
| DragonEEG (NeuroEEG) | 21 (full 10-20) + 3 aux | 500 Hz | the full 5×5 qEEG sheet, 9 homologous pairs, ~200 coherence pairs |

Aux channels (D1/D2/D3) are recorded, stored and shown as raw curves and contact rows — they never enter band analysis or the head maps.

## Quick start

1. **First time on a computer:** double-click `setup.cmd` — creates the local `.venv` and installs the pinned requirements (needs Python 3.11).
2. Double-click `Start NeuroSync.cmd`, or run `.venv\Scripts\python.exe -u main.py` (add `--simulate brainbit` / `--simulate dragon` for the clearly-labelled simulator).
3. Press **Scan devices** (one scan covers every supported headset), select one, press **Connect**.
4. **DragonEEG first connection:** put the headset in pairing mode before scanning; the app tells you when the SDK reports pairing is required. Configure channel mode / frequency / gain and read the settings back before `StartSignal` — an unconfigured Dragon streams flat zeros.
5. With no headset powered on, the app shows "No devices found" and keeps running — scan again once a device is on.

Per-headset operating steps, in the order the app drives them: [`docs/STEPS.md`](docs/STEPS.md).

## Panels

- **Live analysis** — per-channel band table, 5 s window plus the stable multi-epoch estimate, quality gates, contact check.
- **qEEG grid** — the 5×5 z-scored sheet: absolute power, relative power, amplitude asymmetry, coherence, phase lag × Delta/Theta/Alpha/Beta/High Beta. Reference selector (this session / saved baseline / loaded norm file / raw), required condition, exported sheet PNG, theta/beta card, region roll-ups, EC/EO comparison, and a visible raw-trace reminder. See [`docs/QEEG-GRID-SPEC.md`](docs/QEEG-GRID-SPEC.md) and the reading method in [`docs/QEEG-RESEARCH.md`](docs/QEEG-RESEARCH.md).
- **BRAIN MAP** — per-band heat map of the schematic head layout (turbo ramp, percent scale bar, electrode dots, nose-up), band selector, **Export head map PNG…**. Sensor space only, not source localisation.
- **Predominant rhythm** — the largest measured band of the latest window with its share of 1–45 Hz and peak frequency.
- **Average brain wave** — channel-mean, band-limited to 1–45 Hz, rebuilt by summing the five band components.
- **Frequency components** — per-band power for the selected raw channel.
- **Second device window…** — an independent live window for a second headset at the same time (own scan/connect, own 60 s buffer, own map, own save). A BrainBit and a DragonEEG can run side by side, one headset per window.

## Study recording (whole session, not the rolling window)

- **Start study (record)** / **Stop study** capture every row between the two presses in a study span kept separately from the rolling 60 s analysis deque — so the start of a session is not lost.
- Hard cap **60 minutes**: reaching it auto-stops and saves with an explicit cap note; the recording is never silently truncated.
- The saved facts are the **actual elapsed duration (monotonic clock) and the actual row count**, so a transport gap shows up as a difference between the two instead of being hidden.

## Sessions, baselines, history

- Set the subject (name + phone) before saving. Sessions land in `recordings/subjects/<subject>/<timestamp>/` as `raw.csv`, `summary.json`, `charts/*.png`; reopen from **History…**; the trend chart compares saved sessions of the selected subject.
- **Baseline** = this subject's own earlier profile ("predominant-wave profile"), with a **required** condition (eyes closed / eyes open / other). Stored at `recordings/subjects/<subject>/baseline.json`; capturing again replaces it. Later estimates show descriptive deltas against it (percentage points, peak-Hz change, dominant-band change), and the source session is marked ★ in History.
- Conditions are never mixed: a baseline and its comparison must share the condition. Deltas are differences of measured numbers — not an assessment of the person.

## Reports

**Patient report (HTML)…** writes one self-contained file for the current subject: every saved session (device, quality, band tables, 5 s + stable estimates, predominant rhythm), all charts embedded as images, the session history table, the stored AI text with model and time, the qEEG grid section with its reference line, and the offline "How to read these numbers" helper. Open it in any browser and print to PDF. Nothing is uploaded.

## AI analysis

Optional, and the app states which backend is active:

- Local model: install Ollama on this PC with `qwen3.8:latest`.
- Ollama elsewhere: `set OLLAMA_URL=http://192.168.1.50:11434`.
- Any API model (OpenAI, OpenRouter, Anthropic, Gemini-compat, LM Studio…): `set AI_API_URL=…`, `set AI_API_KEY=…`, `set AI_API_MODEL=…` — or put those lines in `ai_settings.env` next to the app (gitignored).

Only derived numbers are ever sent (never raw EEG); with an API backend they go to that provider. The AI is asked to describe the measured numbers, never to diagnose.

## Tests and smoke check

```
.venv\Scripts\python.exe -m unittest discover -p 'test_*.py' -v   # 370 tests, no hardware, no BLE
QT_QPA_PLATFORM=offscreen .venv\Scripts\python.exe main.py --smoke
```

## Use it on another computer

Everything is local — "anywhere" means: install Python 3.11, clone this repo (public, no login needed) or copy the folder, run `setup.cmd`, then `Start NeuroSync.cmd`, and pair the headset **on that PC** (only one app may hold a headset at a time).

## Docs

| File | Contents |
|---|---|
| [`METHODS.md`](METHODS.md) | what is computed, units, limits, what still needs verification on real DragonEEG hardware |
| [`docs/STEPS.md`](docs/STEPS.md) | operator steps per headset, honesty notes first |
| [`docs/QEEG-GRID-SPEC.md`](docs/QEEG-GRID-SPEC.md) | the 5×5 sheet contract (rows, bands, references, colour ramp) |
| [`docs/QEEG-RESEARCH.md`](docs/QEEG-RESEARCH.md) | the reading method, sourced from @supernormalbydrtrishleigh, and what we deliberately do not claim |
| [`docs/QEEG-NORMS-SPEC.md`](docs/QEEG-NORMS-SPEC.md) | how references are imported (licensed exports) and how a local norm is accumulated (≥ 20 subjects) |

## Integration with the rest of the lab

LabFlow (`mission-control/labflow`) reads this app's `recordings/` directory as a **read-only** probe to confirm the EEG baseline station was completed. Nothing is written back and no data leaves the PC. See `mission-control/labflow/INTEGRATIONS.md`.
