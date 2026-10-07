# NeuroSync

Live EEG study viewer for BrainBit headsets (BrainBit Classic/Black, BrainBit 2/Pro/Flex, DragonEEG) driven by the manufacturer's pyneurosdk2 BLE SDK. Everything runs and stays on this PC — recordings are never uploaded.

## Run

- First time on a computer: double-click `setup.cmd` (one time — creates the local `.venv` and installs the pinned requirements; needs Python 3.11).
- Then double-click `Start NeuroSync.cmd`, or from this folder:
  `.venv\Scripts\python.exe -u main.py` (add `--simulate brainbit` / `--simulate dragon` for the simulator).
- Press **Scan devices** (one scan covers every supported headset), select one and press **Connect**.
- DragonEEG first connection: put the headset in pairing mode before scanning; the app will tell you when the SDK reports pairing is required.
- Before saving a session, set the subject (name + phone). Sessions are written under `recordings/subjects/<subject>/<timestamp>/` (`raw.csv`, `summary.json`, `charts/*.png`) and can be reopened from **History…**; the trend chart compares saved sessions of the selected subject.
- The **BRAIN MAP** panel shows the per-band heat map (turbo colour ramp, percent scale bar, electrode dots, nose-up) of the schematic head layout; pick the band with the **Head map band** selector and press **Export head map PNG…** to save a presentation-quality PNG (high-resolution interpolation, larger canvas, electrode name labels, honesty footer). It is an engineering sensor-space picture, not source localisation — see `METHODS.md`.
- Set a per-subject **baseline** ("predominant-wave profile"): with a subject set, record a passing estimate (the stable multi-epoch estimate is preferred, a valid 5 s window is the fallback), choose the condition (eyes closed / eyes open / other) and press **Set as baseline** — or mark a saved session in **History…** with "Set as baseline…". Later estimates and saved sessions then show descriptive deltas against that subject's own earlier numbers (per-band percentage points, peak-Hz change, dominant-band change), and the baseline's source session is marked ★ in History. It is stored locally at `recordings/subjects/<subject>/baseline.json`; capturing again replaces it.
- Baselines are condition-specific: the condition is recorded (and required) because it changes what the numbers mean, so only compare measurements taken in the same condition. A baseline and its deltas are descriptive EEG pattern numbers of your own measurements — not a diagnosis and not an assessment of the person.
- No headset powered on? The app shows "No devices found" and keeps running — scan again once a device is on.
- **Predominant rhythm strip** (right column): the largest measured band of the latest 5 s window (or the stable multi-epoch estimate when available) with its share of 1–45 Hz and its peak frequency — one readable "most predominant brain wave" line.
- **AVERAGE BRAIN WAVE panel** (middle column): the channel-mean of the current device, band-limited to 1–45 Hz, rebuilt by summing the five band components — one descriptive "average wave" display; there is also the per-band FREQUENCY COMPONENTS panel for the selected raw channel.
- **Second device window…** (toolbar): an independent live window for a SECOND headset at the same time — own scan/connect, own 60 s buffer, own 5 s analysis, own BRAIN MAP (with band selector and PNG export), predominant-rhythm line, per-channel table, contact check, AI doctor and *Save session (this device)*. Run a BrainBit and a DragonEEG side by side, one headset per window. Works with real devices or the labelled simulator.
- **Patient report (HTML)…** (toolbar): writes ONE self-contained HTML file for the current subject — every saved session (device, quality, band tables, 5 s + stable estimates, predominant), all charts embedded as images, the session history table, the stored AI-doctor text and the offline interpretation helper ("How to read these numbers"). Open it in any browser and print to PDF to hand over or archive. Nothing is uploaded.
- **Simulator mode** (`--simulate brainbit` / `--simulate dragon`, or pass it to the app when no hardware is present): a clearly-labelled SIMULATED stream (device label starts with "SIMULATED", amber banner, `"simulated": true` in every saved summary, flagged in reports and in the AI payload) generated mathematically on this PC — deterministic alpha-dominated EEG per channel. Purpose: rehearse the whole live pipeline (analysis, maps, saving, reports) without hardware, and verify the app end-to-end. Simulated data is never mixed into real recordings; the AI doctor is instructed to say the data is simulated.

## Test and smoke check

- Full suite (fixture-based, no hardware, no BLE):
  `.venv\Scripts\python.exe -m unittest discover -p 'test_*.py' -v`
- Headless launch check:
  `QT_QPA_PLATFORM=offscreen .venv\Scripts\python.exe main.py --smoke`

## Use it on another computer (headset connects there)

Everything is local — "anywhere" means: set the app up once on that PC, pair the
headset there. (Only one app may connect to a headset at a time.)

1. Install Python 3.11 (https://www.python.org/downloads/ — check "Add python.exe to PATH").
2. Get this folder onto the PC: `git clone` the repo (public — no login needed) or copy the folder.
3. Run `setup.cmd` (one time; creates `.venv`, installs the pinned requirements).
4. Run `Start NeuroSync.cmd`, scan for the headset, connect.
5. Optional — AI doctor / live chat: install Ollama on that PC with the `qwen3.8:latest` model.
   If Ollama runs on another computer on your network, point the app at it before
   launching: `set OLLAMA_URL=http://192.168.1.50:11434` (and optionally `set OLLAMA_MODEL=...`).

## Honesty note

Engineering signal-quality estimates only — including baselines and their deltas, which are differences of measured numbers. Not a diagnosis, not a medical device, no mental-state/emotion/sleep claims. Units and limits are documented in `METHODS.md`, including what still needs verification on physical DragonEEG hardware.
