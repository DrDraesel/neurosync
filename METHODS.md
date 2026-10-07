# NeuroSync live EEG study

This application reads real BrainBit BLE headsets (BrainBit Classic/Black, the BrainBit 2/Pro/Flex family and the 21-channel DragonEEG) via the manufacturer's pyneurosdk2 BLE interface. There is no demo/synthetic mode. Test-only fixtures exist exclusively in test modules. It is an exploratory signal viewer, not a medical device, diagnosis, brain map or validated mental-state classifier. The topographic head maps and trend charts below are engineering displays of this session's numbers, not clinical images.

## Use

Run `python main.py` in this folder, using the existing Hermes Python environment, or double-click `Start NeuroSync.cmd`. Only one app should connect to a headset at a time. Close mobile BrainBit apps. Discovery never silently selects another person's headset: the device picker requires an explicit choice, and a saved Classic address is used only to preselect/short-circuit when that exact device is found again. Press Scan devices for one scan that covers every supported family (BrainBit Classic/Black, BrainBit 2/Pro/Flex, DragonEEG); found devices are listed with their name, address and serial, and nothing is connected until one is selected and Connect is pressed. When nothing is powered on the app shows "No devices found" and keeps running — switch a headset on (DragonEEG: put it in pairing mode the first time) and scan again.

On connection the app confirms the SDK-reported sampling rate and channel table, performs a five-second electrode-resistance check, then starts EEG. Resistance and EEG commands are never run simultaneously. A new contact check stops EEG, collects resistance, stops resistance, and restarts EEG. It intentionally clears the old analysis window.

The raw charts show every channel the connected device reports (O1, O2, T3, T4 for the Classic; the device's own channel table for the DragonEEG), in microvolts, without software filtering. The SDK returns volts, converted exactly once by multiplying by 1e6. The colored charts show five frequency components of the selected channel. Select a channel above the plots.

Delta: purple, 1–4 Hz
Theta: blue, 4–8 Hz
Alpha: green, 8–13 Hz
Beta: orange, 13–30 Hz
Gamma: pink, 30–45 Hz

These are operational band definitions, not universal diagnostic boundaries. Delta begins at 1 Hz; frequencies below 1 Hz are deliberately excluded. Gamma means only the 30–45 Hz portion, not the entire gamma range. Dry-electrode movement can inflate delta; jaw/scalp muscle activity can contaminate beta/gamma.

## Devices: Classic, BrainBit 2 and DragonEEG

One transport and one picker serve all supported headsets; the device type is decided by the scanner's SensorFamily, and the type-specific configuration is applied automatically after connecting.

- **BrainBit Classic / Black** (`LEBrainBit`, `LEBrainBitBlack`): four monopolar channels O1, O2, T3, T4 at 250 Hz; volts x1e6 = microvolts; one resistance value per channel. This is the app path that was verified with physical hardware.
- **BrainBit 2 / Pro / Flex** (`LEBrainBit2`, `LEBrainBitPro`, `LEBrainBitFlex`): generic channel packets configured through the BrainBit2 amplifier shape (`ChGain`, `ChSignalMode`, `ChResistUse`). Implemented from the shipped SDK sample; NOT verified on physical hardware by this project.
- **DragonEEG** (`LENeuroEEG`): 250/500/1000 Hz supported, 250 Hz requested. Frequency, ChannelGain and ChannelMode (gain 6, mode EEGChModeSignalResist) are written AFTER connecting and verified by read-back BEFORE StartSignal: an unconfigured DragonEEG streams flat zeros, so a read-back mismatch aborts the connection instead of showing fake data. First pairing requires the physical headset to be in pairing mode; the app only reports the SDK's `PairingRequired` flag and asks the user to perform that step — it never fakes pairing.

DragonEEG channels: the device's own channel table is read after connecting (`supported_channels`, ordered by the device's numbering). The documented channel set is the 21 standard 10-20 scalp sites (Fp1 F7 F3 Fz Fpz Fp2 F8 F4 T3 C3 Cz C4 T4 T5 P3 Pz P4 T6 O1 Oz O2) plus three auxiliary identifiers D1/D2/D3. The auxiliary channels are NOT scalp sites: the manufacturer describes three poly channels (ECG/EMG/EOG), but which identifier is which cannot be established from the SDK identifiers alone, so the app keeps the SDK identifiers and never labels them as a body signal. The documented 21-site list is the fallback used only when the device reports no readable table; anything the app cannot read back from the device is labelled as such in the session summary (`channel_source`).

DragonEEG resistance arrives as a per-channel value in ohms plus A1/A2/Bias reference values; the app displays a value for every channel and the A1/A2/Bias numbers separately. The same <=1 MOhm and 120-second freshness display limits as the Classic apply; resistance alone does not establish signal quality, and EEG and resistance are never acquired at the same time.

The DragonEEG and BrainBit 2 paths are written against the installed SDK and its shipped sample and have NOT been exercised on physical hardware by this project; see Verification for the open items.

## Five-second measurement

Every second, analyze the newest 1250 uninterrupted samples (5 seconds at nominal 250 Hz). This is a rolling, overlapping window, not an independent five-second trial each second. Require all channels to pass the quality checks before displaying an aggregate predominant band. (The 1250-sample figure is for 250 Hz; at other sampling rates the window is the same five seconds of samples.)

Use scipy.signal.welch with Hann window, 500-sample segments (2 seconds), 250-sample overlap, constant detrending, density scaling. Frequency resolution is 0.5 Hz. Optional 50/60 Hz notch, Q=30, is applied to a mean-removed copy before PSD estimation. 60 Hz is the local default; change it to match the measurement environment. The original/raw readings and CSV export are not filtered.

Integrate PSD (microvolts squared per Hz) over exact band edges using trapezoidal integration with edge interpolation. Resulting band powers are microvolts squared. Relative power is each band's percentage of summed 1–45 Hz power. Aggregate power is the mean of channel absolute band powers, not a mean of channel percentages. A top-two difference of at most five percentage points is reported as mixed. Per-channel summaries remain separate.

Colored display components use independent order-four Butterworth bandpasses with zero-phase window filtering. They retain physical microvolt amplitude rather than arbitrary normalization. They have edge effects and are NOT causal filters; do not infer event timing/latency from them. Spectrum calculations use the waveform directly, never the colored band traces.

Five seconds cannot establish a diagnosis, emotion, attention level, sleep stage, or whole-brain state. The largest spectral band is only a description of this short measurement. No sleep/relaxation/focus labels are inferred.

## Stable multi-epoch estimate (more data, artifact-aware)

A second, longer estimate is computed every two seconds from the newest whole 2-second epochs in a 60-second rolling buffer. Per channel, an epoch is excluded when it fails the same engineering artifact heuristics used above (flatline, peak-to-peak, abrupt change, saturation, excess mains power); each epoch's abrupt-change check includes one preceding sample so a step exactly at an epoch boundary rejects the entering epoch instead of escaping detection. Contact flags and known stream discontinuities invalidate the whole estimate, matching the five-second gate.

Band power for each channel is the mean over its clean epochs and is reported with the standard error of that mean (`±SEM`), so more clean data narrows the interval and one movement or blink no longer spreads across a long window. The aggregate is the mean of the channel means and requires every channel to have at least six clean epochs; its SEM combines channel SEMs as sqrt(sum(sem^2))/N. At least 12 seconds of buffer is required before any stable value is shown; the status line reports clean epochs used, seconds used, and per-channel exclusion counts are kept in `runtime_status.json`. Epoch rejection is a heuristic: it can discard real activity, and it cannot certify artifact-free neural data. The estimate is descriptive, not a diagnosis.

## Baseline (per-subject predominant-wave reference)

A baseline is ONE quality-gated measurement of one subject, stored as `recordings/subjects/<subject>/baseline.json` and used afterwards as that subject's own descriptive reference — a "predominant-wave profile". It records the mean absolute band powers (µV²), the mean relative shares (% of 1–45 Hz), the dominant band, the peak frequency of the mean density between 1 and 45 Hz, a per-channel dominant/peak table, the measurement condition the user picked, device, channel source, quality information and the app version. It compares the subject only against their own earlier numbers; it is never a comparison against other people or a normative database. Capturing a new baseline REPLACES the previous baseline of that subject; saved sessions are never touched.

A baseline is a reference, not a placeholder, so both capture paths require a PASSING estimate and record which one was used:

- **From the live window**: set a subject, wait for a passing estimate — the stable multi-epoch estimate is preferred, and a valid five-second window is used only when no stable estimate exists yet — pick the condition and press "Set as baseline".
- **From a saved session** (History → "Set as baseline…"): the selected session's stable estimate is preferred, its five-second analysis is the fallback; the baseline remembers the session it came from, and that session is marked ★ in the History list.

The measurement condition ('eyes closed', 'eyes open' or 'other') is required at capture and stored with the baseline because it changes what the numbers mean: eyes-closed and eyes-open spectra differ substantially. It is displayed with every profile and every delta, and only measurements taken in the same condition should be compared.

Deltas shown for a later estimate are purely descriptive differences of measured numbers (current − baseline):

- per band, the change of the relative share in percentage points (pp), plus the band with the largest absolute change;
- the peak-frequency change in Hz (positive = the current peak is higher);
- the dominant-band change (e.g. Alpha → Theta, or "dominant unchanged").

When a subject has a baseline, each newly saved session's `summary.json` additionally carries a `vs_baseline` block with those numbers (written only when a baseline existed at save time; the baseline's condition is inside the block). The main window shows the profile, one delta line and a per-band delta table, refreshed with each analysis tick; the History dialog shows the comparison line for every listed session of that subject.

Limits, all shown in the UI as well: five seconds is a short window, and a baseline captured from a single five-second window is a weak reference — the app therefore prefers the stable multi-epoch estimate and records which source was used; repeated electrode placements, contact and movement change these numbers too; the main window adds a caution when the baseline used a different channel set or device; incomplete band numbers leave the affected delta blank instead of guessing; and deltas are descriptive numbers of the measured pattern — not a diagnosis, not a medical measurement and not a measurement of mood, attention, emotion, personality or sleep.

One optional interpretive sentence is allowed (the baseline note in the main window). It describes the measured pattern only (e.g. "Alpha-dominant rhythm, peak 10.2 Hz, 43%"), always continues with the fixed caveat "EEG pattern description, not an assessment of the person", and only for an eyes-closed alpha-dominant baseline may add the literature association "in the literature, eyes-closed alpha dominance is commonly associated with relaxed wakefulness". A literature association is allowed only in that exact form, attached to the explicit non-assessment caveat; no other interpretation, prediction, advice or person-level claim is produced anywhere in the app or stored in `baseline.json`.

## AI doctor (local model assist)

"Ask AI doctor" sends ONLY derived numbers from the latest measurements to a local Ollama server at 127.0.0.1:11434 (default model `qwen3.8:latest`, configurable in `ai_doctor.py`): band powers, relative shares, SEMs, per-channel flags and exclusion counts, contact resistances, transport counters and device metadata. Raw EEG samples, recordings and personal identifiers are never included, and nothing is sent to the internet — the request fails with a visible error if the local server is unreachable, and no substitute analysis is generated in that case.

The system prompt hard-limits the reply: no diagnosis, no disease suggestions, no treatment advice, no emotions/attention/sleep/personality inference, no invented values, plain text under 180 words covering data status, what the numbers show, signal-quality suggestions and a limits sentence, ending with a fixed non-diagnostic disclaimer. The reply, model name and UTC timestamp are stored in `runtime_status.json` and exported with the snapshot. Treat every reply as assistive description of engineering numbers, not medical interpretation; a model can still be wrong and its text is not a measurement.

## Visualisation (engineering display)

- **Raw traces**: one small plot per channel in a grid (3–4 columns, so 4 channels and 21+ channels both stay readable), each channel in its own color, microvolts against time relative to now. Each plot draws the newest 5 seconds only, which keeps drawing cost bounded as the channel count grows to the DragonEEG's 24. The channel list is the connected device's own table.
- **Live band bars**: the five relative band shares (percent of summed 1–45 Hz power) of the latest five-second analysis.
- **Spectrum**: the mean Welch PSD curve (µV²/Hz vs Hz) of the newest window, using exactly the estimation parameters of the five-second section.
- **Head map**: a schematic topographic raster per band, built by `topomap.py` from the per-channel band powers. The layout holds APPROXIMATE schematic 2D electrode positions on a unit head (nose up, ears at x = ±1); they are not measured sensor locations, not digitised coordinates and not a head mesh. Interpolation is a multiquadric radial-basis fit on a regular grid, masked outside the head circle (NaN). In the app and in exports the raster is drawn as a **heat map**: a turbo colour ramp with a percent scale bar, electrode dots with name labels, and a "not source localisation" footer on exports; the export interpolates on a 288×288 grid at 1400×1280 logical pixels (the live panel at 160×160), and rows are flipped for a nose-up display so the schematic front sits under the nose marker. It is an engineering sensor-space picture — NOT source localisation, NOT a clinical brain map — and it says nothing about the depth, origin or anatomical location of the activity. A BrainBit Classic map is built from four electrodes and is mostly extrapolation; read it as such. One PNG per band is exported with each saved session.
- **Cross-session trend**: per-band relative power (percent) across the saved sessions of the selected subject, oldest to newest, taken from a session's stable estimate when it has a valid one, else from its five-second analysis. Only sessions with usable numbers for all five bands are plotted; incomplete sessions are shown as a gap (not zero) via the reduced session count in the status line.

## Contact, artifacts and transport

BrainBit recommends investigating contact resistance above 1 MOhm and stresses that resistance alone does not establish signal quality. The app conservatively requires every resistance finite, positive, <=1 MOhm, measured within 120 seconds. That two-minute freshness policy is an app engineering choice, not a manufacturer requirement. Press Check electrodes to refresh it; this pauses acquisition and restarts the analysis window. Allow electrodes to settle, move hair aside and keep the headband correctly positioned, including forehead reference/common contacts.

Additional conservative engineering heuristics (not clinically validated artifact rejection):

- Raw standard deviation <0.2 microvolts: flatline.
- Raw peak-to-peak >500 microvolts: excessive amplitude.
- Adjacent-sample change >100 microvolts: abrupt artifact.
- Absolute raw value >10000 microvolts: saturation/large offset.
- Excess mains-band power: inspect raw PSD before filtering; reject when line-band power exceeds 25% of 1–100 Hz power.
- Nonfinite values, sample discontinuities, missing data and stale data suppress conclusions.

These checks cannot reliably separate all eye, cardiac, muscle or motion artifacts. Passing them does not certify artifact-free neural activity. The monopolar channels share a reference, so common-reference artifacts may affect several channels. There is no claim of validated bipolar artifact cancellation.

Packet counters are tracked modulo 2048. The Classic stream supplies two consecutive samples per packet number, which is not a missing/duplicate packet. Sequence jumps, excess repeats, >0.5-second callback interruptions or queue overflow invalidate the previous window. Missing samples are never interpolated. No sample for >0.5 seconds suppresses a current conclusion; eight seconds without EEG closes the connection. The UI exposes received rate, battery, sample count and discontinuity count. Reconnect is manual and explicitly selects the known headset.

## Local storage

Save latest 5 seconds writes a local `recordings/<timestamp>/raw.csv` and `report.json`. It preserves raw four-channel microvolts, packet numbers, a relative sample-time axis, analysis parameters and quality flags. It does not upload data. Relative sample times are reconstructed from the confirmed nominal sample rate, not hardware absolute timestamps. A saved flagged window is still flagged; it must not be presented as a valid conclusion.

`runtime_status.json` is a local, overwritten health/analysis snapshot used for verification, not a continuous EEG recording. Its monotonic timestamps are meaningful only in the current boot/session. It now also carries the latest `stable_estimate` and the last `ai_doctor` reply. Treat these files and screenshots as personal EEG data.

### Per-subject permanent storage

Before a recording can be saved the user sets the subject (name + phone) in a dialog; the app refuses to save a session without a subject name. Each saved session lives under `recordings/subjects/<slug(name)>_<phone digits or 'nophone'>/<YYYYmmdd_HHMMSS_ffffff>/`:

- `raw.csv` — raw samples of all streamed channels in microvolts with `sample_index`, `relative_time_s` and `packet_number` columns, plus one `raw_<name>_uV` column per channel. It holds the rolling 60-second analysis buffer, NOT the whole session length (recorded honestly in `raw_span_note`).
- `summary.json` — the analysis summary: per-channel absolute and relative band powers, quality metrics and exclusion counts, contact and transport status, the stable estimate, timestamps (the machine's UTC wall clock, not a hardware timestamp), device type/label, channel list with `channel_source`, subject fields and chart names.
- `charts/*.png` — the per-session charts: raw channel traces, band-power bars, spectrum, the current head map and one head map per band (nine PNGs).
- `baseline.json` — the subject's baseline ("predominant-wave profile", see the Baseline section) once one has been captured: written atomically like `summary.json`, replaced on recapture, and it lives in the subject folder, never inside a session folder.

Subject folder tokens are lowercase ASCII slugs (name) and digits (phone); every built path is re-checked to stay inside the recordings root, so subject input cannot escape the folder tree or touch another subject's data. `summary.json` is written atomically (temporary file, fsync, replace), so a crash cannot leave a truncated summary; `raw.csv` is streamed in one pass and a hard crash can leave it short — readers must treat it as recoverable input, not a journal. Reading a session back returns at most 200,000 raw rows and sets `raw_truncated: true` rather than presenting a clipped recording as complete. A flagged window stays flagged after saving.

The History dialog lists saved subjects and their sessions (newest first), reopens one by loading its `summary.json` and showing the saved summary text plus its PNG charts, and can store a selected session as the subject's baseline ("Set as baseline…"): the source session is then marked ★ and, when the subject has a baseline, every listed session carries its descriptive vs-baseline line. New saves never overwrite or delete earlier sessions. Everything stays on this PC: the store performs no network access and nothing is uploaded.

## Verification and limitations

Run `python -m unittest discover -p 'test_*.py' -v` (164 tests; all fixture-based — no hardware, no BLE). The suite covers the channel maps (Dragon 21-site order and fallback, BrainBit 4-channel), device-selection logic with fake scanners, storage save/list/load round-trips for subjects and sessions, topomap interpolation on the expected grid, the band math, the Dragon engine configuration path with fake sensors, the Qt UI logic, and the baseline paths (capture/validation/storage/delta wording, the Qt baseline panel, History marking and the guarded trend redraw). Numerical fixtures are TEST ONLY. Passing software tests does not prove hardware contact or clinical validity.

Hardware-free launch check: `QT_QPA_PLATFORM=offscreen python main.py --smoke` builds the window, keeps it open for a few seconds (default 4 s; `--smoke-seconds N` to change) and exits 0. `--no-autostart` skips the automatic initial scan.

DragonEEG items that require the physical device (the app cannot establish any of these offline):

- first-time pairing flow, including the pairing-mode step, end to end;
- amplifier write + read-back on the real device (250 Hz, gains, modes) and a non-zero stream after StartSignal;
- the device's actual channel table and order versus the documented 21+3 fallback;
- per-channel resistance values and A1/A2/Bias in ohms, and the <=1 MOhm display path;
- 500 / 1000 Hz streaming (the app currently requests 250 Hz);
- BrainBit 2 / Pro / Flex connection and amplifier configuration;
- the volts x1e6 unit assumption on the generic channel packets (the Classic callback contract is known; the generic path assumes the same units).

Live verification separately requires actual device samples, fresh transport status, populated charts and an honest quality outcome. The head-map electrode positions are schematic drawings, not measured coordinates, and none of the charts is a clinical image.

## Sources

- BrainBit hardware, electrode placement, 250 Hz, contact and signal-quality guidance: https://sdk.brainbit.com/device-recommendation
- BrainBit Classic SDK fields, resistance and packet-counter documentation: https://sdk.brainbit.com/sdk2_bb/
- Python SDK commands and callback interfaces: https://pypi.org/project/pyneurosdk2/
- DragonEEG (NeuroEEG) channel identifiers, amplifier parameter block and resistance API: the `neurosdk.neuro_eeg_sensor` module and the sample shipped with pyneurosdk2 (https://pypi.org/project/pyneurosdk2/)
- Welch PSD method and units: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
