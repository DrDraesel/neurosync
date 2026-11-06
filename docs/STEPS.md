# NeuroSync — operator steps (STEPS.md)

Step-by-step operator instructions for every supported headset, in the order the
app drives them through the manufacturer's pyneurosdk2 BLE interface. Read this
together with METHODS.md (what the app computes and why) and the honesty list at
the end of this file.

Honesty first: NeuroSync is not a medical device. It records and describes
measured EEG numbers; nothing it shows is a diagnosis, a clinical measurement or
a mental-state claim.

## 1. Before you start

1. Run the app (`python main.py`, or double-click `Start NeuroSync.cmd`).
2. Only one app may connect to a headset at a time — close mobile BrainBit apps.
3. Set the subject (name + phone) before recording: the app refuses to save a
   session without a subject name.
4. Place the headset/electrodes per the manufacturer's guidance and let the
   electrodes settle before judging contact quality.

## 2. BrainBit Classic / Black — operator steps

1. **Power the headset on.**
2. Press **Scan devices** — one scan covers every supported family. The headset
   appears with its name, address and serial.
3. Select it and press **Connect**. On connect the app verifies the
   SDK-reported **250 Hz** sampling rate and the channel table
   (O1, O2, T3, T4).
4. **Electrode-contact check (optional but recommended):** press
   **Check electrodes**. This pauses EEG, reads one resistance value per
   channel, then restarts EEG. Resistance ≤ 1 MOhm and measured within the last
   120 seconds is the app's engineering gate — resistance alone does not
   establish signal quality.
5. **Stream:** EEG starts automatically after the contact check. Keep still,
   relax the jaw and avoid blinking during a short study.

There is **NO calibration step for the Classic in the pyneurosdk2 interface** —
none exists for this device, and the app does not invent one. The closest the
app does is the on-connect verification of the 250 Hz rate and the channel
table, plus the optional resistance check.

## 3. DragonEEG / NeuroEEG — operator steps

1. **Pairing mode before the first scan:** for the first use, put the physical
   headset into pairing mode, then press **Scan devices**. The app shows the
   SDK's `PairingRequired` flag and asks the operator to perform that step — it
   never fakes pairing.
2. Select the headset and press **Connect**.
3. **Mandatory pre-stream configuration (automatic):** after connecting and
   **before every stream**, the app WRITES the amplifier parameter block
   (sampling frequency, per-channel gain, per-channel mode) AND READS IT BACK.
   An unconfigured DragonEEG streams flat zeros, so a read-back mismatch aborts
   the connection instead of showing fake data. This write + read-back IS the
   mandatory pre-stream configuration step for this device — the operator does
   not perform it; the app performs it and refuses to stream unless the
   read-back matches.
4. **Electrode-contact check (optional):** resistance is reported per channel
   in ohms plus the A1/A2/Bias reference values.
5. **Stream.** The device's own channel table is read after connecting
   (21 scalp sites of the 10-20 set plus auxiliary D1/D2/D3, which are not
   scalp sites and are never treated as EEG).

## 4. BrainBit 2 / Pro / Flex — operator steps

1. **Power on**, **Scan**, select, **Connect**.
2. On connect the app **writes the BrainBit2 amplifier parameter block**
   (channel gain, signal mode, resistance use) in the BrainBit2 shape from the
   shipped SDK sample.
3. Optional electrode-contact check, then **stream**.

(Implemented from the SDK sample; NOT verified on physical hardware by this
project.)

## 5. If the connection drops

1. **Auto-reconnect first:** let the app attempt its automatic reconnect path
   (the status line shows the loss; acquisition stops after ~8 seconds without
   samples).
2. **Then power-cycle:** if the headset does not come back, switch it **off,
   wait, switch it on** (DragonEEG: pairing mode again), press **Scan**, and
   connect to the same known device. Discovery never silently selects another
   person's headset — the app always reconnects to the device the operator
   picks.

## 6. Study-record flow

1. **Set subject** (name + phone) — required before any save.
2. **Start study** — the app records the full duration from start to stop.
3. **Stop** — the session is saved with the true recording span
   (`recorded_span_s`) and the true raw row count; the summary carries the
   honest span, never the rolling 60 s buffer wording.
4. **AI analysis** (optional) — derived numbers only, to a local model or a
   configured API, with its provenance stored and printed with the text.
5. **Patient report** — one self-contained HTML file with every session, the
   charts, the qEEG sheet, the reading-method section for the clinician and the
   honesty banners. Open it in a browser and print to PDF to share.

## 7. What this data can and cannot support

**The numbers can support:**

- a descriptive spectral summary of the recorded sites (band powers, relative
  shares, dominant band);
- within-subject comparisons against that subject's own saved baseline in the
  same condition (eyes closed / eyes open / other);
- documentation of recording conditions and measured signal quality (contact
  resistance, artifact flags, sampling rate).

**The numbers cannot support:**

- a diagnosis, a screening decision or a treatment decision of any kind;
- mental-state, attention, emotion or sleep-stage claims;
- comparison with an age-matched clinical normative database — none is used,
  and no norms are invented;
- a 19-channel qEEG reading from a 4-electrode BrainBit: four sites with a
  shared reference are a screening-grade view, and the four-site map is mostly
  extrapolation;
- a clinical result from a DragonEEG recording: the headset and this app are
  not medical devices, so the output stays a descriptive research recording;
- any conclusion from a single number or a single head — read clusters across
  rows and bands, and check the raw traces first (artifact looks like a
  region).

**Interpretation is clinician-led.** Whether any number is clinically
meaningful is for a qualified clinician to decide by reviewing the original
recording; neither the app nor this document interprets the numbers medically.

## Related docs

- `METHODS.md` — what the app computes, band definitions, quality gates, limits.
- `docs/QEEG-RESEARCH.md` — the qEEG reading-method source (Dr. Trish Leigh's
  channel, `@supernormalbydrtrishleigh`), which the report's "Reading method
  (for the clinician)" section condenses and quotes.
- `docs/QEEG-GRID-SPEC.md` — the 5×5 grid contract (reference block kinds,
  allowed wording).
- `docs/QEEG-NORMS-SPEC.md` — the local-norm rules (never normative below
  n = 20 subjects, age band recorded).
