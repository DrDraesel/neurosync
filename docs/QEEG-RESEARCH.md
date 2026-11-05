# qEEG research notes — what the source channel actually teaches

Source: **@supernormalbydrtrishleigh** (Dr. Trish Leigh, Leigh Brain & Spine), reviewed 2026-11-03.
Purpose: ground the NeuroSync qEEG grid in the same reading method the reference sheet uses, so the app and the clinician speak the same language.

## Videos used

| ID | Title | What it contributes |
|---|---|---|
| `7VkppFW2yyw` | How to Interpret Your qEEG Brain Map | The row/column layout and the colour rule (the reference sheet in the screenshot) |
| `IQRr0P5fSYo` | Brain Map Interpretation — 19 Channel | Band semantics, eyes-closed vs eyes-open, what each band drives |
| `vyAPl6kULks` | Brain Map Interpretation — Low Beta | Worked example: theta/beta ratio, low beta, alpha excess |
| `PeuFXQD-3dg` | What a "Normal" qEEG Brain Map Looks Like | Age-matched normative database, "all green" as optimal, interpolation from 19 sites |
| `AXrP85phCYQ` | What Does the ADHD Brain Map Look Like? | The ADHD pattern: excess theta, low beta, TBR ≈ 3:1 |
| `aJLnTMyhyOQ` | What information does it give you? | Slow excess → attention/memory/sleep; fast excess → anxiety/OCD/sensory |
| `8pb7Kd7WjRA` | qEEG Brain Map Session for NFB-X | Session workflow and how findings are discussed |

## The reading method (as taught)

**Five bands = five "speeds", each with a state it serves**

| Band | Range used | State when optimal | When excessive |
|---|---|---|---|
| Delta | 1–4 Hz | restorative sleep | slow processing, fog, burnout pattern |
| Theta | 4–8 Hz | drowsy, shifting into sleep | the classic ADHD marker (with low beta) |
| Alpha | 8–13 Hz | calm, relaxed | under-arousal, "stuck on the couch" |
| Beta | 13–30 Hz | focused thinking | tension, over-thinking |
| High Beta | 30–45 Hz | handling stress | anxiety, hypervigilance |

**Rows = what is being measured**

1. **Absolute power** — how "charged" that band is at that site (µV²).
2. **Relative power** — that band's share of the total; which speed dominates.
3. **Amplitude asymmetry** — left vs right at homologous sites.
4. **Coherence** — how similarly two sites behave in that band (shared information; too much = rigidity, too little = poor communication).
5. **Phase lag** — the timing offset between sites (late/early arrival).

**Colour rule on the reference sheet:** green = at the reference, blue = below, yellow → orange → red = above (1, 2, 3 levels). The sheet's legend is a z-scale from −3 to +3. The sheet is generated against an **age-matched normative database**, typically shown twice: eyes closed and eyes open, and in more than one montage (linked ears, Laplacian).

**Patterns she names** (educational, not diagnostic):
- ADHD: high theta + low beta, **theta/beta ratio ≈ 3:1**.
- Anxiety/OCD: excess fast activity (beta/high beta), poor sensory gating.
- Under-arousal/depression pattern: excess alpha, low beta.
- Read **clusters** across rows and bands, never a single head.

## What NeuroSync now does — and what it must not claim

Implemented: all five rows, all five bands, z-coloured heads, selectable reference, condition field, exportable sheet, report section.

**Not implemented, and must not be implied:** a normative database. NeuroGuide-class z-scores require a licensed, age-stratified norm set. Without it the app compares against:
- `session` — the channel mean of this recording (self-referenced),
- `baseline` — a saved baseline of the same subject and condition,
- `norm` — a **local** norm file, only valid once ≥20 subjects exist, with the age band recorded.

Every grid and every report line prints which of these was used. The words "NeuroGuide", "normative clinical database" and any diagnostic claim are forbidden in the UI and the report.

## Device capability, honestly

| | BrainBit Classic/Black/2/Pro/Flex | DragonEEG (NeuroEEG) |
|---|---|---|
| Scalp sites | 4 (O1, O2, T3, T4) | 21 (full 10-20) |
| Bands / absolute / relative power | yes | yes |
| Interpolation map | 4 sites → mostly extrapolation; label it | 21 sites → the real grid |
| Asymmetry pairs | 2 (O1/O2, T3/T4) | 9 homologous pairs |
| Coherence / phase pairs | 6 pairs max | ~200 pairs |
| Verdict | screening view only, never a 19-channel map | the device the 5×5 sheet is designed for |

This is the practical answer to "which headset": the **DragonEEG (more electrodes) is the one that can actually fill the sheet**; the BrainBit gives a coarse four-site view that must be labelled as such.

## Follow-ups worth building next

1. **Eyes-closed / eyes-open pairing** — two recordings, two sheets side by side, since the method always reads both.
2. **Theta/beta ratio card** — the one number she names explicitly; we already compute it in `qeeg.ratios()`.
3. **Local norm accumulation** — every saved session with a condition and age band feeds a local norm file; the app refuses to call it normative until n ≥ 20.
4. **Region roll-ups** — left/right medial/lateral/posterior means (the multivariate regions in the BSI reading guide), so the clinician sees a region table under the grid.
5. **Raw-trace check before believing a map** — the app already stores raw CSV; add a "review raw first" reminder in the panel.
