# qEEG norms + region roll-ups — wave 2 contract

Extends `docs/QEEG-GRID-SPEC.md`. Wave 1 gives us the 5×5 sheet with `session | baseline | norm | none` references.
This wave makes `norm` real, in the only two honest ways available, and adds the reading aids that make the sheet clinically usable.

## 1. `norms.py` — two sources of reference, never mixed

```python
@dataclass
class NormFile:
    schema: str            # "lab.norm/1"
    source: str            # "local" | "neuroguide" | "bsi" | "generic"
    label: str
    n: int
    age_band: str          # "18-25", "26-35", … required
    condition: str         # "eyes_closed" | "eyes_open" | "other"
    device_scope: str      # "19ch" | "21ch" | "4ch" | "any"
    created_at: str
    rows: dict             # {row: {band: {channel|pair: {"mean": float, "sd": float}}}}
    provenance: dict       # file name/hash, exporter, notes
```

### 1a. Import a licensed export (NeuroGuide / Brain Science / generic CSV)

`import_csv(path, source="generic", age_band=..., condition=..., device_scope=...) -> NormFile`

Accept three shapes, detect by header:
- **long**: `channel,band,row,mean,sd` (preferred)
- **wide-mean-sd**: `channel,band,absolute_mean,absolute_sd,relative_mean,relative_sd,…`
- **z-only**: `channel,band,row,z` → **refused**: keep `mean`, `sd`; a z alone cannot seed a reference. The importer must raise `ValueError("z-only exports cannot seed a reference — export mean and SD")`.

Rules: channels not in the 10-20 montage are dropped with a note; a band/row with <1 channel is dropped; every imported file keeps its hash in `provenance`; the UI shows `source` verbatim and **never** relabels an import as local.

### 1b. Accumulate a local norm from our own recordings

`build_local_norm(root, min_n=20, condition=..., age_band=...) -> NormFile`

- Scans `recordings/subjects/*/*/` for summaries with `valid == True`, a stored `condition`, a stored `age`/`age_band`, and a device whose scalp count matches `device_scope`.
- Recomputes `qeeg.compute_metrics` from the stored `raw_*.csv` (never from cached numbers of unknown version) — stamp `qeeg_version` in provenance.
- Aggregates **across subjects** (one row per subject; mean of subject values, SD across subjects). A subject with two sessions in the same condition contributes the mean of those sessions, not two rows.
- Refuses to build with `n < min_n` → returns `None` + reason string. No "almost a norm".

### 1c. Guard rails

- `load()` validates schema, `n`, `age_band`, `source`; refuses a file whose `source == "local"` but lacks `provenance.subjects`.
- The grid footer must print: `reference: <source> · n=<n> · age <age_band> · <condition>`.
- A norm whose `device_scope` does not match the current device may still be used, but the footer appends `· device scope mismatch (21ch norm on 4ch recording)` and the panel shows an amber badge.

## 2. Region roll-ups

`regions.py` (or a function in `qeeg.py`): the ten multivariate regions used in qEEG reading:

```
left_lateral   F7 T3 T5        right_lateral  F8 T4 T6
left_medial    Fp1 F3 C3 P3 O1 right_medial   Fp2 F4 C4 P4 O2
left_anterior  Fp1 F7 F3       right_anterior Fp2 F8 F4
left_central   T3 C3           right_central  T4 C4
left_posterior T5 P3 O1        right_posterior T6 P4 O2
midline        Fz Cz Pz
```

`region_table(grid)` → for each band: mean z per region, plus a `flag` when `|mean z| ≥ 2` and `n_sites ≥ 2`. Render as a compact table under the 5×5 grid, sorted by |z| desc. Regions with <2 available sites are omitted, never filled from neighbours.

## 3. Eyes-closed / eyes-open pairing

`pair_conditions(session_ec, session_eo)` → renders **two 5×5 sheets side by side** plus:
- a delta strip per band/row (`EO z − EC z`), same colour ramp, clamped;
- a header line stating both session ids, conditions, devices and references;
- a refusal if the two sessions come from different devices or different references (mixing them is meaningless).

UI: `qEEG grid` tab gets a **Compare EC/EO…** button that opens two pickers over saved sessions (filtered to the same subject), then shows the paired view. Export writes a single PNG with both sheets.

## 4. Theta/beta card

Under the grid: for each channel, `theta_beta = Theta absolute / Beta absolute`, plus the channel mean. Print it with the reference note from the source teaching: *"a theta/beta ratio near 3:1 is the pattern Dr. Leigh names for ADHD — one number is not a diagnosis; read it with the rest of the sheet."* Flag only when the mean is ≥3.0 AND both bands are usable.

## 5. Raw-trace reminder

Above the grid, a one-line reminder with a link to the raw tab: *"Check the raw traces before trusting a map — artifact looks like a region."* It must be visible, not a tooltip.

## Tests

- `test_norms.py`: long-format import builds a NormFile with correct n/mean/sd; z-only import raises; a local build from 19 synthetic subjects returns None + reason; from 20 returns a file whose `provenance.subjects` has 20 entries; a `source=local` file without subjects is refused on load; device-scope mismatch is reported.
- `test_regions.py`: a synthetic grid with F3/C3/P3 at z=2.5 makes `left_medial` flagged at ≥2; a region with 1 site is omitted.
- `test_qeeg_pairing.py`: EC/EO pairing refuses different devices; delta strip is `EO − EC`; export writes one PNG.
- `test_qeeg_ui.py` (extend): TBR card present, raw-trace reminder present, region table present.

## Do not

- Do not ship any pre-baked norm values in the repo.
- Do not let an imported norm silently become "normative" in the report — source string is printed verbatim.
- Do not compute a local norm from fewer than 20 subjects, and never from simulated sessions (`simulated: true` is excluded).
