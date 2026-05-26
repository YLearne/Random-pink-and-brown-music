# Noise Music Generation & Statistical Analysis

A controlled experiment comparing 1/f (pink) and 1/f² (brown) noise as the basis for random melody generation. Part of a *Music & Mathematics* course assignment.

---

## Experiment Purpose & Design

We generate random melodies by mapping two different noise processes onto pitch and rhythm, then compare their statistical properties:

| Noise type | Spectral law | Algorithm |
|---|---|---|
| Pink noise | PSD ∝ 1/f, slope α ≈ −1 | Voss–McCartney (binary counter, 16 sources) |
| Brown noise | PSD ∝ 1/f², slope α ≈ −2 | Cumulative random walk with reflective boundaries |

**Single-variable control**: pitch uses the noise type's own signal; duration also uses a separate but same-type noise stream (pink→pink, brown→brown). Pitch and rhythm use different seeds so they are statistically independent.

**Reproducibility**: all randomness is seeded from a fixed global seed (666). No parameter in `generator.py` should be changed.

**Dual-track architecture**: two independent data streams share the same `(seed_pitch, seed_dur)` pair but are generated at different lengths. Short sequences (~400-note upper bound, truncated to exactly 16 bars) are used for qualification checks and short-segment statistics. Long sequences (1000 notes, same seeds) are used for power spectral density and ACF — they are never truncated. This separation prevents brown noise's global-std normalization from compressing the pitch range of short excerpts.

---

## 简谱 (Jianpu) Notation Convention

| Symbol | Meaning |
|---|---|
| `1 2 3 4 5 6 7` | C D E F G A B (C-major, movable do) |
| `#1`, `b3` | Sharps/flats (accidental prefix) |
| `,1` | One octave below middle |
| `1` | Middle octave |
| `1'` | One octave above middle |
| `-` (suffix) | Half note (2 beats) |
| *(none)* (suffix) | Quarter note (1 beat) |
| `_` (suffix) | Eighth note (0.5 beats) |
| `=` (suffix) | Sixteenth note (0.25 beats) |
| ` \| ` | Bar line |

Example: `#4_ ,7 3- | 5 2' =`  means  F♯ eighth note (low octave), B quarter (low), E half note, bar line, G quarter (middle), D quarter (high octave), sixteenth rest placeholder.

Output files display 4 bars per line.

---

## Qualification Criteria

Each 16-bar melody segment is classified as **qualified** if it simultaneously satisfies:

1. **Pitch range ≥ 24 semitones** (spans at least 2 full octaves)
2. **All 4 duration types present**: must include half (2.0), quarter (1.0), eighth (0.5), and sixteenth (0.25) notes
3. **Length ≥ 16 bars** (guaranteed by construction)

These criteria mirror the course assignment's requirements for a valid musical excerpt. Brown noise typically fails criterion 1 because its short-term clustering keeps the pitch within a narrow register over 16 bars.

---

## How to Run

```bash
# Install dependencies (Python 3.9+)
pip install -r requirements.txt

# Run the full experiment (one command)
python -m src.main
```

This takes ~1–2 minutes and produces all output files.

---

## Output Files

```
output/
├── pink_demo.txt        Human-readable jianpu score (16 bars, pink noise)
├── pink_demo.csv        Note-level data: idx, bar, midi, jianpu, dur_beats, dur_symbol
├── brown_demo.txt       Same for brown noise
├── brown_demo.csv
├── stats_report.txt     Full statistical report (qualification rates, α values, t-tests)
└── figures/
    ├── fig1_pitch_dist.png       MIDI pitch histogram (all vs qualified)          [short seqs]
    ├── fig2_duration_dist.png    Duration type counts                              [short seqs]
    ├── fig3_interval_dist.png    |Δmidi| interval distribution                     [short seqs]
    ├── fig4_acf.png              Mean autocorrelation function (lag 0–50)          [long seqs]
    ├── fig5_psd.png              ★ Power spectral density, log-log, with fitted α  [long seqs]
    ├── fig6_run_length.png       Monotone run length distribution                  [long seqs]
    ├── fig7_range_dist.png       Pitch range distribution (24-semitone threshold)  [short seqs]
    └── fig8_duration_variety.png Duration variety (1–4 types) per melody           [short seqs]
```

In each figure, **left subplot = pink**, **right subplot = brown**. Solid/dark = all 100 trials; dashed/light = qualified subset.

---

## Key Parameters

| Parameter | Value | Note |
|---|---|---|
| `RANDOM_SEED` | 666 | **Do not change** — fixes all experiment results |
| `MIDI_LOW / MIDI_HIGH` | 48 / 72 | C3 – C5, 25 semitones total |
| `N_BARS_DEMO` | 16 | Demo melody length |
| `N_TRIALS` | 1000 | Large-sample experiment size |
| `N_NOTES_PER_TRIAL` | 1000 | Notes per trial (for PSD estimation) |
| `DURATIONS` | [2.0, 1.0, 0.5, 0.25] | Permitted note durations in beats |

Seed range: large-sample 1000 trials use pitch=1666+2i, dur=pitch+1 (i=0..999), covering seeds 1666–3665. Demo melodies are picked as the first qualified trial within the large sample — no separate demo seeds.

---

## Module Overview

| File | Role |
|---|---|
| `src/generator.py` | **Paper display file.** Noise generators + pitch/rhythm mapping. Every non-blank line has a Chinese comment. |
| `src/notation.py` | MIDI → jianpu conversion; score text and CSV export. |
| `src/stats.py` | Statistical analysis: PSD slope, ACF, run lengths, qualification checks, 8 figures, text report. |
| `src/main.py` | Orchestrates the full experiment pipeline. |

---

## Expected Results

- Pink α ≈ −1.03 (1/f spectrum preserved through tanh+quantize mapping)
- Brown α ≈ −1.63 (steeper than pink; tanh compression attenuates the full −2 slope on MIDI integers)
- Brown ρ(1) ≈ 0.99 vs Pink ρ(1) ≈ 0.76, confirming stronger pitch persistence in brown melodies
- Qualification rate: Pink ~7%, Brown ~0% (brown's local clustering prevents spanning 24 semitones in 16 bars)
