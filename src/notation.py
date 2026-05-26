# =============================================================================
# notation.py — MIDI-to-jianpu conversion and melody export utilities
# =============================================================================

import csv
import os

from .generator import DUR_SYMBOLS, DURATIONS

# ---- pitch class mapping (C major, first-degree C) --------------------------

PITCH_MAP = {
    0: "1", 1: "#1", 2: "2", 3: "#2", 4: "3",
    5: "4", 6: "#4", 7: "5", 8: "#5", 9: "6",
    10: "#6", 11: "7",
}


def midi_to_jianpu(midi: int) -> str:
    """Convert a MIDI integer to a jianpu (numbered musical notation) string.

    MIDI 60 = C4 = "1" (middle C).  Uses sharps for accidentals.
    """
    pitch_class = midi % 12
    octave_offset = midi // 12 - 5
    base = PITCH_MAP[pitch_class]
    if octave_offset < 0:
        return "," * abs(octave_offset) + base
    elif octave_offset > 0:
        return base + "'" * octave_offset
    else:
        return base


def _nearest_dur_symbol(dur_beats: float) -> str:
    """Return the DUR_SYMBOLS value whose key is closest to *dur_beats*."""
    nearest_key = min(DUR_SYMBOLS.keys(), key=lambda k: abs(k - dur_beats))
    return DUR_SYMBOLS[nearest_key]


def melody_to_jianpu_text(melody: list, bars_per_line: int = 4) -> str:
    """Convert a melody (with ``bar`` field set) to a formatted jianpu string.

    Parameters
    ----------
    melody : list of dict
        Note dictionaries as produced by ``truncate_to_bars``.
    bars_per_line : int
        Number of bars after which a newline is inserted.
    """
    # group notes by bar number
    bars: dict[int, list[dict]] = {}
    max_bar = 0
    for note in melody:
        b = note["bar"]
        max_bar = max(max_bar, b)
        bars.setdefault(b, []).append(note)

    result_parts: list[str] = []
    for b in range(1, max_bar + 1):
        # build the symbol string for each note in this bar
        note_symbols: list[str] = []
        for note in bars.get(b, []):
            jp = midi_to_jianpu(note["midi"])
            dur_beats = note["dur_beats"]
            if dur_beats in DUR_SYMBOLS:
                ds = DUR_SYMBOLS[dur_beats]
            else:
                ds = _nearest_dur_symbol(dur_beats)
            note_symbols.append(jp + ds)
        bar_str = " ".join(note_symbols)
        result_parts.append(bar_str)
        if b != max_bar:
            result_parts.append(" | ")
        if b % bars_per_line == 0 and b != max_bar:
            result_parts.append("\n")

    result_parts.append(" |")
    return "".join(result_parts)


def melody_to_csv(melody: list, path: str):
    """Export a melody to a CSV file.

    Columns: idx, bar, midi, jianpu, dur_beats, dur_symbol.
    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = ["idx", "bar", "midi", "jianpu", "dur_beats", "dur_symbol"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for note in melody:
            jp = midi_to_jianpu(note["midi"])
            dur_beats = note["dur_beats"]
            if dur_beats in DUR_SYMBOLS:
                ds = DUR_SYMBOLS[dur_beats]
            else:
                ds = _nearest_dur_symbol(dur_beats)
            writer.writerow({
                "idx": note["idx"],
                "bar": note["bar"],
                "midi": note["midi"],
                "jianpu": jp,
                "dur_beats": dur_beats,
                "dur_symbol": ds,
            })
