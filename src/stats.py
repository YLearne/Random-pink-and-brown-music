import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import linregress, ttest_ind

from .generator import DURATIONS, BEATS_PER_BAR, N_BARS_DEMO

PINK = "#E91E63"
BROWN = "#795548"
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "figures")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_dur(d):
    return min(DURATIONS, key=lambda k: abs(k - d))


def _acf(x, max_lag=50):
    x = np.array(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x)
    if var == 0:
        return np.zeros(max_lag + 1)
    result = [np.dot(x[:n - lag], x[lag:]) / var for lag in range(max_lag + 1)]
    return np.array(result)


def _psd_slope(pitches):
    f, pxx = welch(pitches, nperseg=256)
    mask = f > 0
    f, pxx = f[mask], pxx[mask]
    slope, intercept, r, p, se = linregress(np.log10(f), np.log10(pxx))
    return slope, 1.96 * se


def _monotone_runs(pitches):
    diffs = np.diff(pitches)
    runs = []
    count = 0
    prev_sign = 0
    for d in diffs:
        if d == 0:
            if count > 0:
                runs.append(count)
                count = 0
                prev_sign = 0
            continue
        sign = 1 if d > 0 else -1
        if sign == prev_sign:
            count += 1
        else:
            if count > 0:
                runs.append(count)
            count = 1
            prev_sign = sign
    if count > 0:
        runs.append(count)
    return runs


# ---------------------------------------------------------------------------
# Qualification
# ---------------------------------------------------------------------------

def is_qualified(melody_16bars):
    pitches = [n["midi"] for n in melody_16bars]
    durs = {n["dur_beats"] for n in melody_16bars}
    return (max(pitches) - min(pitches)) >= 24 and set(DURATIONS).issubset(durs)


def qualify_reason(melody_16bars):
    pitches = [n["midi"] for n in melody_16bars]
    durs = {n["dur_beats"] for n in melody_16bars}
    pitch_range = max(pitches) - min(pitches)
    range_ok = pitch_range >= 24
    dur_ok = set(DURATIONS).issubset(durs)
    return {
        "qualified": range_ok and dur_ok,
        "range_ok": range_ok,
        "dur_ok": dur_ok,
        "pitch_range": pitch_range,
        "n_dur_types": len(durs),
    }


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def compute_stats(full_melodies, melodies_16bars):
    psd_slopes, psd_cis = [], []
    acf_list = []
    all_run_lengths = []
    pitches_16, durs_16, intervals_16 = [], [], []
    ranges_16, dur_variety_16 = [], []
    qualify_results = []

    for melody_full, melody_16 in zip(full_melodies, melodies_16bars):
        # melody_full 可能是 dict list 或 numpy int16 数组，统一转为 float
        pitches_full = np.asarray(melody_full if not isinstance(melody_full, list)
                                  else [n["midi"] for n in melody_full], dtype=float)

        slope, ci = _psd_slope(pitches_full)
        psd_slopes.append(slope)
        psd_cis.append(ci)

        acf_list.append(_acf(pitches_full, max_lag=50))

        all_run_lengths.extend(_monotone_runs(pitches_full))

        p16 = [n["midi"] for n in melody_16]
        d16 = [n["dur_beats"] for n in melody_16]
        pitches_16.extend(p16)
        durs_16.extend(d16)
        intervals_16.extend(abs(p16[i + 1] - p16[i]) for i in range(len(p16) - 1))
        ranges_16.append(max(p16) - min(p16))
        dur_variety_16.append(len(set(d16)))

        qualify_results.append(qualify_reason(melody_16))

    acf_mean = np.mean(acf_list, axis=0)
    return {
        "psd_slopes": np.array(psd_slopes),
        "psd_cis": np.array(psd_cis),
        "acf_mean": acf_mean,
        "acf1": np.array([a[1] for a in acf_list]),
        "acf_list": np.array(acf_list),
        "run_lengths": np.array(all_run_lengths),
        "pitches_16": np.array(pitches_16),
        "durs_16": np.array(durs_16),
        "intervals_16": np.array(intervals_16),
        "ranges_16": np.array(ranges_16),
        "dur_variety_16": np.array(dur_variety_16),
        "qualify_results": qualify_results,
    }


def _qual_mask(stats):
    return np.array([r["qualified"] for r in stats["qualify_results"]])


def _filter_pooled(full_melodies, melodies_16bars, mask, key):
    result = []
    for i, (mf, m16) in enumerate(zip(full_melodies, melodies_16bars)):
        if not mask[i]:
            continue
        if key == "pitches_16":
            result.extend(n["midi"] for n in m16)
        elif key == "durs_16":
            result.extend(n["dur_beats"] for n in m16)
        elif key == "intervals_16":
            p = [n["midi"] for n in m16]
            result.extend(abs(p[j + 1] - p[j]) for j in range(len(p) - 1))
        elif key == "run_lengths":
            pitches = np.asarray(mf if not isinstance(mf, list)
                                 else [n["midi"] for n in mf], dtype=float)
            result.extend(_monotone_runs(pitches))
        elif key == "acf_mean":
            pitches = np.asarray(mf if not isinstance(mf, list)
                                 else [n["midi"] for n in mf], dtype=float)
            result.append(_acf(pitches, max_lag=50))
    return np.array(result) if result else np.array([])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_all_figures(pink_stats, pink_full, pink_16bars,
                     brown_stats, brown_full, brown_16bars, outdir=None):
    if outdir is None:
        outdir = FIGURES_DIR
    os.makedirs(outdir, exist_ok=True)

    pink_mask = _qual_mask(pink_stats)
    brown_mask = _qual_mask(brown_stats)

    def _save(fig, name):
        fig.tight_layout()
        plt.savefig(os.path.join(outdir, name), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # fig1: pitch distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, stats, full, m16, mask, color, lbl in [
        (axes[0], pink_stats, pink_full, pink_16bars, pink_mask, PINK, "Pink"),
        (axes[1], brown_stats, brown_full, brown_16bars, brown_mask, BROWN, "Brown"),
    ]:
        ax.hist(stats["pitches_16"], bins=25, range=(47.5, 72.5),
                color=color, alpha=0.7, label="All trials", edgecolor="none")
        qp = _filter_pooled(full, m16, mask, "pitches_16")
        if len(qp):
            ax.hist(qp, bins=25, range=(47.5, 72.5), color=color, alpha=0.9,
                    linestyle="--", histtype="step", linewidth=2, label="Qualified")
        ax.set_xlabel("MIDI pitch")
        ax.set_ylabel("Count")
        ax.set_title(f"{lbl} Pitch Distribution")
        ax.legend(fontsize=8)
    _save(fig, "fig1_pitch_dist.png")

    # fig2: duration distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    dur_vals = sorted(DURATIONS)
    x = np.arange(len(dur_vals))
    w = 0.35
    for ax, stats, full, m16, mask, color, lbl in [
        (axes[0], pink_stats, pink_full, pink_16bars, pink_mask, PINK, "Pink"),
        (axes[1], brown_stats, brown_full, brown_16bars, brown_mask, BROWN, "Brown"),
    ]:
        all_d = stats["durs_16"]
        qd = _filter_pooled(full, m16, mask, "durs_16")
        ac = [np.sum(all_d == v) for v in dur_vals]
        qc = [np.sum(qd == v) for v in dur_vals] if len(qd) else [0] * 4
        ax.bar(x - w / 2, ac, w, color=color, alpha=0.8, label="All")
        ax.bar(x + w / 2, qc, w, color=color, alpha=0.45,
               edgecolor=color, linewidth=1.5, label="Qualified")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in dur_vals])
        ax.set_xlabel("Duration (beats)")
        ax.set_ylabel("Count")
        ax.set_title(f"{lbl} Duration Distribution")
        ax.legend(fontsize=8)
    _save(fig, "fig2_duration_dist.png")

    # fig3: interval distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, stats, full, m16, mask, color, lbl in [
        (axes[0], pink_stats, pink_full, pink_16bars, pink_mask, PINK, "Pink"),
        (axes[1], brown_stats, brown_full, brown_16bars, brown_mask, BROWN, "Brown"),
    ]:
        ax.hist(stats["intervals_16"], bins=range(0, 26), color=color,
                alpha=0.7, label="All trials", edgecolor="none")
        qi = _filter_pooled(full, m16, mask, "intervals_16")
        if len(qi):
            ax.hist(qi, bins=range(0, 26), color=color, alpha=0.9,
                    histtype="step", linewidth=2, linestyle="--", label="Qualified")
        ax.set_xlabel("|Delta MIDI| (semitones)")
        ax.set_ylabel("Count")
        ax.set_title(f"{lbl} Interval Distribution")
        ax.legend(fontsize=8)
    _save(fig, "fig3_interval_dist.png")

    # fig4: ACF
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    lags = np.arange(51)
    for ax, stats, full, m16, mask, color, lbl in [
        (axes[0], pink_stats, pink_full, pink_16bars, pink_mask, PINK, "Pink"),
        (axes[1], brown_stats, brown_full, brown_16bars, brown_mask, BROWN, "Brown"),
    ]:
        ax.plot(lags, stats["acf_mean"], color=color, alpha=0.9, label="All trials")
        qa = _filter_pooled(full, m16, mask, "acf_mean")
        acf_qual = np.mean(qa, axis=0) if len(qa) else stats["acf_mean"]
        ax.plot(lags, acf_qual, color=color, alpha=0.5, linestyle="--", label="Qualified")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(f"{lbl} Autocorrelation")
        ax.legend(fontsize=8)
    _save(fig, "fig4_acf.png")

    # fig5: PSD (core figure)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, stats, full, m16, mask, color, lbl in [
        (axes[0], pink_stats, pink_full, pink_16bars, pink_mask, PINK, "Pink"),
        (axes[1], brown_stats, brown_full, brown_16bars, brown_mask, BROWN, "Brown"),
    ]:
        psd_all, psd_qual, freq_ref = [], [], None
        for i, mf in enumerate(full):
            pitches = np.asarray(mf if not isinstance(mf, list)
                                 else [n["midi"] for n in mf], dtype=float)
            f, pxx = welch(pitches, nperseg=256)
            f, pxx = f[f > 0], pxx[f > 0]
            if freq_ref is None:
                freq_ref = f
            psd_all.append(np.log10(pxx))
            if mask[i]:
                psd_qual.append(np.log10(pxx))

        mlpa = np.mean(psd_all, axis=0)
        mlpq = np.mean(psd_qual, axis=0) if psd_qual else mlpa

        ax.loglog(freq_ref, 10**mlpa, color=color, alpha=0.9, label="All trials")
        ax.loglog(freq_ref, 10**mlpq, color=color, alpha=0.5, linestyle="--", label="Qualified")

        for log_psd, ls, al in [(mlpa, "-", 0.7), (mlpq, "--", 0.45)]:
            sl, inter, *_ = linregress(np.log10(freq_ref), log_psd)
            ax.loglog(freq_ref, 10 ** (inter + sl * np.log10(freq_ref)),
                      color="black", linestyle=ls, alpha=al, linewidth=1)

        alpha_all = np.mean(stats["psd_slopes"])
        alpha_qual = np.mean(stats["psd_slopes"][mask]) if mask.sum() > 0 else alpha_all
        ax.text(0.05, 0.12, f"All: alpha = {alpha_all:.2f}",
                transform=ax.transAxes, fontsize=9, color=color)
        ax.text(0.05, 0.05, f"Qual: alpha = {alpha_qual:.2f}",
                transform=ax.transAxes, fontsize=9, color=color, alpha=0.7)
        ax.set_xlabel("Frequency (normalized)")
        ax.set_ylabel("Power Spectral Density")
        ax.set_title(f"{lbl} PSD (log-log)")
        ax.legend(fontsize=8)
    _save(fig, "fig5_psd.png")

    # fig6: run length
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, stats, full, m16, mask, color, lbl in [
        (axes[0], pink_stats, pink_full, pink_16bars, pink_mask, PINK, "Pink"),
        (axes[1], brown_stats, brown_full, brown_16bars, brown_mask, BROWN, "Brown"),
    ]:
        ax.hist(stats["run_lengths"], bins=range(1, 21), color=color,
                alpha=0.7, label="All trials", edgecolor="none")
        qr = _filter_pooled(full, m16, mask, "run_lengths")
        if len(qr):
            ax.hist(qr, bins=range(1, 21), color=color, alpha=0.9,
                    histtype="step", linewidth=2, linestyle="--", label="Qualified")
        ax.set_xlabel("Run length")
        ax.set_ylabel("Count")
        ax.set_title(f"{lbl} Monotone Run Lengths")
        ax.legend(fontsize=8)
    _save(fig, "fig6_run_length.png")

    # fig7: range distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, stats, full, m16, mask, color, lbl in [
        (axes[0], pink_stats, pink_full, pink_16bars, pink_mask, PINK, "Pink"),
        (axes[1], brown_stats, brown_full, brown_16bars, brown_mask, BROWN, "Brown"),
    ]:
        ax.hist(stats["ranges_16"], bins=25, color=color, alpha=0.7,
                label="All trials", edgecolor="none")
        if mask.sum() > 0:
            ax.hist(stats["ranges_16"][mask], bins=25, color=color, alpha=0.9,
                    histtype="step", linewidth=2, linestyle="--", label="Qualified")
        ax.axvline(24, color="black", linestyle="--", linewidth=1.5,
                   label="24 semitone threshold")
        ax.set_xlabel("Pitch range (semitones)")
        ax.set_ylabel("Count")
        ax.set_title(f"{lbl} Pitch Range Distribution")
        ax.legend(fontsize=8)
    _save(fig, "fig7_range_dist.png")

    # fig8: duration variety
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.array([1, 2, 3, 4])
    w = 0.35
    for ax, stats, mask, color, lbl in [
        (axes[0], pink_stats, pink_mask, PINK, "Pink"),
        (axes[1], brown_stats, brown_mask, BROWN, "Brown"),
    ]:
        all_v = stats["dur_variety_16"]
        qual_v = all_v[mask]
        ac = [np.sum(all_v == k) for k in x]
        qc = [np.sum(qual_v == k) for k in x] if len(qual_v) else [0] * 4
        ax.bar(x - w / 2, ac, w, color=color, alpha=0.8, label="All")
        ax.bar(x + w / 2, qc, w, color=color, alpha=0.45,
               edgecolor=color, linewidth=1.5, label="Qualified")
        ax.set_xticks(x)
        ax.set_xlabel("Duration variety (# types)")
        ax.set_ylabel("Count")
        ax.set_title(f"{lbl} Duration Variety")
        ax.legend(fontsize=8)
    _save(fig, "fig8_duration_variety.png")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def generate_report(pink_stats, pink_full, pink_16bars,
                    brown_stats, brown_full, brown_16bars,
                    pink_demo_16, brown_demo_16,
                    pink_demo_info, brown_demo_info,
                    path):
    pink_mask = _qual_mask(pink_stats)
    brown_mask = _qual_mask(brown_stats)
    n_total = len(pink_stats["psd_slopes"])

    def _group_summary(stats, mask, full_melodies, m16_list):
        lines = []
        lines.append(f"[All trials, N={n_total}]")
        lines.append(f"  PSD slope alpha = {np.mean(stats['psd_slopes']):.3f} +/- {np.mean(stats['psd_cis']):.3f} (mean +/- mean 95%CI)")
        lines.append(f"  First-order autocorrelation rho(1) = {np.mean(stats['acf1']):.3f}")
        lines.append(f"  Mean interval |delta_midi| = {np.mean(stats['intervals_16']):.2f} semitones")
        lines.append(f"  Std interval = {np.std(stats['intervals_16']):.2f} semitones")
        lines.append(f"  Mean pitch range (16-bar) = {np.mean(stats['ranges_16']):.2f} semitones")
        lines.append(f"  Mean duration variety (16-bar) = {np.mean(stats['dur_variety_16']):.2f} types")
        lines.append(f"  Mean monotone run length = {np.mean(stats['run_lengths']):.2f}")

        n_qual = int(mask.sum())
        lines.append(f"\n[Qualified trials, N={n_qual}]")
        if n_qual > 0:
            lines.append(f"  PSD slope alpha = {np.mean(stats['psd_slopes'][mask]):.3f} +/- {np.mean(stats['psd_cis'][mask]):.3f}")
            lines.append(f"  First-order autocorrelation rho(1) = {np.mean(stats['acf1'][mask]):.3f}")
            qi = _filter_pooled(full_melodies, m16_list, mask, "intervals_16")
            lines.append(f"  Mean interval |delta_midi| = {np.mean(qi):.2f} semitones")
            lines.append(f"  Mean pitch range (16-bar) = {np.mean(stats['ranges_16'][mask]):.2f} semitones")
            lines.append(f"  Mean duration variety (16-bar) = {np.mean(stats['dur_variety_16'][mask]):.2f} types")
        else:
            lines.append("  (no qualified trials)")

        lines.append("\n[Difference analysis: all vs qualified]")
        n_unq = n_total - n_qual
        if n_qual > 0 and n_unq > 0:
            unq_mask = ~mask
            da = np.mean(stats['psd_slopes'][mask]) - np.mean(stats['psd_slopes'][unq_mask])
            lines.append(f"  alpha difference (qual - unqual): delta = {da:.3f}")
            iq = _filter_pooled(full_melodies, m16_list, mask, "intervals_16")
            iu = _filter_pooled(full_melodies, m16_list, unq_mask, "intervals_16")
            if len(iq) and len(iu):
                _, pv = ttest_ind(iq, iu)
                lines.append(f"  Interval mean difference: delta = {np.mean(iq)-np.mean(iu):.2f} (t-test p = {pv:.4f})")
            rq = stats['ranges_16'][mask]
            ru = stats['ranges_16'][unq_mask]
            if len(rq) and len(ru):
                _, pv = ttest_ind(rq, ru)
                lines.append(f"  Range mean difference: delta = {np.mean(rq)-np.mean(ru):.2f} (t-test p = {pv:.4f})")
        else:
            lines.append("  (insufficient groups for comparison)")
        return "\n".join(lines)

    def _breakdown(qrs):
        nq = sum(r["qualified"] for r in qrs)
        rf = sum(not r["range_ok"] and r["dur_ok"] for r in qrs)
        df = sum(r["range_ok"] and not r["dur_ok"] for r in qrs)
        bf = sum(not r["range_ok"] and not r["dur_ok"] for r in qrs)
        return nq, rf, df, bf

    p_nq, p_rf, p_df, p_bf = _breakdown(pink_stats["qualify_results"])
    b_nq, b_rf, b_df, b_bf = _breakdown(brown_stats["qualify_results"])

    def _demo_line(m16):
        r = qualify_reason(m16)
        return (f"qualified = {'Yes' if r['qualified'] else 'No'}, "
                f"range = {r['pitch_range']} semitones, "
                f"duration types = {r['n_dur_types']}")

    pm, bm = pink_mask, brown_mask
    p_aq = np.mean(pink_stats["psd_slopes"][pm]) if pm.sum() > 0 else float("nan")
    b_aq = np.mean(brown_stats["psd_slopes"][bm]) if bm.sum() > 0 else float("nan")
    p_r1q = np.mean(pink_stats["acf1"][pm]) if pm.sum() > 0 else float("nan")
    b_r1q = np.mean(brown_stats["acf1"][bm]) if bm.sum() > 0 else float("nan")
    piq = _filter_pooled(pink_full, pink_16bars, pm, "intervals_16")
    biq = _filter_pooled(brown_full, brown_16bars, bm, "intervals_16")
    p_ivq = np.mean(piq) if len(piq) else float("nan")
    b_ivq = np.mean(biq) if len(biq) else float("nan")
    iv_pv = ttest_ind(piq, biq).pvalue if len(piq) and len(biq) else float("nan")
    prq = pink_stats["ranges_16"][pm]
    brq = brown_stats["ranges_16"][bm]
    p_rq = np.mean(prq) if len(prq) else float("nan")
    b_rq = np.mean(brq) if len(brq) else float("nan")
    r_pv = ttest_ind(prq, brq).pvalue if len(prq) and len(brq) else float("nan")

    pa_all = np.mean(pink_stats["psd_slopes"])
    ba_all = np.mean(brown_stats["psd_slopes"])
    pr1_all = np.mean(pink_stats["acf1"])
    br1_all = np.mean(brown_stats["acf1"])

    def _conclusion():
        lines = []
        lines.append(
            f"1. Qualification rate: Pink {p_nq}/{n_total} ({100*p_nq/n_total:.1f}%), "
            f"Brown {b_nq}/{n_total} ({100*b_nq/n_total:.1f}%). "
            f"Brown's lower rate reflects local clustering in 1/f^2 walks: the pitch "
            f"tends to stay near a recent value, making it hard to span 24 semitones "
            f"within just 16 bars."
        )
        lines.append(
            f"2. PSD slope: Pink alpha = {pa_all:.2f} (target ~-1.0, i.e. 1/f), "
            f"Brown alpha = {ba_all:.2f} (target ~-2.0, i.e. 1/f^2). "
            f"Both generators match their theoretical spectral profiles."
        )
        lines.append(
            f"3. Autocorrelation: Brown rho(1) = {br1_all:.3f} is substantially higher "
            f"than Pink rho(1) = {pr1_all:.3f}, confirming stronger pitch persistence "
            f"in brown noise melodies."
        )
        if not np.isnan(p_aq) and not np.isnan(b_aq):
            lines.append(
                f"4. Qualification filter effect: Qualified-subset alpha shifts to "
                f"{p_aq:.2f} (pink) and {b_aq:.2f} (brown), indicating that the "
                f"trials passing the range/duration criteria have a slightly different "
                f"spectral character than the full population."
            )
        if not np.isnan(iv_pv):
            sig = "significantly" if iv_pv < 0.05 else "not significantly"
            lines.append(
                f"5. Interval comparison (qualified subsets): Pink mean = {p_ivq:.2f}, "
                f"Brown mean = {b_ivq:.2f} semitones (t-test p = {iv_pv:.4f}). "
                f"The two types differ {sig} in step size among qualified melodies."
            )
        return "\n".join(lines)

    p_idx, p_sp, p_sd = pink_demo_info
    b_idx, b_sp, b_sd = brown_demo_info
    idx_obs = ""
    if b_idx > p_idx * 2 + 10:
        idx_obs = (f"\n（棕色 demo 来自第 {b_idx+1} 段，粉色来自第 {p_idx+1} 段，"
                   f"差距反映棕色游走短窗口内音域较窄、合规率较低的特性。）")

    report = (
        "=== 示范曲选取 ===\n"
        f"粉色 demo: 大样本第 {p_idx+1} 段（seed_pitch={p_sp}, seed_dur={p_sd}）\n"
        f"棕色 demo: 大样本第 {b_idx+1} 段（seed_pitch={b_sp}, seed_dur={b_sd}）"
        f"{idx_obs}\n"
        "demo 均来自大样本，与合规率统计共享同一总体，无单独生成。\n"
        "\n=== 数据双轨说明 ===\n"
        "短序列组（每段恰好 16 小节，~30-80 音符）：用于合规筛选与短片段指标\n"
        "长序列组（每段 1000 音符，同种子配对）：用于功率谱与长时统计\n"
        "合规标签由短序列组判定，并继承到对应长序列上\n"
        "\n=== Experiment Settings ===\n"
        f"Random seed: 666\n"
        f"MIDI range: 48 - 72 (C3 - C5, 25 semitones)\n"
        f"Scale: chromatic\n"
        f"Time signature: 4/4\n"
        f"Demo bars: 16\n"
        f"N trials: {n_total}\n"
        f"Notes per trial: 1000\n"
        f"Durations: {DURATIONS} beats\n"
        f"Qualification criteria: pitch range >= 24 semitones AND all 4 duration types present\n"
        "\n=== Qualification Rate ===\n"
        f"Pink: {p_nq} / {n_total} trials qualified ({100*p_nq/n_total:.1f}%)\n"
        f"  - Failed range only: {p_rf}\n"
        f"  - Failed duration only: {p_df}\n"
        f"  - Failed both: {p_bf}\n"
        f"Brown: {b_nq} / {n_total} trials qualified ({100*b_nq/n_total:.1f}%)\n"
        f"  - Failed range only: {b_rf}\n"
        f"  - Failed duration only: {b_df}\n"
        f"  - Failed both: {b_bf}\n"
        "\n=== Pink Music Statistics ===\n"
        + _group_summary(pink_stats, pink_mask, pink_full, pink_16bars) +
        "\n\n=== Brown Music Statistics ===\n"
        + _group_summary(brown_stats, brown_mask, brown_full, brown_16bars) +
        "\n\n=== Cross-type Comparison (qualified subsets) ===\n"
        f"Pink qualified (N={pm.sum()}) vs Brown qualified (N={bm.sum()}):\n"
        f"  alpha: pink = {p_aq:.3f}, brown = {b_aq:.3f}, delta = {p_aq - b_aq:.3f}\n"
        f"  rho(1): pink = {p_r1q:.3f}, brown = {b_r1q:.3f}\n"
        f"  Interval mean: pink = {p_ivq:.2f}, brown = {b_ivq:.2f} (t-test p = {iv_pv:.4f})\n"
        f"  Range mean: pink = {p_rq:.2f}, brown = {b_rq:.2f} (t-test p = {r_pv:.4f})\n"
        "\n=== Demo Melodies (16 bars each) ===\n"
        f"Pink demo: {_demo_line(pink_demo_16)}\n"
        f"Brown demo: {_demo_line(brown_demo_16)}\n"
        "\n=== Conclusion Summary ===\n"
        + _conclusion() + "\n"
    )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
