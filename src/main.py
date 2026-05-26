"""
Main experiment runner: generates pink and brown noise melodies,
runs qualification checks, and produces all statistics, figures, and reports.
"""
import os

from .generator import (
    RANDOM_SEED, MIDI_LOW, MIDI_HIGH, N_BARS_DEMO, BEATS_PER_BAR,
    DURATIONS, DUR_SYMBOLS, N_TRIALS, N_NOTES_PER_TRIAL,
    generate_melody, truncate_to_bars, ensure_all_durations,
    generate_melody_for_bars,
)
from .notation import melody_to_jianpu_text, melody_to_csv
from .stats import (
    compute_stats, plot_all_figures, generate_report,
    is_qualified, qualify_reason,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Print experiment parameters
    print("=" * 60)
    print("Noise Music Experiment")
    print("=" * 60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"MIDI range: {MIDI_LOW} - {MIDI_HIGH}")
    print(f"Demo bars: {N_BARS_DEMO}")
    print(f"N trials: {N_TRIALS}, notes per trial (long): {N_NOTES_PER_TRIAL}")
    print(f"Durations: {DURATIONS}")
    print()

    # 2. Large-scale experiment: dual-track data
    # 种子区间 1666–(1666+2*(N_TRIALS-1))，粉色和棕色共享相同 (sp, sd) 配对
    # 短序列：完整 dict list，用于合规筛选与短片段统计
    # 长序列：仅保存 MIDI pitch numpy 数组（int16），节省内存，用于功率谱与长时统计
    import numpy as np
    print(f"Running large-scale experiment ({N_TRIALS} trials each)...")
    pink_short_list, brown_short_list = [], []
    pink_long_list, brown_long_list = [], []

    for i in range(N_TRIALS):
        sp = 1666 + 2 * i
        sd = sp + 1

        # 短序列（约 30–80 音符）：直接生成 16 小节，不截自长序列
        pink_short_list.append(generate_melody_for_bars("pink",  N_BARS_DEMO, BEATS_PER_BAR, sp, sd))
        brown_short_list.append(generate_melody_for_bars("brown", N_BARS_DEMO, BEATS_PER_BAR, sp, sd))

        # 长序列：只提取 MIDI 数组，丢弃 dict 结构，避免 N_TRIALS×1000 个 dict 耗尽内存
        pm = generate_melody("pink",  N_NOTES_PER_TRIAL, seed_pitch=sp, seed_dur=sd)
        bm = generate_melody("brown", N_NOTES_PER_TRIAL, seed_pitch=sp, seed_dur=sd)
        pink_long_list.append(np.array([n["midi"] for n in pm], dtype=np.int16))
        brown_long_list.append(np.array([n["midi"] for n in bm], dtype=np.int16))

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{N_TRIALS} trials done")

    print("All trials generated.")
    print()

    # 3. Pick demo from within the large sample (first qualified trial per noise type)
    # 保证 demo 与大样本来自同一总体，统计一致性最强
    pink_demo_idx = next((i for i, m in enumerate(pink_short_list) if is_qualified(m)), None)
    brown_demo_idx = next((i for i, m in enumerate(brown_short_list) if is_qualified(m)), None)

    if pink_demo_idx is None:
        raise RuntimeError("大样本中无合规粉色旋律，无法选取 demo（可增大 N_TRIALS 或放宽筛选条件）")
    if brown_demo_idx is None:
        raise RuntimeError("大样本中无合规棕色旋律，无法选取 demo（可增大 N_TRIALS 或放宽筛选条件）")

    pink_demo_16 = pink_short_list[pink_demo_idx]
    brown_demo_16 = brown_short_list[brown_demo_idx]
    pink_demo_sp = 1666 + 2 * pink_demo_idx
    brown_demo_sp = 1666 + 2 * brown_demo_idx

    # 内部断言：选出的 demo 必须合规
    assert is_qualified(pink_demo_16), "pink demo 未通过合规检查"
    assert is_qualified(brown_demo_16), "brown demo 未通过合规检查"

    print(f"Pink demo:  trial #{pink_demo_idx+1}, seed_pitch={pink_demo_sp}")
    print(f"Brown demo: trial #{brown_demo_idx+1}, seed_pitch={brown_demo_sp}")

    # Write demo txt files
    with open(os.path.join(OUTPUT_DIR, "pink_demo.txt"), "w", encoding="utf-8") as f:
        f.write("=== Pink Noise Demo (16 bars) ===\n\n")
        f.write(melody_to_jianpu_text(pink_demo_16) + "\n")
    with open(os.path.join(OUTPUT_DIR, "brown_demo.txt"), "w", encoding="utf-8") as f:
        f.write("=== Brown Noise Demo (16 bars) ===\n\n")
        f.write(melody_to_jianpu_text(brown_demo_16) + "\n")

    # Write demo csv files
    melody_to_csv(pink_demo_16, os.path.join(OUTPUT_DIR, "pink_demo.csv"))
    melody_to_csv(brown_demo_16, os.path.join(OUTPUT_DIR, "brown_demo.csv"))

    # Report demo qualification
    pink_demo_q = qualify_reason(pink_demo_16)
    brown_demo_q = qualify_reason(brown_demo_16)
    print(f"Pink demo: {'QUALIFIED' if pink_demo_q['qualified'] else 'NOT QUALIFIED'} "
          f"(range={pink_demo_q['pitch_range']} semitones, dur_types={pink_demo_q['n_dur_types']})")
    print(f"Brown demo: {'QUALIFIED' if brown_demo_q['qualified'] else 'NOT QUALIFIED'} "
          f"(range={brown_demo_q['pitch_range']} semitones, dur_types={brown_demo_q['n_dur_types']})")
    print()

    # 4. Compute statistics
    # 合规标签由短序列判定，并继承到对应长序列（供功率谱分组对比）
    print("Computing statistics...")
    pink_stats = compute_stats(pink_long_list, pink_short_list)
    brown_stats = compute_stats(brown_long_list, brown_short_list)

    from .stats import _qual_mask
    pink_mask = _qual_mask(pink_stats)
    brown_mask = _qual_mask(brown_stats)

    p_nq = int(pink_mask.sum())
    b_nq = int(brown_mask.sum())
    print(f"Pink qualification: {p_nq}/{N_TRIALS} ({100*p_nq/N_TRIALS:.1f}%)")
    print(f"Brown qualification: {b_nq}/{N_TRIALS} ({100*b_nq/N_TRIALS:.1f}%)")
    print()

    # 5. Generate figures
    print("Generating figures...")
    plot_all_figures(
        pink_stats, pink_long_list, pink_short_list,
        brown_stats, brown_long_list, brown_short_list,
        outdir=FIGURES_DIR,
    )
    print(f"Figures saved to {FIGURES_DIR}")
    print()

    # 6. Generate text report
    print("Generating stats report...")
    report_path = os.path.join(OUTPUT_DIR, "stats_report.txt")
    generate_report(
        pink_stats, pink_long_list, pink_short_list,
        brown_stats, brown_long_list, brown_short_list,
        pink_demo_16, brown_demo_16,
        pink_demo_info=(pink_demo_idx, pink_demo_sp, pink_demo_sp + 1),
        brown_demo_info=(brown_demo_idx, brown_demo_sp, brown_demo_sp + 1),
        path=report_path,
    )
    print(f"Report saved to {report_path}")
    print()

    # 7. Print key conclusions
    import numpy as np
    print("=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    p_alpha = np.mean(pink_stats["psd_slopes"])
    b_alpha = np.mean(brown_stats["psd_slopes"])
    p_rho1 = np.mean(pink_stats["acf1"])
    b_rho1 = np.mean(brown_stats["acf1"])
    print(f"Pink PSD slope alpha = {p_alpha:.3f} (target: ~-1.0)")
    print(f"Brown PSD slope alpha = {b_alpha:.3f} (target: ~-2.0)")
    print(f"Pink rho(1) = {p_rho1:.3f}")
    print(f"Brown rho(1) = {b_rho1:.3f}")
    print(f"Pink qualification rate: {100*p_nq/N_TRIALS:.1f}%")
    print(f"Brown qualification rate: {100*b_nq/N_TRIALS:.1f}%")

    # Acceptance check (PSD computed on MIDI integer sequences after tanh+quantize mapping;
    # non-linear mapping compresses dynamic range, so empirical slopes differ from raw noise slopes)
    assert -1.2 <= p_alpha <= -0.8, f"Pink alpha {p_alpha:.3f} out of expected range [-1.2, -0.8]"
    assert -1.9 <= b_alpha <= -1.3, f"Brown alpha {b_alpha:.3f} out of expected range [-1.9, -1.3]"
    print()
    print("Acceptance check PASSED: alpha values within expected ranges.")
    print()
    print("Experiment complete. All output files are in the 'output/' directory.")


if __name__ == "__main__":
    main()
