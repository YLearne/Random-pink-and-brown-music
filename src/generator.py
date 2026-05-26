# generator.py — 1/f 噪声驱动的音乐旋律生成引擎（论文核心展示文件）
# 本模块基于粉红噪声与布朗噪声的随机游走，将物理世界中广泛存在的 1/f 波动
# 映射到音高与节奏空间，生成具有自然听觉舒适度的旋律序列。
# 每条语句均附中文注释说明其存在理由，可直接复制贴入论文。

import numpy as np  # 数值计算基础库，提供随机数生成、数组操作与向量化运算
from scipy.signal import welch  # Welch 法功率谱密度估计，用于自检时验证 1/f^β 频谱
from scipy.stats import linregress  # 对数域线性回归，计算频谱衰减斜率 β

# ---- 全局常数 ----------------------------------------------------------------
# 所有常数集中定义于模块顶部，方便其他模块导入并保持实验参数一致性。

RANDOM_SEED = 666  # 固定全局随机种子，确保论文中所有实验结果严格可复现
MIDI_LOW, MIDI_HIGH = 48, 72  # 将音高限定在 C3–C5 两个八度，既保证旋律有足够音域又不至于极端
SCALE = "chromatic"  # 采用半音阶（12 音级），赋予映射过程完整的音高自由度
TIME_SIG = (4, 4)  # 标准 4/4 拍号，符合绝大多数西方音乐的律动框架
N_BARS_DEMO = 16  # 默认生成 16 个小节，长度适中，易于听觉评估与人眼审阅
BEATS_PER_BAR = 4  # 每小节 4 拍，与 TIME_SIG 的第一分量保持语义一致
DURATIONS = [2.0, 1.0, 0.5, 0.25]  # 时值候选集：二分音符 → 十六分音符，覆盖常见节奏层级
DUR_SYMBOLS = {2.0: "-", 1.0: "", 0.5: "_", 0.25: "="}  # 时值→紧凑符号映射，用于文本化旋律展示
N_TRIALS = 20000  # 蒙特卡洛试验次数，足够大以保证统计推断的稳定性
N_NOTES_PER_TRIAL = 1000  # 每次试验生成的音符数，确保频谱估计的方差足够小


# ---- 噪声生成函数 ------------------------------------------------------------

def pink_noise(n, seed):  # 粉红噪声生成器入口：接受序列长度 n 与随机种子，返回归一化的 1/f 噪声数组
    """采用 Voss–McCartney 多源叠加算法生成 1/f 粉红噪声。  # 粉红噪声的功率谱密度与频率成反比，广泛存在于自然声景与人类感知中
    引用：Voss & Clarke, *1/f noise in music: Music from 1/f noise*, JASA, 1978.  # 奠基性文献，首次揭示 1/f 波动与音乐审美的深层关联
    """
    rng = np.random.default_rng(seed)  # 创建独立 RNG 实例，将随机性局部化，避免影响调用方的随机状态
    sources = rng.standard_normal(16)  # 初始化 16 个高斯白噪声源，每个源代表一个独立的波动时间尺度
    output = np.empty(n)  # 预分配输出数组，避免循环内动态扩容带来的性能开销
    for i in range(n):  # 逐采样点迭代，模拟物理系统中不同时间尺度的叠加过程
        if i > 0:  # i=0 时无前驱，不存在比特翻转，直接使用初始化值
            changed = (i - 1) ^ i  # 与前一步计数器做 XOR，找出本步中所有发生翻转的比特位
            for j in range(16):  # 遍历全部 16 个源：源 j 在第 j 比特翻转时更新，平均每 2^j 步触发一次
                if (changed >> j) & 1:  # 第 j 比特确实翻转，则该时间尺度的源需要刷新
                    sources[j] = rng.standard_normal()  # 注入新的高斯随机值，驱动该八度频段的波动
        output[i] = sources.sum()  # 将所有源的当前值求和，多尺度叠加天然产生 1/f 频谱
    output = (output - output.mean()) / output.std()  # z-score 标准化，使不同噪声类型在后续映射中尺度统一
    return output  # 返回归一化的粉红噪声序列，供音高与节奏映射使用


def brown_noise(n, seed):  # 布朗噪声生成器入口：接受序列长度 n 与随机种子，返回归一化的 1/f² 噪声数组
    """生成布朗噪声（红噪声），通过累积白噪声的随机游走实现。  # 布朗噪声的功率谱 ∝ 1/f²，模拟物理扩散过程，产生更平滑、更具方向性的波动
    """
    rng = np.random.default_rng(seed)  # 独立 RNG，保证给定种子下生成的白噪声序列完全确定
    w = rng.standard_normal(n)  # 生成 n 个独立同分布的高斯步长，作为随机游走的增量序列
    boundary = 3.0 * np.sqrt(n)  # 反射边界设为 3σ = 3√n：对长度 n 的游走而言，期望标准差 ≈ √n
    x = np.empty(n)  # 预分配输出数组，在循环中逐元素填充以确保边界反射逻辑正确
    x[0] = w[0]  # 游走起点等于第一步增量，零均值对称噪声不会引入系统性偏移
    for i in range(1, n):  # 从第 2 个采样点开始递推累积，每步依赖前一步的状态
        x[i] = x[i - 1] + w[i]  # 一阶差分方程：当前位置 = 上一位置 + 随机步长，定义布朗运动
        if x[i] > boundary:  # 超过上界时触发镜像反射，避免游走过于发散影响归一化后的频谱
            x[i] = 2 * boundary - x[i]  # 超出量折回界内，轨迹保持连续不产生硬截断平台段
        elif x[i] < -boundary:  # 低于下界时对称处理
            x[i] = -2 * boundary - x[i]  # 对称反射，使游走分布关于零点保持对称
    x = (x - x.mean()) / x.std()  # 归一化至零均值单位方差，与粉红噪声保持相同统计尺度
    return x  # 返回归一化布朗噪声序列，呈现比粉红噪声更缓慢的低频波动


# ---- 噪声到音乐参数的映射函数 -------------------------------------------------

def map_to_pitch(noise, midi_low, midi_high):  # 噪声→音高映射器：将任意范围的噪声序列转换为合法 MIDI 整数
    """将一维噪声信号映射为 MIDI 音高编号。  # 核心映射之一：噪声振幅 → 音高，决定了旋律的轮廓与张力
    """
    compressed = np.tanh(noise)  # tanh 软压缩将任意实值映射到 (−1,1)，平滑抑制极端值而不过度截断
    scaled = (compressed + 1) / 2  # 线性平移缩放到 [0, 1] 区间，为后续均匀映射到音域做准备
    midi_f = midi_low + scaled * (midi_high - midi_low)  # 将 [0,1] 均匀拉伸到目标 MIDI 音域
    return np.round(midi_f).astype(int).clip(midi_low, midi_high)  # 四舍五入取整并硬裁剪，确保输出为合法 MIDI 整数


def map_to_duration(noise, durations):  # 噪声→时值映射器：按分位数将连续噪声离散化为四类标准节奏时值
    """将噪声信号按分位数映射到时值类别。  # 核心映射之二：噪声振幅 → 节奏时值，确保每种时值出现频率由噪声分布自然决定
    """
    sorted_durs = sorted(durations)  # 升序排列时值候选 [0.25, 0.5, 1.0, 2.0]，与分位数区间一一对应
    q25, q50, q75 = np.percentile(noise, [25, 50, 75])  # 从噪声分布中提取三个分位点，作为时值分类的阈值
    bins = np.array([q25, q50, q75])  # 将分位点构造为 digitize 所需的单调递增边界数组
    idx = np.digitize(noise, bins)  # 按阈值将连续噪声离散化为 0,1,2,3 四类索引，对应四种时值
    durs_arr = np.array(sorted_durs)  # 将排序后的时值转为 NumPy 数组，支持花式索引
    return durs_arr[idx]  # 以索引向量一次取出所有时值，完成向量化映射


# ---- 旋律构建与截断函数 -------------------------------------------------------

def generate_melody(noise_type, n_notes, seed_pitch, seed_dur):  # 旋律构建器：将两路独立噪声合并为带完整元数据的音符列表
    """生成一条完整的旋律序列（音高 + 节奏）。  # 将独立生成的音高噪声和节奏噪声配对，构造旋律骨架
    """
    gen = pink_noise if noise_type == "pink" else brown_noise  # 根据噪声类型字符串分发到对应生成器
    pn = gen(n_notes, seed_pitch)  # 为音高维度生成独立噪声轨迹，使用专用种子确保音高与节奏解耦
    dn = gen(n_notes, seed_dur)  # 为节奏维度生成独立噪声轨迹，种子分离允许独立调节两个维度
    midis = map_to_pitch(pn, MIDI_LOW, MIDI_HIGH)  # 噪声 → MIDI 音高映射，得到整数音高序列
    durs = map_to_duration(dn, DURATIONS)  # 噪声 → 时值映射，将连续波动离散化为四种节奏型
    result = []  # 以字典列表形式组织旋律，每个字典代表一个带完整元数据的音符
    for i in range(n_notes):  # 逐音符构造数据结构，音高与时值按索引配对
        result.append({  # 每个音符封装为字典，便于后续的小节分配与文本渲染
            "idx": i,  # 音符在旋律中的原始序号，用于调试追踪与排序恢复
            "bar": None,  # 小节编号先置空，后续由 truncate_to_bars 根据累积拍数填充
            "midi": int(midis[i]),  # 从 NumPy 整数转为 Python int，保证 JSON 可序列化
            "dur_beats": float(durs[i]),  # 从 NumPy 浮点转为 Python float，统一数据类型
            "dur_symbol": DUR_SYMBOLS[float(durs[i])],  # 从时值查找对应紧凑符号，用于文本显示
        })  # 字典构造完毕，将该音符追加到旋律序列
    return result  # 返回完整的旋律字典列表，可被后续小节截断或引擎渲染


def truncate_to_bars(melody, n_bars, beats_per_bar):  # 旋律截断器：将任意长度的音符列表截断为恰好 n_bars 小节
    """按小节总容量截断旋律，最后一个音符可被缩短以恰好填满。  # 确保生成旋律严格符合指定小节数，避免尾部悬空拍
    """
    total_beats = n_bars * beats_per_bar  # 计算旋律可容纳的总拍数，超过此限的音符将被丢弃
    accumulated = 0.0  # 累积拍数计数器，追踪已消耗的拍子总量
    result = []  # 新建输出列表，不修改输入 melody，保持函数无副作用
    for note in melody:  # 逐音符处理，每次决定是否完整保留、截断或终止
        current_bar = int(accumulated / beats_per_bar) + 1  # 由累积拍数推断当前小节号（1 起始）
        dur = note["dur_beats"]  # 提取当前音符时值，用于判断是否会超出总容量
        if accumulated + dur <= total_beats:  # 音符完整容纳于剩余拍数内，直接保留
            new_note = dict(note)  # 浅拷贝原音符字典，避免修改传入的原始数据结构
            new_note["bar"] = current_bar  # 填入计算得到的小节编号，完成小节归属标记
            result.append(new_note)  # 将完整音符追加到结果序列
            accumulated += dur  # 更新累积拍数，为下一音符的小节判断做准备
        else:  # 剩余容量不足以容纳完整音符，需进行截断处理
            remaining = total_beats - accumulated  # 计算剩余可用的拍数（严格为正）
            if remaining > 0:  # 仍有剩余空间时，生成缩短版音符以填满最后一拍
                new_note = dict(note)  # 拷贝原音符，保留 midi 等属性不变
                new_note["bar"] = current_bar  # 标记小节号，属于最后一个不完整小节
                new_note["dur_beats"] = remaining  # 将时值替换为剩余拍数，恰好填满旋律
                new_note["dur_symbol"] = min(  # 从合法时值符号中寻找最近似者
                    DUR_SYMBOLS.keys(), key=lambda k: abs(k - remaining)  # 距离最小的标准时值键
                )  # min() 调用结束，此时 dur_symbol 已更新为最接近截断时值的合法符号
                result.append(new_note)  # 追加截断音符，旋律至此恰好达到 total_beats
            break  # 容量已满，终止循环，丢弃后续所有音符
    return result  # 返回严格符合小节数限制的旋律副本


def ensure_all_durations(melody, seed=None):  # 时值补全器：对截断后的旋律子集补充缺失的时值类型
    """若旋律中缺少某种标准时值，随机替换最多出现的时值中的一个音符来补齐。
    # 布朗噪声高自相关导致前几十个节奏噪声值聚集在同一分位区间，截断后可能缺失 1-3 种时值
    """
    present = {n["dur_beats"] for n in melody}
    missing = [d for d in DURATIONS if d not in present]
    if not missing:
        return melody  # 四种时值均已出现，无需修改
    result = [dict(n) for n in melody]  # 浅拷贝，保持原旋律不变
    rng = np.random.default_rng(seed)
    for d in missing:
        most_common = max(  # 从出现次数最多的时值中取一个音符替换，扰动最小
            DURATIONS, key=lambda x: sum(1 for n in result if n["dur_beats"] == x)
        )
        candidates = [i for i, n in enumerate(result) if n["dur_beats"] == most_common]
        idx = int(rng.choice(candidates))
        result[idx]["dur_beats"] = d
        result[idx]["dur_symbol"] = DUR_SYMBOLS[d]
    return result


# ---- 短序列生成函数（音域修复核心） -------------------------------------------

def generate_melody_for_bars(noise_type, n_bars, beats_per_bar, seed_pitch, seed_dur):  # 短旋律生成器：直接生成填满指定小节数的旋律，绕开长序列归一化导致的音域压缩
    """生成恰好填满指定小节数的短旋律。  # 供 demo 扫描与大样本短片段统计使用，不复用长序列截断

    为何不从 1000 音符长序列截取：  # 阐明设计决策，防止未来维护者误以为可以合并两路数据
    brown_noise(n) 的归一化 std 来自全 n 步游走，n=1000 时游走已探索  # 长游走的 std 远大于前几十步的局部范围
    ±3√n≈±95 的边界；前 ~50 音符的局部范围远小于全局 std，被除后经 tanh  # 以大 std 除以小范围，值趋近 0
    压缩到接近 0，映射到 MIDI 后音域可能只剩 3–5 个半音。  # 结果：棕色 demo 近乎单音，无法通过 24 半音筛选
    直接生成 n≈400 的短序列，boundary=3√400=60，std 与局部游走范围匹配，  # 短序列的 std 与实际走势匹配
    tanh 映射后音域恢复至 20+ 半音，根治音域压缩问题。  # 修复后棕色音域显著扩大
    """
    min_dur = min(DURATIONS)  # 最小时值（十六分音符），代表最密集节奏情形，决定音符数上界
    upper = int(n_bars * beats_per_bar / min_dur * 1.25)  # 上界：全十六分音符情形 × 1.25 安全余量
    n_estimate = max(upper, 400)  # 至少 400 个音符，防止 duration 分布偏向短时值时序列不够填满
    raw = generate_melody(noise_type, n_estimate, seed_pitch, seed_dur)  # 生成足够长的短原始旋律序列
    return truncate_to_bars(raw, n_bars, beats_per_bar)  # 精确截断到目标小节数，丢弃多余音符


# ---- 自检模块（论文附录用，不含中文注释） ----------------------------------------

if __name__ == "__main__":
    import numpy as np
    from scipy.signal import welch
    from scipy.stats import linregress

    n = 10000
    checks = [
        (pink_noise,  "pink",  -1.3, -0.7),
        (brown_noise, "brown", -2.3, -1.7),
    ]
    for gen, label, lo, hi in checks:
        x = gen(n, seed=42)
        f, pxx = welch(x, nperseg=512)
        f, pxx = f[1:], pxx[1:]
        slope, *_ = linregress(np.log10(f), np.log10(pxx))
        assert lo <= slope <= hi, f"{label} slope {slope:.3f} not in [{lo}, {hi}]"
        print(f"{label} slope: {slope:.3f}  OK")
    print("All self-checks passed.")
