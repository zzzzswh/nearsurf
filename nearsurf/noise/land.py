"""nearsurf.noise.land — 陆上（及海底节点）背景噪声 + 道缺陷合成

（源自 fb-dataset-v3 的 synth/noise.py，算法未改。简便入口：nearsurf.land_noise。）

设计要点:
  1. 绝对水平定标：所有分量参照 gather 的全局参考水平（强道 RMS 的
     ref_q 分位数），不逐道跟随该道自身信号 —— 远道信号弱、噪声水平
     不变，SNR 随偏移距自然劣化，这正是初至拾取的真实难度来源。
  2. 每道等概率：受染道由逐道独立 Bernoulli 决定，每道水平在样本基准上
     再乘 ±trace_var_db 的随机波动。
  3. 显式参数化：各分量以 SNR(dB)/相对幅度参数化，实际抽取值全部写入
     返回的 info dict，供 metadata 落盘与消融分析。
  4. 纯 numpy + 注入式 Generator：逐样本比特级可复现（无全局 RNG）。

噪声分量:
  ambient    带限有色背景噪声（随机频带 + 随机谱倾斜 + 时/空非平稳包络），全道覆盖
  powerline  工频 50/60 Hz + 衰减谐波 + 慢幅度调制，逐道 Bernoulli；
             另有相干模式（共享相位沿排列慢漂移 = 电缆串扰条纹）
  spikes     尖脉冲（仪器故障 / 踩踏），逐道 Bernoulli + 泊松计数
  bursts     短时爆发（风 / 交通），随机时窗内的带限噪声包
  patchy     空间相关低频斑块（竖条纹，移植自毕设并随机化）
  hum        共模低频窄带（横向波纹，移植自毕设并随机化）
  machinery  矿山机械/风机谐波梳：随机基频 + 衰减谐波 + 慢节律 AM，
             空间高斯足迹（矿区工区标配）
  gusts      阵风锋：共享时窗/足迹、锋面推进、道间载波独立的时空
             相关爆发（区别于逐道独立 bursts）
  airblast   空气波：沿 |offset|/340 m/s 的线性同相轴，高频子波
             （需要 offsets；真空法自由地表没有空气层，物理上缺失，
              这里作为噪声分量补回）

总水平：master_db —— 样本级总闸，对合成后的总噪声整体加
一个 U(master_db) dB 的偏移，用于整体水平校准（对生成的道集
目检后定先验）；各分量参数只管配比。

道缺陷（corrupt() 统一编排）:
  dead_hard   全零死道（label → no_fb 哨兵）
  dead_noisy  信号消失、只剩噪声的死道（label → no_fb；训练难例，
              强迫模型依据波形形态而非能量判别）
  polarity    极性反转（label 不变）
  gain        逐道耦合增益抖动，log-normal（label 不变）
  clip        削波（label 不变）
  ringy       谐振耦合道：全输入过阻尼谐振支路 → 拖尾振铃（label 不变）
  dropout     telemetry 掉段：随机零段；覆盖初至 ±guard 时 label → 哨兵

入口:
    noisy, label_out, alive, info = corrupt(
        clean, label, dt, rng,
        ncfg=NoiseConfig(), dcfg=DefectConfig(),
        offsets=shot.offsets, t0=wavelet_ref_delay,
    )

既可离线烘焙进数据集，也可放进 dataloader 在线增强（每 epoch 换
rng 种子即得无限噪声多样性）。
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# 配置
# =============================================================================
@dataclasses.dataclass(frozen=True)
class NoiseConfig:
    # ---- 全局水平 ----
    snr_db: tuple = (3.0, 30.0)       # ambient 样本级 SNR 范围（相对参考 RMS）
    ref_q: float = 0.95               # 参考水平 = 逐道 RMS/峰值 的该分位数
    trace_var_db: float = 6.0         # 逐道水平等概率随机波动 ±dB
    master_db: tuple = (0.0, 0.0)     # 样本级总闸：总噪声整体偏移 U(lo,hi) dB
                                      #   （先验由对生成道集的目检确定）
    # ---- 背景的参照水平 ----
    # "global"：背景 = 参考 RMS（强道分位，基本由近道面波决定）× 10^(−snr_db/20)，再过总闸。
    #           远道弱折射到底被淹到什么程度是间接的、不可控的。
    # "far_fb"：背景直接按**远道初至**定：取 |炮检距| 最远 far_frac 的有标签道，量干净波场
    #           [fb, fb+far_win_ms] 的 RMS 中位数 A_far，背景 RMS = A_far × 10^(−far_snr_db/20)。
    #           背景不过总闸（总闸只管尖脉冲/突发/工频等偶发分量）。标签来自干净波场，不受影响。
    amb_ref: str = "global"
    far_frac: float = 0.3
    far_win_ms: float = 60.0
    far_snr_db: tuple = (-6.0, 12.0)
    far_other_cap_db: float = 0.0     # far_fb 下，偶发分量（工频/机械/斑块…）在远道的 RMS
                                      # 不超过背景 + 此值 dB；否则整体压下来。防止它们按
                                      # 近道参考定级后把远道初至淹到 −20 dB 以下（标签仍在、
                                      # 波形已不可见 → 教模型"凭空"拾取）

    # ---- ambient（带限有色背景）----
    amb_f_lo: tuple = (0.5, 4.0)      # Hz，低截频率抽取范围
    amb_f_hi: tuple = (25.0, 120.0)   # Hz，高截（自动压到 0.8×Nyquist）
    amb_tilt: tuple = (0.0, 1.2)      # 谱倾斜指数 α，|F| ∝ (f/f_lo)^(−α)
    amb_t_mod_db: tuple = (0.0, 6.0)  # 时间慢包络摆幅 ±dB（风起风停；(0,0) 关闭）
    amb_x_mod_db: tuple = (0.0, 6.0)  # 沿排列水平剖面摆幅 ±dB（一头靠路；(0,0) 关闭）
    # ambient 纹理：背景里"有方向的"那部分（地表各处噪声源激发的面波/散射在排列上
    # 留下的倾斜、交叉、弯曲的短同相轴 —— 真实背景噪声的主要观感）。
    # 背景 = √(1−r)·白噪 + √r·纹理，r ~ U(amb_tex_frac)，总水平不变。(0,0) = 关闭（旧行为）
    amb_tex_frac: tuple = (0.0, 0.0)
    tex_v: tuple = (1500.0, 15000.0)  # m/s，纹理视速度（log-uniform；正负倾向各半）
                                      #   下限受画布道距 20~30 m 约束：再慢就空间假频、成了白噪
    tex_n_dips: tuple = (2, 6)        # 每个小区的主倾向数
    tex_width: float = 0.15           # 倾向扇面宽度（相对慢度的比例）
    tex_band: tuple = (3.0, 30.0)     # Hz，纹理频带
    tex_patches: tuple = (3, 8)       # 小区个数：各小区倾向组合不同，平滑拼接 → 局部方向多变
    tex_warp_ms: tuple = (0.0, 40.0)  # 沿排列平滑时移幅度 → 直线变弯（波浪状）
    tex_warp_corr: float = 1.0 / 12.0 # 时移的横向平滑尺度（占道数比例）；小 → 弯得更勤

    # ---- powerline（工频）----
    pl_freqs: tuple = (50.0, 60.0)    # 每样本随机选一
    pl_p_trace: tuple = (0.0, 0.5)    # 受染道比例（样本级抽取 → 逐道 Bernoulli）
    pl_amp: tuple = (0.02, 0.5)       # 基波峰值 / 参考峰值，log-uniform
    pl_n_harmonics: int = 2           # 谐波个数（2f, 3f, …），幅度按 0.3^k 递减
    pl_am_freq: tuple = (0.1, 1.0)    # Hz，慢幅度调制频率
    pl_am_depth: float = 0.3          # 幅度调制深度
    pl_coherent_p: float = 0.4        # 相干模式概率：受染道共享相位并沿排列慢漂移
                                      #   （电缆串扰的横向条纹形态）；否则逐道乱相
    pl_phase_drift: tuple = (0.0, 2.0)  # rad，相干模式下相位沿排列的总漂移量

    # ---- spikes（尖脉冲）----
    spk_p_trace: tuple = (0.0, 0.05)  # 受染道比例
    spk_amp: tuple = (0.5, 6.0)       # 峰值 / 参考峰值，log-uniform
    spk_width: tuple = (3, 9)         # 脉冲半宽（采样点，hanning）

    # ---- bursts（短时爆发）----
    bst_p_trace: tuple = (0.0, 0.08)
    bst_dur: tuple = (0.02, 0.30)     # s
    bst_band: tuple = (6.0, 45.0)     # Hz，载波频带（压到 0.8×Nyquist）
    bst_amp: tuple = (0.3, 3.0)       # RMS / 参考 RMS，log-uniform

    # ---- airblast（空气波）----
    air_prob: float = 0.4             # 样本出现空气波的概率（需 offsets）
    air_v: float = 340.0              # m/s
    air_f: tuple = (25.0, 60.0)       # 子波主频（压到 0.8×Nyquist）
    air_amp: tuple = (0.1, 1.5)       # 峰值 / 参考峰值 @ 100 m，log-uniform
    air_decay: float = 1.0            # 幅度 ∝ (|off|/100 m)^(−decay)

    # ---- patchy（空间相关低频斑块；移植自毕设 land_noise v5 §4.1）----
    ptc_prob: float = 0.5             # 样本出现概率
    ptc_f_lo: tuple = (0.5, 5.0)      # 带下限（Hz）
    ptc_f_ratio: tuple = (1.5, 4.0)   # 带上限 = f_lo × ratio（压 0.8×Nyquist）
    ptc_corr_m: tuple = (5.0, 60.0)   # 空间相关长度（m），log-uniform
    ptc_amp: tuple = (0.1, 1.0)       # RMS / 参考 RMS，log-uniform

    # ---- hum（共模低频窄带；移植自毕设 BG_linear，参数全随机化）----
    hum_prob: float = 0.35            # 样本出现概率
    hum_f: tuple = (0.8, 6.0)         # 基频（Hz），log-uniform
    hum_period_tr: tuple = (10.0, 100.0)  # 空间相位波动周期（道）
    hum_phase_amp: tuple = (0.5, 4.0)     # 空间相位摆幅（rad）
    hum_amp: tuple = (0.05, 0.6)      # RMS / 参考 RMS，log-uniform

    # ---- machinery（矿山机械/风机谐波梳；HardPicks 四工区均为矿区）----
    mch_prob: float = 0.35            # 样本出现概率
    mch_f0: tuple = (10.0, 45.0)      # 基频 Hz，log-uniform
    mch_n_harm: tuple = (2, 5)        # 谐波条数（含基波）
    mch_harm_decay: tuple = (0.4, 0.8)  # 逐阶幅度衰减因子
    mch_footprint_m: tuple = (100.0, 1500.0)  # 空间足迹高斯半宽（m），log-uniform
    mch_amp: tuple = (0.05, 0.8)      # 足迹中心 RMS / 参考 RMS，log-uniform
    mch_am_freq: tuple = (0.05, 0.8)  # Hz，慢节律幅度调制
    mch_am_depth: tuple = (0.0, 0.6)  # 调制深度

    # ---- gusts（阵风锋：时空相关爆发包，区别于逐道独立的 bursts）----
    gst_prob: float = 0.35            # 样本出现概率
    gst_n: tuple = (1, 3)             # 事件条数
    gst_dur: tuple = (0.15, 0.80)     # s，事件时窗
    gst_footprint_m: tuple = (200.0, 3000.0)  # 空间足迹高斯半宽（m），log-uniform
    gst_front_v: tuple = (30.0, 1000.0)  # m/s，锋面视速度（log-uniform，来向随机；
                                      #   慢→斜穿的游走噪声条，快→近同时的宽爆发）
    gst_band: tuple = (2.0, 20.0)     # Hz，载波频带
    gst_amp: tuple = (0.3, 3.0)       # 足迹中心 RMS / 参考 RMS，log-uniform


@dataclasses.dataclass(frozen=True)
class DefectConfig:
    dead_prob: float = 0.6            # 样本出现死道的概率
    dead_frac: tuple = (0.0, 0.10)    # 死道比例（占总道数）
    dead_noisy_ratio: float = 0.5     # 死道中"只剩噪声"型的占比
    polarity_p: tuple = (0.0, 0.03)   # 极性反转道比例
    gain_db_std: float = 2.0          # 逐道增益抖动 std（dB）；0 关闭
    clip_p: tuple = (0.0, 0.05)       # 削波道比例
    clip_level: tuple = (0.4, 0.9)    # 削波阈 / 该道峰值
    ring_p: tuple = (0.0, 0.04)       # 谐振耦合道比例（坏耦合 → 拖尾振铃，label 不变）
    ring_f: tuple = (8.0, 30.0)       # Hz，耦合谐振频率
    ring_q: tuple = (4.0, 15.0)       # 品质因子（大 → 振铃长）
    ring_gain: tuple = (0.5, 2.0)     # 谐振支路相对增益
    drop_p: tuple = (0.0, 0.03)       # telemetry 掉段道比例
    drop_dur: tuple = (0.010, 0.250)  # s，掉段时长
    drop_guard: int = 12              # 掉段覆盖 fb±guard（采样点）→ label 置哨兵


# =============================================================================
# 小工具
# =============================================================================
def _logu(rng: np.random.Generator, lo: float, hi: float) -> float:
    """log-uniform 标量。"""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _logu_arr(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))


def _db_jitter(rng: np.random.Generator, var_db: float, n: int) -> np.ndarray:
    """逐道等概率水平波动：10^(U(−v, v)/20)。"""
    return 10.0 ** (rng.uniform(-var_db, var_db, size=n) / 20.0)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """[0,1] 升余弦过渡。"""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def _band_taper(f: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    """平滑带通响应（升余弦边沿，避免 brickwall 的 Gibbs 振铃）。"""
    w_lo = max(0.5 * f_lo, float(f[1]) if f.size > 1 else 0.1)
    w_hi = max(0.3 * f_hi, w_lo)
    rise = _smoothstep((f - (f_lo - w_lo)) / w_lo)          # f_lo−w → f_lo
    fall = 1.0 - _smoothstep((f - f_hi) / w_hi)             # f_hi → f_hi+w
    return rise * fall


def _shaped_noise(nrec: int, nt: int, dt: float,
                  f_lo: float, f_hi: float, tilt: float,
                  rng: np.random.Generator) -> np.ndarray:
    """带限有色高斯噪声，全局 RMS 归一到 1。"""
    w = rng.standard_normal((nrec, nt))
    f = np.fft.rfftfreq(nt, d=dt)
    H = _band_taper(f, f_lo, f_hi)
    if tilt > 0:
        ff = np.maximum(f, f_lo)                            # 低于 f_lo 不再抬升
        H = H * (ff / f_lo) ** (-tilt)
    x = np.fft.irfft(np.fft.rfft(w, axis=1) * H[None, :], n=nt, axis=1)
    std = float(x.std())
    return x / std if std > 1e-12 else x


def _ricker(f0: float, dt: float) -> tuple[np.ndarray, int]:
    """短 Ricker 子波；返回 (wavelet, 峰值样点索引)。"""
    half = max(int(round(1.5 / f0 / dt)), 2)
    t = (np.arange(2 * half + 1) - half) * dt
    a = (np.pi * f0 * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float64), half


# =============================================================================
# 各噪声分量（返回未定标或已定标的 (nrec, nt) 数组 + 参数 dict）
# =============================================================================
def _textured(nrec: int, nt: int, dt: float, dx: float, cfg: NoiseConfig,
              rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """有倾向的相干背景：f-k 扇面滤波的白噪，分小区换倾向组合并平滑拼接，再沿排列
    平滑时移弯曲。全局 RMS 归一到 1。

    纹理频带低（≤ tex_band[1]），在 q 倍粗化的时间网格上合成再插回原网格（快 q 倍）。
    """
    q = max(1, int(0.125 / (dt * cfg.tex_band[1])))         # 粗网格 Nyquist ≥ 4×带上限
    ntc, dtc = -(-nt // q), dt * q
    f = np.fft.rfftfreq(ntc, d=dtc)[None, :]               # (1, F) Hz
    k = np.fft.fftfreq(nrec, d=dx)[:, None]                # (nrec, 1) 1/m
    band = _band_taper(f[0], *cfg.tex_band)[None, :]
    n_p = int(rng.integers(cfg.tex_patches[0], cfg.tex_patches[1] + 1))
    # 小区权重：沿 x、t 都平滑变化的随机划分（softmax 一组平滑随机场）
    logits = np.stack([gaussian_filter1d(rng.standard_normal(nrec), nrec / 4.0)[:, None] +
                       gaussian_filter1d(rng.standard_normal(ntc), ntc / 6.0)[None, :]
                       for _ in range(n_p)])
    logits *= 2.5 / max(float(logits.std()), 1e-12)
    wts = np.exp(logits - logits.max(0, keepdims=True))
    wts /= wts.sum(0, keepdims=True)
    field = np.zeros((nrec, ntc))
    for j in range(n_p):
        nd = int(rng.integers(cfg.tex_n_dips[0], cfg.tex_n_dips[1] + 1))
        p = np.exp(rng.uniform(np.log(1 / cfg.tex_v[1]), np.log(1 / cfg.tex_v[0]), nd))
        p *= rng.choice([-1.0, 1.0], nd)                    # s/m，正负倾向
        M = np.zeros((nrec, f.shape[1]))
        for pj in p:
            sig = f * abs(pj) * cfg.tex_width + 1e-5
            M += rng.uniform(0.5, 1.0) * np.exp(-0.5 * ((k + pj * f) / sig) ** 2)
        W = np.fft.rfft2(rng.standard_normal((nrec, ntc)))
        field += np.fft.irfft2(W * M * band, s=(nrec, ntc)) * wts[j]
    if q > 1:                                               # 频域补零上采样（无插值毛刺）
        field = np.fft.irfft(np.fft.rfft(field, axis=1), n=ntc * q, axis=1)[:, :nt] * q
    warp = rng.uniform(*cfg.tex_warp_ms) * 1e-3
    if warp > 0:                                            # 沿排列平滑时移 → 弯曲
        sh = gaussian_filter1d(rng.standard_normal(nrec), nrec * cfg.tex_warp_corr)
        sh = sh / max(float(np.abs(sh).max()), 1e-12) * warp / dt
        t = np.arange(nt, dtype=np.float64)
        field = np.stack([np.interp(t - sh[i], t, field[i]) for i in range(nrec)])
    std = float(field.std())
    return (field / std if std > 1e-12 else field), dict(n_patches=n_p, warp_ms=warp * 1e3)


def _ambient(shape: tuple[int, int], dt: float, ref_rms: float,
             cfg: NoiseConfig, rng: np.random.Generator, dx: float | None = None,
             snr_range: tuple | None = None):
    nrec, nt = shape
    nyq = 0.5 / dt
    f_lo = rng.uniform(*cfg.amb_f_lo)
    f_hi = min(rng.uniform(*cfg.amb_f_hi), 0.8 * nyq)
    tilt = rng.uniform(*cfg.amb_tilt)
    snr = rng.uniform(*(snr_range or cfg.snr_db))      # 抽取次数不变 → 随机流与旧行为一致

    base = _shaped_noise(nrec, nt, dt, f_lo, f_hi, tilt, rng)
    level = ref_rms * 10.0 ** (-snr / 20.0)                 # 绝对水平（不逐道跟随信号）
    per_trace = _db_jitter(rng, cfg.trace_var_db, nrec)     # 逐道等概率波动
    # 非平稳：时间慢包络（风起风停）× 沿排列水平剖面（局地噪声源梯度）
    d_t = rng.uniform(*cfg.amb_t_mod_db)
    d_x = rng.uniform(*cfg.amb_x_mod_db)
    env_t = np.ones(nt)
    if d_t > 0:
        s = gaussian_filter1d(rng.standard_normal(nt), nt / 10.0)
        env_t = 10.0 ** (d_t * s / max(float(np.abs(s).max()), 1e-12) / 20.0)
    env_x = np.ones(nrec)
    if d_x > 0:
        s = gaussian_filter1d(rng.standard_normal(nrec), nrec / 6.0)
        env_x = 10.0 ** (d_x * s / max(float(np.abs(s).max()), 1e-12) / 20.0)
    tex_info = dict(present=False)
    if cfg.amb_tex_frac[1] > 0 and dx is not None:
        # 纹理用派生的独立随机流：开关纹理不改变其余分量的随机序列
        trng = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])
        r = float(trng.uniform(*cfg.amb_tex_frac))
        tex, ti = _textured(nrec, nt, dt, dx, cfg, trng)
        base = np.sqrt(1.0 - r) * base + np.sqrt(r) * tex
        tex_info = dict(present=True, frac=r, **ti)
    noise = base * (level * per_trace * env_x)[:, None] * env_t[None, :]
    return noise, dict(snr_db=snr, f_lo=f_lo, f_hi=f_hi, tilt=tilt,
                       t_mod_db=float(d_t), x_mod_db=float(d_x), texture=tex_info)


def _powerline(shape: tuple[int, int], dt: float, ref_peak: float,
               cfg: NoiseConfig, rng: np.random.Generator):
    nrec, nt = shape
    nyq = 0.5 / dt
    freq = float(rng.choice(cfg.pl_freqs))
    p = rng.uniform(*cfg.pl_p_trace)
    hit = rng.random(nrec) < p                              # 逐道等概率
    n_hit = int(hit.sum())
    noise = np.zeros(shape)
    coherent = False
    if n_hit > 0:
        t = np.arange(nt) * dt
        amps = _logu_arr(rng, *cfg.pl_amp, n_hit) * ref_peak
        # 相干模式：电缆串扰共模拾取 → 受染道共享相位、沿排列慢漂移，
        # 呈现真实数据里典型的工频横向条纹；否则逐道乱相（局地拾取）
        coherent = rng.random() < cfg.pl_coherent_p
        drift = rng.uniform(*cfg.pl_phase_drift) * float(rng.choice([-1.0, 1.0]))
        pos = np.linspace(0.0, 1.0, n_hit)
        sig = np.zeros((n_hit, nt))
        for k in range(cfg.pl_n_harmonics + 1):             # 基波 + 谐波
            fk = freq * (k + 1)
            if fk >= 0.9 * nyq:
                break
            ak = (0.3 ** k) * rng.uniform(0.5, 1.5, size=n_hit)
            if coherent:
                ph = rng.uniform(0, 2 * np.pi) + drift * pos
            else:
                ph = rng.uniform(0, 2 * np.pi, size=n_hit)
            sig += (ak[:, None] * np.sin(2 * np.pi * fk * t[None, :]
                                         + ph[:, None]))
        # 慢幅度调制（供电波动）
        f_am = rng.uniform(*cfg.pl_am_freq)
        ph_am = rng.uniform(0, 2 * np.pi, size=n_hit)
        am = 1.0 + cfg.pl_am_depth * np.sin(
            2 * np.pi * f_am * t[None, :] + ph_am[:, None])
        noise[hit] = amps[:, None] * sig * am
    return noise, dict(freq=freq, p_trace=p, n_hit=n_hit,
                       coherent=bool(coherent))


def _spikes(shape: tuple[int, int], ref_peak: float,
            cfg: NoiseConfig, rng: np.random.Generator):
    nrec, nt = shape
    p = rng.uniform(*cfg.spk_p_trace)
    hit = np.where(rng.random(nrec) < p)[0]
    noise = np.zeros(shape)
    n_total = 0
    for i in hit:
        n_spk = 1 + rng.poisson(0.7)
        for _ in range(n_spk):
            w = int(rng.integers(*cfg.spk_width))
            it = int(rng.integers(0, nt))
            amp = _logu(rng, *cfg.spk_amp) * ref_peak * rng.choice([-1.0, 1.0])
            kern = np.hanning(2 * w + 1) * amp
            a, b = max(0, it - w), min(nt, it + w + 1)
            noise[i, a:b] += kern[(a - (it - w)):(a - (it - w)) + (b - a)]
        n_total += n_spk
    return noise, dict(p_trace=p, n_traces=len(hit), n_spikes=n_total)


def _bursts(shape: tuple[int, int], dt: float, ref_rms: float,
            cfg: NoiseConfig, rng: np.random.Generator):
    nrec, nt = shape
    nyq = 0.5 / dt
    p = rng.uniform(*cfg.bst_p_trace)
    hit = np.where(rng.random(nrec) < p)[0]
    noise = np.zeros(shape)
    for i in hit:
        dur = rng.uniform(*cfg.bst_dur)
        n_dur = max(int(dur / dt), 8)
        it0 = int(rng.integers(0, max(nt - n_dur, 1)))
        f_lo, f_hi = cfg.bst_band
        carrier = _shaped_noise(1, n_dur, dt, f_lo,
                                min(f_hi, 0.8 * nyq), 0.0, rng)[0]
        amp = _logu(rng, *cfg.bst_amp) * ref_rms
        noise[i, it0:it0 + n_dur] += amp * np.hanning(n_dur) * carrier
    return noise, dict(p_trace=p, n_traces=len(hit))


def _patchy(shape: tuple[int, int], dt: float, ref_rms: float,
            cfg: NoiseConfig, rng: np.random.Generator,
            offsets: np.ndarray | None = None):
    """空间相关低频斑块（竖条纹）：风振 / 微震 / 耦合不良区。

    白噪场 → 沿道向高斯平滑（相关长度以【米】计，与 layout 道距解耦）
    → 时间带通 → 全局 RMS 定标 + 逐道 dB 抖动。
    移植自毕设 land_noise.generate_patchy_noise_gpu：保留各向异性纹理，
    把固定参数（2–6 Hz、σ=1.5 道、系数 0.9×0.005）全部随机化并改为
    绝对定标。
    """
    nrec, nt = shape
    if rng.random() >= cfg.ptc_prob:
        return np.zeros(shape), dict(present=False)
    nyq = 0.5 / dt
    f_lo = rng.uniform(*cfg.ptc_f_lo)
    f_hi = min(f_lo * rng.uniform(*cfg.ptc_f_ratio), 0.8 * nyq)
    corr_m = _logu(rng, *cfg.ptc_corr_m)
    if offsets is not None and len(offsets) > 1:
        dxm = float(np.median(np.abs(np.diff(np.asarray(offsets, np.float64)))))
    else:
        dxm = 10.0                                          # 无几何时按 10 m/道
    sig_tr = max(corr_m / max(dxm, 1e-9), 0.3)
    field = gaussian_filter1d(rng.standard_normal(shape), sig_tr,
                              axis=0, mode="nearest")
    f = np.fft.rfftfreq(nt, d=dt)
    field = np.fft.irfft(np.fft.rfft(field, axis=1)
                         * _band_taper(f, f_lo, f_hi)[None, :], n=nt, axis=1)
    rms = float(np.sqrt(np.mean(field ** 2)))
    amp = _logu(rng, *cfg.ptc_amp) * ref_rms
    if rms > 1e-12:
        field *= amp / rms
    field *= _db_jitter(rng, cfg.trace_var_db, nrec)[:, None]
    return field, dict(present=True, f_lo=f_lo, f_hi=float(f_hi),
                       corr_m=corr_m, amp=amp)


def _hum(shape: tuple[int, int], dt: float, ref_rms: float,
         cfg: NoiseConfig, rng: np.random.Generator):
    """共模低频窄带（横向波纹）：记录系统 / 文化干扰的共模拾取。

    同一正弦贯穿全排列，相位沿道向缓慢摆动 → 慢起伏的水平条带。
    移植自毕设 land_noise.BG_linear_noise_gpu（f0=1.5 Hz、周期 40 道
    固定值 → 全随机化，幅度改绝对 RMS 定标）。
    """
    nrec, nt = shape
    if rng.random() >= cfg.hum_prob:
        return np.zeros(shape), dict(present=False)
    f0 = _logu(rng, *cfg.hum_f)
    per = _logu(rng, *cfg.hum_period_tr)
    ph_amp = rng.uniform(*cfg.hum_phase_amp)
    t = np.arange(nt) * dt
    i = np.arange(nrec)
    phase_x = ph_amp * np.sin(2 * np.pi * i / per + rng.uniform(0, 2 * np.pi))
    w = np.sin(2 * np.pi * f0 * t[None, :] + phase_x[:, None]
               + rng.uniform(0, 2 * np.pi))
    amp = _logu(rng, *cfg.hum_amp) * ref_rms
    hum = (np.sqrt(2.0) * amp) * w * _db_jitter(rng, cfg.trace_var_db,
                                                nrec)[:, None]
    return hum, dict(present=True, f=f0, period_tr=per, amp=amp)


def _machinery(shape: tuple[int, int], dt: float, ref_rms: float,
               offsets: np.ndarray | None,
               cfg: NoiseConfig, rng: np.random.Generator):
    """矿山机械/风机谐波梳：随机基频 + 衰减谐波 + 慢节律 AM。

    设备是点源 → 空间高斯足迹（沿排列衰减的竖向亮带 + 水平细条纹）。
    与 powerline 的区别：基频不固定 50/60、谐波结构随机、空间局域。
    HardPicks 四工区均为在产矿区，这类干扰是标配。
    """
    nrec, nt = shape
    if rng.random() >= cfg.mch_prob:
        return np.zeros(shape), dict(present=False)
    nyq = 0.5 / dt
    x = (np.asarray(offsets, np.float64) if offsets is not None
         else np.arange(nrec) * 10.0)
    f0 = _logu(rng, *cfg.mch_f0)
    n_h = int(rng.integers(cfg.mch_n_harm[0], cfg.mch_n_harm[1] + 1))
    decay = rng.uniform(*cfg.mch_harm_decay)
    t = np.arange(nt) * dt
    sig = np.zeros(nt)
    used = 0
    for k in range(n_h):
        fk = f0 * (k + 1)
        if fk >= 0.9 * nyq:
            break
        sig += (decay ** k) * np.sin(2 * np.pi * fk * t
                                     + rng.uniform(0, 2 * np.pi))
        used += 1
    if used == 0:
        return np.zeros(shape), dict(present=False)
    f_am = rng.uniform(*cfg.mch_am_freq)
    depth = rng.uniform(*cfg.mch_am_depth)
    sig *= 1.0 + depth * np.sin(2 * np.pi * f_am * t
                                + rng.uniform(0, 2 * np.pi))
    sig /= max(float(np.sqrt(np.mean(sig ** 2))), 1e-12)     # 时域 RMS=1
    c = rng.uniform(float(x.min()), float(x.max()))
    fp = _logu(rng, *cfg.mch_footprint_m)
    w = np.exp(-0.5 * ((x - c) / fp) ** 2)
    amp = _logu(rng, *cfg.mch_amp) * ref_rms
    noise = ((amp * w * _db_jitter(rng, cfg.trace_var_db, nrec))[:, None]
             * sig[None, :])
    return noise, dict(present=True, f0=f0, n_harm=used, center_m=float(c),
                       footprint_m=fp, amp=amp)


def _gusts(shape: tuple[int, int], dt: float, ref_rms: float,
           offsets: np.ndarray | None,
           cfg: NoiseConfig, rng: np.random.Generator):
    """阵风锋：时空相关爆发包（区别于逐道独立的 bursts）。

    共享时窗 / 空间足迹 / 幅度剖面，锋面以视速度沿排列推进（慢 →
    斜穿的游走噪声条，形似步行/车辆经过；快 → 近同时的宽爆发）；
    道间载波独立（风压包络相干、湍流本身不相干）。
    """
    nrec, nt = shape
    noise = np.zeros(shape)
    if rng.random() >= cfg.gst_prob:
        return noise, dict(present=False)
    nyq = 0.5 / dt
    x = (np.asarray(offsets, np.float64) if offsets is not None
         else np.arange(nrec) * 10.0)
    n_ev = int(rng.integers(cfg.gst_n[0], cfg.gst_n[1] + 1))
    events = []
    for _ in range(n_ev):
        dur = rng.uniform(*cfg.gst_dur)
        n_dur = max(int(dur / dt), 16)
        c = rng.uniform(float(x.min()), float(x.max()))
        fp = _logu(rng, *cfg.gst_footprint_m)
        v = _logu(rng, *cfg.gst_front_v) * float(rng.choice([-1.0, 1.0]))
        t0e = rng.uniform(0.0, max(nt * dt - dur, dt))       # 足迹中心处起时
        f_hi = min(cfg.gst_band[1], 0.8 * nyq)
        amp = _logu(rng, *cfg.gst_amp) * ref_rms
        w = np.exp(-0.5 * ((x - c) / fp) ** 2)
        sel = np.where(w > 0.05)[0]
        if sel.size == 0:
            continue
        carrier = _shaped_noise(sel.size, n_dur, dt,
                                cfg.gst_band[0], f_hi, 0.0, rng)
        env = np.hanning(n_dur)
        jit = _db_jitter(rng, cfg.trace_var_db, sel.size)
        it0 = np.round((t0e + (x[sel] - c) / v) / dt).astype(np.int64)
        for j, i in enumerate(sel):
            a, b = int(it0[j]), int(it0[j]) + n_dur
            ca, cb = max(a, 0), min(b, nt)
            if cb <= ca:                                     # 锋面出记录 → 自然截断
                continue
            noise[i, ca:cb] += ((amp * w[i] * jit[j])
                                * env[ca - a:cb - a] * carrier[j, ca - a:cb - a])
        events.append(dict(t0=float(t0e), dur=float(dur), center_m=float(c),
                           footprint_m=float(fp), front_v=float(v),
                           amp=float(amp)))
    return noise, dict(present=bool(events), events=events)


def _airblast(shape: tuple[int, int], dt: float, ref_peak: float,
              offsets: np.ndarray, t0: float,
              cfg: NoiseConfig, rng: np.random.Generator):
    nrec, nt = shape
    nyq = 0.5 / dt
    present = rng.random() < cfg.air_prob
    noise = np.zeros(shape)
    if not present:
        return noise, dict(present=False)

    f_air = min(rng.uniform(*cfg.air_f), 0.8 * nyq)
    amp0 = _logu(rng, *cfg.air_amp) * ref_peak
    wav, half = _ricker(f_air, dt)
    off = np.abs(np.asarray(offsets, dtype=np.float64))
    t_arr = off / cfg.air_v + t0                            # 峰值到时
    amp = amp0 * (np.maximum(off, 10.0) / 100.0) ** (-cfg.air_decay)
    ipk = np.round(t_arr / dt).astype(np.int64)
    for i in range(nrec):
        a, b = ipk[i] - half, ipk[i] + half + 1
        if b <= 0 or a >= nt:
            continue
        ca, cb = max(a, 0), min(b, nt)
        noise[i, ca:cb] += amp[i] * wav[ca - a:cb - a]
    return noise, dict(present=True, f=f_air, amp=amp0)


# =============================================================================
# 噪声合成
# =============================================================================
def far_fb_amplitude(clean: np.ndarray, label, offsets, dt: float,
                     far_frac: float = 0.3, win_ms: float = 60.0) -> float:
    """远道初至的干净振幅：|炮检距| 最远 far_frac 的有标签道，[fb, fb+win] 的 RMS 中位数。"""
    nrec, nt = clean.shape
    lab = np.asarray(label).astype(np.int64)
    ok = (lab > 0) & (lab < nt)
    if offsets is None or ok.sum() < 8:
        return float("nan")
    off = np.abs(np.asarray(offsets, dtype=np.float64))
    thr = np.quantile(off[ok], 1.0 - far_frac)
    sel = np.flatnonzero(ok & (off >= thr))
    w = max(int(round(win_ms * 1e-3 / dt)), 4)
    a = [np.sqrt(np.mean(clean[i, lab[i]:lab[i] + w] ** 2)) for i in sel if lab[i] + w <= nt]
    return float(np.median(a)) if a else float("nan")


def make_noise(clean: np.ndarray, dt: float, rng: np.random.Generator,
               cfg: NoiseConfig = NoiseConfig(),
               offsets: np.ndarray | None = None,
               t0: float = 0.0, label=None) -> tuple[np.ndarray, dict]:
    """合成总噪声（不修改 clean）。

    Parameters
    ----------
    clean   : (nrec, nt) float 数组，正演输出（未归一化）
    dt      : 采样间隔（s）
    rng     : numpy Generator（逐样本可复现）
    offsets : (nrec,) 带符号炮检距（m）；None 时跳过 airblast
    t0      : 子波峰值延迟（s），与初至标签的时间约定对齐
    label   : (nrec,) 初至标签（哨兵 = nt）；cfg.amb_ref == "far_fb" 时用来定背景水平

    Returns
    -------
    (noise, info)：noise (nrec, nt) float32；info 各分量实际参数
    """
    clean = np.asarray(clean, dtype=np.float64)
    nrec, nt = clean.shape

    # ---- 全局参考水平（强道分位数；死 gather 早退）----
    tr_rms = np.sqrt(np.mean(clean ** 2, axis=1))
    tr_peak = np.max(np.abs(clean), axis=1)
    live = tr_peak > 0
    if not live.any():
        return np.zeros_like(clean, dtype=np.float32), dict(empty=True)
    ref_rms = float(np.quantile(tr_rms[live], cfg.ref_q))
    ref_peak = float(np.quantile(tr_peak[live], cfg.ref_q))

    dx = (float(np.median(np.abs(np.diff(offsets)))) if offsets is not None
          and len(offsets) > 1 else None)
    a_far = float("nan")
    if cfg.amb_ref == "far_fb" and label is not None:
        a_far = far_fb_amplitude(clean, label, offsets, dt, cfg.far_frac, cfg.far_win_ms)
    use_far = np.isfinite(a_far) and a_far > 0
    if use_far:
        amb, i_amb = _ambient((nrec, nt), dt, a_far, cfg, rng, dx=dx, snr_range=cfg.far_snr_db)
    else:                               # global，或远道无标签时回退
        amb, i_amb = _ambient((nrec, nt), dt, ref_rms, cfg, rng, dx=dx)
    pl, i_pl = _powerline((nrec, nt), dt, ref_peak, cfg, rng)
    spk, i_spk = _spikes((nrec, nt), ref_peak, cfg, rng)
    bst, i_bst = _bursts((nrec, nt), dt, ref_rms, cfg, rng)
    ptc, i_ptc = _patchy((nrec, nt), dt, ref_rms, cfg, rng, offsets=offsets)
    hum, i_hum = _hum((nrec, nt), dt, ref_rms, cfg, rng)
    mch, i_mch = _machinery((nrec, nt), dt, ref_rms, offsets, cfg, rng)
    gst, i_gst = _gusts((nrec, nt), dt, ref_rms, offsets, cfg, rng)
    if offsets is not None:
        air, i_air = _airblast((nrec, nt), dt, ref_peak, offsets, t0, cfg, rng)
    else:
        air, i_air = np.zeros((nrec, nt)), dict(present=False)

    other = pl + spk + bst + ptc + hum + mch + gst + air
    # 样本级总闸（恒抽取一次，保证 rng 流与 master_db 取值无关）
    m_db = float(rng.uniform(*cfg.master_db))
    g_m = 10.0 ** (m_db / 20.0)
    other_scale = 1.0
    if use_far:                         # far_fb：背景不过总闸；偶发分量在远道封顶
        far = np.abs(np.asarray(offsets, float)) >= np.quantile(np.abs(np.asarray(offsets, float)),
                                                                1.0 - cfg.far_frac)
        # 背景的时间/空间慢包络、逐道起伏会让远道实际水平偏离目标 ±6 dB → 按远道实测
        # RMS 把背景整体校到目标（形态不变，只校总水平）
        # 在"初至前背景窗" [fb−300, fb−20 ms] 上量（与 generate.far_snr_db 同一把尺子；
        # 时间慢包络让整道 RMS 与初至附近的背景不一样）
        lab = np.asarray(label).astype(np.int64)
        p0, p1 = int(round(0.3 / dt)), int(round(0.02 / dt))
        rows = [i for i in np.flatnonzero(far) if p0 <= lab[i] < nt]
        if len(rows) >= 4:
            r_amb = float(np.median([np.sqrt(np.mean(amb[i, lab[i] - p0:lab[i] - p1] ** 2))
                                     for i in rows]))
        else:
            r_amb = float(np.median(np.sqrt(np.mean(amb[far] ** 2, axis=1))))
        want = a_far * 10.0 ** (-float(i_amb["snr_db"]) / 20.0)
        if r_amb > 0:
            amb = amb * (want / r_amb)
            r_amb = want
        r_oth = float(np.median(np.sqrt(np.mean((other[far] * g_m) ** 2, axis=1))))
        cap = r_amb * 10.0 ** (cfg.far_other_cap_db / 20.0)
        if r_oth > cap > 0:
            other_scale = cap / r_oth
        noise = amb + other * (g_m * other_scale)
    else:
        noise = (amb + other) * g_m
    info = dict(ref_rms=ref_rms, ref_peak=ref_peak, master_db=m_db,
                amb_ref="far_fb" if use_far else "global", far_amp=a_far,
                other_scale=float(other_scale),
                ambient=i_amb, powerline=i_pl,
                spikes=i_spk, bursts=i_bst, patchy=i_ptc, hum=i_hum,
                machinery=i_mch, gusts=i_gst,
                airblast=i_air)
    return noise.astype(np.float32), info


# =============================================================================
# 道缺陷 + 总编排
# =============================================================================
def corrupt(clean: np.ndarray, label: np.ndarray, dt: float,
            rng: np.random.Generator,
            ncfg: NoiseConfig = NoiseConfig(),
            dcfg: DefectConfig = DefectConfig(),
            offsets: np.ndarray | None = None,
            t0: float = 0.0,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """步骤 7 入口：clean gather → noisy gather + 修订标签。

    顺序：抽死道 → 清信号 → 增益/极性（作用于信号）→ 加噪
          → 硬零死道抹平 → 削波。

    Returns
    -------
    noisy     : (nrec, nt) float32
    label_out : (nrec,) 与输入同 dtype；死道 → nt 哨兵
    alive     : (nrec,) bool，False 仅表示"硬零死道"
                （dead_noisy 道保留在训练里学习 no_fb，见 info）
    info      : 全部实际抽取参数（可整体 json 化落盘）
    """
    clean = np.asarray(clean, dtype=np.float64)
    nrec, nt = clean.shape
    label_out = np.array(label, copy=True)
    sentinel = label_out.dtype.type(nt)

    # ---- 死道抽取（硬零 / 只剩噪声两型）----
    dead_hard = np.zeros(nrec, dtype=bool)
    dead_noisy = np.zeros(nrec, dtype=bool)
    if rng.random() < dcfg.dead_prob:
        frac = rng.uniform(*dcfg.dead_frac)
        n_dead = int(nrec * frac)
        if n_dead > 0:
            idx = rng.choice(nrec, size=n_dead, replace=False)
            noisy_type = rng.random(n_dead) < dcfg.dead_noisy_ratio
            dead_noisy[idx[noisy_type]] = True
            dead_hard[idx[~noisy_type]] = True

    sig = clean.copy()
    sig[dead_hard | dead_noisy] = 0.0
    label_out[dead_hard | dead_noisy] = sentinel

    # ---- 逐道增益抖动（耦合差异，作用于信号）----
    if dcfg.gain_db_std > 0:
        gain = 10.0 ** (rng.normal(0.0, dcfg.gain_db_std, size=nrec) / 20.0)
        sig *= gain[:, None]

    # ---- 极性反转 ----
    p_flip = rng.uniform(*dcfg.polarity_p)
    flip = rng.random(nrec) < p_flip
    sig[flip] *= -1.0

    # ---- 噪声（参考水平用原始 clean 计算，避免死道/增益扰动基准）----
    noise, n_info = make_noise(clean, dt, rng, ncfg, offsets=offsets, t0=t0, label=label)
    noise = noise.astype(np.float64)
    noise[dead_hard] = 0.0                                  # 硬零死道无任何输出

    noisy = sig + noise

    # ---- 谐振耦合道（坏耦合：全部输入过阻尼谐振支路 → 拖尾振铃）----
    # 作用于 sig+noise 之后：耦合谐振对地动与环境噪声一视同仁。
    # 恒等支路保留原 onset，label 不变 —— 训练难例：波形拖尾但初至仍在。
    p_ring = rng.uniform(*dcfg.ring_p)
    ring_tr = np.where(rng.random(nrec) < p_ring)[0]
    ring_info = []
    for i in ring_tr:
        fr = rng.uniform(*dcfg.ring_f)
        Q = rng.uniform(*dcfg.ring_q)
        gn = rng.uniform(*dcfg.ring_gain)
        n_h = int(np.clip(3.0 * Q / (np.pi * fr) / dt, 16, nt))  # ~3 个衰减时长
        th = np.arange(n_h) * dt
        h = np.exp(-np.pi * fr * th / Q) * np.sin(2 * np.pi * fr * th)
        s_h = float(np.abs(h).sum())
        if s_h > 1e-12:
            noisy[i] += gn * np.convolve(noisy[i], h / s_h)[:nt]
        ring_info.append(dict(trace=int(i), f=float(fr), q=float(Q),
                              gain=float(gn)))

    # ---- 削波 ----
    p_clip = rng.uniform(*dcfg.clip_p)
    clip_tr = np.where(rng.random(nrec) < p_clip)[0]
    for i in clip_tr:
        pk = np.abs(noisy[i]).max()
        if pk > 0:
            lv = rng.uniform(*dcfg.clip_level) * pk
            np.clip(noisy[i], -lv, lv, out=noisy[i])

    # ---- telemetry 掉段：随机零段；覆盖初至 ±guard → label 置哨兵 ----
    p_drop = rng.uniform(*dcfg.drop_p)
    drop_tr = np.where(rng.random(nrec) < p_drop)[0]
    drops = []
    for i in drop_tr:
        n_gap = max(int(rng.uniform(*dcfg.drop_dur) / dt), 2)
        it0 = int(rng.integers(0, max(nt - n_gap, 1)))
        noisy[i, it0:it0 + n_gap] = 0.0
        fb_i = int(label_out[i])
        erased = (fb_i != nt
                  and it0 - dcfg.drop_guard <= fb_i <= it0 + n_gap + dcfg.drop_guard)
        if erased:                       # onset 被抹掉 → 无可学的初至
            label_out[i] = sentinel
        drops.append(dict(trace=int(i), it0=it0, n=int(n_gap),
                          fb_erased=bool(erased)))

    alive = ~dead_hard
    info = dict(
        noise=n_info,
        dead_hard_idx=np.where(dead_hard)[0].tolist(),
        dead_noisy_idx=np.where(dead_noisy)[0].tolist(),
        polarity_idx=np.where(flip)[0].tolist(),
        clip_idx=clip_tr.tolist(),
        ring=ring_info,
        dropouts=drops,
        gain_db_std=dcfg.gain_db_std,
    )
    return noisy.astype(np.float32), label_out, alive, info