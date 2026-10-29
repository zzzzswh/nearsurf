"""
nearsurf.land —— 在 2D 速度模型上加起伏地表 + 风化带（陆上近地表）

    from nearsurf import land as ns                   # 或 nearsurf.land_model(vp_deep, dh, "loess", seed=0)
    res = ns.build(vp_deep, dh, "loess", rng)        # vp_deep: (nz, nx) m/s，第 0 行是模型顶
    res.vp            # (nz, nx)，地表以上 = 0（空气）
    res.surf_idx      # (nx,) int，每列第一个固体格点的行号（SWEEP 的 topography）
    res.truth         # 地表 / 潜水面 / 风化层底界 等真值曲线
    ns.plot(path, res, dh)

场景（SCENARIOS，均匀抽样用 ns.assign）
-----------------------------------------
  plain      平原      起伏 5~30 m，风化层薄，潜水面浅，可能有古河道、钙质硬壳
  hills      丘陵      起伏 40~150 m，山顶浑圆，沟谷风化层厚
  foothills  山前带    起伏 150~350 m，山脊尖锐，沟谷砾石层厚，潜水面深，非均质强
  loess      黄土塬    平坦塬面 + 深切冲沟，厚层低速黄土（塬上厚、沟底薄），潜水面深
  desert     沙漠      不对称沙丘（缓迎风坡、陡背风坡），极低速干沙，戈壁钙质硬壳
  flat       平坦地表  没有起伏，风化层厚度、潜水面、速度结构照常（作为起伏地表的对照组）
  shield     结晶地盾  起伏 10~60 m，薄冰碛（3~20 m，横向变化大）直接盖在 5.5 km/s 以上的基岩上，
                       潜水面浅，冰碛/基岩接触突变（硬岩矿区；配合 configs/synth2d_hardrock.yaml）

构建步骤
--------
1. 地形 elev(x)：按场景生成（分形布朗运动 + 坡面扩散 / 塬面+冲沟 / 沙丘），
   缩放到抽到的起伏幅度并限坡度；向下"切"进模型（侵蚀截断地层），
   最高点 = 模型顶 + air_top。
2. 风化层厚度 th(x)：随地貌变化 —— 凹处（沟谷）厚、陡坡薄（土壤生成模型的思路）；
   黄土按高程（塬上厚、沟底薄），沙漠 = 沙丘体本身 + 底部薄层。
3. 潜水面：地形的平缓版（深度在山下大、沟谷里小）。
4. 速度（d = 地表以下深度）：
      干燥层   v = v0 · (1 + d/d0)^n                      松散介质幂律梯度
      潜水面下 v = max(v_dry, v_sat + g_sat·(d - wt))      饱和跳变
      钙质硬壳 / 古河道充填（按概率）                       速度反转 / 低速透镜体
      风化层底 smoothstep 过渡到深部模型，深部顶面先降 frac 再指数恢复（裂隙化基岩）
      2D 各向异性随机场 × 随深度衰减                       近地表非均质
   最后 v >= v_floor（数值频散的下限：正演网格分辨不了更慢的波，由调用方按网格和主频给）。

单位：长度 m，速度 m/s；elev 向上为正，depth 向下为正。

依据（参数区间的来源，详见 docs/references.md）
------------------------------------------------------------
  地形谱       地形剖面功率谱 ~ k^-2（Hurst ≈ 0.5），短波长端由坡面过程截断出特征尺度
               （Turcotte 1992；Perron et al. 2008）-> hurst 0.4~0.75 + lmin + 扩散
  坡面扩散     q = K·S / (1 - (S/Sc)^2)（Roering et al. 1999）：山顶凸、坡面趋直
  土层厚度     与曲率线性相关，斜率随曲率标准差缩放（Patton et al. 2018）-> th_curv·cn
  风化底界     平行或"镜像"地形取决于构造应力（St. Clair et al. 2015）-> mirror
  潜水面       湿润、浅潜水面时近似地形的平缓版，干旱 / 深潜水面时基本脱钩
               （Haitjema & Mitchell-Bruker 2005）-> wt_follow 按场景递减
  干燥层梯度   颗粒介质 Hertz-Mindlin 理论 v ∝ σ^(1/6) ≈ z^(1/6)，实测幂指数略高
               （Bachrach et al. 2000）-> n 0.15~0.30
  饱和跳变     饱和松散沉积物 Vp 1500~2000 m/s；实测降速层 200~1150 m/s，
               高速层 1350~2460 m/s（多地微测井统计）
  沙丘         迎风坡 5~10°，滑落面接近休止角 32~34°（Pelletier et al. 2015 等）
  高速夹层     近地表薄高速层造成速度反转、初至"叠瓦"，层析静校正难以处理
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# =============================================================== 场景参数
# 每项是 (lo, hi) 均匀抽样区间，或标量。
SCENARIOS: dict[str, dict] = {
    "plain": dict(
        shape="fbm", relief=(5.0, 30.0), hurst=(0.4, 0.6), lmin=(150.0, 300.0),
        diffusion=(10.0, 40.0), max_slope=0.15, tilt=(-0.004, 0.004),
        v0=(450.0, 700.0), n=(0.15, 0.25), d0=(1.0, 3.0),
        th_base=(4.0, 12.0), th_curv=(2.0, 5.0), th_slope=(0.0, 10.0), mirror=(0.0, 0.1),
        th_lo=2.0, th_hi=20.0,
        wt_mean=(2.0, 8.0), wt_follow=(0.4, 0.8), v_sat=(1500.0, 1700.0), g_sat=(5.0, 15.0),
        trans=(3.0, 8.0), frac=(0.0, 0.10), het=(0.04, 0.08),
        p_caliche=0.2, p_channel=0.5,
    ),
    "hills": dict(
        shape="fbm", relief=(40.0, 150.0), hurst=(0.45, 0.65), lmin=(80.0, 150.0),
        diffusion=(30.0, 80.0), max_slope=0.45, tilt=(-0.01, 0.01),
        v0=(500.0, 900.0), n=(0.15, 0.25), d0=(1.0, 3.0),
        th_base=(6.0, 15.0), th_curv=(4.0, 10.0), th_slope=(5.0, 15.0), mirror=(0.0, 0.3),
        th_lo=2.0, th_hi=40.0,
        wt_mean=(5.0, 20.0), wt_follow=(0.3, 0.7), v_sat=(1600.0, 1900.0), g_sat=(5.0, 15.0),
        trans=(5.0, 12.0), frac=(0.05, 0.15), het=(0.05, 0.10),
        p_caliche=0.0, p_channel=0.3,
    ),
    "foothills": dict(
        shape="fbm", relief=(150.0, 350.0), hurst=(0.5, 0.75), lmin=(50.0, 100.0),
        diffusion=(5.0, 25.0), max_slope=0.70, tilt=(-0.03, 0.03),
        v0=(600.0, 1200.0), n=(0.15, 0.22), d0=(1.0, 4.0),
        th_base=(5.0, 12.0), th_curv=(8.0, 18.0), th_slope=(5.0, 12.0), mirror=(0.1, 0.5),
        th_lo=1.0, th_hi=60.0,
        wt_mean=(15.0, 60.0), wt_follow=(0.2, 0.5), v_sat=(1800.0, 2200.0), g_sat=(5.0, 20.0),
        trans=(5.0, 15.0), frac=(0.10, 0.25), het=(0.06, 0.12),
        p_caliche=0.0, p_channel=0.0,
    ),
    "loess": dict(
        shape="loess", relief=(60.0, 160.0), hurst=(0.4, 0.6), lmin=(300.0, 600.0),
        diffusion=(5.0, 15.0), max_slope=0.90, tilt=(-0.005, 0.005),
        n_gully=(2, 6), gully_w=(80.0, 300.0), gully_p=(0.4, 0.8),
        v0=(450.0, 700.0), n=(0.20, 0.30), d0=(2.0, 6.0),
        th_top=(50.0, 150.0), th_floor=(3.0, 15.0), th_q=(0.7, 1.5),
        wt_mean=(30.0, 80.0), wt_follow=(0.0, 0.2), v_sat=(1700.0, 2000.0), g_sat=(3.0, 10.0),
        trans=(5.0, 12.0), frac=(0.0, 0.10), het=(0.03, 0.06),
        p_caliche=0.0, p_channel=0.0,
    ),
    "shield": dict(               # 冰川作用后的结晶地盾（硬岩矿区：Sudbury / Lalor / Brunswick / Halfmile 类）
        # 薄层冰碛覆盖在结晶基岩上：起伏小（湖泊、鼓丘、基岩露头）、覆盖层薄且横向变化大，
        # 潜水面浅（湿润）、冰碛与基岩接触面突变（trans 小），基岩顶部有裂隙降速带（frac）。
        # 速度参考：冰碛 1500~2500 m/s（饱和），干燥表层更低；基岩 5500~6200 m/s 由深部模型给出。
        shape="fbm", relief=(10.0, 60.0), hurst=(0.5, 0.7), lmin=(80.0, 200.0),
        diffusion=(5.0, 20.0), max_slope=0.30, tilt=(-0.005, 0.005),
        v0=(600.0, 1000.0), n=(0.15, 0.25), d0=(1.0, 3.0),
        th_base=(3.0, 20.0), th_curv=(3.0, 8.0), th_slope=(2.0, 8.0), mirror=(0.0, 0.2),
        th_lo=0.5, th_hi=45.0,
        wt_mean=(1.0, 4.0), wt_follow=(0.6, 0.9), v_sat=(1700.0, 2300.0), g_sat=(5.0, 20.0),
        trans=(1.0, 3.0), frac=(0.05, 0.20), het=(0.05, 0.10),
        p_caliche=0.0, p_channel=0.2,
    ),
    "flat": dict(                 # 平坦地表：没有起伏，风化层 / 潜水面 / 速度结构照常（对照组）
        shape="flat", relief=0.0, hurst=0.5, lmin=200.0, diffusion=0.0, max_slope=0.15, tilt=0.0,
        v0=(450.0, 800.0), n=(0.15, 0.25), d0=(1.0, 3.0),
        th_base=(4.0, 15.0), th_curv=0.0, th_slope=0.0, mirror=0.0, th_lo=2.0, th_hi=25.0,
        wt_mean=(2.0, 10.0), wt_follow=0.0, v_sat=(1500.0, 1800.0), g_sat=(5.0, 15.0),
        trans=(3.0, 10.0), frac=(0.0, 0.10), het=(0.04, 0.08),
        p_caliche=0.2, p_channel=0.3,
    ),
    "desert": dict(
        shape="dunes", relief=(8.0, 40.0), stoss_deg=(5.0, 10.0), lee_deg=(28.0, 33.0),
        dune_gap=(0.0, 1.0), base_relief=(2.0, 10.0), diffusion=(2.0, 5.0),
        max_slope=0.68, tilt=(-0.003, 0.003),
        v0=(450.0, 600.0), n=(0.15, 0.25), d0=(0.5, 2.0),
        th_sand=(2.0, 10.0),
        wt_mean=(20.0, 80.0), wt_follow=(0.0, 0.1), v_sat=(1700.0, 2000.0), g_sat=(3.0, 10.0),
        trans=(3.0, 8.0), frac=(0.0, 0.10), het=(0.03, 0.06),
        p_caliche=0.6, p_channel=0.0,
    ),
}
NAMES = tuple(SCENARIOS)


@dataclasses.dataclass
class NearSurface:
    vp: np.ndarray            # (nz, nx)，空气 = 0
    surf_idx: np.ndarray      # (nx,) int64，第一个固体格点行号
    surf_z: np.ndarray        # (nx,) 地表深度（相对模型顶，m）
    elev: np.ndarray          # (nx,) 地形高程（最高点 = air_top 处，相对量，m）
    wt_z: np.ndarray          # (nx,) 潜水面深度（相对模型顶，m）
    base_z: np.ndarray        # (nx,) 风化层底界深度（相对模型顶，m）
    scenario: str
    params: dict
    vs: np.ndarray | None = None   # (nz, nx) 传了 vs_deep 才有：横波速度，空气 = 0


# =============================================================== 抽样小工具
def _u(rng, v):
    if isinstance(v, tuple):
        lo, hi = v
        if isinstance(lo, int) and isinstance(hi, int):
            return int(rng.integers(lo, hi + 1))
        return float(rng.uniform(lo, hi))
    return v


def sample_params(scenario: str, rng, overrides: dict | None = None) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError(f"未知场景 {scenario!r}，可选 {NAMES}")
    spec = dict(SCENARIOS[scenario])
    spec.update(overrides or {})
    return {k: _u(rng, tuple(v) if isinstance(v, list) else v) for k, v in spec.items()}


def assign(n: int, scenarios, seed: int = 0) -> list[str]:
    """n 个模型均匀分配场景：每一轮把场景打乱一次再依次发，任意前缀都近似均匀。"""
    rng = np.random.default_rng(seed)
    out: list[str] = []
    while len(out) < n:
        out += list(rng.permutation(list(scenarios)))
    return out[:n]


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / max(hi - lo, 1e-12)


def _corr1d(nx, dh, length_m, rng):
    """相关长度 length_m 的零均值、单位标准差随机曲线。"""
    s = gaussian_filter1d(rng.standard_normal(nx), max(length_m / dh, 0.5), mode="reflect")
    return s / max(s.std(), 1e-12)


# =============================================================== 地形
def fbm(nx, dh, hurst, lmin, rng):
    """1D 分形布朗运动：P(k) ∝ k^-(2H+1)，波长短于 lmin 的分量去掉。归一化到 [0, 1]。"""
    k = np.fft.rfftfreq(nx, dh)
    amp = np.zeros_like(k)
    amp[1:] = k[1:] ** (-(2 * hurst + 1) / 2)
    amp[k > 1.0 / lmin] = 0.0
    spec = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, k.size))
    return _norm01(np.fft.irfft(spec, n=nx))


def hillslope_diffusion(h, dh, length_m, s_crit=0.8):
    """非线性坡面扩散（Roering 1999 型）：山顶磨圆，陡处搬运加快、沟谷保持尖锐。

    length_m 是扩散长度尺度（≈ 被磨圆的特征宽度）。显式格式，步长按稳定条件取。
    """
    if length_m <= 0:
        return h
    h = h.astype(np.float64).copy()
    K = 1.0
    t_total = length_m ** 2 / (2 * K)
    t = 0.0
    while t < t_total:
        hx = np.diff(h) / dh                                   # 半格点坡度
        denom = np.clip(1.0 - (hx / s_crit) ** 2, 0.1, None)
        D = K / denom
        dt = min(0.4 * dh * dh / D.max(), t_total - t)
        q = -D * hx                                            # 通量
        div = np.zeros_like(h)
        div[1:-1] = (q[1:] - q[:-1]) / dh
        h[1:-1] -= dt * div[1:-1]
        h[0], h[-1] = h[1], h[-2]
        t += dt
    return h


def limit_slope(h, dh, max_slope):
    """压缩起伏使最大坡度不超过 max_slope（保形状）。"""
    s = np.abs(np.gradient(h, dh)).max()
    if s <= max_slope or s == 0:
        return h, 1.0
    f = max_slope / s
    m = h.mean()
    return m + (h - m) * f, f


def topo_fbm(nx, dh, p, rng):
    h = fbm(nx, dh, p["hurst"], p["lmin"], rng)
    h = hillslope_diffusion(h * p["relief"], dh, p["diffusion"])
    return _norm01(h)


def topo_loess(nx, dh, p, rng):
    """黄土塬：平缓塬面（低起伏 fBm）+ 若干深切冲沟（陡壁、底部较平）。"""
    x = np.arange(nx) * dh
    plateau = 0.12 * fbm(nx, dh, p["hurst"], p["lmin"], rng)
    cut = np.zeros(nx)
    L = x[-1]
    for _ in range(p["n_gully"]):
        c = rng.uniform(0.05, 0.95) * L
        w = p["gully_w"] * rng.uniform(0.6, 1.4)
        depth = rng.uniform(0.5, 1.0)
        # 沟壁占半宽的 gully_p，中间是平沟底；smoothstep 让沟沿和沟脚圆滑
        r = np.clip((1.0 - np.abs(x - c) / w) / p["gully_p"], 0.0, 1.0)
        cut = np.maximum(cut, depth * r * r * (3 - 2 * r))
    h = plateau + 1.0 - cut
    h = hillslope_diffusion(h * p["relief"], dh, p["diffusion"], s_crit=1.2)
    return _norm01(h)


def topo_dunes(nx, dh, p, rng):
    """横向沙丘：迎风坡 stoss_deg（5~10°）、背风滑落面 lee_deg（接近休止角 32~34°），
    丘高沿测线缓变，丘间留 dune_gap 倍丘宽的平坦低地；坐在低起伏的戈壁面上。

    丘长由丘高和两侧坡角决定：L = H/tan(stoss) + H/tan(lee)，不单独抽波长，
    这样坡角始终落在实测范围内。返回 (归一化地形, 沙丘高度/relief)。
    """
    H = p["relief"] * np.clip(0.6 + 0.35 * _corr1d(nx, dh, 1500.0, rng), 0.2, 1.0)
    ls = H / np.tan(np.radians(p["stoss_deg"]))
    ll = H / np.tan(np.radians(p["lee_deg"]))
    gap = p["dune_gap"] * (ls + ll) * np.clip(1 + 0.5 * _corr1d(nx, dh, 800.0, rng), 0.2, 2.0)
    lam = ls + ll + gap
    phase = np.cumsum(dh / lam) + rng.uniform()
    f = phase % 1.0
    a_s, a_l = ls / lam, ll / lam
    prof = np.where(f < a_s, f / a_s,
                    np.where(f < a_s + a_l, 1.0 - (f - a_s) / a_l, 0.0))
    dunes = prof * H / p["relief"]                            # 相对 relief 的沙丘高度
    base = p["base_relief"] / max(p["relief"], 1e-6) * fbm(nx, dh, 0.5, 300.0, rng)
    h = hillslope_diffusion((dunes + base) * p["relief"], dh, p["diffusion"], s_crit=0.75)
    return _norm01(h), dunes


# =============================================================== 主函数
def build(vp_deep: np.ndarray, dh: float, scenario: str, rng: np.random.Generator,
          overrides: dict | None = None, v_floor: float = 450.0,
          air_top: float = 0.0, vs_deep: np.ndarray | None = None,
          vs_floor: float = 250.0, vpvs_dry: tuple = (1.5, 2.0),
          th_scale: float = 1.0, cap_ratio: float | None = None) -> NearSurface:
    """vp_deep (nz, nx) m/s，第 0 行 = 模型顶。返回加了近地表的模型。

    传入 vs_deep 时同时构建横波速度（弹性波用）：
      干燥层   Vs = Vp_dry / r，r ∈ vpvs_dry（干燥松散介质泊松比低，Vp/Vs 约 1.5~2）
      潜水面下 Vs 不随饱和跳变（流体不承受剪切，Gassmann），沿干燥层的梯度继续
      风化层底 与深部 Vs 用同一个 smoothstep 过渡；非均质场与 Vp 共用
      最后 vs_floor <= Vs <= Vp/1.42（保证泊松比为正）
    th_scale 把风化层厚度整体放大（面波强度控制，见 synth2d 的 nearsurface.groundroll）。
    Vs 相关的随机数全部在 Vp 之后抽，所以同一个种子下 Vp 与只做声波时完全一致。
    """
    nz, nx = vp_deep.shape
    p = sample_params(scenario, rng, overrides)
    x = np.arange(nx) * dh
    if p["v0"] < v_floor:          # 网格粗、速度下限高时：整体抬高，保留梯度形态（而不是削平）
        p["v0_sampled"] = p["v0"]
        p["v0"] = v_floor * float(rng.uniform(1.0, 1.25))
    p["v_sat"] = max(p["v_sat"], 1.3 * p["v0"])                # 饱和层始终明显快于干燥层

    # ---------- 1. 地形
    dunes = None
    if p["shape"] == "fbm":
        u = topo_fbm(nx, dh, p, rng)
    elif p["shape"] == "loess":
        u = topo_loess(nx, dh, p, rng)
    elif p["shape"] == "dunes":
        u, dunes = topo_dunes(nx, dh, p, rng)
    elif p["shape"] == "flat":
        u = np.zeros(nx)
    else:
        raise ValueError(p["shape"])
    elev = u * p["relief"] + p["tilt"] * (x - x.mean())
    elev, slope_f = limit_slope(elev, dh, p["max_slope"])
    elev = elev - elev.max()                                   # 最高点 = 0
    surf_z = air_top - elev                                    # 地表深度（向下为正）
    relief = float(-elev.min()) + 0.0                         # +0.0：平坦时避免 -0.0
    if surf_z.max() > 0.5 * (nz - 1) * dh:
        raise ValueError(f"起伏 {relief:.0f} m 超过模型深度的一半，模型太浅")

    # ---------- 2. 风化层厚度
    slope = np.abs(np.gradient(elev, dh))
    if scenario == "loess":
        hn = _norm01(elev)                                     # 塬上 1，沟底 0
        th = p["th_floor"] + (p["th_top"] - p["th_floor"]) * hn ** p["th_q"]
        th *= 1 + 0.1 * _corr1d(nx, dh, 150.0, rng)
    elif scenario == "desert":
        sand = (dunes * p["relief"] * slope_f) if dunes is not None else 0.0
        th = p["th_sand"] + sand
    else:
        # 浅部（土壤 / 可移动风化物）：厚度与曲率线性相关，斜率随流域曲率标准差缩放，
        # 截距 = 平直坡（曲率 0）处的厚度（Patton et al. 2018）。凹处厚、凸处薄。
        curv = gaussian_filter1d(np.gradient(np.gradient(elev, dh), dh), 40.0 / dh)
        cn = curv / max(float(curv.std()), 1e-12)
        sn = slope / max(p["max_slope"], 1e-6)
        # 深部（裂隙化 / 风化基岩底界）：长波长上要么平行地表（mirror=0），
        # 要么"镜像"地形、山脊下更厚（mirror>0，水平构造挤压强时；St. Clair et al. 2015）
        es = gaussian_filter1d(elev, 400.0 / dh, mode="reflect")
        th = (p["th_base"] + p["th_curv"] * cn - p["th_slope"] * sn
              + p["mirror"] * (es - es.mean())
              + 0.25 * p["th_base"] * _corr1d(nx, dh, 100.0, rng))
    th = gaussian_filter1d(np.clip(th, p.get("th_lo", 1.0), p.get("th_hi", 1e9)), 10.0 / dh)
    th = np.clip(th, 1.0, None) * th_scale                  # th_scale：面波强度控制用
    base_z = surf_z + th

    # ---------- 3. 潜水面（地形的平缓版）
    sm = gaussian_filter1d(elev, 300.0 / dh, mode="reflect")
    wt_elev = p["wt_follow"] * (sm - sm.mean())
    wt_d = elev - wt_elev
    wt_d = wt_d - wt_d.mean() + p["wt_mean"]
    wt_d = np.clip(wt_d, 0.5, None)
    wt_z = surf_z + wt_d

    # ---------- 4. 速度
    z = (np.arange(nz) * dh)[:, None]
    d = z - surf_z[None, :]                                    # 地表以下深度
    dp = np.clip(d, 0, None)
    v_dry = p["v0"] * (1.0 + dp / p["d0"]) ** p["n"]
    # cap_ratio：风化层（低速带）本体不快于该列风化层底下方的深部速度 × cap_ratio。
    # 幂律梯度在厚风化层（黄土塬可达 200 m）底部会冲到 3000 m/s 以上（v0 被 v_floor 抬高后更严重），
    # 比下面的深部还快 —— 整片速度倒转，远道初至沿高速"硬壳"传播、被冲沟切断后几乎消失。
    # 钙质硬壳（薄、斑块状）不受限：薄的局部高速层是真实存在的近地表倒转。None = 不限（旧行为）。
    cap = None
    if cap_ratio is not None:
        brow = np.clip(np.ceil(base_z / dh).astype(np.int64) + 1, 0, nz - 1)
        cap = (cap_ratio * vp_deep[brow, np.arange(nx)])[None, :]
        cap = np.maximum(cap, v_floor * 1.05)
        v_dry = np.minimum(v_dry, cap)
        p["cap_ratio"] = float(cap_ratio)
    v_wet = np.maximum(v_dry, p["v_sat"] + p["g_sat"] * (dp - wt_d[None, :]))
    if cap is not None:
        v_wet = np.minimum(v_wet, cap)
    v_ns = np.where(d < wt_d[None, :], v_dry, v_wet)

    extras = {}
    cal_mask, cal_v, ch_mask, ch_fac = None, None, None, None
    if rng.random() < p["p_caliche"]:                          # 钙质硬壳：浅部高速薄层，斑块状
        c_top = rng.uniform(0.5, 4.0)
        c_th = rng.uniform(1.5, 5.0)
        c_v = rng.uniform(1500.0, 2500.0)
        patch = _corr1d(nx, dh, rng.uniform(100.0, 400.0), rng) > rng.uniform(-0.8, 0.3)
        if dunes is not None:                                  # 沙漠：只在丘间低地
            patch &= dunes < 0.25
        m = (d >= c_top) & (d < c_top + c_th) & patch[None, :] & (d < th[None, :])
        v_ns = np.where(m, c_v, v_ns)
        cal_mask, cal_v = m, c_v
        extras["caliche"] = dict(top=c_top, thick=c_th, v=c_v, cover=float(patch.mean()))
    if rng.random() < p["p_channel"]:                          # 古河道充填：低处的低速透镜体
        if np.ptp(elev) > 0:
            lows = np.argsort(elev)[: max(1, nx // 20)]
            cx = x[int(rng.choice(lows))]
        else:                                                  # 平坦地表：没有"低处"，随机放
            cx = float(rng.uniform(0.1, 0.9) * x[-1])
        w = rng.uniform(80.0, 300.0)
        dep = rng.uniform(5.0, 25.0)
        r = np.clip(1 - ((x - cx) / w) ** 2, 0, None)
        lens_bot = dep * np.sqrt(r)
        m = (d >= 0) & (d < lens_bot[None, :])
        ch_fac = float(rng.uniform(0.7, 0.85))
        v_ns = np.where(m, np.maximum(v_ns * ch_fac, v_floor), v_ns)
        ch_mask = m
        base_z = np.maximum(base_z, surf_z + lens_bot)
        extras["channel"] = dict(x=float(cx), half_width=float(w), depth=float(dep))

    # 风化层底：smoothstep 过渡到深部模型；深部顶面裂隙化（先降 frac 再恢复）
    s = np.clip((d - (base_z - surf_z)[None, :]) / p["trans"] + 0.5, 0.0, 1.0)
    w = s * s * (3 - 2 * s)
    below = np.clip(d - (base_z - surf_z)[None, :], 0, None)
    deep = vp_deep * (1.0 - p["frac"] * np.exp(-below / 20.0))
    v = (1.0 - w) * v_ns + w * deep

    # 2D 各向异性非均质（水平相关长、垂直短），随深度衰减
    g = gaussian_filter(rng.standard_normal((nz, nx)),
                        (max(3.0 / dh, 0.5), max(rng.uniform(30.0, 150.0) / dh, 1.0)))
    g /= max(g.std(), 1e-12)
    v = v * (1.0 + p["het"] * g * np.exp(-dp / 60.0))

    v = np.maximum(v, v_floor)
    v[d < 0] = 0.0
    surf_idx = np.ceil(surf_z / dh - 1e-6).astype(np.int64)    # 第一个固体格点
    for ix in range(nx):                                       # 与离散化后的空气掩码一致
        v[: surf_idx[ix], ix] = 0.0
        v[surf_idx[ix]:, ix] = np.maximum(v[surf_idx[ix]:, ix], v_floor)

    vs_out = None
    if vs_deep is not None:
        r_dry = float(rng.uniform(*vpvs_dry))
        vs_ns = v_dry / r_dry                                   # 饱和不改变 Vs
        if cal_mask is not None:
            vs_ns = np.where(cal_mask, cal_v / 1.8, vs_ns)
        if ch_mask is not None:
            vs_ns = np.where(ch_mask, vs_ns * ch_fac, vs_ns)
        deep_s = vs_deep * (1.0 - p["frac"] * np.exp(-below / 20.0))
        vs_out = (1.0 - w) * vs_ns + w * deep_s
        vs_out = vs_out * (1.0 + p["het"] * g * np.exp(-dp / 60.0))
        vs_out = np.minimum(np.maximum(vs_out, vs_floor), v / 1.42)
        vs_out[v == 0] = 0.0
        vs_out = vs_out.astype(np.float32)
        p["vpvs_dry"] = r_dry

    p_out = {k: (float(v_) if isinstance(v_, (float, np.floating)) else v_)
             for k, v_ in p.items()}
    if vs_out is not None:
        p_out["vs_solid_min"] = float(vs_out[vs_out > 0].min())
    p_out.update(relief_final=relief, slope_scale=float(slope_f),
                 th_min=float(th.min()), th_max=float(th.max()),
                 wt_min=float(wt_d.min()), wt_max=float(wt_d.max()),
                 v_solid_min=float(v[v > 0].min()), extras=extras)
    return NearSurface(vp=v.astype(np.float32), surf_idx=surf_idx, surf_z=surf_z.astype(np.float32),
                       elev=elev.astype(np.float32), wt_z=wt_z.astype(np.float32),
                       base_z=base_z.astype(np.float32), scenario=scenario, params=p_out,
                       vs=vs_out)


# =============================================================== 作图
def plot(path, res: NearSurface, dh: float, title: str = "", depth_below: float = 200.0,
         vmax: float | None = None):
    """近地表放大图：上 = 速度 + 地表/潜水面/风化层底，下 = 三列速度-深度曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nz, nx = res.vp.shape
    x = np.arange(nx) * dh
    zb = float(min(res.surf_z.max() + depth_below, (nz - 1) * dh))
    zt = float(res.surf_z.min()) - 10.0
    vm = np.ma.masked_equal(res.vp, 0.0)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw=dict(height_ratios=[2.2, 1]))
    im = a0.imshow(vm, cmap="jet", extent=[0, x[-1], (nz - 1) * dh, 0], aspect="auto",
                   vmin=float(vm.min()), vmax=vmax or float(np.percentile(vm.compressed(), 99)))
    a0.plot(x, res.wt_z, "w:", lw=1.0, label="water table")
    a0.plot(x, res.base_z, "k--", lw=0.8, label="weathering base")
    a0.set_ylim(zb, zt)
    a0.set_ylabel("depth (m)")
    a0.legend(loc="lower right", fontsize=8)
    a0.set_title(title or f"near surface: {res.scenario}", fontsize=10)
    fig.colorbar(im, ax=a0, pad=0.01, label="Vp (m/s)")
    cols = [int(np.argmin(res.elev)), int(np.argmax(res.elev)), nx // 2]
    for c, lab in zip(cols, ("lowest", "highest", "middle")):
        col = res.vp[:, c]
        k = col > 0
        dd = np.arange(nz)[k] * dh - res.surf_z[c]
        m = dd <= depth_below
        a1.plot(col[k][m], dd[m], label=f"{lab}  x={x[c]:.0f} m")
        a0.axvline(x[c], color="0.3", lw=0.5)
    a1.invert_yaxis()
    a1.set_xlabel("Vp (m/s)")
    a1.set_ylabel("depth below surface (m)")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
