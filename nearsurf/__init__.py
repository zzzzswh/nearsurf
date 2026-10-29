"""nearsurf —— 2D 合成地震数据的近地表：陆上近地表（7 种场景）、海水层与海底，以及配套的背景噪声

纯 numpy / scipy，和正演程序无关：给一个深部速度模型，返回加了近地表（或海水层）的模型；
给一个干净道集，返回加了背景噪声的道集。原本是 PreStack-data-factory 的一部分，拆出来单独用。

    import nearsurf
    ns = nearsurf.land_model(vp_deep, dh=6.25, scenario="foothills", seed=0, vs_deep=vs_deep)
    ns.vp, ns.vs, ns.surf_z, ns.wt_z, ns.base_z            # 模型 + 地表 / 潜水面 / 风化层底真值
    sea = nearsurf.marine_model(vp_deep, vs_deep, dh=5.0, seed=0)
    sea.vp, sea.vs, sea.rho, sea.seafloor_z

    noisy, label, info = nearsurf.land_noise(gather, dt=0.002, preset="rich_v3", seed=0,
                                            offsets=offsets, label=fb_index, defects=True)
    noisy = nearsurf.streamer_noise(gather, dt=0.002, dx_rec=12.5, seed=0)

模块
  land      陆上近地表：7 种场景（SCENARIOS：plain / hills / foothills / loess / desert / flat / shield），
            起伏地表 + 风化层 + 潜水面 + Vp / Vs + 非均质；build(...) 是完整接口
  marine    海水层 + 海底地形（5 种形态）+ 海底沉积物
  noise     背景噪声：noise.land（陆上 / 海底节点 + 道缺陷）、noise.streamer（拖缆）、NOISE_PRESETS
约定
  模型 (nz, nx)，第 0 行 = 模型顶，深度向下；长度 m、速度 m/s；空气 / 海水以上 vp = 0
  道集 (ntr, nt)；初至标签是样点下标，== nt 表示无标签
  随机性只来自传入的种子 / Generator：同一种子逐比特可复现
"""
from __future__ import annotations

import numpy as np

from . import land, marine, noise
from .land import SCENARIOS, NearSurface
from .marine import MarineModel
from .noise import NOISE_PRESETS

__version__ = "0.1.0"
__all__ = ["land", "marine", "noise", "SCENARIOS", "NOISE_PRESETS", "NearSurface", "MarineModel",
           "land_model", "marine_model", "land_noise", "streamer_noise"]


def _rng(seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def land_model(vp_deep, dh: float, scenario: str, seed=0, vs_deep=None, **kw) -> NearSurface:
    """在深部模型上加陆上近地表。kw 透传给 land.build（overrides、v_floor、vs_floor、air_top、cap_ratio …）。"""
    return land.build(np.asarray(vp_deep, np.float32), dh, scenario, _rng(seed),
                      vs_deep=None if vs_deep is None else np.asarray(vs_deep, np.float32), **kw)


def marine_model(vp_deep, vs_deep, dh: float, seed=0, **kw) -> MarineModel:
    """在深部模型上加海水层 + 海底。vs_deep 可为 None（只做声波）。kw 透传给 marine.build。"""
    return marine.build(np.asarray(vp_deep, np.float32),
                        None if vs_deep is None else np.asarray(vs_deep, np.float32), dh, _rng(seed), **kw)


def land_noise(gather, dt: float, preset: str = "rich_v3", seed=0, offsets=None, t0: float = 0.0,
               label=None, defects: bool = False, overrides: dict | None = None):
    """给一个道集加陆上 / 海底节点背景噪声（可选道缺陷）。返回 (加噪道集, 标签, info)。

    preset   NOISE_PRESETS 的键：base / rich / rich_v3 / obn；overrides 覆盖 NoiseConfig 字段
    offsets  (ntr,) 带符号炮检距（m）：空气波、按远道定水平都要它
    label    (ntr,) 初至样点下标（== nt 无标签）：rich_v3 / obn 按远道初至定背景水平；开 defects 时必须给
    defects  True：死道 / 极性 / 增益 / 削波 / 振铃 / 掉段，坏道的标签会改成 nt
    """
    from .noise import land as nl
    if preset not in NOISE_PRESETS:
        raise ValueError(f"preset 只能是 {list(NOISE_PRESETS)}")
    kw = dict(NOISE_PRESETS[preset])
    kw.update({k: (tuple(v) if isinstance(v, list) else v) for k, v in (overrides or {}).items()})
    cfg = nl.NoiseConfig(**kw)
    g = np.asarray(gather)
    rng = _rng(seed)
    if defects:
        if label is None:
            raise ValueError("defects=True 需要 label（初至样点下标）")
        noisy, lab, _, info = nl.corrupt(g, np.asarray(label), dt, rng, cfg, nl.DefectConfig(),
                                         offsets=offsets, t0=t0)
        return noisy, lab, info
    nz, info = nl.make_noise(g, dt, rng, cfg, offsets=offsets, t0=t0, label=label)
    return g + nz, label, info


def streamer_noise(gather, dt: float, dx_rec: float, seed=0, amp_coeffs: dict | None = None,
                   gain: float = 1.0):
    """给拖缆炮集 (ntr, nt) 加海上拖缆噪声（涌浪 / 地震干扰 / 电缆振动 / 串音），返回加噪道集。

    幅度相对炮集信号峰值：amp_coeffs 覆盖各类的系数（默认 noise.streamer.DEFAULT_AMP_COEFFS），gain 总增益。
    想按远道初至定总水平，可先取 noise.streamer.make_streamer_noise 的噪声，再按需缩放。
    """
    from .noise import streamer as ns_
    g = np.asarray(gather)
    return g + ns_.make_streamer_noise(g, dt, dx_rec, rng=_rng(seed), amp_coeffs=amp_coeffs, gain=gain)
