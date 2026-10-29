"""
nearsurf.marine —— 在 2D 深部模型上加海水层 + 海底地形 + 海底沉积物（海洋场景）

    from nearsurf import marine                       # 或 nearsurf.marine_model(vp_deep, vs_deep, dh, seed=0)
    res = marine.build(vp_deep, vs_deep, dh, rng, min_depth=20.0, vs_floor=500.0)
    res.vp / res.vs / res.rho   (nz, nx)；水里 vs = 0，rho = 海水密度
    res.seafloor_z              (nx,) 海底深度（m，相对海面）
    res.surf_idx                (nx,) 全 0：自由表面就是平的海面

结构（从上到下）
--------------
  海面 z = 0（平的自由表面）
  海水    vp = vw0 + g·z（声速剖面），vs = 0，rho = 1025
  海底    seafloor_z(x)：五种形态（flat / gentle_slope / rough / canyon / ridge，
          沿用 streamer_shots_generator 的分类），平均水深按分档抽样
  沉积物  vp = v_sf + grad·d（d = 海底以下深度），vs = vp / r(d)，
          r 在海底处很大（软泥 Vp/Vs 4~8），随深度指数降到 ~2.2；
          rho 软泥约 1.4~1.8 g/cc，随 vp 增大
  深部    沉积物在 sed_thick 处 smoothstep 过渡到深部模型（data7 映射后的速度），
          深部被海底"切"开（侵蚀截断），与陆上近地表的做法一致

参数来源
--------
  水深分档、海底形态、海底速度 1550~2400、梯度 0.6~2.0 s^-1：streamer_shots_generator
  （seabed_model.py）的取值；水深分档改成偏浅（初至拾取在浅水区最难）。
  海底软泥 Vs 很低（~100 m/s 量级）、Vp/Vs 很大，随深度增加：Hamilton (1979) 等；
  受网格限制，Vs 不低于 vs_floor（弹性波由它定网格）。
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

DEFAULTS = dict(
    depth_bins={"very_shallow": (15.0, 60.0), "shallow": (60.0, 200.0),
                "mid": (200.0, 600.0), "deep": (600.0, 1500.0)},
    depth_weights={"very_shallow": 0.3, "shallow": 0.35, "mid": 0.25, "deep": 0.1},
    styles=("flat", "gentle_slope", "rough", "canyon", "ridge"),
    v_water=(1475.0, 1500.0),       # 海面声速
    v_water_grad=0.016,             # (m/s)/m
    rho_water=1025.0,
    v_sf=(1550.0, 1900.0),          # 海底面沉积物 Vp
    v_grad=(0.6, 2.0),              # 沉积物 Vp 梯度 (1/s)
    sed_thick=(50.0, 600.0),        # 沉积物厚度：在此深度过渡到深部模型
    trans=(20.0, 80.0),             # 过渡带宽度
    vpvs_sf=(4.0, 8.0),             # 海底面 Vp/Vs
    vpvs_deep=2.2,                  # 沉积物深处 Vp/Vs
    vpvs_len=(30.0, 150.0),         # Vp/Vs 衰减深度尺度
    het=(0.02, 0.05),               # 沉积物 2D 非均质
)


@dataclasses.dataclass
class MarineModel:
    vp: np.ndarray
    vs: np.ndarray | None
    rho: np.ndarray
    seafloor_z: np.ndarray
    surf_idx: np.ndarray
    params: dict


def _u(rng, v):
    return float(rng.uniform(*v)) if isinstance(v, (tuple, list)) else v


def _lowpass_noise(nx, dh, wavelength, rng):
    s = gaussian_filter1d(rng.standard_normal(nx), max(wavelength / dh / 4.0, 0.5), mode="reflect")
    return s / max(s.std(), 1e-12)


def seafloor_profile(nx, dh, mean_depth, style, rng):
    """海底深度 (m)。起伏幅度随水深增大（深水区地形起伏更大），形态同 seabed_model.py。"""
    x = np.arange(nx) * dh
    L = x[-1]
    amp_long = (0.05 + 0.15 * rng.random()) * mean_depth + 3.0
    amp_short = 0.15 * amp_long
    wl_long = rng.uniform(1500.0, 5000.0)
    if style == "flat":
        long = 0.2 * amp_long * _lowpass_noise(nx, dh, wl_long, rng)
        amp_short *= 0.3
    elif style == "gentle_slope":
        slope = rng.uniform(0.005, 0.03) * rng.choice([-1, 1])
        long = slope * (x - L / 2) + 0.5 * amp_long * _lowpass_noise(nx, dh, wl_long, rng)
        amp_short *= 0.5
    elif style == "rough":
        long = amp_long * _lowpass_noise(nx, dh, wl_long, rng)
    elif style in ("canyon", "ridge"):
        base = 0.4 * amp_long * _lowpass_noise(nx, dh, wl_long, rng)
        c = rng.uniform(0.25, 0.75) * L
        w = rng.uniform(0.04, 0.12) * L
        h = rng.uniform(1.0, 3.0) * amp_long * (1 if style == "canyon" else -0.8)
        long = base + h * np.exp(-0.5 * ((x - c) / w) ** 2)
        amp_short *= 0.6
    else:
        raise ValueError(f"未知海底形态 {style!r}")
    short = amp_short * _lowpass_noise(nx, dh, rng.uniform(100.0, 400.0), rng)
    return mean_depth + long + short


def build(vp_deep, vs_deep, dh, rng, overrides=None, min_depth=20.0, vs_floor=500.0,
          v_floor=1450.0, max_frac=0.5) -> MarineModel:
    """vp_deep / vs_deep (nz, nx)，第 0 行 = 海面。vs_deep 为 None 时不做 Vs。"""
    p = dict(DEFAULTS)
    p.update(overrides or {})
    nz, nx = vp_deep.shape
    zmax = (nz - 1) * dh

    # ---------- 水深与海底
    names = list(p["depth_bins"])
    w = np.array([p["depth_weights"][k] for k in names], float)
    bin_ = names[int(rng.choice(len(names), p=w / w.sum()))]
    mean_depth = _u(rng, tuple(p["depth_bins"][bin_]))
    style = str(rng.choice(list(p["styles"])))
    sf = seafloor_profile(nx, dh, mean_depth, style, rng)
    sf = np.clip(sf, min_depth, max_frac * zmax)

    # ---------- 速度
    z = (np.arange(nz) * dh)[:, None]
    d = z - sf[None, :]                                   # 海底以下深度
    dp = np.clip(d, 0, None)
    vw0, g = _u(rng, p["v_water"]), float(p["v_water_grad"])
    v_water = vw0 + g * z
    v_sf = _u(rng, p["v_sf"]) + 60.0 * _lowpass_noise(nx, dh, 3000.0, rng)
    v_sf = np.maximum(v_sf, vw0 + g * sf + 30.0)          # 海底面一定比海水快
    grad = _u(rng, p["v_grad"])
    thick = _u(rng, p["sed_thick"])
    tr = _u(rng, p["trans"])
    dsed = np.minimum(dp, thick + tr)                     # 沉积物梯度只到过渡带为止
    v_sed = v_sf[None, :] + grad * dsed
    s = np.clip((d - thick) / tr + 0.5, 0.0, 1.0)
    wgt = s * s * (3 - 2 * s)
    het_amp = _u(rng, p["het"])
    gfield = gaussian_filter(rng.standard_normal((nz, nx)),
                             (max(3.0 / dh, 0.5), max(rng.uniform(50.0, 200.0) / dh, 1.0)))
    gfield /= max(gfield.std(), 1e-12)
    vp_solid = (1 - wgt) * v_sed * (1 + het_amp * gfield) + wgt * vp_deep
    vp = np.where(d < 0, v_water, np.maximum(vp_solid, v_floor)).astype(np.float32)

    # ---------- Vs：软泥 Vp/Vs 很大，随深度指数降到 vpvs_deep；水里 0
    vs = None
    r0 = _u(rng, p["vpvs_sf"])
    Lr = _u(rng, p["vpvs_len"])
    if vs_deep is not None:
        r = p["vpvs_deep"] + (r0 - p["vpvs_deep"]) * np.exp(-dsed / Lr)
        vs_sed = v_sed / r
        vs_solid = (1 - wgt) * vs_sed * (1 + het_amp * gfield) + wgt * vs_deep
        vs_solid = np.minimum(np.maximum(vs_solid, vs_floor), vp / 1.42)
        vs = np.where(d < 0, 0.0, vs_solid).astype(np.float32)

    # ---------- 密度：海水 1025；沉积物软泥约 1.45 g/cc 起，随 vp 增大，不超过 Gardner
    gard = 310.0 * np.maximum(vp, 1.0) ** 0.25
    rho_sed = np.clip(1400.0 + 0.6 * (vp - 1450.0), 1400.0, None)
    rho = np.where(d < 0, p["rho_water"], np.minimum(rho_sed, gard)).astype(np.float32)

    params = dict(scenario="marine", depth_bin=bin_, mean_water_depth=mean_depth, style=style,
                  water_depth_min=float(sf.min()), water_depth_max=float(sf.max()),
                  v_water_surface=vw0, v_water_grad=g, v_seafloor_mean=float(v_sf.mean()),
                  v_grad=grad, sed_thick=thick, trans=tr, het=het_amp,
                  vpvs_seafloor=r0 if vs_deep is not None else None,
                  vs_solid_min=float(vs[vs > 0].min()) if vs is not None else None)
    return MarineModel(vp=vp, vs=vs, rho=rho, seafloor_z=sf.astype(np.float32),
                       surf_idx=np.zeros(nx, np.int64), params=params)
