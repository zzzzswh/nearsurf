"""nearsurf 自测（CPU，几秒）：python tests/test_nearsurf.py   或   uv run --package nearsurf python tests/test_nearsurf.py

[1] 陆上 7 种场景都能建：形状、空气 = 0、地表行号与 surf_z 一致、Vs < Vp、风化层底在地表以下
[2] 同一种子逐比特复现，换种子结果不同；只做声波时 Vp 与带 Vs 时一致
[3] 海洋：水里 Vs = 0、rho = 1025，海底深度 ≥ min_depth
[4] 陆上噪声：各预设能出、可复现、道缺陷会把坏道标签改成 nt
[5] 拖缆噪声：形状对、可复现、幅度随信号峰值缩放
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nearsurf  # noqa: E402

FAIL = []


def ck(name, ok, msg=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({msg})" if msg else ""))
    if not ok:
        FAIL.append(name)


def deep(nz=240, nx=400):
    z = np.linspace(0, 1, nz)[:, None]
    vp = ((2200 + 2000 * z) * np.ones((1, nx))).astype(np.float32)
    return vp, (vp / 1.9).astype(np.float32)


def main():
    dh = 10.0
    vp, vs = deep()
    print("[1] 陆上场景")
    for sc in nearsurf.SCENARIOS:
        r = nearsurf.land_model(vp, dh, sc, seed=1, vs_deep=vs, v_floor=900.0, vs_floor=500.0)
        rows = np.arange(vp.shape[0])[:, None]
        air = rows < r.surf_idx[None, :]
        ok = (r.vp.shape == vp.shape and (r.vp[air] == 0).all() and (r.vp[~air] > 0).all()
              and np.array_equal(r.surf_idx, np.ceil(r.surf_z / dh - 1e-6).astype(np.int64))
              and (r.vs[~air] < r.vp[~air]).all() and (r.base_z >= r.surf_z - 1e-6).all())
        ck(f"{sc:<9}", ok, f"起伏 {r.surf_z.max() - r.surf_z.min():.0f} m，"
                           f"风化层 {np.min(r.base_z - r.surf_z):.0f}~{np.max(r.base_z - r.surf_z):.0f} m，"
                           f"Vp ≥ {r.vp[~air].min():.0f}")
    print("[2] 可复现")
    a = nearsurf.land_model(vp, dh, "hills", seed=7, vs_deep=vs)
    b = nearsurf.land_model(vp, dh, "hills", seed=7, vs_deep=vs)
    c = nearsurf.land_model(vp, dh, "hills", seed=8, vs_deep=vs)
    d = nearsurf.land_model(vp, dh, "hills", seed=7)
    ck("同一种子逐比特相同", np.array_equal(a.vp, b.vp) and np.array_equal(a.vs, b.vs))
    ck("换种子不同", not np.array_equal(a.vp, c.vp))
    ck("只做声波时 Vp 与带 Vs 时一致", np.array_equal(a.vp, d.vp) and d.vs is None)
    print("[3] 海洋")
    m = nearsurf.marine_model(vp, vs, dh, seed=3, min_depth=60.0)
    water = m.vs == 0
    ck("水里 Vs = 0、rho = 1025", water.any() and np.allclose(m.rho[water], 1025.0), f"{water.sum()} 个水格点")
    ck("海底深度 ≥ min_depth", m.seafloor_z.min() >= 60.0 - dh, f"{m.seafloor_z.min():.0f}~{m.seafloor_z.max():.0f} m")
    ck("自由表面是平的海面", (m.surf_idx == 0).all())
    print("[4] 陆上噪声")
    rng = np.random.default_rng(0)
    ntr, nt, dt = 96, 1000, 0.002
    off = (np.arange(ntr) - 10) * 25.0
    lab = np.clip((0.15 + np.abs(off) / 2500.0) / dt, 0, nt - 1).astype(np.int64)
    g = np.zeros((ntr, nt), np.float32)
    for i, k in enumerate(lab):
        g[i, k:k + 40] = np.sin(np.linspace(0, 3 * np.pi, 40)) / np.sqrt(1 + abs(off[i]) / 100)
    for pre in nearsurf.NOISE_PRESETS:
        n1, _, info = nearsurf.land_noise(g, dt, preset=pre, seed=5, offsets=off, label=lab)
        n2, _, _ = nearsurf.land_noise(g, dt, preset=pre, seed=5, offsets=off, label=lab)
        ck(f"预设 {pre:<8}可复现、确实加了噪声", np.array_equal(n1, n2) and not np.array_equal(n1, g)
           and np.isfinite(n1).all())
    nd, lab_d, _ = nearsurf.land_noise(g, dt, preset="rich_v3", seed=11, offsets=off, label=lab, defects=True,
                                      overrides={"master_db": [-30.0, -30.0]})
    ck("道缺陷：形状不变，坏道标签改成 nt", nd.shape == g.shape and set(np.unique(lab_d)) <= set(np.unique(lab)) | {nt},
       f"{int((lab_d == nt).sum())} 道无标签")
    print("[5] 拖缆噪声")
    s1 = nearsurf.streamer_noise(g, dt, 12.5, seed=2)
    s2 = nearsurf.streamer_noise(g, dt, 12.5, seed=2)
    s3 = nearsurf.streamer_noise(10 * g, dt, 12.5, seed=2)
    ck("形状对、可复现", s1.shape == g.shape and np.array_equal(s1, s2))
    ck("幅度随信号峰值缩放", np.allclose(s3 - 10 * g, 10 * (s1 - g), rtol=1e-4, atol=1e-6))
    if FAIL:
        print(f"\n{len(FAIL)} 项失败：{FAIL}")
        sys.exit(1)
    print("\n== 全部通过 ==")


if __name__ == "__main__":
    main()
