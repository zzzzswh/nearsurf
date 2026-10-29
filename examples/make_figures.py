"""Regenerate the figures in docs/images/ (used by the README).

    pip install -e ".[plot]"
    python examples/make_figures.py

Everything is seeded, so the figures are identical on every run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import nearsurf  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

BLURB = {
    "plain": "plain: low relief, thin weathering, shallow water table",
    "hills": "hills: rounded hilltops, thick colluvium in valleys",
    "foothills": "foothills: steep ridges, deep water table, strong heterogeneity",
    "loess": "loess: flat plateau cut by deep gullies, thick slow loess",
    "desert": "desert: asymmetric dunes of very slow dry sand",
    "flat": "flat: no relief (control case), weathering as usual",
    "shield": "shield: thin glacial till on fast crystalline bedrock",
}


def layered_deep(nz, nx, dh, v_top=1800.0, v_bot=3600.0, seed=0):
    """A simple deep model: linear gradient plus a few gently dipping layers."""
    rng = np.random.default_rng(seed)
    z = np.arange(nz)[:, None] * dh
    x = np.arange(nx)[None, :] * dh
    v = v_top + (v_bot - v_top) * z / z.max()
    for d0 in (0.35, 0.6, 0.8):
        top = d0 * z.max() + 15 * np.sin(2 * np.pi * x / (nx * dh) * rng.uniform(0.5, 1.5))
        v = v + 250.0 * (z > top)
    return v.astype(np.float32)


def fig_scenarios():
    dh, nx, nz = 2.5, 1601, 361                      # 4 km wide, 900 m deep
    vp = layered_deep(nz, nx, dh)
    vs = (vp / 1.8).astype(np.float32)
    names = list(nearsurf.SCENARIOS)
    res = {s: nearsurf.land_model(vp, dh, s, seed=3, vs_deep=vs, v_floor=350.0, vs_floor=180.0) for s in names}
    x = np.arange(nx) * dh
    fig, axes = plt.subplots(4, 2, figsize=(14, 13))
    vmin, vmax = 300.0, 3800.0
    for ax, s in zip(axes.flat, names):
        r = res[s]
        elev = -r.surf_z                              # elevation relative to the model top
        top, bot = elev.max() + 15, elev.min() - 160
        img = np.ma.masked_equal(r.vp, 0.0)
        im = ax.imshow(img, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto",
                       extent=[0, x[-1] / 1000, -(nz - 1) * dh, 0], interpolation="nearest")
        ax.plot(x / 1000, elev, "k-", lw=1.0)
        ax.plot(x / 1000, -r.wt_z, color="white", ls=":", lw=1.2)
        ax.plot(x / 1000, -r.base_z, color="k", ls="--", lw=0.9)
        ax.set_ylim(bot, top)
        ax.set_facecolor("#dfe9f3")
        ax.set_title(BLURB[s], fontsize=10, loc="left")
        ax.set_ylabel("elevation (m)")
    ax = axes.flat[len(names)]                        # last panel: Vp below the surface at mid-line
    for s in names:
        r = res[s]
        c = nx // 2
        col = r.vp[:, c]
        k = col > 0
        d = np.arange(nz)[k] * dh - r.surf_z[c]
        m = d <= 150
        ax.plot(col[k][m], d[m], lw=1.4, label=s)
    ax.invert_yaxis()
    ax.set_xlabel("Vp (m/s)")
    ax.set_ylabel("depth below surface (m)")
    ax.set_title("Vp profiles at x = 2 km", fontsize=10, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("x (km)" if ax is not axes.flat[len(names)] else ax.get_xlabel())
    axes[-1, 0].set_xlabel("x (km)")
    fig.colorbar(im, ax=axes[:, 0].tolist() + axes[:-1, 1].tolist(), shrink=0.6, pad=0.01, label="Vp (m/s)")
    fig.suptitle("Onshore scenarios (same deep model, seed 3).  black: surface   "
                 "white dotted: water table   black dashed: weathering base", fontsize=11)
    fig.savefig(OUT / "scenarios.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def fig_marine():
    dh, nx, nz = 10.0, 1201, 201                     # 12 km wide, 2 km deep
    vp = layered_deep(nz, nx, dh, v_top=1900.0, v_bot=4200.0, seed=1)
    vs = (vp / 1.9).astype(np.float32)
    styles = ("flat", "gentle_slope", "rough", "canyon", "ridge")
    ov = dict(depth_bins={"mid": (250.0, 450.0)}, depth_weights={"mid": 1.0})
    res = {s: nearsurf.marine_model(vp, vs, dh, seed=5, overrides=dict(ov, styles=(s,)), min_depth=30.0)
           for s in styles}
    x = np.arange(nx) * dh / 1000
    fig, axes = plt.subplots(3, 2, figsize=(14, 9.5))
    for ax, s in zip(axes.flat, styles):
        r = res[s]
        im = ax.imshow(r.vs, cmap="viridis", vmin=0.0, vmax=2000.0, aspect="auto",
                       extent=[0, x[-1], (nz - 1) * dh, 0], interpolation="nearest")
        ax.plot(x, r.seafloor_z, "w-", lw=1.0)
        ax.set_ylim(1200, 0)
        ax.set_title(f"seafloor: {s}", fontsize=10, loc="left")
        ax.set_ylabel("depth (m)")
    ax = axes.flat[-1]
    r = res["rough"]
    c = nx // 2
    k = r.seafloor_z[c]
    z = np.arange(nz) * dh
    m = (z >= k) & (z <= k + 600)
    ax.plot(r.vp[m, c], z[m] - k, label="Vp")
    ax.plot(r.vs[m, c], z[m] - k, label="Vs")
    ax.plot(r.rho[m, c], z[m] - k, label="density (kg/m³)")
    ax.invert_yaxis()
    ax.set_xlabel("m/s  or  kg/m³")
    ax.set_ylabel("depth below seafloor (m)")
    ax.set_title("sediment profile below the seafloor", fontsize=10, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("x (km)")
    fig.colorbar(im, ax=axes[:, 0].tolist() + axes[:-1, 1].tolist(), shrink=0.6, pad=0.01,
                 label="Vs (m/s); water = 0")
    fig.suptitle("Offshore: water layer + seafloor + soft sediments (Vs shown; white: seafloor)", fontsize=11)
    fig.savefig(OUT / "marine.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def ricker(t, f):
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def toy_gather(offsets, dt, nt, t0=0.12, f=22.0):
    """A clean shot gather with a refracted first arrival, a reflection and slow ground roll."""
    t = np.arange(nt) * dt
    g = np.zeros((offsets.size, nt), np.float32)
    fb = np.empty(offsets.size)
    for i, o in enumerate(np.abs(offsets)):
        t_dir = o / 900.0
        t_ref = 0.06 + o / 2600.0
        t_fb = min(t_dir, t_ref)
        fb[i] = t0 + t_fb
        amp = 1.0 / np.sqrt(1.0 + o / 50.0)
        g[i] += amp * ricker(t - t0 - t_fb, f)
        g[i] += 0.6 * amp * ricker(t - t0 - np.sqrt(0.8 ** 2 + (o / 2200.0) ** 2), f * 0.8)
        g[i] += 2.5 * amp * ricker(t - t0 - o / 350.0, 8.0)
    return g, np.minimum(np.rint(fb / dt).astype(np.int64), nt)


def show(ax, g, offsets, dt, clip, title, picks=None):
    ax.imshow(g.T, cmap="gray", vmin=-clip, vmax=clip, aspect="auto",
              extent=[offsets[0], offsets[-1], g.shape[1] * dt, 0], interpolation="bilinear")
    if picks is not None:
        ok = picks < g.shape[1]
        ax.plot(offsets[ok], picks[ok] * dt, "r.", ms=2)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("offset (m)")


def fig_noise():
    dt, nt = 0.002, 1250
    off = (np.arange(241) - 120) * 12.5                # split spread, ±1.5 km
    g, fb = toy_gather(off, dt, nt)
    noisy, _, _ = nearsurf.land_noise(g, dt, preset="rich_v3", seed=4, offsets=off, label=fb, t0=0.12)
    bad, fb_bad, _ = nearsurf.land_noise(g, dt, preset="rich_v3", seed=9, offsets=off, label=fb, t0=0.12,
                                         defects=True, overrides={"master_db": [-26.0, -26.0]})
    off_s = 100.0 + np.arange(240) * 12.5              # towed streamer, one side
    gs, _ = toy_gather(off_s, dt, nt)
    st = nearsurf.streamer_noise(gs, dt, 12.5, seed=2)
    clip = float(np.percentile(np.abs(g), 99.0)) * 0.35
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.6), sharey=True)
    show(axes[0], g, off, dt, clip, "clean (red: first-break labels)", fb)
    show(axes[1], noisy, off, dt, clip, "land background noise (rich_v3)")
    show(axes[2], bad, off, dt, clip, "+ trace defects (labels updated)", fb_bad)
    show(axes[3], st, off_s, dt, float(np.percentile(np.abs(gs), 99.0)) * 0.35, "towed-streamer noise")
    axes[0].set_ylabel("time (s)")
    fig.suptitle("Background noise on a toy shot gather (same display clip within each panel pair)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "noise.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_scenarios()
    fig_marine()
    fig_noise()
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.relative_to(OUT.parents[1])}  {p.stat().st_size / 1024:.0f} KiB")
