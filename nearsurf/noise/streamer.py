# 拷自 streamer_shots_generator（refs/noise/streamer_noise.py）：四类噪声生成函数原样保留；
# 去掉 torch 依赖，把组合入口 make_streamer_noise_gpu 换成文件末尾的 numpy 版 make_streamer_noise。
"""Streamer-scene noise synthesis — v2 (Hlebnikov 2021 taxonomy).

Four additive noise types for towed marine streamer data, following
Hlebnikov et al. 2021 Table 1. Replaces the earlier v1 {swell, dc_drift,
white} design with a more literature-grounded set:

    swell        — 2–15 Hz, non-coherent waveforms, cluster-shaped envelope
    si           — seismic interference, coherent LMO (H/T + T/H + B/S)
    strumming    — 1–8 Hz, coherent LMO from head/tail, ~10% of cable
    crossfeed    — broadband, flat (zero-moveout) coherent spikes

Integration with the pipeline
-----------------------------
The four component generators are numpy + scipy.filtfilt, dropped in from
`marine_seismic_noise.py` v2 (verbatim save for namespacing). The only
addition is `make_streamer_noise_gpu`, a torch wrapper that:
  - accepts (B=1, NREC, NT) torch tensor,
  - transposes to (NT, NREC) numpy (user-convention) per shot,
  - scales per-component amplitude to `k × signal_max × N(1, 0.2²)`,
  - returns composed noise as (1, NREC, NT) torch on the same device.
The pipeline's per-sample `np.random.Generator` is threaded through, so
(global_seed, sample_idx) fully reproduces the noise realisation.

Deviation from v5 §4.2 spec
---------------------------
v5 §4.2 listed {swell 2–10 Hz, DC drift 0–2 Hz} following an older
taxonomy. v2 switches to Hlebnikov 2021's four dominant types (drops
DC drift — it is subsumed into swell in this taxonomy; adds SI,
strumming, cross-feed). The tech-delivery doc and project plan should
be updated at P3.6.

References
----------
[1] Hlebnikov, V., Elboth, T., Vinje, V., & Gelius, L.-J. (2019).
    Onboard de-noise processing for improving towed marine seismic
    acquisition efficiency. SEG Expanded Abstracts, doi:10.1190/segam2019-3212338.1
[2] Hlebnikov, V., Elboth, T., Vinje, V., & Gelius, L.-J. (2021).
    Noise types and their attenuation in towed marine seismic: A tutorial.
    Geophysics, 86(2), W1–W19, doi:10.1190/geo2019-0808.1
[3] Elboth, T., Reif, B. A. P., & Andreassen, O. (2009). Flow and swell
    noise in marine seismic data. Geophysics, 74(2), Q17–Q25.
[4] Elboth, T., Presterud, I. V., & Hermansen, D. (2010). Time-frequency
    seismic data de-noising. Geophysical Prospecting, 58(3), 441–453.
[5] Gulunay, N. (2008). Two different algorithms for seismic interference
    noise attenuation. The Leading Edge, 27(2), 176–181.
[6] Parrish, J. F. (2005). Streamer string waves and swell noise. SEG EA, 72–75.
[9] Teigen, O., et al. (2012). Characterization of noise modes in
    multicomponent (4C) towed-streamer. EAGE EA, Z013.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import signal

__all__ = [
    "generate_swell_noise",
    "generate_seismic_interference",
    "generate_strumming_noise",
    "generate_crossfeed_noise",
    "make_streamer_noise",
    "DEFAULT_AMP_COEFFS",
]


# =====================================================================
# Utility helpers (module-private)
# =====================================================================
def _bandpass(data, fmin, fmax, fs, order=4):
    """Zero-phase Butterworth bandpass along axis-0 (time axis)."""
    nyq = 0.5 * fs
    low = float(np.clip(fmin / nyq, 1e-5, 1 - 1e-5))
    high = float(np.clip(fmax / nyq, 1e-5, 1 - 1e-5))
    b, a = signal.butter(order, [low, high], btype="band")
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    return signal.filtfilt(b, a, data, axis=0)


def _ricker(t, f):
    """Ricker wavelet centred at t=0."""
    u = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * u) * np.exp(-u)


# =====================================================================
# 1. Swell noise
# =====================================================================
def generate_swell_noise(
    n_samples, n_traces, dt,
    amplitude=1.0,
    fmin=2.0,
    fmax=15.0,
    cluster_size=15,
    n_clusters=None,
    rng=None,
):
    """Synthesize swell noise.

    Swell noise is caused by transversal vibrations of the streamer induced
    by the interaction between the streamer, its mounted equipment, and the
    turbulent water column [3]. Band-limited to 2–15 Hz [2, Table 1], affects
    the full record length continuously, and clusters of adjacent traces show
    elevated amplitude [2, p.W3]. Within each cluster the individual trace
    waveforms differ — NON-COHERENT trace to trace [2, Table 1].

    Synthesis (after Liu et al. [8]):
      (a) Gaussian-shaped amplitude envelopes on random trace clusters.
      (b) Independent band-limited Gaussian noise per trace (non-coherent).
      (c) Full-record-length noise (no temporal windowing).

    Returns: (n_samples, n_traces) ndarray.
    References: [2], [3], [8].
    """
    if rng is None:
        rng = np.random.default_rng()
    fs = 1.0 / dt
    if n_clusters is None:
        n_clusters = max(1, n_traces // (cluster_size * 2))

    # (a) Spatial amplitude envelope
    amp_env = np.full(n_traces, 0.05)
    for _ in range(n_clusters):
        center = rng.integers(0, n_traces)
        half_w = rng.integers(max(1, cluster_size // 2), cluster_size * 2)
        peak = rng.uniform(0.5, 1.0)
        sigma = half_w / 2.5
        for tr in range(n_traces):
            g = peak * np.exp(-0.5 * ((tr - center) / sigma) ** 2)
            amp_env[tr] = max(amp_env[tr], g)

    # (b) Independent band-limited noise per trace — full record length
    raw = rng.standard_normal((n_samples, n_traces))
    filtered = _bandpass(raw, fmin, fmax, fs)

    noise = np.zeros_like(filtered)
    for tr in range(n_traces):
        rms = np.std(filtered[:, tr])
        if rms > 0:
            noise[:, tr] = amplitude * amp_env[tr] * filtered[:, tr] / rms

    return noise


# =====================================================================
# 2. Seismic Interference (SI)
# =====================================================================
def generate_seismic_interference(
    n_samples, n_traces, dt, dx,
    amplitude=1.0,
    n_events=2,
    v_apparent_range=(1200.0, 5000.0),
    dominant_freq=35.0,
    train_duration=1.5,
    include_broadside=True,
    rng=None,
):
    """Synthesize seismic interference (SI) noise.

    SI records energy from a nearby seismic vessel's source. COHERENT from
    trace to trace within one shot (constant apparent velocity), NON-COHERENT
    shot-to-shot [2, p.W4]. Three geometric sub-types [2, p.W4, Fig. 2]:
      H/T (head-to-tail)  : source ahead  → positive dip
      T/H (tail-to-head)  : source behind → negative dip
      B/S (broadside)     : source abeam  → ~zero moveout (v_app > 20 km/s)
    Arrives as ~1.5-s trains [2, p.W4]; broadband [2, Table 1].

    Returns: (n_samples, n_traces) ndarray.
    References: [1], [2], [4], [5].
    """
    if rng is None:
        rng = np.random.default_rng()
    fs = 1.0 / dt
    noise = np.zeros((n_samples, n_traces))
    n_train = max(4, int(train_duration * fs))

    def _si_burst(n):
        raw = rng.standard_normal(n)
        f_lo = max(1.0, dominant_freq * 0.5)
        f_hi = min(fs / 2.0 - 1.0, dominant_freq * 2.0)
        filt = _bandpass(raw, f_lo, f_hi, fs)
        return filt * np.hanning(n)

    def _inject_event(v_app, ev_amp):
        t0 = rng.uniform(0.0, max(dt, n_samples * dt - train_duration))
        i0_base = int(t0 / dt)
        src = _si_burst(n_train)
        for tr in range(n_traces):
            delay_s = tr * dx / v_app
            i_start = i0_base + int(delay_s / dt)
            s0 = max(0, -i_start)
            d0 = max(0, i_start)
            length = min(n_train - s0, n_samples - d0)
            if length > 0:
                noise[d0:d0 + length, tr] += ev_amp * src[s0:s0 + length]

    # H/T and T/H events
    for _ in range(n_events):
        v = rng.uniform(*v_apparent_range)
        v = v if rng.random() > 0.5 else -v
        _inject_event(v, rng.uniform(0.5, 1.0) * amplitude)

    # Broadside — nearly flat in t-x
    if include_broadside:
        v_bs = rng.uniform(20_000.0, 100_000.0)
        if rng.random() > 0.5:
            v_bs = -v_bs
        _inject_event(v_bs, rng.uniform(0.3, 0.8) * amplitude)

    return noise


# =====================================================================
# 3. Strumming / Tugging
# =====================================================================
def generate_strumming_noise(
    n_samples, n_traces, dt, dx,
    amplitude=1.0,
    fmin=1.0,
    fmax=8.0,
    v_strum_range=(1500.0, 3000.0),
    n_events_head=2,
    n_events_tail=2,
    affected_frac=0.10,
    burst_duration=0.8,
    rng=None,
):
    """Synthesize strumming / tugging noise.

    Strumming arises from longitudinal vibrations triggered by sudden tension
    changes in rough seas [2, p.W5]. Velocity 1.5–3.0 km/s (polyurethane skin,
    Kevlar, copper wires) [2]; frequency 1–8 Hz [2, Table 1]; normally affects
    only the first few hundred metres from each end [2, p.W5]. COHERENT trace
    to trace [2, Table 1].

    Default `affected_frac=0.10` ≈ first few hundred metres on a 3 km cable
    (matches the literature description). For a longer streamer consider
    `0.15`. Use `1.0` to model the severe / debris-induced case.

    Synthesis reverses the LMO model of [1, App. C]:
      (a) Band-limited (1–8 Hz) burst generated once per event.
      (b) Propagated trace-by-trace with `dx / v_strum` per-trace delay.
      (c) Head-end events: positive dip (H/T). Tail-end: negative dip (T/H).

    Returns: (n_samples, n_traces) ndarray.
    References: [1], [2], [6], [9].
    """
    if rng is None:
        rng = np.random.default_rng()
    fs = 1.0 / dt
    noise = np.zeros((n_samples, n_traces))
    n_burst = max(4, int(burst_duration * fs))
    n_affected = max(1, int(n_traces * affected_frac))

    def _make_burst():
        raw = rng.standard_normal(n_burst)
        filt = _bandpass(raw, fmin, fmax, fs)
        return filt * np.hanning(n_burst)

    def _inject(burst, t0_samp, start_tr, direction, ev_amp, v_strum):
        """BUG FIX (v2.1) vs v2: write directly at correct time position."""
        for k in range(n_affected):
            tr = start_tr + direction * k
            if not (0 <= tr < n_traces):
                break
            prop_samp = int(k * dx / v_strum / dt)
            dst_start = t0_samp + prop_samp
            src_start = 0
            if dst_start < 0:
                src_start = -dst_start
                dst_start = 0
            dst_end = min(n_samples, dst_start + n_burst - src_start)
            length = dst_end - dst_start
            if length > 0:
                noise[dst_start:dst_end, tr] += (
                    ev_amp * burst[src_start:src_start + length]
                )

    # Head-end strumming (positive dip, H/T)
    for _ in range(n_events_head):
        v_strum = rng.uniform(*v_strum_range)
        t0_samp = rng.integers(
            int(0.05 * n_samples),
            max(int(0.05 * n_samples) + 1, int(0.85 * n_samples)),
        )
        _inject(_make_burst(), t0_samp, start_tr=0, direction=1,
                ev_amp=rng.uniform(0.5, 1.0) * amplitude, v_strum=v_strum)

    # Tail-end strumming (negative dip, T/H)
    for _ in range(n_events_tail):
        v_strum = rng.uniform(*v_strum_range)
        t0_samp = rng.integers(
            int(0.05 * n_samples),
            max(int(0.05 * n_samples) + 1, int(0.85 * n_samples)),
        )
        _inject(_make_burst(), t0_samp, start_tr=n_traces - 1, direction=-1,
                ev_amp=rng.uniform(0.5, 1.0) * amplitude, v_strum=v_strum)

    return noise


# =====================================================================
# 4. Cross-feed (electrical leakage)
# =====================================================================
def generate_crossfeed_noise(
    n_samples, n_traces, dt,
    amplitude=1.0,
    n_spikes=6,
    affected_frac=1.0,
    spike_bw=0.05,
    rng=None,
):
    """Synthesize cross-feed (electrical leakage) noise.

    Cross-feed is instrument noise from electrical leakage between channels
    [2, p.W10]. The disturbance propagates at the speed of light, so the
    identical signal appears simultaneously on all affected channels —
    zero moveout, perfectly coherent flat energy in the shot gather.
    Broadband [2, Table 1].

    Synthesis:
      (a) A short (~50 ms) windowed pulse generated once per event.
      (b) IDENTICAL waveform injected at the same time on all affected traces.

    Returns: (n_samples, n_traces) ndarray.
    References: [1], [2].
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = np.zeros((n_samples, n_traces))
    n_pulse = max(3, int(spike_bw / dt))
    n_aff = max(1, int(n_traces * affected_frac))
    affected = np.sort(rng.choice(n_traces, n_aff, replace=False))

    for _ in range(n_spikes):
        raw_pulse = rng.standard_normal(n_pulse) * np.hanning(n_pulse)
        sp_amp = rng.uniform(0.5, 1.5) * amplitude
        t_center = rng.integers(n_pulse, n_samples - n_pulse)
        i0 = max(0, t_center - n_pulse // 2)
        i1 = min(n_samples, i0 + n_pulse)
        length = i1 - i0
        for tr in affected:
            noise[i0:i1, tr] += sp_amp * raw_pulse[:length]

    return noise


# =====================================================================
# Torch composer (pipeline integration layer)
# =====================================================================
DEFAULT_AMP_COEFFS: Dict[str, float] = {
    "swell": 0.30,      # visible but not dominant
    "si": 0.25,
    "strum": 0.30,
    "crossfeed": 0.30,
}




# =====================================================================
# numpy 组合入口（替代原来的 torch 版 make_streamer_noise_gpu，逻辑相同）
# =====================================================================
def make_streamer_noise(shot_gather, dt, dxrec, *, rng=None, amp_coeffs=None, gain=1.0):
    """(nrec, nt) 干净炮集 -> (nrec, nt) 噪声（不含信号）。

    每个分量的幅度 = 系数 × 本炮信号峰值 × N(1, 0.2²)（截到 >= 0），与原
    make_streamer_noise_gpu 一致；随机数抽取顺序也一致。
    """
    g = np.asarray(shot_gather)
    nrec, nt = g.shape
    rng = np.random.default_rng() if rng is None else rng
    amp_coeffs = DEFAULT_AMP_COEFFS if amp_coeffs is None else amp_coeffs
    signal_max = float(np.abs(g).max())

    def _scaled_amp(key):
        pert = max(0.0, float(rng.normal(loc=1.0, scale=0.2)))
        return amp_coeffs[key] * pert * signal_max

    total = generate_swell_noise(n_samples=nt, n_traces=nrec, dt=dt,
                                 amplitude=_scaled_amp("swell"), rng=rng)
    total = total + generate_seismic_interference(n_samples=nt, n_traces=nrec, dt=dt, dx=dxrec,
                                                  amplitude=_scaled_amp("si"), rng=rng)
    total = total + generate_strumming_noise(n_samples=nt, n_traces=nrec, dt=dt, dx=dxrec,
                                             amplitude=_scaled_amp("strum"), rng=rng)
    total = total + generate_crossfeed_noise(n_samples=nt, n_traces=nrec, dt=dt,
                                             amplitude=_scaled_amp("crossfeed"), rng=rng)
    return (np.asarray(total).T * gain).astype(np.float32)
