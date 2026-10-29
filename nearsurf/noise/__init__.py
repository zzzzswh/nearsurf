"""nearsurf.noise —— 合成数据的背景噪声

  land.py      陆上 / 海底节点背景噪声 + 道缺陷：ambient、工频、尖脉冲、爆发、斑块、共模低频、机械谐波、
               阵风、空气波；道缺陷：死道、极性反转、增益抖动、削波、振铃、掉段（源自 fb-dataset-v3）
  streamer.py  海上拖缆噪声（Hlebnikov et al. 2021 四类：涌浪、地震干扰、电缆振动、串音）

NOISE_PRESETS：land.NoiseConfig 的常用预设（键是 NoiseConfig 的字段）。
    from nearsurf.noise import NOISE_PRESETS, land
    cfg = land.NoiseConfig(**NOISE_PRESETS["rich_v3"])
简便入口见 nearsurf.land_noise / nearsurf.streamer_noise。
"""

NOISE_PRESETS = {
    # 基线
    "base": dict(master_db=(-45.0, -28.0)),
    # PRESET = rich
    "rich": dict(master_db=(-38.0, -20.0), snr_db=(-5.0, 22.0), trace_var_db=3.0,
                 amb_tex_frac=(0.4, 0.9), tex_band=(2.0, 15.0), tex_warp_ms=(40.0, 150.0),
                 tex_warp_corr=1.0 / 40.0, tex_width=0.3),
    # PRESET = rich_v3：背景按远道初至定水平（远道初至 SNR 10~22 dB），纹理只要直线
    "rich_v3": dict(master_db=(-38.0, -20.0), snr_db=(-5.0, 22.0), trace_var_db=3.0,
                    amb_tex_frac=(0.4, 0.9), tex_band=(2.0, 15.0), tex_warp_ms=(0.0, 0.0),
                    tex_warp_corr=1.0 / 40.0, tex_width=0.3,
                    amb_ref="far_fb", far_frac=0.05, far_win_ms=60.0,
                    far_snr_db=(10.0, 22.0), far_other_cap_db=-3.0),
    # 海底节点：rich_v3 去掉空气波（海底没有）、工频少量、整体更安静
    "obn": dict(master_db=(-38.0, -20.0), snr_db=(-5.0, 22.0), trace_var_db=3.0,
                amb_tex_frac=(0.4, 0.9), tex_band=(2.0, 15.0), tex_warp_ms=(0.0, 0.0),
                tex_warp_corr=1.0 / 40.0, tex_width=0.3,
                amb_ref="far_fb", far_frac=0.05, far_win_ms=60.0,
                far_snr_db=(12.0, 25.0), far_other_cap_db=-6.0,
                air_prob=0.0, pl_p_trace=(0.0, 0.1)),
}
