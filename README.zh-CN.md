# nearsurf

[English](README.md)

**给 2D 合成地震数据加上真实的近地表。** 输入一个深部速度模型，它会加上起伏地表、风化层、潜水面和小尺度非均质
（陆上 7 种场景），或者海水层、海底地形和海底软沉积物（海上），并给出每个界面的真值。输入一个干净的炮集，它会加上
像野外一样的背景噪声和道缺陷。

纯 numpy / scipy，和正演程序无关；给定种子，结果逐比特可复现。

![陆上场景](docs/images/scenarios.png)

## 为什么需要它

合成数据最"干净"的地方，恰恰是实测数据最难的地方：近地表。静校正、初至拾取、面波压制、近地表反演，都取决于低速风化层、
起伏地形和噪声。`nearsurf` 把任意深部模型（地质模型、Marmousi、层状模型……）变成带有合理近地表的模型，在它上面正演出来的
数据更接近实测，同时保留真值，可用于训练或验证。

## 安装

```bash
pip install git+https://github.com/zzzzswh/nearsurf.git
pip install "nearsurf[plot] @ git+https://github.com/zzzzswh/nearsurf.git"   # 需要画图时，连同 matplotlib 一起装
```

需要 Python ≥ 3.10、numpy、scipy。

## 快速上手

代码见[英文 README 的 Quick start](README.md#quick-start)（可直接运行）。核心是四个函数：

| 函数 | 作用 |
| --- | --- |
| `land_model(vp_deep, dh, scenario, seed, vs_deep=...)` | 加陆上近地表，返回 vp / vs、真实地表 `surf_z`、第一个固体格点 `surf_idx`、潜水面 `wt_z`、风化层底 `base_z`、本次抽到的参数 `params` |
| `marine_model(vp_deep, vs_deep, dh, seed)` | 加海水层 + 海底 + 沉积物，返回 vp / vs / rho、海底深度 `seafloor_z` |
| `land_noise(gather, dt, preset, seed, offsets=..., label=..., defects=...)` | 陆上 / 海底节点背景噪声，可选道缺陷（坏道的初至标签会被去掉） |
| `streamer_noise(gather, dt, dx_rec, seed)` | 海上拖缆噪声 |

## 陆上场景

| 场景 | 地貌 | 起伏 | 特点 |
| --- | --- | --- | --- |
| `plain` | 平原 | 5–30 m | 风化层薄，潜水面浅，可能有古河道和钙质硬壳 |
| `hills` | 丘陵 | 40–150 m | 山顶浑圆，沟谷里风化层厚 |
| `foothills` | 山前带 | 150–350 m | 山脊尖锐，沟谷砾石层厚，潜水面深，非均质强 |
| `loess` | 黄土塬 | 60–160 m | 平坦塬面被深沟切割，厚层低速黄土 |
| `desert` | 沙漠 | 沙丘 | 不对称沙丘（迎风坡缓、背风坡陡），极低速干沙，戈壁硬壳 |
| `flat` | 平地表 | 0 | 风化层、潜水面照常，作为起伏地表的对照组 |
| `shield` | 结晶地盾 | 10–60 m | 薄冰碛直接盖在 5.5 km/s 以上的基岩上 |

构建步骤：分形地形加坡面扩散 → 风化层厚度随地貌变化（凹处厚、陡坡薄）→ 潜水面 → 速度（干燥层幂律梯度、
潜水面跳变、可选硬壳 / 古河道、过渡到裂隙化深部、随深度衰减的各向异性随机场）。每一步的依据见
[`nearsurf/land.py`](nearsurf/land.py) 和 [`docs/references.md`](docs/references.md)。

每个场景的参数都是区间，每次实现按种子各抽一个值。用 `overrides` 改区间，比如
`overrides={"relief": [100, 100], "het": [0, 0]}`。注意写成 `[a, a]` 区间而不是标量：这样随机数的抽取顺序不变，
模型里其他部分也就不会跟着变。`v_floor` / `vs_floor` 是速度下限，请按你的正演网格和主频来设。

## 海上

![海上](docs/images/marine.png)

平的海面、海水声速剖面、五种海底形态（`flat`、`gentle_slope`、`rough`、`canyon`、`ridge`），以及 Vp/Vs
从海底处的 4–8 随深度降到约 2.2 的软沉积物；深部模型被海底截断。

## 背景噪声

![噪声](docs/images/noise.png)

| 场合 | 函数 | 内容 |
| --- | --- | --- |
| 陆上 / 海底节点 | `land_noise` | 有色背景噪声、工频、尖脉冲、爆发、斑块、共模低频、机械谐波、阵风、空气波；道缺陷：死道、极性反转、增益抖动、削波、振铃、掉段 |
| 海上拖缆 | `streamer_noise` | 涌浪、地震干扰、电缆振动、串音（Hlebnikov et al., 2022） |

预设 `NOISE_PRESETS`：`base`、`rich`、`rich_v3`（按远道初至定背景水平，远道 SNR 10–22 dB）、`obn`（无空气波，更安静）。

## 约定

- 模型 `(nz, nx)`，第 0 行是模型顶，深度向下；长度单位 m，速度单位 m/s；地面以上（空气）或海面以上 `vp = 0`。
- 道集 `(ntr, nt)`；初至标签是样点下标，`nt` 表示无标签。
- 随机性只来自 `seed`（整数或 `numpy.random.Generator`）：同一种子总是得到同一个结果。

## 开发

```bash
git clone https://github.com/zzzzswh/nearsurf.git && cd nearsurf
pip install -e ".[plot]"
python tests/test_nearsurf.py        # GitHub Actions 也会在 Python 3.10–3.13 上跑
python examples/make_figures.py      # 重新生成 docs/images/ 里的图
```

## 许可证

[MIT](LICENSE)
