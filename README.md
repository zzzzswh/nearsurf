# nearsurf

2D 合成地震数据的**近地表**：陆上近地表（7 种场景）、海水层与海底，以及配套的背景噪声。纯 numpy / scipy，和正演程序无关。

- 给一个深部速度模型 → 加上起伏地表、风化层、潜水面、非均质（陆上 7 种场景），或海水层 + 海底地形 + 沉积物（海上），同时给出地表 / 潜水面 / 风化层底 / 海底的真值
- 给一个干净道集 → 加上陆上 / 海底节点背景噪声（9 类分量 + 6 类道缺陷）或海上拖缆噪声（4 类）
- 同一种子逐比特可复现

原本是 [PreStack-data-factory](../../README.md) 的一部分（`prestack_factory.nearsurface / marine / noise` 现在都转发到这里），拆出来给别的项目、别的正演程序用。

## 安装

```bash
pip install ./packages/nearsurf            # 或 uv add ./packages/nearsurf；画图另装 nearsurf[plot]
```

在 PreStack-data-factory 里不用单独装：它是 uv 工作区成员，`uv sync` 时一起装好。

## 快速上手

```python
import numpy as np
import nearsurf

# 深部模型：(nz, nx)，第 0 行 = 模型顶，深度向下，m/s
vp_deep = ...; vs_deep = ...; dh = 6.25

# 陆上近地表
ns = nearsurf.land_model(vp_deep, dh, "foothills", seed=0, vs_deep=vs_deep, v_floor=1100, vs_floor=700)
ns.vp, ns.vs                # (nz, nx)，地表以上 = 0（空气）
ns.surf_z, ns.surf_idx      # 每列真实地表深度（m）/ 第一个固体格点行号（阶梯地表的正演程序用）
ns.wt_z, ns.base_z          # 潜水面、风化层底界深度（m）—— 真值
ns.params                   # 这次抽到的全部参数

# 海上
sea = nearsurf.marine_model(vp_deep, vs_deep, dh, seed=0, min_depth=50.0)
sea.vp, sea.vs, sea.rho, sea.seafloor_z     # 水里 vs = 0、rho = 1025

# 背景噪声：道集 (ntr, nt)，初至标签是样点下标（== nt 表示无）
noisy, label, info = nearsurf.land_noise(gather, dt=0.002, preset="rich_v3", seed=0,
                                        offsets=offsets, label=fb, defects=True)
noisy = nearsurf.streamer_noise(shot, dt=0.002, dx_rec=12.5, seed=0)
```

完整接口：`nearsurf.land.build(...)`、`nearsurf.marine.build(...)`、`nearsurf.noise.land.make_noise / corrupt(...)`、
`nearsurf.noise.streamer.make_streamer_noise(...)`，参数含义见各模块文件头。

## 陆上场景（`nearsurf.SCENARIOS`）

| 场景 | 地貌 | 起伏 | 特点 |
| --- | --- | --- | --- |
| `plain` | 平原 | 5–30 m | 风化层薄，潜水面浅，可能有古河道、钙质硬壳 |
| `hills` | 丘陵 | 40–150 m | 山顶浑圆，沟谷风化层厚 |
| `foothills` | 山前带 | 150–350 m | 山脊尖锐，沟谷砾石层厚，潜水面深，非均质强 |
| `loess` | 黄土塬 | 60–160 m | 平坦塬面 + 深切冲沟，厚层低速黄土 |
| `desert` | 沙漠 | 沙丘 | 不对称沙丘，极低速干沙，戈壁钙质硬壳 |
| `flat` | 平地表 | 0 | 风化层、潜水面照常（起伏地表的对照组） |
| `shield` | 结晶地盾 | 10–60 m | 薄冰碛直接盖在 5.5 km/s 以上的基岩上 |

每个场景的参数是区间，每次按种子抽一组；`overrides={"relief": [100, 100], "het": [0, 0]}` 改区间
（写成 `[a, a]` 而不是标量，随机数抽取顺序才不变）。构建步骤和每个参数的依据见 `nearsurf/land.py` 文件头与
[docs/references.md](docs/references.md)。

## 海上（`nearsurf.marine`）

海面是平的自由表面；海水声速剖面；海底五种形态（flat / gentle_slope / rough / canyon / ridge）；
海底沉积物 Vp 随深度增加、Vp/Vs 从软泥的 4–8 降到约 2.2；深部被海底截断。

## 背景噪声（`nearsurf.noise`）

| 场合 | 入口 | 内容 |
| --- | --- | --- |
| 陆上 / 海底节点 | `land_noise`（`noise.land`） | ambient、工频、尖脉冲、爆发、斑块、共模低频、机械谐波、阵风、空气波；道缺陷：死道、极性反转、增益抖动、削波、振铃、掉段 |
| 海上拖缆 | `streamer_noise`（`noise.streamer`） | 涌浪、地震干扰、电缆振动、串音（Hlebnikov et al. 2022） |

预设 `NOISE_PRESETS`：`base`、`rich`、`rich_v3`（背景按远道初至定水平，远道初至 SNR 10–22 dB）、`obn`（去掉空气波、更安静）。

## 测试

```bash
python tests/test_nearsurf.py
```

## 独立成仓库

```bash
git subtree split --prefix packages/nearsurf -b nearsurf     # 带着历史拆出一个分支，推到新仓库即可
```
