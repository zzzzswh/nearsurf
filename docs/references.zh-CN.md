# 建模依据与参考文献

本文档说明 `nearsurf` 里每一步建模选择的来源，以及参数区间是怎么定的。[English](references.md)

> **2026-10 修订说明。**逐条核对了参考文献：补全了卷期、页码、DOI 和作者；更正了 Pelletier et al. 2015 的题目；补上了正文引用、但文献表里漏掉的 Newman & Turcotte 1990、Castagna et al. 1985 和 Bachrach et al. 1998。
> 每一步都标了依据的类型，分三种：
> - **【文献】**：有文献直接支撑；
> - **【文献·外推】**：文献只支持定性规律，或者被外推到了文献没有覆盖的尺度；
> - **【经验取值】**：没有文献依据，是工程上的设定。
>
> 没有出处的说法单独列在第 8 节。

## 1. 地形形态（`nearsurf/land.py`）

**谱特征【文献】。**地形剖面的功率谱在很宽的尺度范围内近似服从 S(k) ∝ k^-2，对应分形布朗运动的 Hurst 指数 H ≈ 0.5（β = 2H + 1）（Turcotte 1992；Newman & Turcotte 1990）。纯分形谱缺少河网和沟谷，不能完整描述真实地形；在坡面尺度上，坡面输运过程会在谱上留下特征尺度，使地形偏离分形（Perron et al. 2008）。

**实现。**用 fBm 谱合成剖面，Hurst 取 0.4～0.75。参数 `lmin` 去掉短于坡面尺度的分量，再做坡面扩散，产生特征尺度。早期版本 Hurst 取 0.6～0.95，谱比实测偏陡（地形过于平滑），已经修正。Hurst 和 `lmin` 的具体区间按场景设定，属于【经验取值】。

**坡面扩散【文献】。**土壤覆盖坡面的输运通量随坡度非线性增加，接近临界坡度时迅速变大：q = K·S / (1 − (S/Sc)²)。这使山顶呈凸形、坡面趋于平直（Roering et al. 1999；2001）。`hillslope_diffusion` 采用的就是这一形式，`diffusion` 参数是扩散长度尺度（取值为【经验取值】）。

**黄土塬【经验取值，待补文献】。**塬是黄土堆积后被流水侵蚀残留的高原面，地表平坦（坡度 1°～3°），四周为深切沟谷；黄土厚度一般 50～200 m，局部超过 300 m。这几个数字目前没有出处，见第 8 节。`topo_loess` 采用"平缓塬面 + 陡壁平底冲沟"的形态，黄土厚度取 50～150 m，沟底黄土被剥蚀后较薄。

**沙丘【文献】。**横向沙丘迎风坡约 5°～10°，背风滑落面接近休止角。Sutton et al.（2013a, 2013b）的实验给出平均临界角约 34°、平均休止角约 32°（转引自 Pelletier et al. 2015；Sutton 原文未核对）。Pelletier et al.（2015）的野外观测还表明，临界角和休止角都会随风速增大而减小，临界角也随滑落面高度略有下降。`topo_dunes` 直接由丘高和两侧坡角决定丘长，不单独抽取波长，保证坡角落在实测范围内。小沙丘（几米高）的滑落面只有几个网格宽，经过扩散后坡角会低于休止角。

## 2. 风化层厚度

**浅部（土壤 / 可移动风化物）【文献·外推】。**厚度与坡面曲率呈强线性关系（凹处厚、凸处薄），斜率随流域曲率标准差变化，截距等于平直坡处的厚度（Patton et al. 2018）。实现上把曲率按标准差归一化，取 `th_base + th_curv · cn`。

注意：Patton et al. 2018 研究的是米级的可移动风化物（拟合 RMSE 约 0.19 m），而地震意义上的低降速带往往有几十米。这里把同样的"凹厚凸薄"规律外推到更厚的风化层，是一个合理但没有直接验证的假设。`th_base`、`th_curv`、`th_slope` 的区间是【经验取值】。

**深部（裂隙化基岩底界）【文献·外推】。**地震折射和电阻率成像表明，裂隙发育带的底界有两种形态：水平构造挤压应力相对近地表重力应力较大时，底界"镜像"地形（山脊下更深）；比值较小时，底界平行于地表（St. Clair et al. 2015）。参数 `mirror` 控制这一点：平原取 0～0.1，丘陵取 0～0.3，山前带（挤压构造背景）取 0.1～0.5。

注意：St. Clair et al. 2015 只给出"镜像 / 平行"的定性判据，`mirror` 系数本身是【经验取值】。另外，实现里这一项是 `mirror × (平滑高程 − 均值)`，单位是米，随起伏幅度线性放大。起伏放大到几百米时，这一项可达 ±100 m 量级，远大于 `th_base`，厚度大面积被裁到 `th_lo` / `th_hi`。如果不希望这样，应改为按起伏幅度归一化。

## 3. 潜水面

**【文献·外推】**非承压含水层的潜水面常被当作"地形的平缓版"，但这一假设并不总成立：潜水面是否跟随地形取决于补给、渗透系数和地形尺度，可以用一个简单判据预先估计（Haitjema & Mitchell-Bruker 2005）。原文还提到另有数值研究表明，潜水面浅时主要受地形控制、近似平行于地表，变深时地形的控制减弱；这句目前没有出处，见第 8 节。

实现上，参数 `wt_follow` 按场景递减：平原 0.4～0.8，丘陵 0.3～0.7，山前带 0.2～0.5，黄土塬 0～0.2，沙漠 0～0.1。文献给的是判据，没有给这个系数，所以具体数值是【经验取值】。

## 4. 速度

**干燥松散层的梯度【文献】。**颗粒介质的 Hertz-Mindlin 接触理论预测速度与有效应力的 1/6 次方成正比。在浅部，有效应力近似与深度成正比，所以 v ∝ z^(1/6)。海滩干砂实测的浅层 P 波速度可低至 150 m/s 左右，并且随深度按幂律增加（Bachrach et al. 1998；2000）。实现上取 `v = v0 (1 + d/d0)^n`，n 为 0.15～0.30。干砂表层真实速度远低于 `v_floor`（默认 450 m/s），这是为控制网格做的妥协（见第 7 节）。

**饱和跳变【文献】。**饱和松散砂的 P 波速度一般为 1500～2000 m/s。微测井统计：

- 尼日尔三角洲 OML-62，61 个点（Akpabio & Onwusiri 2004）：风化层（降速层）210～994 m/s（平均 552），次风化层（高速层）1358～2464 m/s（平均 1734），风化层厚 2.8～52.8 m（平均 19.3）；
- 另一组统计：降速层 482～1147 m/s，高速层 1601～1832 m/s，厚 3～20.5 m。**这组数据没有出处**，见第 8 节。

`v_sat` 取 1500～2200 m/s，`v0` 取 450～1200 m/s，与这些统计一致。

**横波速度。**
- 深部 Vs【文献】：用 Castagna 泥岩线 Vs = (Vp − 1360)/1.16（Castagna et al. 1985）。原关系是对饱水碎屑岩拟合的，用到所有深部地层上是一个近似。
- 干燥层 Vp/Vs 取 1.5～2.0【文献·外推】：依据是干燥松散介质的泊松比低。Bachrach et al. 2000 专门研究了浅层松散砂的泊松比，可以作为依据，但具体数值还没有和原文对照过。
- 潜水面以下 Vs 不跳变：依据是 Gassmann 理论中流体不改变剪切模量。文献待补，见第 8 节。

**沙漠区的分层【待补文献】。**沙丘区的折射静校正通常把近地表分为三层：干燥风化层、潜水面以下层、高速层。

**高速夹层【文献·定性 + 经验取值】。**近地表薄高速层（如钙质硬壳）会造成速度反转，使初至呈"叠瓦"状；常规回折波层析不能正确处理速度反转。这是静校正中的经典难题，实现上以概率 `p_caliche` 加入。原文的出处只写了"见相关专利与技术文献"，没有具体文献，见第 8 节。硬壳的深度、厚度、速度、覆盖率都是【经验取值】。另外，硬壳厚 1.5～5 m，在大偏移距档的网格（6.25 m）上只表现为零星的单个格点。

**黄土的衰减【文献】。**厚层第四系黄土的强衰减会削弱深层反射；黄土塬区地貌多样（塬、梁、峁、沟），地形起伏复杂。黄土塬区正演一般需要同时考虑起伏地表和衰减 Q（Hu et al. 2023）。`nearsurf` 不提供 Q（见第 7 节）。

**过渡、非均质与其他【经验取值】。**下面这些都没有文献依据，是为了让模型看起来合理、并控制数值误差而设定的：

- 风化层底的 smoothstep 过渡（带宽 `trans`）；
- 裂隙化基岩顶面"先降 `frac`、再在约 20 m 内指数恢复"的形式。St. Clair et al. 2015 只支持"存在裂隙发育的低速带"这一定性结论；
- 2D 非均质随机场：强度 `het`、垂直相关 3 m、水平相关 30～150 m、按 e^(−d/60) 随深度衰减；
- 古河道低速透镜体：位置、尺寸、降速比例；
- `v_sat ≥ 1.3·v0` 的约束。

## 5. 海上（`nearsurf/marine.py`）

**沉积物 Vp/Vs【文献·改编】。**Hamilton（1979）把海底沉积物的 Vp、Vs 测量按深度对应起来：软的陆源沉积物 Vp/Vs
从海底处的 13 以上降到 1000 m 处的约 2.6；砂质沉积物从 5 m 处的约 9 降到 20 m 处的约 6。`nearsurf` 让 Vp/Vs
从海底的 4～8 随深度衰减到约 2.2，并用 `vs_floor` 截断 Vs：真实海底 Vs（常在 100 m/s 量级）对常见的正演网格太慢。
所以海底附近的取值是数值上的妥协，不是对 Hamilton 曲线的复现。

**海水、海底形态与沉积物速度【经验取值】。**海水声速 1475～1500 m/s、梯度 0.016 s⁻¹，密度 1025 kg/m³；
海底 Vp 1550～1900 m/s、梯度 0.6～2.0 s⁻¹；水深分档（偏向浅水，浅水区初至拾取最难）和五种海底形态沿用作者早先的
拖缆合成数据生成器。

## 6. 背景噪声（`nearsurf/noise`）

**陆上 / 海底节点噪声【经验取值】。**各分量（有色背景、工频、尖脉冲、爆发、斑块、共模低频、机械谐波、阵风、空气波）
和道缺陷（死道、极性、增益、削波、振铃、掉段）描述的是陆上数据里常见的噪声；水平和概率是对生成的道集目检后设定的，
不来自已发表的模型。`rich_v3` / `obn` 预设按远道初至的幅度定背景水平，所以信噪比和实测数据一样随炮检距变差。

**拖缆噪声【文献】。**四类噪声（涌浪、地震干扰、电缆振动、串音）及其频带、时差形态按 Hlebnikov et al.（2021）的分类；
涌浪 / 流噪声的物理机制见 Elboth et al.（2009）。各类幅度相对每炮信号峰值，是【经验取值】。`nearsurf/noise/streamer.py`
里还为个别噪声类型引用了 Elboth et al.（2010）、Gulunay（2008）、Parrish（2005）、Teigen et al.（2012），
这些是从原代码沿用的，尚未与原文核对。

## 7. 数值上的妥协与目前没有模拟的

- **速度下限 `v_floor` / `vs_floor`**：场景抽到的 v0 低于 `v_floor` 时，整条速度曲线整体抬高。这是为控制正演网格
  （频散判据）做的妥协，不是地质依据。下限较高时（比如 1100 m/s）有两个副作用：一是潜水面处的饱和跳变明显变弱；
  二是风化层 Vs 可能高于其下浅部基岩的 Vs，形成 Vs 反转。
- **衰减（Q）**：没有模拟，`nearsurf` 只给速度和密度。
- **三维效应**：模型是 2D 的；侧向散射和三维地形的影响没有体现。
- **黄土微地貌**：黄土陷穴、垂直节理带来的速度非均质和方位各向异性。

## 8. 待补出处的说法

下面这些在正文里出现过，但目前没有文献支撑。需要补出处，或者明确改成"经验取值"：

1. 黄土塬地貌数据：坡度 1°～3°，黄土厚一般 50～200 m、局部超过 300 m。
2. 潜水面："潜水面浅时近似平行于地表、变深时地形控制减弱"的数值研究。
3. 微测井的"另一组统计"（降速层 482～1147 m/s，高速层 1601～1832 m/s，厚 3～20.5 m）。
4. 沙漠区折射静校正的"三层划分"。
5. 钙质硬壳 / 速度反转对回折波层析的影响（原文只写"见相关专利与技术文献"）。
6. Gassmann 理论（潜水面以下 Vs 不跳变的依据）。
7. Sutton et al. 2013a, 2013b 的原文（沙丘临界角 / 休止角的一手数据）。

## 参考文献

- Akpabio, I.O., Onwusiri, H.N., 2004. The computation of LVL correction parameters in parts of Western Niger Delta (OML 62). *Global Journal of Geological Sciences* 2(1), 45–50.
- Bachrach, R., Dvorkin, J., Nur, A., 1998. High-resolution shallow-seismic experiments in sand, Part II: Velocities in shallow unconsolidated sand. *Geophysics* 63(4), 1234–1240.
- Bachrach, R., Dvorkin, J., Nur, A., 2000. Seismic velocities and Poisson's ratio of shallow unconsolidated sands. *Geophysics* 65, 559–564.
- Castagna, J.P., Batzle, M.L., Eastwood, R.L., 1985. Relationships between compressional-wave and shear-wave velocities in clastic silicate rocks. *Geophysics* 50(4), 571–581. doi:10.1190/1.1441933
- Elboth, T., Reif, B.A.P., Andreassen, Ø., 2009. Flow and swell noise in marine seismic data. *Geophysics* 74(2), Q17–Q25. doi:10.1190/1.3078403
- Haitjema, H.M., Mitchell-Bruker, S., 2005. Are water tables a subdued replica of the topography? *Ground Water* 43(6), 781–786. doi:10.1111/j.1745-6584.2005.00090.x
- Hamilton, E.L., 1979. Vp/Vs and Poisson's ratios in marine sediments and rocks. *Journal of the Acoustical Society of America* 66(4), 1093–1101.
- Hlebnikov, V., Elboth, T., Vinje, V., Gelius, L.-J., 2021. Noise types and their attenuation in towed marine seismic: A tutorial. *Geophysics* 86, W1–W19. doi:10.1190/geo2019-0808.1
- Hu, Z., Yang, J., Han, L., Huang, J., Qin, S., Sun, J., Yu, Y., 2023. Modeling seismic wave propagation in the Loess Plateau using a viscoacoustic wave equation with explicitly expressed quality factor. *Frontiers in Earth Science* 10, 1069166. doi:10.3389/feart.2022.1069166
- Newman, W.I., Turcotte, D.L., 1990. Cascade model for fluvial geomorphology. *Geophysical Journal International* 100.
- Patton, N.R., Lohse, K.A., Godsey, S.E., Crosby, B.T., Seyfried, M.S., 2018. Predicting soil thickness on soil mantled hillslopes. *Nature Communications* 9, 3329. doi:10.1038/s41467-018-05743-y
- Pelletier, J.D., Sherman, D.J., Ellis, J.T., Farrell, E.J., Jackson, N.L., Li, B., Nordstrom, K.F., Maia, L.P., Omidyeganeh, M., 2015. Dynamics of sediment storage and release on aeolian dune slip faces: A field study in Jericoacoara, Brazil. *Journal of Geophysical Research: Earth Surface* 120(9), 1911–1934. doi:10.1002/2015JF003636
- Perron, J.T., Kirchner, J.W., Dietrich, W.E., 2008. Spectral signatures of characteristic spatial scales and nonfractal structure in landscapes. *Journal of Geophysical Research* 113. doi:10.1029/2007JF000866
- Roering, J.J., Kirchner, J.W., Dietrich, W.E., 1999. Evidence for nonlinear, diffusive sediment transport on hillslopes and implications for landscape morphology. *Water Resources Research* 35, 853–870. doi:10.1029/1998WR900090
- Roering, J.J., Kirchner, J.W., Dietrich, W.E., 2001. Hillslope evolution by nonlinear, slope-dependent transport: Steady state morphology and equilibrium adjustment timescales. *Journal of Geophysical Research* 106, 16499–16513. doi:10.1029/2001JB000323
- St. Clair, J., Moon, S., Holbrook, W.S., Perron, J.T., Riebe, C.S., Martel, S.J., Carr, B., Harman, C., Singha, K., Richter, D.deB., 2015. Geophysical imaging reveals topographic stress control of bedrock weathering. *Science* 350(6260), 534–538. doi:10.1126/science.aab2210
- Turcotte, D.L., 1992. *Fractals and Chaos in Geology and Geophysics*. Cambridge University Press.