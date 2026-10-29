# Physical basis and references

This document explains where each modelling choice in `nearsurf` comes from and how the parameter ranges were set.
[中文版](references.zh-CN.md)

Every step is tagged with the kind of support it has:

- **[literature]**: directly supported by the cited work;
- **[literature, extrapolated]**: the literature supports the qualitative behaviour, or the result is applied
  outside the scale the literature covers;
- **[engineering choice]**: no literature support; a setting chosen to make models plausible and numerically usable.

Statements that are used in the text but still lack a source are listed in [Open points](#8-open-points).

## 1. Topography (`nearsurf/land.py`)

**Spectrum [literature].** Topographic profiles have a power spectrum close to S(k) ∝ k⁻² over a wide range of
scales, i.e. fractional Brownian motion with Hurst exponent H ≈ 0.5 (β = 2H + 1) (Turcotte, 1992; Newman &
Turcotte, 1990). A purely fractal spectrum has no valleys or channel networks; at hillslope scales, transport
processes imprint a characteristic length and the topography departs from fractal behaviour (Perron et al., 2008).

*Implementation.* Profiles are synthesised from an fBm spectrum with H = 0.4–0.75. The parameter `lmin` removes
wavelengths shorter than the hillslope scale and hillslope diffusion then creates the characteristic scale.
The per-scenario ranges of H and `lmin` are [engineering choice].

**Hillslope diffusion [literature].** Soil flux on soil-mantled slopes grows non-linearly with slope and diverges
near a critical slope, q = K·S / (1 − (S/S_c)²), which produces convex hilltops and planar side slopes
(Roering et al., 1999, 2001). `hillslope_diffusion` uses this form; the diffusion length `diffusion` is an
[engineering choice].

**Loess plateau [engineering choice].** A plateau is the remnant of loess deposition dissected by fluvial erosion:
a nearly flat top bounded by deeply incised gullies. `topo_loess` uses "gentle plateau + steep-walled, flat-floored
gullies" with 50–150 m of loess, thinner on gully floors where it has been eroded. The geomorphic numbers behind
this are listed under [Open points](#8-open-points).

**Dunes [literature].** Transverse dunes have stoss slopes of about 5–10° and lee (slip) faces near the angle of
repose. Experiments by Sutton et al. (2013a, b) give a mean critical angle of about 34° and a mean angle of repose of
about 32° (as reported by Pelletier et al., 2015; the Sutton papers were not checked). Field observations by
Pelletier et al. (2015) show that both angles decrease with wind speed and that the critical angle decreases slightly
with slip-face height. `topo_dunes` sets the dune length from the dune height and the two slope angles, so the slopes
stay within the observed range. Small dunes (a few metres high) have slip faces only a few grid cells wide and end up
below the angle of repose after diffusion.

## 2. Weathering thickness

**Shallow part (soil / mobile regolith) [literature, extrapolated].** Thickness is strongly and linearly related to
hillslope curvature (thick in hollows, thin on noses); the slope of the relation scales with the standard deviation
of curvature in the catchment and the intercept equals the thickness on planar slopes (Patton et al., 2018).
`nearsurf` normalises curvature by its standard deviation and uses `th_base + th_curv · cn`.

Patton et al. (2018) studied metre-scale mobile regolith (fit RMSE about 0.19 m), whereas seismic low-velocity
layers are often tens of metres thick. Applying the same "thick in hollows, thin on noses" rule to the thicker
weathering layer is a reasonable but unverified assumption. The ranges of `th_base`, `th_curv` and `th_slope` are
[engineering choice].

**Deep part (base of fractured bedrock) [literature, extrapolated].** Seismic refraction and resistivity imaging
show two shapes of the base of the fractured zone: it mirrors the topography (deeper under ridges) when horizontal
tectonic compression is large relative to near-surface gravitational stress, and parallels the surface when the ratio
is small (St. Clair et al., 2015). The parameter `mirror` controls this: 0–0.1 for plains, 0–0.3 for hills and
0.1–0.5 for foothills (compressional setting).

St. Clair et al. (2015) give only the qualitative mirror / parallel criterion; the value of `mirror` is an
[engineering choice]. In the implementation the term is `mirror × (smoothed elevation − mean)` in metres, so it
grows linearly with relief: with relief scaled to several hundred metres it reaches ±100 m, far more than `th_base`,
and thickness is often clipped at `th_lo` / `th_hi`. Normalising it by the relief would avoid this.

## 3. Water table

**[literature, extrapolated].** The water table of an unconfined aquifer is often assumed to be a subdued copy of the
topography, but whether it follows the topography depends on recharge, hydraulic conductivity and the scale of the
terrain, and can be estimated with a simple criterion (Haitjema & Mitchell-Bruker, 2005).

`wt_follow` decreases by scenario: plain 0.4–0.8, hills 0.3–0.7, foothills 0.2–0.5, loess 0–0.2, desert 0–0.1.
The literature gives a criterion, not this coefficient, so the values are an [engineering choice].

## 4. Velocities

**Gradient in dry unconsolidated material [literature].** Hertz–Mindlin contact theory for granular media predicts
velocity proportional to effective stress to the power 1/6; near the surface, effective stress is roughly
proportional to depth, so v ∝ z^(1/6). Shallow P-wave velocities in dry beach sand can be as low as about 150 m/s and
increase with depth following a power law (Bachrach et al., 1998, 2000). `nearsurf` uses `v = v0 (1 + d/d0)^n` with
n = 0.15–0.30. Real dry-sand velocities near the surface are far below the default `v_floor` (450 m/s); that floor is
a numerical compromise (Section 7).

**Jump at the water table [literature].** Saturated unconsolidated sand typically has Vp of 1500–2000 m/s.
Uphole statistics from 61 points in OML 62, Niger Delta (Akpabio & Onwusiri, 2004): weathering layer 210–994 m/s
(mean 552), sub-weathering layer 1358–2464 m/s (mean 1734), weathering thickness 2.8–52.8 m (mean 19.3).
`v_sat` = 1500–2200 m/s and `v0` = 450–1200 m/s are consistent with these values.

**Shear-wave velocity.**
- Deep Vs [literature]: the mudrock line Vs = (Vp − 1360)/1.16 (Castagna et al., 1985). It was fitted to
  water-saturated clastic rocks; applying it to all deep formations is an approximation.
- Dry-layer Vp/Vs = 1.5–2.0 [literature, extrapolated]: dry unconsolidated media have low Poisson's ratios;
  Bachrach et al. (2000) studied Poisson's ratio of shallow unconsolidated sands, but the exact values have not been
  compared with that paper.
- No jump in Vs at the water table: fluids do not change the shear modulus in Gassmann's theory (source to be added,
  see Open points).

**Fast crust (caliche) [literature, qualitative + engineering choice].** A thin, fast near-surface layer causes a
velocity inversion and "shingled" first arrivals that conventional diving-wave tomography cannot handle. It is added
with probability `p_caliche`; its depth, thickness, velocity and coverage are [engineering choice]. A 1.5–5 m thick
crust shows up only as isolated cells on a coarse (e.g. 6.25 m) grid.

**Loess attenuation [literature].** Strong attenuation in thick Quaternary loess weakens deep reflections, and loess
landforms (plateaus, ridges, hills, gullies) make the topography complex; modelling in loess areas generally needs
both irregular topography and attenuation Q (Hu et al., 2023). `nearsurf` does not provide Q (Section 7).

**Transitions, heterogeneity and the rest [engineering choice].** These are set to make the models look plausible
and to control numerical error:

- the smoothstep transition at the weathering base (width `trans`);
- the fractured top of the deep model, lowered by `frac` and recovering exponentially over about 20 m
  (St. Clair et al. (2015) support only the existence of a fractured low-velocity zone);
- the anisotropic random field: strength `het`, vertical correlation 3 m, horizontal 30–150 m, decaying as e^(−d/60);
- low-velocity channel fill: position, size and velocity reduction;
- the constraint `v_sat ≥ 1.3·v0`.

## 5. Offshore (`nearsurf/marine.py`)

**Sediment Vp/Vs [literature, adapted].** Hamilton (1979) linked Vp and Vs measurements in marine sediments: in
soft terrigenous sediments Vp/Vs falls from about 13 or more at the seafloor to about 2.6 at 1000 m, and in sands from
about 9 at 5 m to about 6 at 20 m. `nearsurf` lets Vp/Vs decay from 4–8 at the seafloor towards about 2.2 at depth,
and clips Vs at `vs_floor`: the real seafloor Vs (often ~100 m/s) is too slow for typical solver grids. The seafloor
values are therefore a numerical compromise rather than a reproduction of Hamilton's curves.

**Water, seafloor shapes and sediment velocities [engineering choice].** Water sound speed 1475–1500 m/s with a
0.016 s⁻¹ gradient, density 1025 kg/m³; seafloor Vp 1550–1900 m/s with a 0.6–2.0 s⁻¹ gradient; the water-depth classes
(weighted towards shallow water, where first-break picking is hardest) and the five seafloor styles follow an earlier
synthetic-streamer generator by the author.

## 6. Background noise (`nearsurf/noise`)

**Land / ocean-bottom noise [engineering choice].** The components (coloured ambient noise, power line, spikes,
bursts, patchy and common-mode low-frequency noise, machinery harmonics, wind gusts, air blast) and the trace
defects (dead, polarity, gain, clipping, ringing, dropouts) describe noise commonly seen in land data; their levels
and probabilities were set by visual QC of the generated gathers and are not taken from a published model. The
`rich_v3` / `obn` presets set the background level from the amplitude of the far-offset first breaks, so the
signal-to-noise ratio degrades with offset as in field data.

**Towed-streamer noise [literature].** The four types (swell, seismic interference, strumming, cross-feed) and their
frequency bands and moveouts follow the taxonomy of Hlebnikov et al. (2021); the physics of swell / flow noise is
discussed by Elboth et al. (2009). Amplitudes are relative to the peak signal of each shot and are an
[engineering choice]. `nearsurf/noise/streamer.py` also cites Elboth et al. (2010), Gulunay (2008), Parrish (2005)
and Teigen et al. (2012) for individual noise types; these were carried over from the original code and have not
been checked against the papers.

## 7. Numerical compromises and what is not modelled

- **Velocity floors `v_floor` / `vs_floor`.** When a scenario draws a `v0` below `v_floor`, the whole velocity curve
  is shifted up. This is a compromise for the solver's grid (dispersion), not geology. With a high floor
  (e.g. 1100 m/s) the jump at the water table becomes weak, and the weathering-layer Vs can exceed the Vs of the
  shallow bedrock below (a Vs inversion).
- **Attenuation (Q)** is not modelled.
- **Three-dimensional effects**: models are 2D; out-of-plane scattering and 3D topography are not represented.
- **Loess micro-relief**: sinkholes and vertical joints (velocity heterogeneity and azimuthal anisotropy) are not modelled.

## 8. Open points

Statements used above that still need a source, or should be relabelled as engineering choices:

1. Loess plateau geomorphology: plateau slopes of 1–3°, loess generally 50–200 m thick and locally more than 300 m.
2. Water table: numerical studies showing that a shallow water table parallels the surface and that topographic
   control weakens as it deepens.
3. A second set of uphole statistics (weathering 482–1147 m/s, sub-weathering 1601–1832 m/s, thickness 3–20.5 m).
4. The three-layer split (dry weathering, below water table, high-velocity layer) used for refraction statics in dunes.
5. The effect of caliche / velocity inversions on diving-wave tomography.
6. Gassmann's theory (why Vs does not jump at the water table).
7. The original Sutton et al. (2013a, b) papers (critical angle and angle of repose of dunes).

Contributions of sources are welcome.

## References

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
