# AutoSPICE – implementation documentation

This document explains how `AutoSPICE.py` is built, what assumptions its Monte Carlo and PVT analyses make, and what it cannot do. It is the reference document. `README.md` and `README_CHANGES.md` are earlier, shorter notes. Where they disagree with this file, this file is correct.

> **Verification status.** The netlist parser, the metric maths (including the Fourier spectrum) and the file syntax were checked on their own. The whole app has **not** been run against ngspice by the author, because ngspice was not available while writing it. Expect to fix small problems on first use, especially in how ngspice output is read.

---

## 1. What the program does

You load a SPICE netlist (`.cir` or `.net`). You choose which parameters the program may change and in which range, and you describe the performance you want as formulas with goals. AutoSPICE then repeatedly edits the netlist, runs ngspice, measures the formulas, and searches for a design that meets all constraints. It then improves the remaining goals. Afterwards you can plot the results, check the design against process/voltage/temperature corners (PVT), and check its statistical spread (Monte Carlo).

Run it with:

```
brew install ngspice
pip install streamlit numpy pandas scipy plotly
streamlit run AutoSPICE.py
```

---

## 2. Architecture

Everything is in one file, `AutoSPICE.py`. It is organised in layers, from the bottom up. Each layer only uses the ones below it.

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ UI (Streamlit): sections 1–7, session state                          │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Optimizer (phase 1 + phase 2)     │  Robustness engine (MC, PVT)     │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Metric engine: formulas, signal library, violation / margin          │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Simulation engine: control block, ngspice run, result parsing        │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Netlist layer: load() → lines + parameters + analyses; render()      │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Helpers: engineering notation (num, eng, val)                        │
 └──────────────────────────────────────────────────────────────────────┘
```

### 2.1 Data flow of one evaluation

```
 parameter values ─► render() ─► netlist text ─► simulate() ─► signals (Sig objects)
                                                                    │
 constraint violation / margin / score  ◄─ calc() per metric  ◄─────┘
```

Every part of the program (optimizer, plots, Monte Carlo, PVT) uses this same chain. They differ only in how they choose the parameter values and what they do with the numbers.

### 2.2 The main objects

| Object | Meaning |
|---|---|
| `L` | List of netlist lines (continuation lines joined; line 0 is the title). |
| `P` | List of tunable parameters. Each has a key (e.g. `M1.W`), the nominal value, the line number, and a regular expression that finds the value in that line. |
| `AN` | The analysis lines (`tran`, `ac`, `noise`, `op`), editable in step 4. |
| `Sig` | A signal: `x` (time or frequency) and `y` (values; complex for AC). |
| `M` | The metrics list: Name, Formula, Goal, Target, Tol. |

---

## 3. Module walkthrough

### 3.1 Engineering notation

- `num("2.2u")` returns `2.2e-6`. Suffixes: `f p n u m k Meg G T` (and `mil`). As in SPICE, `m` is milli and mega is `Meg`. The micro signs `µ` and `μ` are accepted. Trailing unit letters are ignored (`10kHz`).
- `eng(2.2e-6)` returns `"2.2u"`; it is used for every table and axis.
- `val()` reads a table cell and raises a readable error if it cannot.
- Numbers inside metric formulas also accept suffixes (`input_noise(1k)`).

### 3.2 Netlist layer

**Loading (`load`)** does the following in order:
1. Replaces `µ`/`μ` by `u` and the LTspice `SINE(` by the ngspice `SIN(`.
2. Joins continuation lines (`+`).
3. Handles `.lib`, `.include`, `.inc`. If the file exists on this computer the line is kept (a relative path is made absolute). If not, as with an LTspice/Windows path, the line is **removed** and listed in a warning.
4. Removes the analysis lines `.tran .ac .noise .op .dc .end .backanno`. The first four are remembered in `AN` (an LTspice `.tran 10m` is converted to the ngspice form). `.dc` is discarded (see section 7).
5. Detects the tunable parameters:

| Element | Parameter names | Notes |
|---|---|---|
| `R`, `C`, `L` | the element name (`R1`) | value is the 4th token; skipped if it is an expression like `{Rload}` |
| `V`, `I` | the element name | DC value (`dc 1.8` or a plain 4th token) |
| `V`, `I` with `SIN(` | `V1.offset`, `V1.amp`, `V1.freq` | first three arguments |
| `V`, `I` with `PULSE(` | `.low .high .delay .rise .fall .width .period` | first seven arguments |
| `M` | `M1.W`, `M1.L` | from `W=` and `L=` |
| `.param` | `param.NAME` | numeric values only |

**Rendering (`render`)** copies the lines and, for each parameter in the value dictionary, replaces the value text with the new number using that parameter's regular expression. The original lines are never modified.

### 3.3 Simulation engine (`simulate`)

1. From the signals the metrics need, it builds one ngspice **control block** (`.control … .endc`) that runs only the required analyses, in this order: `op`, `ac`, `noise`, `tran`.
2. An analysis whose box in step 4 is empty raises a clear error. An empty box counts as "not defined".
3. Results are written with `wrdata` (AC data as separate real and imaginary vectors; noise from `noise1` via `inoise_spectrum`). Operating-point values are read from the text printed by `print`.
4. ngspice runs in batch mode (`ngspice -b`) in a temporary folder with a 180 s timeout. The folder is deleted afterwards, so parallel runs do not collide.
5. Everything is returned as a dictionary `{(analysis, name): Sig}`.

### 3.4 Metric engine

- **Signals** are written `analysis.name` (`tran.out`, `ac.out`, `op.i(vdd)`). The program scans all formulas to find which analyses to run (`needs`).
- **Formulas** are Python expressions evaluated on the signals with the built-in functions listed below.

| Group | Functions |
|---|---|
| AC | `dc_gain_db`, `bandwidth` (−3 dB), `ugf`, `phase_margin`, `gain_margin_db`, `pole(s,k,order)`, `zero(s,k,order)` |
| Noise | `input_noise(f)` (V/√Hz, interpolated) |
| Transient | `amplitude`, `gain`, `gain_db`, `slew_rate`, `overshoot`, `settling_time`, `mean`, `pk2pk` |
| Spectral | `snr`, `sndr`, `thd`, `sfdr`, `enob` |
| Power | `power(op.i(vdd), V)` |

- **Crossings.** If a crossing does not exist (for example the gain never reaches 0 dB), `ugf`, `bandwidth` and the margins extrapolate at −20 dB/decade rather than return `nan`. This gives the optimizer a direction to follow.
- **Phase margin** is `180° + phase` at the unity-gain frequency, with the DC phase taken as 0°. The AC signal is therefore treated as a loop gain.
- **Poles and zeros** come from a rational-function fit (Sanathanan–Koerner iterations) of the AC response. They are estimates.
- **Spectral metrics** use the last up to 32 whole periods of the transient, resampled on 16384 points (coherent FFT, no window). Harmonics 2 to 9 count as distortion.

### 3.5 Violation, margin and score

For a constraint metric, with `d` the shortfall (zero or negative when met):

```
violation = ln(1 + max(0, d) / scale)
scale     = |target|, or the metric value on the original netlist when the target is 0
```

The logarithm keeps one exploding metric (say 1e21) from dominating. The violation is exactly 0 when the constraint is met. A metric that is `nan` counts as 60. A crashed simulation counts as 60 for every constraint.

The **margin** (used in PVT) is the same shortfall, signed (negative when failing) and divided by the same scale.

For *maximize* and *minimize* goals, the **score** is the sum of `∓value/reference`, where the reference is that metric's value at the first feasible design. Lower is better. All goals have equal weight.

### 3.6 Optimizer

Parameters are mapped to numbers `u` in [0, 1]. The mapping is logarithmic when `min > 0` and `max/min > 10`, otherwise linear. The search works only on `u`, so your Min and Max are always respected.

**Phase 1: find a feasible design.** The nominal design is simulated first. If it already meets all constraints, phase 1 ends. Otherwise the program runs *rounds* until one design has violation 0 (or a limit is reached). Each round does:
1. **Differential evolution** (global). The population is made of the best designs so far, random perturbations of them, and Latin-hypercube samples.
2. **Nelder–Mead** (local) from the best design.

Each round is larger and longer than the last, and uses a different mutation strategy and random seed. Results are cached (points are compared at 5 decimals), so repeated points are not simulated again.

**Phase 2: improve the goals.** It runs only if a feasible design exists and there is at least one maximize/minimize goal. It uses **Powell's method** from the first feasible design. Infeasible points get a penalty of `100 + violation`.

**Limits.** Phase 1: simulations (default unlimited) and minutes (default 60). Phase 2: simulations (default 100) and minutes (default 10). The optimizer simulates one design at a time.

If no feasible design is found, the closest one is shown, with ❌ on the constraints that are not met and a list of parameters stuck at their range limit.

### 3.7 Results and plots

**Section 5** shows the metric table (initial vs optimum), the parameter table and a convergence plot.

**Section 6** re-simulates the initial and the optimum design with the signals you choose. It draws:

1. **Transient:** one subplot per signal.
2. **Fourier:** one spectrum per transient signal, below the transient plot (section 3.8).
3. **Bode:** magnitude and phase.
4. **Noise:** input-referred noise.
5. **Poles and zeros:** fitted poles and zeros on a signed-log scale.

It also fills the **summary table**: extra metrics (same Name/Formula structure) reported for both designs but never used as constraints.

### 3.8 Fourier plot

- Input: the fundamental frequency (optional). The DC component is removed, and the vertical axis is the peak amplitude in dBV.
- **With a fundamental frequency:** the last whole periods (at most 32) are linearly interpolated onto 16384 evenly spaced points and a plain FFT is taken. The fundamental falls exactly on a bin, so there is no leakage and no window is needed. The x-axis is linear up to 20 times the fundamental, and a table lists harmonics 1 to 6 in volts and dBc, for the initial and optimum designs.
- **Without a fundamental frequency:** the whole simulation is resampled and a Hann window is applied. The x-axis is logarithmic. The peak amplitude can read up to about 1.4 dB low if the tone falls between two bins. There is no harmonic table.
- The transient must contain at least 2 periods. The accuracy of the spectrum is limited by the transient time step (see section 7).

I checked the spectrum function against a synthetic signal: a 0.5 V tone with a 3rd harmonic at −20 dBc gave 0.49999959 V and −20.00006 dBc.

### 3.9 Parallel batch engine (Monte Carlo and PVT)

`batch()` receives a list of ready netlists. It simulates them in parallel threads (CPU cores minus one), measures the formulas on each, and updates a progress bar with an estimate of the time left. Only the simulation runs in the threads; all Streamlit calls stay in the main thread. A failed simulation gives an `_error` entry instead of values.

### 3.10 Session state

Streamlit re-runs the whole script at every interaction, so results are kept in `st.session_state`: the loaded netlist (`L`, `P`, `AN`), the metrics tables, the optimization result (`res`), the plot data (`plots`), and the Monte Carlo and PVT results (`mc`, `pvt`). Reloading the browser page clears them.

---

## 4. Monte Carlo analysis: assumptions

**What it does.** For each of the N runs, every selected parameter is multiplied by `(1 + e)`, where `e` is a random number. The netlist is simulated and the metrics measured. The design being varied is the optimum (or the initial design, your choice).

**Assumptions:**

1. **Every selected parameter varies independently.** R1 and R2 get different random numbers. There is no global (shared) variation and no correlation, so a shift that moves all resistors together is not modelled. Mismatch between two devices is modelled only in the sense that every selected parameter is independent.
2. **The variation is multiplicative and symmetric.** It is Gaussian with `σ = tolerance / 3` (so the tolerance is the 3σ value), or uniform between ±tolerance.
3. **The Gaussian is not truncated.** About 0.27 % of values lie outside the tolerance. With a very large tolerance a value could even become negative, and the program does not stop that.
4. **Only netlist parameters vary** (component values, W/L, `.param` values, source values). The `.model` parameters, the process, the temperature and the supply are not varied unless you select the matching parameter.
5. **Yield** is the share of runs that meet **every constraint** in the metrics table (including `=` with its tolerance). A failed simulation counts as a failure. If there are no constraints, no yield is computed.
6. **The confidence interval** is a 95 % Wilson interval on the yield. It is wide for small N. For a true yield of 50 %, N = 100 gives about ±10 % and N = 1000 about ±3 %.
7. **Reproducibility.** The seed seeds NumPy's random generator, so the same seed and settings give the same samples. The seed is not written into the CSV, so note it yourself.
8. **Histograms** have about √N bins, with the metric on the x-axis and the number of samples on the y-axis. Dashed lines mark the mean and ±3σ, and a red line marks the spec.
9. Only the metrics you choose are histogrammed, but all constraint metrics are measured for the yield.

**Not implemented:** global/local variation groups, Cpk, sensitivity ranking (which component limits the yield), and running Monte Carlo at the worst PVT corner.

---

## 5. PVT analysis: assumptions

**What it does.** A corner is a combination (process, supply set, temperature). The program simulates the full Cartesian product of the process corners, supply sets and temperatures you choose, and measures every metric in the metrics table.

**Process corners (TT, FF, SS, SF, FS):**

1. They are made by **editing the `.model` nmos/pmos cards written in the netlist**. The threshold parameter (`VTO` or `VTH0`) is shifted by ΔVth volts, and the mobility parameter (`KP` or `U0`) is scaled by ΔKP percent. Both amounts are constants you enter (defaults 50 mV and 10 %). They are not derived from foundry statistics, and they are not linked to the Monte Carlo tolerances.
2. "Fast" means a lower |Vth| and a higher KP, "slow" the opposite. The shift is applied to the threshold with the correct sign for NMOS (positive VTO) and PMOS (negative VTO).
3. **The same shift is applied to every nmos and pmos `.model` card.** TT leaves the netlist unchanged. SF and FS shift NMOS and PMOS in opposite directions.
4. If a card has none of the four parameters, it is left unchanged silently. The program only warns when the netlist contains no nmos/pmos `.model` cards at all.
5. **Models loaded from a library file are not changed.** The program does not switch `.lib` sections (tt/ff/ss) of a PDK.
6. Resistors, capacitors and other elements have **no process corner**.

**Voltage:**

1. You select voltage sources (those with a DC value) and give each a min, nominal and max value. The defaults are ±10 % of the design value.
2. By default all selected rails move together to min, nom or max. Unticking the option runs every combination, which grows quickly with the number of rails.
3. Waveform parameters (`V1.amp` and so on) are not treated as rails.

**Temperature:** a `.temp` line is added to the netlist for each temperature. How much the result changes depends entirely on the models. Resistors and capacitors given as plain values have no temperature coefficient, and the effect on transistors depends on the model level. Expect smaller temperature effects with simple models than with a real PDK.

**Pass/fail and margin.** A corner passes if all constraints are met. For each constraint the program shows the corner with the smallest margin (the worst case) and that margin. A simulation that fails or produces no value counts as a failure with no margin value.

**Not implemented:** re-optimizing at the worst corner, and linking PVT corners to the statistical model so that they bracket the Monte Carlo spread.

---

## 6. Practical notes

- **Analyses must be defined.** Metrics that use `tran`, `ac` or `noise` need the matching line in step 4. For the example `.ac dec 50 1 1G`, `.tran 1u 1m`, `.noise v(out) vin dec 20 1 1G`.
- **AC signals** assume an AC source of magnitude 1, so that `ac.out` is a gain.
- **Spectral metrics and the Fourier plot** need a transient of at least 2 periods, and a time step small enough that the tone is well sampled.
- **Node names** are lower-cased by ngspice. Use lower case in formulas.
- The files `single_stage_amp.cir` and `Draft43.net` were used to test the netlist parsing.

---

## 7. Limitations

**Netlist and simulator**
- Only `tran`, `ac`, `noise` and `op` analyses are used. `.dc`, `.step`, `.meas`, `.tf`, `.four` and similar lines are removed or ignored, and LTspice-only syntax is not converted, apart from the few cases listed in section 3.2.
- Only plain numbers can be tuned. Values written as expressions (`{Rload}`, `{2*R1}`) are not parameters. In a `.param` line only numeric values are found, and expressions that refer to other parameters are not evaluated by the program.
- Elements inside `.subckt` definitions are detected like any other lines, so changing one changes every instance. Subcircuit instances (`X…`) have no parameters of their own. This was not tested.
- Library files (`.lib`) with an unreadable path are discarded. If your design needs those models, define them with `.model` in the netlist.
- Operating-point values are read from ngspice's printed text, and branch currents rely on the form `name#branch`. A different ngspice version could print them differently.

**Metrics**
- Poles and zeros are fitted estimates, and a too-high order gives spurious ones.
- Phase margin and gain margin are meaningful only for a loop-gain signal.
- Spectral metrics resample the transient by linear interpolation, so a coarse or uneven time step adds error. Set a small maximum time step in `.tran`.
- Input noise requires a `.noise` analysis with a named input source.
- There is no area metric, and no built-in CMRR/PSRR.
- Metric formulas are evaluated with Python's `eval` with built-ins removed. That is **not a security sandbox**. Only use formulas you wrote yourself.

**Optimizer**
- The search is heuristic: a feasible design is not guaranteed, even when one exists, and an unlimited phase 1 will run forever if the constraints cannot be met. Use the time limit.
- Phase 2 is a local search from the first feasible design. It weighs all goals equally and can stall at a constraint boundary.
- The optimizer runs one simulation at a time, so it is slow for long simulations.
- Metrics with discontinuities or noise make the search harder.

**Robustness analyses**
- No mismatch/global model, no Cpk, no sensitivity ranking, no `.lib` corner switching (see sections 4 and 5).
- A Monte Carlo run of N simulations costs N times one simulation, divided by the number of cores. PVT cost is the number of process corners × supply sets × temperatures.

**Application**
- It is a local, single-user app. Results exist only in the browser session, so a page reload loses them. Use the CSV downloads to keep data.
- Streamlit's own *Stop* button cancels a run and discards its results.
- The app has not been tested on large netlists, and a very large netlist would make the tables long.
