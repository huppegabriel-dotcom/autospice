# SPICE Auto-Sizer – what changed in this version

Two changes: (1) engineering-notation input and display of numbers, and (2) a much more persistent optimizer that keeps searching until it finds a design meeting all constraints. The file is still `spice_autosizer.py`. Section 4 of the original README is superseded by section 2 below; the rest still applies.

## 1. Engineering notation (n, u, m, k, Meg …)

You can now type values with SPICE-style suffixes everywhere a number is entered, and results are displayed the same way.

| Suffix | Factor | Suffix | Factor |
|---|---|---|---|
| `f` | 1e-15 | `k` | 1e3 |
| `p` | 1e-12 | `Meg` | 1e6 |
| `n` | 1e-9 | `G` | 1e9 |
| `u` | 1e-6 | `T` | 1e12 |
| `m` | 1e-3 | | |

Where it applies:
- **Parameter table:** Nominal, Min and Max are shown as e.g. `2.2u`, `10k`, `1Meg`. Type the same way. Plain `1e-9` also still works.
- **Metrics table:** the Target and Tol columns accept suffixes (`10Meg`, `50n`).
- **Formulas:** numbers inside formulas accept suffixes too, e.g. `input_noise(1k)`, `snr(tran.out, 10k)`.
- **Results:** the metric table, the parameter table and the plot axes use the same notation. Plot axes use SI prefixes.

Important: as in SPICE, **`m` means milli, and mega is `Meg`** (never `M`). Trailing unit letters are ignored (`10kHz`, `5V`). An unreadable value gives a clear error message instead of crashing.

## 2. Optimizer: why it did not converge, and what is different now

I could not reproduce your run, so I can't say for certain which one caused the ~1e+21 values. I found three likely causes in the old code and fixed all of them.

### 2.1 Problems fixed

1. **Division by a tiny number.** A constraint violation was divided by `|target|`. When a target was `0` (the default target when you add a metric from the list), it was divided by 1e-30, so even a tiny miss became about 1e+21. Now, if the target is 0, the violation is scaled by the metric's value on your original netlist (or 1 if that is 0).
2. **Unbounded values.** Metrics such as fitted zeros can return extreme numbers. The violation is now `ln(1 + shortfall / scale)`. It is still exactly 0 when a constraint is met, but it grows slowly, so a metric at 1e21 gives about 48 instead of 1e21. Small violations (a few %) are almost unchanged.
3. **Flat plateaus.** When a measurement didn't exist (for example, the gain never crossed 0 dB), the metric was `nan` and the optimizer received the same penalty everywhere, so it had nothing to follow. `ugf`, `bandwidth`, `phase_margin` and `gain_margin_db` now extrapolate at −20 dB/decade beyond the sweep, so a design that is "closer" to crossing scores better than one that is farther. A metric that is still `nan` or infinite gets a fixed penalty of 60.

### 2.2 The new Phase 1: keep searching until feasible

Phase 1 is now a loop of search rounds. It continues until it finds a design with violation 0, or until a limit you set is reached (by default there is no simulation limit).

Each round, in order:
1. **Differential evolution** (global search). The starting population contains:
   - the best designs found so far,
   - random perturbations of those designs, and
   - Latin-hypercube samples (evenly spread random points).
2. **Nelder–Mead** (local search) from the best design found so far, to refine it.

Each new round differs from the last:
- the population grows (from `max(10, 3 × parameters)` up to 100),
- each differential-evolution run is longer (15 + 5 × round generations),
- it uses a different mutation strategy (cycling through four),
- it uses a different random seed.

Because each round starts from the best designs already found, nothing is lost between rounds, and results are cached so repeated points are never re-simulated.

The search stops as soon as one design meets every constraint.

### 2.3 Limits and time

Four new inputs in step 4:

| Input | Default | Meaning |
|---|---|---|
| Phase 1 · max simulations | 0 (unlimited) | stop searching after this many simulations |
| Phase 1 · max minutes | 60 | stop searching after this time (0 = unlimited) |
| Phase 2 · max simulations | 100 | simulation budget for improving the goals |
| Phase 2 · max minutes | 10 | time limit for improving the goals (0 = unlimited) |

- The Phase 2 budget now counts **from the moment Phase 2 starts**. Before, Phase 1 and Phase 2 shared one total, so a long Phase 1 could leave Phase 2 nothing.
- With no limits, the progress panel shows "searching until feasible" instead of an estimate.
- If a limit is reached, you still get a result: the closest design found, with ❌ marking the constraints that are not met.
- Streamlit's own **Stop** button (top right) cancels the run and **discards its results**. For long runs, use the time limit instead.

### 2.4 Better feedback when no feasible design exists

If the search ends without a feasible design, the results page now says so and lists any parameters that ended up at the limit of their range (within 2% of min or max). A parameter stuck at a limit is the usual sign that the range is too narrow.

### 2.5 Other robustness changes
- The original netlist is simulated first. If that fails, you get the ngspice error immediately instead of a silent failure inside the optimizer.
- A formula that fails (typo, missing signal) now only affects its own metric, and the error is shown in the progress line. Before, it made every metric `nan`.
- Min and Max are checked (`Min < Max`) before starting.

## 3. Using it on your example

1. Select only a few parameters (for example `M1.W`, `M6.W`, `M5.W`, `Cc`, `Ibias`) with ranges such as `1u` to `200u` for widths.
2. Add the metrics, and type the targets in engineering notation (`10Meg`, `50n`).
3. Leave the Phase 1 limits at the defaults, or set a time limit you are comfortable with.
4. If it still ends with no feasible design, look at the ❌ rows and the "parameters at their range limit" message, then widen those ranges or relax that constraint.

## 4. Honest limits

- A feasible design is **not guaranteed**. If the constraints cannot be met together within your parameter ranges (for example, a gain-bandwidth that needs more current than Level-1 models with 1 µm devices can give), the search will not find one however long it runs. With no limits it would run indefinitely, which is why the Phase 1 time limit defaults to 60 minutes.
- Phase 2 is still a local search from the first feasible design, with equal weights on all goals.
- I checked the updated file for syntax errors, but I have not run it against ngspice here, so please test it on your example and send me any error.
