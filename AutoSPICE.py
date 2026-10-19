"""AutoSPICE: ngspice + Streamlit.   Run:  streamlit run AutoSPICE.py
Setup (macOS): brew install ngspice && pip install streamlit numpy pandas scipy plotly"""
import re, time, tempfile, subprocess, os
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.interpolate import CubicSpline
from scipy.stats import qmc

# ───────────────────────── netlist parsing ─────────────────────────
SI = {'t': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15, 'mil': 25.4e-6}
def num(s):
    m = re.fullmatch(r'([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)(meg|mil|[tgkmunpf])?[a-z]*', s.lower().replace('µ', 'u').replace('μ', 'u'))
    return float(m[1]) * SI.get(m[2], 1) if m else None

def eng(v):
    """1.5e-6 -> '1.5u', 2e6 -> '2Meg' (SPICE-style engineering notation)."""
    if v is None or not np.isfinite(v): return str(v)
    if v == 0: return "0"
    for p_, s_ in ((1e12, "T"), (1e9, "G"), (1e6, "Meg"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f")):
        if abs(v) >= p_ * 0.9999999: return f"{v / p_:.4g}{s_}"
    return f"{v:.3g}"
def val(x, default=0.0):
    if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "": return default
    v = num(str(x).strip())
    if v is None: raise ValueError(f"Cannot read the number '{x}' (use e.g. 4.7k, 2u, 1Meg, 1e-9)")
    return v
SUFFIX = re.compile(r"(?<![\w.])(?:\d+\.?\d*|\.\d+)(?:meg|[tgkmunpf])(?![\w(])", re.I)  # 10k, 2.2u inside formulas

LIBS = {}  # uploaded library files: lower-case file name -> path on disk (filled when a netlist is loaded)
HINT = dict(tran="tran 1u 1m", ac="ac dec 50 1 1G", noise="noise v(out) vin dec 20 1 1G")
DIR = re.compile(r'^\s*\.(tran|ac|noise|op|dc|end|backanno)\b', re.I)

WAVE = {'sin': ['offset', 'amp', 'freq'], 'pulse': ['low', 'high', 'delay', 'rise', 'fall', 'width', 'period']}

def load(text):
    text = text.replace('µ', 'u').replace('μ', 'u')            # LTspice writes a real micro sign
    text = re.sub(r'(?i)\bsine\s*\(', 'SIN(', text)            # LTspice SINE( -> ngspice SIN(
    dropped = []
    L = []
    for l in text.replace("\r", "").split("\n"):
        if l.startswith("+") and L: L[-1] += " " + l[1:]
        else: L.append(l)
    P, AN = [], {}; cur, scope = None, {}
    for i, l in enumerate(L):
        s = l.strip()
        ml = re.match(r'^\.(lib|include|inc)\s+(.*)$', s, re.I)
        if ml:  # use library files that exist on this computer or were uploaded (matched by file name); drop the others
            raw_ = ml[2].strip(); q_ = re.match(r'"([^"]+)"\s*(.*)$', raw_)
            if q_: path_, rest_ = q_[1], q_[2]
            elif os.path.exists(raw_) or " " not in raw_: path_, rest_ = raw_, ""
            else: path_, rest_ = raw_.split(None, 1)
            hit = path_ if os.path.exists(path_) else LIBS.get(os.path.basename(path_.replace("\\", "/")).lower())
            if hit:
                kw = "include" if (ml[1].lower() != "lib" or not rest_) else "lib"  # ngspice .lib needs a section name; LTspice `.lib file` = include
                L[i] = f'.{kw} "{os.path.abspath(hit)}"' + (f" {rest_}" if kw == "lib" else "")
            else: dropped.append(s); L[i] = ""
            continue
        ms = re.match(r"^\.subckt\s+(\S+)(.*)$", s, re.I)
        if ms:  # remember the scope; default values after `params:` are tunable as  subcktname:PARAM
            cur = ms[1]
            for k_, v_ in re.findall(r"(\w+)\s*=\s*([^\s=]+)", ms[2]):
                if num(v_) is not None: P.append(dict(key=f"{cur}:{k_}", nom=num(v_), i=i, kind="p", pat=rf"(^\s*\.subckt\s.*?(?<!\w){k_}\s*=\s*)([^\s=]+)"))
            continue
        if re.match(r"^\.ends\b", s, re.I): cur = None; continue
        if cur: scope[i] = cur
        m = DIR.match(s)
        if m:
            k = m[1].lower()
            if k in ("tran", "ac", "noise", "op"): AN[k] = s[1:].strip()
            L[i] = ""; continue
        if s.lower().startswith(".param"):
            for mm in re.finditer(r"(\w+)\s*=\s*([^\s=]+)", s):
                v = num(mm[2])
                if v is not None: P.append(dict(key="param." + mm[1], nom=v, i=i, pat=rf"(\b{mm[1]}\s*=\s*)(\S+)", kind="p"))
            continue
        if i == 0 or not s or s[0] in "*;$.": continue
        t = s.split(); c = t[0][0].upper(); nm = t[0]
        pre = rf"^(\s*{re.escape(nm)}\s+\S+\s+\S+\s+)(\S+)"
        if c in "RCL" and len(t) > 3 and num(t[3]) is not None:
            P.append(dict(key=nm, nom=num(t[3]), i=i, pat=pre, kind=c))
        elif c in "VI":
            mw = re.search(r'\b(sin|pulse)\s*\(', s, re.I)
            if mw:  # waveform arguments: SIN(offset amp freq ...), PULSE(low high delay rise fall width period)
                for k, wn in enumerate(WAVE[mw[1].lower()]):
                    m3 = re.search(rf'\b{mw[1]}\s*\(\s*(?:[^\s)]+\s+){{{k}}}([^\s)]+)', s, re.I)
                    if m3 and num(m3[1]) is not None:
                        P.append(dict(key=f'{nm}.{wn}', nom=num(m3[1]), i=i, kind='w', pat=rf'(\b{mw[1]}\s*\(\s*(?:[^\s)]+\s+){{{k}}})([^\s)]+)'))
            m2 = re.search(r"\bdc\s+(\S+)", s, re.I)
            if m2 and num(m2[1]) is not None: P.append(dict(key=nm, nom=num(m2[1]), i=i, pat=r"(\bdc\s+)(\S+)", kind=c))
            elif len(t) > 3 and num(t[3]) is not None: P.append(dict(key=nm, nom=num(t[3]), i=i, pat=pre, kind=c))
        elif c == "M":
            for k in "WL":
                m2 = re.search(rf"(?<!\w){k}\s*=\s*(\S+)", s, re.I)
                if m2 and num(m2[1]) is not None:
                    P.append(dict(key=f"{nm}.{k}", nom=num(m2[1]), i=i, pat=rf"((?<!\w){k}\s*=\s*)(\S+)", kind="M"))
        elif c == "X":  # instance parameters:  X1 a b out opamp A0=100k  ->  X1.A0
            for mm in re.finditer(r"(?<![\w.])(\w+)\s*=\s*([^\s=]+)", s):
                if num(mm[2]) is not None: P.append(dict(key=f"{nm}.{mm[1]}", nom=num(mm[2]), i=i, kind="x", pat=rf"((?<![\w.]){mm[1]}\s*=\s*)(\S+)"))
    for p_ in P:  # elements inside a .subckt definition are named  subckt:element  (a change affects every instance)
        if p_["i"] in scope: p_["key"] = f'{scope[p_["i"]]}:{p_["key"]}'
    if "tran" in AN:  # LTspice -> ngspice tran syntax
        p = AN["tran"].split()
        if len(p) == 2 and num(p[1]): AN["tran"] = f"tran {num(p[1])/1000:g} {p[1]}"
        elif len(p) > 2 and num(p[1]) == 0 and num(p[2]): p[1] = f"{num(p[2])/1000:g}"; AN["tran"] = " ".join(p)
    return L, P, AN, dropped

def render(L, P, vals):
    L = L[:]
    for p in P:
        if p["key"] in vals:
            L[p["i"]] = re.sub(p["pat"], lambda m: m[1] + f"{vals[p['key']]:.6g}", L[p["i"]], count=1, flags=re.I)
    return L

# ───────────────────────── simulation ─────────────────────────
class Sig:
    """A signal (x = time or frequency). Signals can be added, subtracted, multiplied: `ac.vout+ - ac.vout-` is a differential signal."""
    def __init__(s, x, y): s.x, s.y = x, y
    def _o(s, o, f): return Sig(s.x, f(s.y, o.y if isinstance(o, Sig) else o))
    __add__ = __radd__ = lambda s, o: s._o(o, lambda a, b: a + b)
    __sub__ = lambda s, o: s._o(o, lambda a, b: a - b)
    __rsub__ = lambda s, o: s._o(o, lambda a, b: b - a)
    __mul__ = __rmul__ = lambda s, o: s._o(o, lambda a, b: a * b)
    __truediv__ = lambda s, o: s._o(o, lambda a, b: a / b)
    __neg__ = lambda s: Sig(s.x, -s.y)

NODEMAP = {}  # original node name (lower case) -> name safe for ngspice expressions; filled from the loaded netlist
def node_set(L):  # node names used by the elements of the netlist
    nodes = set()
    for i, l in enumerate(L):
        t = l.split()
        if i == 0 or not t or t[0][0] in "*;$.": continue
        c = t[0][0].upper()
        if c in "RCLVID": nodes.update(t[1:3])
        elif c in "MEG": nodes.update(t[1:5])
        elif c == "Q":
            nodes.update(t[1:4])
            if len(t) >= 6 and num(t[5]) is None and "=" not in t[5]: nodes.add(t[4])  # optional substrate node
        elif c == "X": nodes.update([x for x in t[1:] if "=" not in x][:-1])
    return nodes
def unknown_nodes(L, names):  # signals that name a node which does not exist (typos like `vout` for `Vout+`)
    ok = {n.lower() for n in node_set(L)} | {"0", "gnd", "inoise"}
    return sorted({n for n in names if "(" not in n and "." not in n and n.lower() not in ok})
def subckt_check(L):  # subcircuit instances whose definition cannot be found
    defined = {m[1].lower() for l in L for m in [re.match(r"^\s*\.subckt\s+(\S+)", l, re.I)] if m}
    for l in L:
        m = re.match(r'^\s*\.(?:include|lib)\s+"([^"]+)"', l, re.I)
        if m and os.path.exists(m[1]): defined |= {x.lower() for x in re.findall(r"(?im)^\s*\.subckt\s+(\S+)", open(m[1], errors="ignore").read())}
    used = {}
    for i, l in enumerate(L):
        t = l.split()
        if i and t and t[0][0] in "Xx":
            used.setdefault([x for x in t[1:] if "=" not in x and x.lower() != "params:"][-1].lower(), t[0])
    return sorted((m, x) for m, x in used.items() if m not in defined)
def node_map(L):
    nodes = node_set(L); names = {n.lower() for n in nodes}; m = {}
    for n in nodes:
        if re.search(r"\W", n):
            sf = re.sub(r"\W", lambda x: {"+": "_p", "-": "_n"}.get(x[0], "_"), n).lower()
            while sf in names or sf in m.values(): sf += "x"
            m[n.lower()] = sf
    return m
def fixnames(text):
    for n in sorted(NODEMAP, key=len, reverse=True):
        text = re.sub(r"(?<![\w+\-.#])" + re.escape(n) + r"(?![\w+\-#])", NODEMAP[n], text, flags=re.I)
    return text

vec = lambda n: n if "(" in n else f"v({n})"

def simulate(L, AN, need):
    A = {k: v.strip().lstrip(".") for k, v in AN.items() if v and v.strip()}  # empty boxes count as "not defined"
    for a in need:
        if a != "op" and a not in A: raise RuntimeError(f"No {a} analysis is defined: fill the '{a}' box in step 4 (e.g. {HINT[a]}).")
    ctl = [".control", "set noaskquit"]
    ns = {a: sorted(need[a]) for a in need}
    if "op" in need: ctl += ["op"] + [f"print {vec(n)}" for n in ns["op"]]
    if "ac" in need:
        ctl += [A["ac"], "let f_=real(frequency)"]
        for j, n in enumerate(ns["ac"]): ctl += [f"let a{j}r=real({vec(n)})", f"let a{j}i=imag({vec(n)})"]
        ctl.append("wrdata ac.txt f_ " + " ".join(f"a{j}r a{j}i" for j in range(len(ns["ac"]))))
    if "noise" in need: ctl += [A["noise"], "setplot noise1", "wrdata noise.txt inoise_spectrum"]
    if "tran" in need: ctl += [A["tran"], "wrdata tran.txt " + " ".join(vec(n) for n in ns["tran"])]
    ctl.append(".endc")
    with tempfile.TemporaryDirectory() as wd:
        open(f"{wd}/c.cir", "w").write(fixnames("\n".join(L) + "\n" + "\n".join(ctl) + "\n.end\n"))
        r = subprocess.run(["ngspice", "-b", "c.cir"], cwd=wd, capture_output=True, text=True, timeout=180)
        D = {}
        def rd(f):
            if not os.path.exists(f"{wd}/{f}"): raise RuntimeError(f"ngspice produced no {f}:\n" + (r.stdout + r.stderr)[-600:])
            return np.loadtxt(f"{wd}/{f}", ndmin=2)
        if "op" in need:
            for m in re.finditer(r"^\s*(\S+)\s*=\s*(\S+)\s*$", r.stdout, re.M):
                k = re.sub(r"^(\w+)#branch$", r"i(\1)", m[1].lower())
                try: D[("op", k)] = Sig(None, float(m[2]))
                except ValueError: pass
        if "ac" in need:
            c = rd("ac.txt")[:, 1::2]
            for j, n in enumerate(ns["ac"]): D[("ac", n)] = Sig(c[:, 0], c[:, 1 + 2*j] + 1j * c[:, 2 + 2*j])
        if "noise" in need:
            A = rd("noise.txt"); D[("noise", "inoise")] = Sig(A[:, 0], A[:, 1])
        if "tran" in need:
            A = rd("tran.txt")
            for j, n in enumerate(ns["tran"]): D[("tran", n)] = Sig(A[:, 0], A[:, 1 + 2*j])
    return D

# ───────────────────────── metric library ─────────────────────────
def _cross(f, y, lv):
    i = np.where((y[:-1] >= lv) & (y[1:] < lv))[0]
    if len(i):
        i = i[0]; a = (lv - y[i]) / (y[i+1] - y[i]); lf = np.log10(f)
        return 10 ** (lf[i] + a * (lf[i+1] - lf[i]))
    # no crossing: extrapolate at -20 dB/dec so "closer" designs still look better than "farther" ones
    if y[0] < lv: return f[0] * 10 ** ((y[0] - lv) / 20)
    return f[-1] * 10 ** ((y[-1] - lv) / 20)
_db = lambda s: 20 * np.log10(np.abs(s.y) + 1e-300)
_ph = lambda s: np.degrees(np.unwrap(np.angle(s.y))) - np.degrees(np.angle(s.y[0]))
_amp = lambda s: (s.y.max() - s.y.min()) / 2

def _at(s, arr, f): return np.interp(np.log10(f), np.log10(s.x), arr)
def dc_gain_db(s): return _db(s)[0]
def bandwidth(s): d = _db(s); return _cross(s.x, d, d[0] - 3.0103)
def ugf(s): return _cross(s.x, _db(s), 0.0)
def phase_margin(s):
    fu = ugf(s); return np.nan if np.isnan(fu) else 180 + _at(s, _ph(s), fu)
def gain_margin_db(s):
    f = _cross(s.x, _ph(s), -180.0); return np.nan if np.isnan(f) else -_at(s, _db(s), f)

def _fit(s, npl, nz):  # Sanathanan-Koerner rational fit of H(jw)
    w = 2 * np.pi * s.x; w0 = np.sqrt(w[w > 0].min() * w.max()); sv = 1j * w / w0; H = s.y
    wt = 1 / np.abs(H)
    for _ in range(10):
        A = np.hstack([sv[:, None] ** np.arange(nz + 1), -H[:, None] * sv[:, None] ** np.arange(1, npl + 1)]) * wt[:, None]
        b = H * wt
        x = np.linalg.lstsq(np.vstack([A.real, A.imag]), np.r_[b.real, b.imag], rcond=None)[0]
        n, d = x[:nz + 1], np.r_[1, x[nz + 1:]]
        wt = 1 / (np.abs(H) * np.abs(np.polyval(d[::-1], sv)))
    return np.roots(n[::-1]) * w0 / (2 * np.pi), np.roots(d[::-1]) * w0 / (2 * np.pi)
def _kth(r, k): r = sorted(np.abs(r)); return r[int(k) - 1] if len(r) >= k else np.nan
def pole(s, k, order=4): return _kth(_fit(s, int(max(order, k)), int(max(order, k)) - 1)[1], k)
def zero(s, k, order=4): return _kth(_fit(s, int(max(order, k)) , int(max(order, k)))[0], k)

def _resample(t, y, tt):  # cubic-spline resampling of the (uneven) transient time grid
    t, k = np.unique(t, return_index=True); y = y[k]
    return CubicSpline(t, y)(tt) if len(t) > 3 else np.interp(tt, t, y)
def _f0(s, N=2 ** 14):  # frequency of the dominant tone: Hann FFT peak, then refined by maximising the single-frequency DFT
    t = s.x; tt = np.linspace(t[0], t[-1], N, endpoint=False); y = _resample(t, s.y, tt); y = y - y.mean(); w = np.hanning(N)
    X = np.abs(np.fft.rfft(y * w)); k = int(np.argmax(X[2:N // 2 - 2])) + 2; df = 1 / (t[-1] - t[0])
    a, b, c = (np.log(X[k + d] + 1e-30) for d in (-1, 0, 1)); den = a - 2 * b + c
    f = (k + (0.5 * (a - c) / den if den else 0)) * df; tr = tt - tt[0]
    return minimize_scalar(lambda fx: -abs(np.sum(y * w * np.exp(-2j * np.pi * fx * tr))), bounds=(f - 1.5 * df, f + 1.5 * df),
                           method="bounded", options=dict(xatol=df * 1e-7)).x
def _coh(s, fin, mmax, N=2 ** 14):
    """Coherent sampling: the last m WHOLE periods (m <= mmax) are resampled on N points, so the tone sits exactly in bin m."""
    t = s.x; fin = fin or _f0(s); m = min(int(np.floor((t[-1] - t[0]) * fin + 1e-9)), mmax)
    if m < 2: raise RuntimeError("the transient is too short: it needs at least 2 periods of the fundamental")
    tt = t[-1] - m / fin + np.arange(N) * (m / fin / N)
    return _resample(t, s.y, tt), m, fin, tt
def _spec(s, fin, nh=9):
    yy, m, *_ = _coh(s, fin, 32); N = len(yy)
    P = np.abs(np.fft.rfft(yy - yy.mean())) ** 2 / N ** 2; P[0] = 0
    hb = [m * h for h in range(2, nh + 1) if m * h < len(P)]
    sig, harm = P[m], P[hb].sum(); noise = P.sum() - sig - harm
    return sig, harm, noise, P, m
def snr(s, fin): a, h, n, *_ = _spec(s, fin); return 10 * np.log10(a / n)
def sndr(s, fin): a, h, n, *_ = _spec(s, fin); return 10 * np.log10(a / (n + h))
def thd(s, fin): a, h, n, *_ = _spec(s, fin); return 10 * np.log10(h / a)
def enob(s, fin): return (sndr(s, fin) - 1.76) / 6.02
def sfdr(s, fin): a, h, n, P, m = _spec(s, fin); Q = P.copy(); Q[m] = 0; return 10 * np.log10(a / Q.max())
def slew_rate(s):
    dt = np.diff(s.x); ok = dt > 0; return np.max(np.abs(np.diff(s.y)[ok] / dt[ok]))
def overshoot(s):
    st_ = s.y[-1] - s.y[0]
    return np.nan if st_ == 0 else ((s.y.max() if st_ > 0 else s.y.min()) - s.y[-1]) / st_ * 100
def settling_time(s, tol=0.01):
    out = np.where(np.abs(s.y - s.y[-1]) > tol * abs(s.y[-1] - s.y[0]))[0]
    return s.x[out[-1]] - s.x[0] if len(out) else 0.0
FUNCS = dict(amplitude=_amp, gain=lambda a, b: _amp(a) / _amp(b), gain_db=lambda a, b: 20 * np.log10(_amp(a) / _amp(b)),
             dc_gain_db=dc_gain_db, bandwidth=bandwidth, ugf=ugf, phase_margin=phase_margin, gain_margin_db=gain_margin_db,
             pole=pole, zero=zero, snr=snr, sndr=sndr, thd=thd, enob=enob, sfdr=sfdr, slew_rate=slew_rate,
             overshoot=overshoot, settling_time=settling_time, power=lambda s, v: -s.y * v,
             mean=lambda s: float(np.mean(s.y)), pk2pk=lambda s: float(s.y.max() - s.y.min()), np=np)
SIGRE = re.compile(r"\b(tran|ac|op)\.((?:i|v)\([^)\s]+\)|\w+(?:\.\w+)*)", re.I)  # x1.n1 = node inside instance X1
def sigsub(f):  # formula text -> python expression; node names containing + or - are recognised from the netlist
    f = SUFFIX.sub(lambda m: repr(num(m[0])), f)
    for n in sorted((n for n in NODEMAP), key=len, reverse=True):
        f = re.sub(r"\b(tran|ac|op)\." + re.escape(n) + r"(?![\w])", lambda m: f'S("{m[1].lower()}","{n}")', f, flags=re.I)
    return SIGRE.sub(lambda m: f'S("{m[1].lower()}","{m[2].lower()}")', f)
def needs(formulas):
    nd = {}
    for f in formulas:
        for a, n in re.findall(r'S\("(\w+)","([^"]+)"\)', sigsub(f)): nd.setdefault(a, set()).add(n)
        if "input_noise(" in f: nd.setdefault("noise", set()).add("inoise")
    return nd
def calc(D, f):
    env = dict(FUNCS); env["S"] = lambda a, n: D[(a, n)]
    env["input_noise"] = lambda fr: float(np.interp(fr, D[("noise", "inoise")].x, D[("noise", "inoise")].y))
    return float(eval(sigsub(f), {"__builtins__": {}}, env))

TEMPLATES = {
    "Gain (tran)": ("Gain", "gain(tran.out, tran.in)"), "DC gain (dB)": ("DCgain_dB", "dc_gain_db(ac.out)"),
    "Bandwidth -3dB (Hz)": ("BW", "bandwidth(ac.out)"), "Unity-gain freq (Hz)": ("UGF", "ugf(ac.out)"),
    "Phase margin (deg)": ("PM", "phase_margin(ac.out)"), "Gain margin (dB)": ("GM", "gain_margin_db(ac.out)"),
    "Dominant pole #1 (Hz)": ("P1", "pole(ac.out, 1)"), "Dominant zero #1 (Hz)": ("Z1", "zero(ac.out, 1)"),
    "Input noise @1kHz (V/√Hz)": ("Vn", "input_noise(1e3)"), "SNR (dB)": ("SNR", "snr(tran.out, 1e3)"),
    "SNDR (dB)": ("SNDR", "sndr(tran.out, 1e3)"), "THD (dB)": ("THD", "thd(tran.out, 1e3)"),
    "ENOB (bits)": ("ENOB", "enob(tran.out, 1e3)"), "SFDR (dB)": ("SFDR", "sfdr(tran.out, 1e3)"),
    "Slew rate (V/s)": ("SR", "slew_rate(tran.out)"), "Settling time (s)": ("Ts", "settling_time(tran.out)"),
    "Overshoot (%)": ("OS", "overshoot(tran.out)"), "Power (W)": ("Pwr", "power(op.i(vdd), 1.8)"),
    "Output swing (V)": ("Swing", "pk2pk(tran.out)")}
GOALS = ["maximize", "minimize", ">=", ">", "<=", "<", "="]

def violation(v, g, t, tol, sc=1.0):
    # 0 when met; otherwise log(1 + shortfall/scale). The log keeps it bounded even if a metric explodes (1e21...),
    # and the scale falls back to the initial metric value when the target is 0.
    if g in ("maximize", "minimize"): return 0.0
    if not np.isfinite(v): return 60.0
    d = t - v if g in (">=", ">") else v - t if g in ("<=", "<") else abs(v - t) - abs(tol or 0)
    return float(np.log1p(max(0.0, d) / max(abs(t) if t else sc, 1e-30)))

# ───────── Fourier analysis of a transient signal ─────────
def spectrum(s, fin=None, mmax=8, N=2 ** 14):
    """Coherent single-sided amplitude spectrum (volts, peak), DC removed. Returns (freq, amplitude, m, fin_used, leakage_dB).
    The last m whole periods are resampled so the fundamental is exactly bin m (bin width = fin/m).
    fin None -> the fundamental is detected automatically. leakage = energy in bins m±1..3 relative to the fundamental."""
    yy, m, fin, tt = _coh(s, fin, mmax, N); yy = yy - yy.mean()
    X = np.abs(np.fft.rfft(yy)) * 2 / N; P = X ** 2
    leak = 10 * np.log10((P[max(1, m - 3):m + 4].sum() - P[m] + 1e-300) / (P[m] + 1e-300))
    return np.fft.rfftfreq(N, tt[1] - tt[0]), X, m, fin, leak

# ───────────────────────── UI ─────────────────────────
st.set_page_config(page_title="AutoSPICE", layout="wide")
st.title("AutoSPICE")
with st.expander("Optional: library / subcircuit files used by the netlist (.sub, .lib, .inc)"):
    libs_up = st.file_uploader("Files referenced by .include / .lib lines (matched by file name)", accept_multiple_files=True, key="libfiles")
up = st.file_uploader("1 · Netlist (.cir / .net)", type=["cir", "net", "sp", "txt"])
if not up: st.stop()
sig_ = (up.name, tuple((f.name, f.size) for f in libs_up))
if st.session_state.get("fname") != sig_:
    st.session_state.clear(); st.session_state.fname = sig_
    d_ = tempfile.mkdtemp(prefix="autospice_")
    for f_ in libs_up:  # saved on disk so that ngspice can .include them
        b_ = f_.getvalue()
        try: tx_ = b_.decode("utf-8")
        except UnicodeDecodeError: tx_ = b_.decode("latin-1")
        LIBS[os.path.basename(f_.name).lower()] = pth_ = os.path.join(d_, os.path.basename(f_.name))
        open(pth_, "w", encoding="utf-8").write(tx_.replace("µ", "u").replace("μ", "u"))
    raw = up.getvalue()
    try: txt = raw.decode("utf-8")
    except UnicodeDecodeError: txt = raw.decode("latin-1")
    st.session_state.L, st.session_state.P, st.session_state.AN, st.session_state.dropped = load(txt)
    st.session_state.metrics = pd.DataFrame(columns=["Name", "Formula", "Goal", "Target", "Tol"]); st.session_state.mk = 0
L, P, AN = st.session_state.L, st.session_state.P, st.session_state.AN
NODEMAP.update(node_map(L))
for mdl_, x_ in subckt_check(L):
    st.warning(f"{x_} uses the subcircuit `{mdl_}`, but no `.subckt {mdl_}` was found in the netlist or in an included file. "
               "Upload the library file in the optional box above the netlist.")
if st.session_state.dropped:
    st.warning("Removed library line(s) that ngspice cannot read (the file does not exist on this computer, e.g. an LTspice/Windows path). "
               "Models written with `.model` in the netlist are kept:\n\n" + "\n".join(f"- `{d}`" for d in st.session_state.dropped))

st.subheader("2 · Parameters the optimizer may change")
fmt = st.column_config.TextColumn(help="Engineering suffixes: f p n u m k Meg G  (e.g. 2.2u, 10k, 1Meg, 1e-9)")
rng_ = lambda p: (p["nom"] * .5, p["nom"] * 1.5) if p["kind"] in "VI" else (p["nom"] * .1, p["nom"] * 10)
pdf = pd.DataFrame([dict(Select=False, Parameter=p["key"], Nominal=eng(p["nom"]), Min=eng(min(rng_(p))), Max=eng(max(rng_(p)))) for p in P])
ped = st.data_editor(pdf, hide_index=True, use_container_width=True, disabled=["Parameter", "Nominal"], key="ptab",
                     column_config={"Nominal": fmt, "Min": fmt, "Max": fmt})

st.subheader("3 · Performance metrics")
c1, c2 = st.columns([3, 1])
tpl = c1.selectbox("Add a built-in metric", list(TEMPLATES))
if c2.button("Add", use_container_width=True):
    n, f = TEMPLATES[tpl]
    st.session_state.metrics = pd.concat([st.session_state.metrics, pd.DataFrame([dict(Name=n, Formula=f, Goal=">=", Target="0", Tol="0")])], ignore_index=True)
    st.session_state.mk += 1; st.rerun()
st.caption("Signals: `tran.node`, `ac.node`, `op.node`, `tran.i(vname)` · all functions: " + ", ".join(k for k in FUNCS if k != "np") + ", input_noise")
med = st.data_editor(st.session_state.metrics, num_rows="dynamic", use_container_width=True, key=f"m{st.session_state.mk}",
                     column_config={"Goal": st.column_config.SelectboxColumn(options=GOALS, required=True),
                                    "Target": fmt, "Tol": fmt})
st.session_state.metrics = med.copy()

st.subheader("4 · Analyses & budget")
a1, a2, a3 = st.columns(3)
for k, col in zip(("tran", "ac", "noise"), (a1, a2, a3)): AN[k] = col.text_input(k, AN.get(k, ""), key="an_" + k, placeholder=HINT[k])
b1, b2, b3, b4 = st.columns(4)
N1 = int(b1.number_input("Phase 1 · max simulations (0 = unlimited)", 0, 1000000, 0))
T1 = b2.number_input("Phase 1 · max minutes (0 = unlimited)", 0, 100000, 60)
N2 = int(b3.number_input("Phase 2 · max simulations", 0, 1000000, 100))
T2 = b4.number_input("Phase 2 · max minutes (0 = unlimited)", 0, 100000, 10)

class Stop(Exception): pass
class Found(Exception): pass

if st.button("Run optimization", type="primary"):
    sel = ped[ped.Select]
    try:
        M = [dict(Name=r.Name if isinstance(r.Name, str) and r.Name else r.Formula, Formula=r.Formula,
                  Goal=r.Goal if r.Goal in GOALS else ">=", Target=val(r.Target), Tol=val(r.Tol))
             for r in med.dropna(subset=["Formula"]).itertuples()]
        lo = np.array([val(x) for x in sel.Min]); hi = np.array([val(x) for x in sel.Max])
    except ValueError as e: st.error(str(e)); st.stop()
    if sel.empty or not M: st.error("Select at least one parameter and one metric."); st.stop()
    if np.any(lo >= hi): st.error("Each Min must be smaller than its Max."); st.stop()
    names = list(sel.Parameter); n = len(names)
    logs = (lo > 0) & (hi / np.maximum(lo, 1e-300) > 10)
    span = np.where(logs, np.log(hi / np.maximum(lo, 1e-300)), hi - lo)
    dec = lambda u: dict(zip(names, np.where(logs, lo * np.exp(u * span), lo + u * span)))
    enc = lambda v: np.clip(np.where(logs, np.log(np.maximum(v, 1e-300) / np.maximum(lo, 1e-300)), v - lo) / span, 0, 1)
    nd = needs([m["Formula"] for m in M]); nom = {p["key"]: p["nom"] for p in P}
    bad = unknown_nodes(L, [n for a in nd for n in nd[a]])
    if bad: st.error(f"Unknown node(s) in the formulas: {', '.join(bad)}. Nodes of this netlist: {', '.join(sorted(node_set(L)))}. "
                     "Use the exact name (e.g. `ac.vout+`), and put spaces around a minus sign: `ac.vout+ - ac.vout-`."); st.stop()
    now = time.time()
    S = dict(limit=N1 if N1 > 0 else np.inf, n0=0, phase=1, round=0, refs=None, sc={}, t0=now, ps=now, sim_t=0.0,
             dl=now + T1 * 60 if T1 > 0 else np.inf)
    hist, cache = [], {}
    box, bar, chart = st.empty(), st.progress(0.0), st.empty()

    def run(vals):
        D = simulate(render(L, P, vals), AN, nd); mv = {}
        for m in M:
            try: mv[m["Name"]] = calc(D, m["Formula"])
            except Exception as e: mv[m["Name"]] = np.nan; S["err"] = f"{m['Name']}: {str(e)[:200]}"
        return D, mv

    try: D0, m0 = run({})
    except Exception as e: st.error(f"The initial netlist could not be simulated:\n\n{e}"); st.stop()
    if S.get("err"): st.warning("Initial design, metric problem → " + S["err"])
    # per-metric scale used when the target is 0 (avoids dividing by ~0)
    S["sc"] = {k: (abs(v) if np.isfinite(v) and v != 0 else 1.0) for k, v in m0.items()}

    def ev(u):
        key = tuple(np.round(u, 5))
        if key in cache: return cache[key]
        if len(hist) >= S["limit"] or time.time() > S["dl"]: raise Stop
        t = time.time()
        try: _, mv = run(dec(np.asarray(u)))
        except Exception as e: mv = {m["Name"]: np.nan for m in M}; S["err"] = str(e)[:300]
        S["sim_t"] += time.time() - t
        v = sum(violation(mv[m["Name"]], m["Goal"], m["Target"], m["Tol"], S["sc"][m["Name"]]) for m in M)
        sc = 0.0
        if S["refs"]:
            for m in M:
                if m["Goal"] in ("maximize", "minimize") and np.isfinite(mv[m["Name"]]):
                    sc += (-1 if m["Goal"] == "maximize" else 1) * mv[m["Name"]] / S["refs"][m["Name"]]
        r = dict(u=np.array(u), m=mv, viol=v, score=sc, phase=S["phase"]); hist.append(r); cache[key] = r
        k = len(hist); avg = S["sim_t"] / k; best = min(h["viol"] for h in hist); fr, eta = [], []
        if np.isfinite(S["limit"]): fr.append((k - S["n0"]) / (S["limit"] - S["n0"])); eta.append(avg * (S["limit"] - k))
        if np.isfinite(S["dl"]): fr.append((time.time() - S["ps"]) / (S["dl"] - S["ps"])); eta.append(S["dl"] - time.time())
        box.info(f"Phase {S['phase']}" + (f" · round {S['round']}" if S["phase"] == 1 else "") + f" · {k} sims · {avg:.2f}s/sim · "
                 f"elapsed {time.time()-S['t0']:.0f}s · " + (f"ETA ≤ {max(min(eta), 0):.0f}s" if eta else "searching until feasible") +
                 f" · best violation {best:.3g}" + (f" · last error: {S['err']}" if S.get("err") else ""))
        bar.progress(float(min(max(fr), 1.0)) if fr else 0.0)
        chart.line_chart(pd.DataFrame({"violation": [h["viol"] for h in hist], "best": np.minimum.accumulate([h["viol"] for h in hist])}))
        return r

    top = lambda k: [h["u"] for h in sorted(hist, key=lambda h: h["viol"])[:k]]
    feasible = lambda: any(h["viol"] == 0 for h in hist)
    def f1(u):
        r = ev(u)
        if r["viol"] == 0: raise Found
        return r["viol"]

    STRAT = ["best1bin", "rand1bin", "currenttobest1bin", "randtobest1bin"]
    try:
        ev(enc(np.array([nom[k] for k in names], float)))
        # ── Phase 1: restart loop (global DE + local Nelder-Mead), widening each round, until feasible or limit ──
        while not feasible():
            S["round"] += 1; rnd = S["round"]; rng = np.random.default_rng(rnd)
            m_ = min(100, max(10, 3 * n) * (1 + rnd // 2))
            seeds = top(max(1, m_ // 10)); k = m_ // 3
            pert = [np.clip(seeds[i % len(seeds)] + rng.normal(0, 0.3 / np.sqrt(rnd) + 0.02, n), 0, 1) for i in range(k)]
            pop = np.vstack(seeds + pert + [qmc.LatinHypercube(d=n, seed=rnd).random(m_ - len(seeds) - k)])
            try:
                differential_evolution(f1, [(0, 1)] * n, init=pop, strategy=STRAT[(rnd - 1) % 4], maxiter=15 + 5 * rnd, tol=0,
                                       polish=False, seed=rnd, updating="immediate", mutation=(0.4, 1.0), recombination=0.8)
                minimize(f1, top(1)[0], method="Nelder-Mead", bounds=[(0, 1)] * n, options=dict(maxfev=40 * n, xatol=1e-4, fatol=1e-9))
            except Found: break
        # ── Phase 2: improve maximize/minimize goals while staying feasible ──
        feas = [h for h in hist if h["viol"] == 0]
        goals = [m for m in M if m["Goal"] in ("maximize", "minimize")]
        if feas and goals and N2 > 0:
            S.update(phase=2, n0=len(hist), limit=len(hist) + N2, ps=time.time(), dl=time.time() + T2 * 60 if T2 > 0 else np.inf)
            S["refs"] = {k: (abs(v) if np.isfinite(v) and v != 0 else 1.0) for k, v in feas[0]["m"].items()}
            cache.clear(); feas[0]["score"] = 0.0
            f2 = lambda u: (lambda r: r["score"] if r["viol"] == 0 else 100 + r["viol"])(ev(u))
            try: minimize(f2, feas[0]["u"], method="Powell", bounds=[(0, 1)] * n, options=dict(xtol=1e-3, ftol=1e-4))
            except Stop: pass
    except (Stop, Found): pass
    feas = [h for h in hist if h["viol"] == 0]
    best = min(feas, key=lambda h: h["score"]) if feas else min(hist, key=lambda h: h["viol"])
    vb = dec(best["u"])
    try: Db, _ = run(vb)
    except Exception: Db = D0
    box.success(f"Done: {len(hist)} simulations, {S['round']} search round(s), {time.time()-S['t0']:.0f}s — " +
                ("all constraints met ✅" if best["viol"] == 0 else "no feasible design found ⚠️ (showing the closest one)"))
    st.session_state.res = dict(hist=hist, best=best, vb=vb, D0=D0, Db=Db, M=M, m0=hist[0]["m"], nd=nd, names=names, sc=S["sc"])

# ───────────────────────── results ─────────────────────────
R = st.session_state.get("res")
if R:
    st.subheader("5 · Results")
    rows = []
    for m in R["M"]:
        v = R["best"]["m"][m["Name"]]
        rows.append({"Metric": m["Name"], "Goal": f"{m['Goal']} {eng(m['Target']) if m['Goal'] not in ('maximize','minimize') else ''}",
                     "Initial": eng(R["m0"][m["Name"]]), "Optimum": eng(v),
                     "Met": "–" if m["Goal"] in ("maximize", "minimize") else ("✅" if violation(v, m["Goal"], m["Target"], m["Tol"], R["sc"][m["Name"]]) == 0 else "❌")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    nom = {p["key"]: p["nom"] for p in P}
    st.dataframe(pd.DataFrame([{"Parameter": k, "Initial": eng(nom[k]), "Optimum": eng(v)} for k, v in R["vb"].items()]), use_container_width=True, hide_index=True)
    pin = [k for k, u in zip(R["names"], R["best"]["u"]) if u < .02 or u > .98]
    if R["best"]["viol"] > 0:
        st.warning("No design met every constraint (see ❌ rows); the closest design is shown." +
                   (f" Parameters at their range limit: {', '.join(pin)} — consider widening them." if pin else ""))
    h = R["hist"]; fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=[x["viol"] + 1e-12 for x in h], name="constraint violation", mode="markers"), secondary_y=False)
    fig.add_trace(go.Scatter(y=[x["score"] if x["viol"] == 0 else None for x in h], name="goal score (feasible)", mode="lines+markers"), secondary_y=True)
    fig.update_yaxes(type="log", secondary_y=False, title="violation"); fig.update_layout(title="Convergence", xaxis_title="simulation #")
    st.plotly_chart(fig, use_container_width=True)
    C = {"initial": "#1f5fbf", "optimum": "#e63b2e"}
    def grid(fg, title, hm="x unified"):
        ax = dict(tickformat=".3~s", showgrid=True, gridcolor="rgba(128,128,128,.40)", showline=True, mirror=True, linecolor="gray",
                  ticks="outside", showticklabels=True, minor=dict(showgrid=True, gridcolor="rgba(128,128,128,.15)", ticks="outside"))
        fg.update_layout(title=title, hovermode=hm, legend=dict(orientation="h", y=1.08)); fg.update_xaxes(**ax); fg.update_yaxes(**ax); return fg

    st.subheader("6 · Result plots & summary table")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    sig_t = c1.text_input("Transient signals (comma-separated)", "out")
    sig_a = c2.text_input("AC signal (Bode + pole-zero)", "out")
    order = int(c3.number_input("Pole/zero fit order", 1, 12, 3))
    fin_txt = c4.text_input("Fourier fundamental (Hz)", "", placeholder="blank = auto", help="Frequency of the input tone, e.g. 10k. Blank: detected automatically. Either way the FFT is coherent (tone in a single bin).")
    e1, e2 = st.columns(2)
    harm = int(e1.number_input("Fourier: harmonics shown", 3, 50, 10)); scale_ = e2.radio("Fourier: amplitude scale", ["Linear (V)", "dBV"], horizontal=True)
    st.caption("Summary metrics: reported for the initial and optimum designs, **not** used as constraints.")
    st.session_state.setdefault("summ", pd.DataFrame(columns=["Name", "Formula"])); st.session_state.setdefault("sk", 0)
    d1, d2 = st.columns([3, 1])
    tpl2 = d1.selectbox("Add a summary metric", list(TEMPLATES), key="tpl2")
    if d2.button("Add", key="add2", use_container_width=True):
        st.session_state.summ = pd.concat([st.session_state.summ, pd.DataFrame([dict(zip(["Name", "Formula"], TEMPLATES[tpl2]))])], ignore_index=True)
        st.session_state.sk += 1; st.rerun()
    sm = st.data_editor(st.session_state.summ, num_rows="dynamic", use_container_width=True, key=f"s{st.session_state.sk}")
    st.session_state.summ = sm.copy()

    if st.button("Generate plots and summary", type="primary"):
        S_ = [(r.Name if isinstance(r.Name, str) and r.Name else r.Formula, r.Formula) for r in sm.dropna(subset=["Formula"]).itertuples()]
        ts = [t.strip().lower() for t in sig_t.split(",") if t.strip()]
        nd2 = needs([f for _, f in S_])
        if ts and AN.get("tran", "").strip(): nd2.setdefault("tran", set()).update(ts)
        if sig_a.strip() and AN.get("ac", "").strip(): nd2.setdefault("ac", set()).add(sig_a.strip().lower())
        if AN.get("noise", "").strip(): nd2.setdefault("noise", set()).add("inoise")
        try: fin_ = val(fin_txt) or None
        except ValueError as e: st.error(str(e)); st.stop()
        bad_ = unknown_nodes(L, [n for a in nd2 for n in nd2[a]])
        if bad_: st.error(f"Unknown node(s): {', '.join(bad_)}. Nodes of this netlist: {', '.join(sorted(node_set(L)))}"); st.stop()
        with st.spinner("Simulating the initial and optimum designs…"):
            try: D0_, Db_ = simulate(render(L, P, {}), AN, nd2), simulate(render(L, P, R["vb"]), AN, nd2)
            except Exception as e: st.error(str(e)); st.stop()
        def mv(D, f):
            try: return calc(D, f)
            except Exception: return np.nan
        st.session_state.plots = dict(D0=D0_, Db=Db_, ts=ts, ac=sig_a.strip().lower(), order=order, fin=fin_, harm=harm, scale=scale_,
            rows=[{"Metric": nm, "Formula": f, "Initial": eng(mv(D0_, f)), "Optimum": eng(mv(Db_, f))} for nm, f in S_])

    Pl = st.session_state.get("plots")
    if Pl:
        D0_, Db_ = Pl["D0"], Pl["Db"]
        both = lambda a, n: [(D0_[(a, n)], "initial"), (Db_[(a, n)], "optimum")]
        if Pl["rows"]: st.dataframe(pd.DataFrame(Pl["rows"]), use_container_width=True, hide_index=True)
        # 1) transient
        ts = [t for t in Pl["ts"] if ("tran", t) in D0_]
        if ts:
            fg = make_subplots(rows=len(ts), cols=1, shared_xaxes=True, subplot_titles=[f"tran.{t}" for t in ts])
            for i, t in enumerate(ts, 1):
                for s_, nm in both("tran", t):
                    fg.add_trace(go.Scatter(x=s_.x, y=s_.y, name=nm, mode="lines", legendgroup=nm, showlegend=(i == 1), line=dict(color=C[nm])), i, 1)
            fg.update_xaxes(title_text="Time (s)", row=len(ts), col=1); fg.update_layout(height=300 * len(ts) + 120)
            st.plotly_chart(grid(fg, "Transient analysis"), use_container_width=True)
            fin = Pl.get("fin"); H = Pl.get("harm", 10); dbs = Pl.get("scale") == "dBV"
            for t in ts:  # coherent Fourier analysis of every transient signal, drawn as a histogram (one bar per FFT bin)
                try:
                    cur, sp, info = [], {}, []
                    for s_, nm in both("tran", t):
                        f_, X_, m_, fu_, lk_ = spectrum(s_, fin); sp[nm] = (X_, m_, fu_)
                        k_ = np.arange(1, min(len(f_), int((H + 0.5) * m_) + 1)); y_ = 20 * np.log10(X_[k_] + 1e-18) if dbs else X_[k_]
                        cur.append((nm, f_[k_], y_, f_[1] - f_[0])); info.append((nm, lk_, fu_, m_))
                    top = max(c[2].max() for c in cur); base = top - 120 if dbs else 0.0
                    fg = go.Figure()
                    for nm, fx, yv, bw in cur:
                        fg.add_trace(go.Bar(x=fx, y=np.maximum(yv - base, 0), base=base, width=bw, name=nm, opacity=0.65,
                                            marker=dict(color=C[nm], line=dict(color="black", width=0.4))))
                    fu_ = info[-1][2]; m_ = info[-1][3]
                    fg.update_layout(barmode="overlay", bargap=0)
                    fg = grid(fg, f"Fourier analysis · tran.{t} · coherent FFT over {m_} periods", hm="x")
                    fg.update_xaxes(title_text="Frequency (Hz) — ticks at the harmonics", tickmode="array", tickvals=[h * fu_ for h in range(H + 1)],
                                    ticktext=[eng(h * fu_) for h in range(H + 1)], range=[0, (H + 0.5) * fu_])
                    fg.update_yaxes(title_text="Amplitude (dBV, peak)" if dbs else "Amplitude (V, peak)", tickformat="" if dbs else ".3~s",
                                    range=[base, top + 5] if dbs else [0, top * 1.1])
                    st.plotly_chart(fg, use_container_width=True)
                    st.caption(" · ".join(f"{nm}: fundamental {eng(fu)}Hz in bin {m}, leakage into neighbouring bins {lk:.0f} dB" for nm, lk, fu, m in info))
                    if any(lk > -40 for _, lk, _, _ in info):
                        st.warning("The fundamental is not exactly in one bin (leakage above −40 dB): the frequency you entered probably differs from the real tone.")
                    tab = []
                    for h in range(1, 7):
                        r_ = {"Harmonic": h, "Frequency": eng(h * fu_)}
                        for nm in ("initial", "optimum"):
                            X_, m_, _u = sp[nm]; a_ = X_[h * m_] if h * m_ < len(X_) else np.nan
                            r_[f"{nm} (V)"] = eng(a_); r_[f"{nm} (dBc)"] = f"{20 * np.log10(a_ / X_[m_] + 1e-18):.1f}"
                        tab.append(r_)
                    st.dataframe(pd.DataFrame(tab), use_container_width=True, hide_index=True)
                except Exception as e: st.warning(f"Fourier plot failed for {t}: {e}")
        else: st.info("No transient plot: define a .tran analysis and enter at least one signal.")
        # 2) Bode
        a_ = Pl["ac"]
        if ("ac", a_) in D0_:
            fg = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Magnitude (dB)", "Phase (deg)"))
            for s_, nm in both("ac", a_):
                kw = dict(name=nm, mode="lines", legendgroup=nm, line=dict(color=C[nm]))
                fg.add_trace(go.Scatter(x=s_.x, y=_db(s_), **kw), 1, 1); fg.add_trace(go.Scatter(x=s_.x, y=_ph(s_), showlegend=False, **kw), 2, 1)
            fg.update_xaxes(type="log"); fg.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
            fg.update_yaxes(title_text="dB", row=1, col=1); fg.update_yaxes(title_text="deg", row=2, col=1); fg.update_layout(height=650)
            st.plotly_chart(grid(fg, f"AC Bode plot · ac.{a_}"), use_container_width=True)
        else: st.info("No Bode plot: define an .ac analysis and enter an AC signal.")
        # 3) noise
        if ("noise", "inoise") in D0_:
            fg = go.Figure()
            for s_, nm in both("noise", "inoise"): fg.add_trace(go.Scatter(x=s_.x, y=s_.y, name=nm, mode="lines", line=dict(color=C[nm])))
            fg.update_xaxes(type="log", title_text="Frequency (Hz)"); fg.update_yaxes(type="log", title_text="Input-referred noise (V/√Hz)")
            st.plotly_chart(grid(fg, "Noise analysis"), use_container_width=True)
        else: st.info("No noise plot: define a .noise analysis.")
        # 4) poles and zeros (rational fit of the AC response)
        if ("ac", a_) in D0_:
            zt = lambda v: np.sign(v) * np.log10(1 + np.abs(v)); fg = go.Figure()
            try:
                for s_, nm in both("ac", a_):
                    z, p_ = _fit(s_, Pl["order"], max(Pl["order"] - 1, 0))
                    for r, sym, lab in ((p_, "x", "pole"), (z, "circle-open", "zero")):
                        fg.add_trace(go.Scatter(x=zt(r.real), y=zt(r.imag), mode="markers", name=f"{nm} {lab}s",
                            marker=dict(symbol=sym, size=12, color=C[nm], line=dict(width=2)),
                            text=[f"{lab}: {eng(c.real)} {'+' if c.imag >= 0 else '-'} {eng(abs(c.imag))}j Hz" for c in r], hoverinfo="text+name"))
                fg.add_hline(y=0, line_color="gray"); fg.add_vline(x=0, line_color="gray")
                fg.update_xaxes(title_text="Real part (signed log10 of Hz)"); fg.update_yaxes(title_text="Imaginary part (signed log10 of Hz)")
                fg = grid(fg, "Poles and zeros (fit of the AC response)", hm="closest"); fg.update_xaxes(tickformat=""); fg.update_yaxes(tickformat="")
                st.plotly_chart(fg, use_container_width=True)
                st.caption("Estimate from a rational fit; axes are sign·log10(1+|f|) so decades fit on one plot, hover shows the true values.")
            except Exception as e: st.warning(f"Pole-zero fit failed: {e}")
    st.download_button("Download optimized netlist", "\n".join(render(L, P, R["vb"])) + "\n.end\n", "optimized.cir")

    # ═════════════ 7 · Robustness: Monte Carlo & PVT ═════════════
    import itertools
    from concurrent.futures import ThreadPoolExecutor, as_completed
    st.divider(); st.header("7 · Robustness: Monte Carlo & PVT")
    design = st.radio("Design to verify", ["Optimum", "Initial"], horizontal=True)
    base = {p["key"]: p["nom"] for p in P}
    if design == "Optimum": base.update(R["vb"])
    Mc, SC = R["M"], R["sc"]; cons = [m for m in Mc if m["Goal"] not in ("maximize", "minimize")]
    avail = {m["Name"]: m["Formula"] for m in Mc}
    for r in st.session_state.summ.dropna(subset=["Formula"]).itertuples(): avail.setdefault(r.Name if isinstance(r.Name, str) and r.Name else r.Formula, r.Formula)
    ok_of = lambda o, m: violation(o.get(m["Name"], np.nan), m["Goal"], m["Target"], m["Tol"], SC[m["Name"]]) == 0
    def margin(o, m):  # signed, normalised distance to the spec (negative = failing)
        v = o.get(m["Name"], np.nan)
        if not np.isfinite(v): return -np.inf
        d = v - m["Target"] if m["Goal"] in (">=", ">") else m["Target"] - v if m["Goal"] in ("<=", "<") else abs(m["Tol"]) - abs(v - m["Target"])
        return d / max(abs(m["Target"]) if m["Target"] else SC[m["Name"]], 1e-30)

    def batch(jobs, forms, label):  # runs ngspice in parallel threads; Streamlit calls stay in the main thread
        nd_ = needs(list(forms.values()))
        def one(Ls):
            try:
                D = simulate(Ls, AN, nd_); out = {}
                for k, f in forms.items():
                    try: out[k] = calc(D, f)
                    except Exception: out[k] = np.nan
                return out
            except Exception as e: return {"_error": str(e)[:200]}
        res_, bar_, txt, t0 = [None] * len(jobs), st.progress(0.0), st.empty(), time.time()
        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
            fut = {ex.submit(one, j): i for i, j in enumerate(jobs)}
            for k, f in enumerate(as_completed(fut), 1):
                res_[fut[f]] = f.result(); el = time.time() - t0
                bar_.progress(k / len(jobs)); txt.info(f"{label}: {k}/{len(jobs)} simulations · elapsed {el:.0f}s · ETA ≈ {el / k * (len(jobs) - k):.0f}s")
        txt.empty(); return res_

    # ───────── Monte Carlo ─────────
    st.subheader("Monte Carlo")
    mdf = pd.DataFrame([{"Select": False, "Parameter": p["key"], "Value": eng(base[p["key"]]), "Error %": 1.0} for p in P])
    mced = st.data_editor(mdf, hide_index=True, use_container_width=True, disabled=["Parameter", "Value"], key="mctab",
                          column_config={"Error %": st.column_config.NumberColumn(min_value=0.0, format="%.3g", help="± tolerance in % (3σ for Gaussian)")})
    m1, m2, m3, m4 = st.columns(4)
    nrun = int(m1.number_input("Number of runs", 5, 100000, 100)); seed = int(m2.number_input("Random seed", 0, 2**31 - 1, 42))
    dist = m3.selectbox("Distribution", ["Gaussian (tolerance = 3σ)", "Uniform (± tolerance)"])
    mets = m4.multiselect("Metrics to measure", list(avail), default=list(avail)[:1])
    if st.button("Run Monte Carlo", type="primary", key="runmc"):
        sel = mced[mced.Select]
        if sel.empty or not mets: st.error("Select at least one parameter and one metric.")
        else:
            rng = np.random.default_rng(seed); forms = {m["Name"]: m["Formula"] for m in cons}; forms.update({k: avail[k] for k in mets})
            jobs, smp = [], []
            for _ in range(nrun):
                v, d = dict(base), {}
                for _, r in sel.iterrows():
                    tol = float(r["Error %"]) / 100; e = rng.normal(0, tol / 3) if dist.startswith("G") else rng.uniform(-tol, tol)
                    v[r["Parameter"]] = base[r["Parameter"]] * (1 + e); d["p:" + r["Parameter"]] = v[r["Parameter"]]
                jobs.append(render(L, P, v)); smp.append(d)
            outs = batch(jobs, forms, "Monte Carlo")
            df = pd.DataFrame([{**d, **{k: o.get(k, np.nan) for k in forms}, "error": o.get("_error", ""),
                                "pass": bool(cons) and "_error" not in o and all(ok_of(o, m) for m in cons)} for d, o in zip(smp, outs)])
            st.session_state.mc = dict(df=df, mets=mets)
    MC = st.session_state.get("mc")
    if MC:
        df = MC["df"]; bad = df[df.error != ""]
        if len(bad): st.warning(f"{len(bad)} of {len(df)} runs failed, e.g.: {bad.error.iloc[0]}")
        if cons:
            k, n_ = int(df["pass"].sum()), len(df); ph = k / n_; z = 1.96
            c_ = (ph + z * z / (2 * n_)) / (1 + z * z / n_); w_ = z * np.sqrt(ph * (1 - ph) / n_ + z * z / (4 * n_ * n_)) / (1 + z * z / n_)
            (st.success if ph >= .95 else st.warning)(f"Yield: {ph:.1%} (95% CI {c_ - w_:.1%} – {c_ + w_:.1%}) · {k}/{n_} runs meet every constraint")
        else: st.info("The metrics table has no constraints, so no yield is computed.")
        stats = []
        for mname in MC["mets"]:
            x = df[mname].astype(float).dropna().values
            if not len(x): continue
            mu, sd = x.mean(), (x.std(ddof=1) if len(x) > 1 else 0.0)
            stats.append({"Metric": mname, "Mean": eng(mu), "σ": eng(sd), "Min": eng(x.min()), "Max": eng(x.max())})
            fg = go.Figure(go.Histogram(x=x, nbinsx=max(10, int(np.sqrt(len(x)))), marker_color="#1f5fbf"))
            for xv, lab, col in ((mu, "mean", "black"), (mu - 3 * sd, "−3σ", "gray"), (mu + 3 * sd, "+3σ", "gray")):
                fg.add_vline(x=xv, line_dash="dash", line_color=col, annotation_text=lab)
            for m in cons:
                if m["Name"] == mname: fg.add_vline(x=m["Target"], line_color="#e63b2e", line_width=2, annotation_text="spec")
            fg.update_xaxes(title_text=mname); fg.update_yaxes(title_text="Number of samples"); fg.update_layout(bargap=.05, showlegend=False)
            st.plotly_chart(grid(fg, f"Monte Carlo · {mname}", hm="closest"), use_container_width=True)
        if stats: st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
        st.download_button("Download all Monte Carlo runs (CSV)", df.to_csv(index=False), "montecarlo_runs.csv")

    # ───────── PVT ─────────
    st.subheader("PVT corners")
    MODEL = re.compile(r"^(\s*\.model\s+\S+\s+)(nmos|pmos)\b(.*)$", re.I)
    def apply_proc(lines, sn, sp, dv, dk):  # shift VTO/VTH0 (volts) and KP/U0 (percent) in .model cards; s=+1 fast, -1 slow
        out = []
        for l in lines:
            m = MODEL.match(l); isn = bool(m) and m[2].lower() == "nmos"
            if m and (sn if isn else sp):
                sg = sn if isn else sp
                def f(mm):
                    v = num(mm[3])
                    if v is None: return mm[0]
                    v = (v - sg * dv if isn else v + sg * dv) if mm[1].lower() in ("vto", "vth0") else v * (1 + sg * dk / 100)
                    return f"{mm[1]}{mm[2]}{v:.6g}"
                l = m[1] + m[2] + re.sub(r"(?i)\b(vto|vth0|kp|u0)(\s*=\s*)([^\s)]+)", f, m[3])
            out.append(l)
        return out
    pc1, pc2, pc3 = st.columns(3)
    procs = pc1.multiselect("Process corners", ["TT", "FF", "SS", "SF", "FS"], ["TT", "FF", "SS"])
    dvs = pc2.text_input("Threshold shift ΔVth (V)", "50m"); dk = pc3.number_input("Mobility shift ΔKP (%)", 0.0, 100.0, 10.0)
    st.caption("FF/SS shift every NMOS and PMOS `.model` (VTO or VTH0, and KP or U0) by the amounts above (faster = lower |Vth|, higher KP); SF/FS shift NMOS and PMOS in opposite directions. TT is unchanged.")
    temps = st.text_input("Temperatures (°C, comma-separated)", "-40, 27, 125")
    vs = [p_ for p_ in P if p_["kind"] == "V"]
    rdf = pd.DataFrame([{"Select": False, "Source": p_["key"], "Min": eng(base[p_["key"]] * .9), "Nom": eng(base[p_["key"]]), "Max": eng(base[p_["key"]] * 1.1)} for p_ in vs])
    st.caption("Supply rails: select the voltage sources to vary.")
    red = st.data_editor(rdf, hide_index=True, use_container_width=True, disabled=["Source"], key="railtab") if len(rdf) else rdf
    v1, v2 = st.columns(2)
    levels = v1.multiselect("Rail levels", ["min", "nom", "max"], ["min", "nom", "max"]) or ["nom"]
    together = v2.checkbox("All selected rails move together (no cartesian product)", True)
    if st.button("Run PVT", type="primary", key="runpvt"):
        try:
            dv = val(dvs); tmp = [float(t) for t in temps.split(",") if t.strip()] or [27.0]
            rails = [(r["Source"], val(r["Min"]), val(r["Nom"]), val(r["Max"])) for _, r in red[red.Select].iterrows()] if len(red) else []
        except ValueError as e: st.error(f"Invalid input: {e}")
        else:
            PROC = {"TT": (0, 0), "FF": (1, 1), "SS": (-1, -1), "SF": (-1, 1), "FS": (1, -1)}; li = {"min": 1, "nom": 2, "max": 3}
            if not rails: rsets = [("nom", {})]
            elif together: rsets = [(l, {r[0]: r[li[l]] for r in rails}) for l in levels]
            else: rsets = [("/".join(c), {r[0]: r[li[l]] for r, l in zip(rails, c)}) for c in itertools.product(levels, repeat=len(rails))]
            if any(x != "TT" for x in procs) and not any(MODEL.match(l) for l in L):
                st.warning("No `.model nmos/pmos` cards in the netlist: process corners will have no effect.")
            labels, jobs = [], []
            for pr in procs or ["TT"]:
                for rl, rv in rsets:
                    for T in tmp:
                        labels.append(f"{pr} | {rl} | {T:g}°C")
                        jobs.append(apply_proc(render(L, P, {**base, **rv}), *PROC[pr], dv, dk) + [f".temp {T:g}"])
            forms = {m["Name"]: m["Formula"] for m in Mc}; outs = batch(jobs, forms, "PVT")
            okm = {m["Name"]: [ok_of(o, m) for o in outs] for m in cons}
            disp = pd.DataFrame({"Corner": labels, **{m["Name"]: [eng(o.get(m["Name"], np.nan)) for o in outs] for m in Mc}})
            worst = []
            for m in cons:
                mg = [margin(o, m) for o in outs]; w_ = int(np.argmin(mg))
                worst.append({"Spec": m["Name"], "Worst corner": labels[w_], "Value": eng(outs[w_].get(m["Name"], np.nan)),
                              "Margin": "fail (no value)" if not np.isfinite(mg[w_]) else f"{mg[w_]:+.1%}", "Passes all corners": "✅" if all(okm[m["Name"]]) else "❌"})
            allok = [all(okm[n][i] for n in okm) for i in range(len(outs))]
            st.session_state.pvt = dict(disp=disp, okm=okm, worst=pd.DataFrame(worst), npass=sum(allok), n=len(outs),
                                        raw=pd.DataFrame([{"Corner": lb, **o} for lb, o in zip(labels, outs)]),
                                        err=next((o["_error"] for o in outs if "_error" in o), ""))
    PV = st.session_state.get("pvt")
    if PV:
        if PV["err"]: st.warning(f"Some corners failed to simulate, e.g.: {PV['err']}")
        (st.success if PV["npass"] == PV["n"] else st.warning)(f"{PV['npass']}/{PV['n']} corners meet every constraint")
        def paint(col):
            if col.name not in PV["okm"]: return [""] * len(col)
            return ["background-color:#c9f2c7;color:#111" if o else "background-color:#f5c2c0;color:#111" for o in PV["okm"][col.name]]
        st.dataframe(PV["disp"].style.apply(paint, axis=0), use_container_width=True, hide_index=True)
        if len(PV["worst"]): st.markdown("**Worst-case corner per spec**"); st.dataframe(PV["worst"], use_container_width=True, hide_index=True)
        st.download_button("Download all PVT runs (CSV)", PV["raw"].to_csv(index=False), "pvt_runs.csv")