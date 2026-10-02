# Cautare de strategii + palnie de validare. Principii:
#  * cautarea vede DOAR segmentul "train"; "validare" se foloseste o singura data pe finalisti (cu corectie pentru numarul lor);
#  * "lockbox" (ultima parte a istoricului) se atinge o singura data, la final, pe supravietuitori;
#  * costurile sunt parametri; implicit sunt PROVIZORII pana sunt masurate pe contul real.
import math, time
import numpy as np
from . import blocks as B, feat, data as D
from .bt import run_bt, stats
from . import ftmo


def zcrit(p):
    """z unilateral pentru probabilitatea p (aproximare Acklam inversa a normalei)."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - p)


def tstat(r):
    n = len(r)
    if n < 3: return 0.0
    s = np.std(r, ddof=1)
    return float(np.mean(r) / (s / math.sqrt(n))) if s > 0 else 0.0


class Ctx:
    """Date pregatite pentru un simbol: bare de executie + trend HTF (doar din bare HTF inchise)."""
    def __init__(self, m1, exec_tf="5m", cost_price=0.0, slip_price=0.0, split=(0.5, 0.25, 0.25), day_shift=7200):
        sec = D.TFSEC[exec_tf]; self.sec = sec
        x = D.resample(m1, sec); self.x = x
        self.t, self.o, self.h, self.l, self.c, self.v = x["t"], x["o"], x["h"], x["l"], x["c"], x["v"]
        self.a = feat.atr(self.h, self.l, self.c, 14)
        self.n = len(self.t)
        self.cost = np.full(self.n, float(cost_price)); self.slip = float(slip_price)
        self.day = ((self.t + day_shift) // 86400).astype(np.int64)
        i1 = int(self.n * split[0]); i2 = int(self.n * (split[0] + split[1]))
        self.seg = {"train": (0, i1), "val": (i1, i2), "lock": (i2, self.n)}
        self.trend = {}
        for name, tf in (("h1", "1h"), ("h4", "4h")):
            hs = D.resample(m1, D.TFSEC[tf]); idx = D.htf_index(self.t, sec, hs["t"], D.TFSEC[tf])
            for ln in (20, 50):
                e = feat.ema(hs["c"], ln); ev = D.map_htf(e, idx); cv = D.map_htf(hs["c"], idx)
                tr = np.zeros(self.n, np.int8); ok = ~np.isnan(ev)
                tr[ok] = np.where(cv[ok] > ev[ok], 1, -1)
                self.trend[(name, ln)] = tr
        self.cache = {}

    def block_signal(self, blk, p):
        key = (blk, tuple(sorted(p.items())))
        if key in self.cache: return self.cache[key]
        o, h, l, c, a = self.o, self.h, self.l, self.c, self.a
        if blk == "sweep_mss": r = B.sig_sweep_mss(o, h, l, c, a, p["n_piv"], p["mss_wait"], p["buf"], p["min_sweep"])
        elif blk == "ob": r = B.sig_ob(o, h, l, c, a, p["n_piv"], p["disp"], p["lookback"], p["max_age"], p["reject"], p["buf"], p["full"])
        elif blk == "fvg": r = B.sig_fvg_retest(o, h, l, c, a, p["gap"], p["max_age"], p["reject"], p["buf"], p["n_piv"], p["sweep"], p["sweep_win"])
        elif blk == "breakout": r = B.sig_breakout(h, l, c, a, p["n"])
        elif blk == "pullback":
            r = B.sig_pullback(c, a, feat.ema(c, p["fast"]), feat.ema(c, p["slow"]), feat.rsi(c, p["rsi_n"]), p["lvl"])
        else: raise ValueError(blk)
        if len(self.cache) > 400: self.cache.clear()
        self.cache[key] = r; return r


# ---- spatiul de cautare (listele sunt ordonate, vecinii = indecsi adiacenti) ----
SPACE = {
    "sweep_mss": {"n_piv": [2, 3, 4, 6, 8], "mss_wait": [3, 6, 10, 16], "buf": [0.0, 0.1, 0.3], "min_sweep": [0.0, 0.1, 0.3]},
    "ob": {"n_piv": [2, 3, 5], "disp": [0.8, 1.2, 1.6, 2.2], "lookback": [4, 8, 14], "max_age": [20, 60, 150], "reject": [0, 1], "buf": [0.0, 0.1, 0.3], "full": [0, 1]},
    "fvg": {"gap": [0.1, 0.2, 0.4, 0.7], "max_age": [10, 30, 80], "reject": [0, 1], "buf": [0.0, 0.1, 0.3], "n_piv": [2, 3, 5], "sweep": [0, 1], "sweep_win": [6, 12, 24]},
    "breakout": {"n": [10, 20, 40, 80, 160]},
    "pullback": {"fast": [8, 13, 21], "slow": [34, 55, 89], "rsi_n": [7, 14], "lvl": [30, 40, 45]},
}
COMMON = {"sl_mult": [0.7, 1.0, 1.5, 2.0], "tp_r": [1.0, 1.5, 2.0, 3.0], "hold": [24, 72, 200],
          "htf": ["none", "h1_20", "h1_50", "h4_20", "h4_50"], "hours": [(0, 24), (7, 11), (7, 16), (13, 17), (0, 7), (13, 21)]}


def random_genome(rng, blocks=None):
    blk = rng.choice(blocks or list(SPACE))
    g = {"blk": blk, "p": {k: v[rng.integers(len(v))] for k, v in SPACE[blk].items()}}
    for k, v in COMMON.items(): g[k] = v[rng.integers(len(v))]
    return g


def _norm(g):
    g = {**g, "blk": str(g["blk"]), "htf": str(g["htf"]), "hours": [int(x) for x in g["hours"]], "sl_mult": float(g["sl_mult"]), "tp_r": float(g["tp_r"]), "hold": int(g["hold"]), "p": {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in g["p"].items()}}
    return g


def key_of(g): return (g["blk"], tuple(sorted(g["p"].items())), g["sl_mult"], g["tp_r"], g["hold"], g["htf"], tuple(g["hours"]))


def neighbours(g):
    """Toate variantele cu UN parametru mutat cu un pas."""
    out = []
    spaces = [("p", k, SPACE[g["blk"]][k]) for k in g["p"]] + [("c", k, COMMON[k]) for k in ("sl_mult", "tp_r", "hold")]
    for kind, k, vals in spaces:
        cur = g["p"][k] if kind == "p" else g[k]; i = vals.index(cur)
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                g2 = {**g, "p": dict(g["p"])}
                if kind == "p": g2["p"][k] = vals[j]
                else: g2[k] = vals[j]
                out.append(g2)
    return out


def mutate(g, rng, k=1):
    g2 = {**g, "p": dict(g["p"])}
    spaces = [("p", n, SPACE[g["blk"]][n]) for n in g["p"]] + [("c", n, COMMON[n]) for n in COMMON]
    for _ in range(k):
        kind, n, vals = spaces[rng.integers(len(spaces))]
        if kind == "p": g2["p"][n] = vals[rng.integers(len(vals))]
        else: g2[n] = vals[rng.integers(len(vals))]
    return g2


def trades(ctx, g, seg, cost_mult=1.0, extra_cost=0.0):
    """Tranzactiile genomului pe un segment. Intoarce (r, ziua intrarii, indexul intrarii)."""
    sig, sd = ctx.block_signal(g["blk"], g["p"])
    if g["htf"] != "none":
        nm, ln = g["htf"].split("_"); tr = ctx.trend[(nm, int(ln))]
        sig = np.where(sig == tr, sig, 0).astype(np.int8)
    h0, h1 = g["hours"]
    if (h0, h1) != (0, 24):
        close_t = ctx.t + ctx.sec
        sig = np.where(B.hour_mask(close_t, h0, h1), sig, 0).astype(np.int8)
    a, b = ctx.seg[seg] if isinstance(seg, str) else seg
    cost = ctx.cost * cost_mult + extra_cost
    ent, ext, dirs, rs, why = run_bt(ctx.o, ctx.h, ctx.l, ctx.c, sig, sd * g["sl_mult"], g["tp_r"], g["hold"], cost, ctx.slip * cost_mult, a, b)
    return rs, ctx.day[ent], ent


def fitness(r, min_n):
    if len(r) < min_n: return -9.0
    m = float(np.mean(r))
    if m <= 0: return m
    return tstat(r)


def search(ctx, budget_s, seed=1, min_n=80, pop=300, progress=None, blocks=None):
    """Cautare pe train: esantionare aleatoare, apoi rafinare locala a celor mai buni. Intoarce (lista sortata, nr. de variante incercate)."""
    rng = np.random.default_rng(seed); t0 = time.time(); seen = {}; best = []
    def ev(g):
        k = key_of(g)
        if k in seen: return
        r, _, _ = trades(ctx, g, "train"); seen[k] = (fitness(r, min_n), len(r), float(np.mean(r)) if len(r) else 0.0, g)
    phase = 0
    while time.time() - t0 < budget_s:
        if phase == 0 or len(seen) < pop:
            for _ in range(50): ev(random_genome(rng, blocks))
        else:
            top = sorted(seen.values(), key=lambda x: -x[0])[:max(10, pop // 10)]
            for _ in range(50):
                ev(mutate(top[rng.integers(len(top))][3], rng, 1 + int(rng.integers(2))))
        phase += 1
        if progress and phase % 5 == 0: progress(len(seen), time.time() - t0, max((v[0] for v in seen.values()), default=0))
    res = sorted(seen.values(), key=lambda x: -x[0])
    return res, len(seen)


# ---------------- palnia de validare ----------------
def validate(ctx, cands, n_trials, k_final=25, alpha=0.05, risk_pct=0.5, use_lock=True, log=lambda *a: None):
    """cands: rezultate de la search (fitness, n, avg, genom), cele mai bune primele. Fiecare etapa elimina; se pastreaza motivul."""
    out = []; finals = []; seen_p = set()
    for c_ in cands:                                   # diversitate: cel mult o varianta pe (bloc, parametri de semnal)
        kp = (c_[3]["blk"], tuple(sorted(c_[3]["p"].items())))
        if kp in seen_p or c_[0] <= 0: continue
        seen_p.add(kp); finals.append(c_)
        if len(finals) >= k_final: break
    K = len(finals)
    zval = zcrit(alpha / max(K, 1))
    for fit, n_tr, avg_tr, g in finals:
        rec = {"genome": _norm(g), "train": {"n": n_tr, "avg_r": avg_tr, "t": fit}, "stages": {}}
        out.append(rec)
        # 1) validare out-of-sample (o singura privire; prag corectat pentru K finalisti)
        rv, _, _ = trades(ctx, g, "val"); tv = tstat(rv)
        rec["val"] = {"n": len(rv), "avg_r": float(np.mean(rv)) if len(rv) else 0.0, "t": tv}
        if len(rv) < 30 or np.mean(rv) <= 0 or tv < zval:
            rec["fail"] = "validare: t=%.2f (n=%d) < prag %.2f" % (tv, len(rv), zval); continue
        rec["stages"]["val"] = True
        # 2) robustete la vecini (train+val)
        def tv_avg(gg, cm=1.0, ex=0.0):
            a, _, _ = trades(ctx, gg, "train", cm, ex); b, _, _ = trades(ctx, gg, "val", cm, ex)
            r = np.concatenate([a, b]); return (float(np.mean(r)) if len(r) else -1.0), len(r)
        base, nb = tv_avg(g); nbs = [tv_avg(x)[0] for x in neighbours(g)]
        pos = float(np.mean([v > 0 for v in nbs])) if nbs else 0.0
        rec["robust"] = {"neighbours": len(nbs), "pos_share": pos, "median_avg_r": float(np.median(nbs)) if nbs else 0.0}
        if pos < 0.7 or np.median(nbs) < 0.4 * base:
            rec["fail"] = "robustete: doar %.0f%% din variantele vecine pozitive" % (pos * 100); continue
        rec["stages"]["robust"] = True
        # 3) stres de costuri (x2 costuri)
        s2, _ = tv_avg(g, 2.0); rec["cost_x2_avg_r"] = s2
        if s2 <= 0: rec["fail"] = "costuri x2: avg R=%.3f <= 0" % s2; continue
        rec["stages"]["cost"] = True
        # 4) stabilitate in timp: 4 blocuri consecutive pe train+val
        a, da, ea = trades(ctx, g, "train"); b, db, eb = trades(ctx, g, "val")
        r = np.concatenate([a, b]); day = np.concatenate([da, db]); chunks = np.array_split(r, 4)
        posc = sum(1 for c in chunks if len(c) and np.mean(c) > 0); rec["blocks_pos"] = posc
        if posc < 3: rec["fail"] = "stabilitate: doar %d/4 perioade pozitive" % posc; continue
        rec["stages"]["time"] = True
        # 5) FTMO: trecere in ferestre mobile + bootstrap pe zile
        ev = ftmo.evaluate(r, day, risk_pct=risk_pct, stride=1)
        bs = ftmo.bootstrap(r, day, risk_pct=risk_pct, n_days=120, reps=400)
        rec["ftmo"] = {"rolling_pass_both": ev["pass_both"], "rolling_fail_daily": ev["fail_daily"], "rolling_fail_total": ev["fail_total"],
                       "bootstrap_pass_both": bs["pass_both"], "starts": ev["starts"]}
        rec["stages"]["ftmo"] = True
        out_n = len([x for x in out if x["stages"].get("ftmo")])
    # 6) lockbox, o singura data, doar pe supravietuitori (corectie pentru numarul lor)
    surv = [x for x in out if x["stages"].get("ftmo")]
    if use_lock and surv:
        zl = zcrit(alpha / len(surv))
        for x in surv:
            rl, dl, _ = trades(ctx, x["genome"], "lock"); tl = tstat(rl)
            x["lock"] = {"n": len(rl), "avg_r": float(np.mean(rl)) if len(rl) else 0.0, "t": tl, "thr": zl}
            x["stages"]["lock"] = bool(len(rl) >= 20 and np.mean(rl) > 0 and tl >= zl)
            if not x["stages"]["lock"]: x["fail"] = "lockbox: t=%.2f (n=%d) < prag %.2f" % (tl, len(rl), zl)
    return out
