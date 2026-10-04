# Contextul, spatiul de cautare si palnia de validare pentru strategia LIQ (DAX M1). Vezi lab/liq.py pentru simulator.
import math, time
import numpy as np
from . import liq as Q, ftmo
from .search import zcrit, tstat

SPACE = {
    "model": [0, 1], "nf": [1, 2, 3, 5], "leg_n": [3, 5, 8, 12, 20], "pre": [0, 5, 10, 20],
    "disp_mode": [0, 1, 2], "disp_mult": [0.5, 0.75, 1.0, 1.5, 2.0], "disp_n": [5, 10, 20], "disp_k": [2, 3, 4], "disp_retr": [0, 1],
    "sl_mode": [0, 1], "buf": [0.0, 0.01, 0.05, 0.5, 1.0, 2.0], "fvg_pre": [0, 3, 10], "fvg_min": [0.0, 0.5, 1.0, 2.0, 4.0],
    "win": [15, 30, 45, 60], "ord_mode": [0, 1, 2], "start": [600, 630, 660], "end": [960, 1020, 1080], "max_tr": [1, 2], "stop_win": [1, 0],
    "tp_r": [1.5, 2.0, 2.5, 3.0], "be_r": [0.0, 1.0, 1.5, 2.0], "src_mask": [7, 1, 2, 4, 3, 5, 6], "dir": [3, 1, 2], "anchor": [0, 1],
    "max_fvg": [2, 3, 5], "sel": [0, 1, 2],
}
# varianta exact din specificatie (parametrii nefixati de specificatie: valori implicite rezonabile)
BASE = {"model": 0, "nf": 2, "leg_n": 8, "pre": 10, "disp_mode": 0, "disp_mult": 1.5, "disp_n": 10, "disp_k": 3, "disp_retr": 1, "sl_mode": 0, "buf": 0.0,
        "fvg_pre": 0, "fvg_min": 0.0, "win": 45, "ord_mode": 0, "start": 600, "end": 1080, "max_tr": 2, "stop_win": 1, "tp_r": 2.0, "be_r": 1.5,
        "src_mask": 7, "dir": 3, "anchor": 0, "max_fvg": 3, "sel": 0}
# parametri care conteaza doar pentru anumite alegeri (restul se ignora la cheie si la mutatii utile)
def relevant(g):
    r = dict(g)
    if g["model"] == 0: r["leg_n"] = 0
    else: r["nf"] = 0; r["pre"] = 0
    if g["disp_mode"] == 0: r["disp_mult"] = 0; r["disp_n"] = 0; r["disp_k"] = 0; r["disp_retr"] = 0
    elif g["disp_mode"] == 1: r["disp_k"] = 0; r["disp_retr"] = 0
    return r


def key_of(g): return tuple(sorted(relevant(g).items()))


def _norm(g): return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in g.items()}


def random_genome(rng): return {k: v[rng.integers(len(v))] for k, v in SPACE.items()}


def mutate(g, rng, k=1):
    g2 = dict(g); keys = list(SPACE)
    for _ in range(k):
        n = keys[rng.integers(len(keys))]; g2[n] = SPACE[n][rng.integers(len(SPACE[n]))]
    return g2


def neighbours(g):
    out = []
    for k, vals in SPACE.items():
        if k in ("model", "dir", "src_mask", "sel", "ord_mode"): continue
        i = vals.index(g[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals): out.append({**g, k: vals[j]})
    return out


class LCtx:
    def __init__(self, m1, tick=0.01, cost_pts=1.5, split=(0.5, 0.25, 0.25), spread_by_hour=None, spread_mult=1.5, slip=0.5, comm=0.0):
        self.tick = tick
        t = np.ascontiguousarray(m1["t"], np.int64); self.t = t
        self.o, self.h, self.l, self.c = [np.ascontiguousarray(m1[k], np.float64) for k in ("o", "h", "l", "c")]
        self.mstep = max(1, int(round(float(np.median(np.diff(t[:200000]))) / 60.0))) if len(t) > 2 else 1   # minute pe bara (M1=1, M5=5)
        off = Q.ro_offset(t); loc = t + off
        self.lday = (loc // 86400).astype(np.int64); self.lmin = ((loc % 86400) // 60).astype(np.int64)
        hr = ((t // 3600) % 24).astype(int)
        if spread_by_hour is not None: cost = spread_mult * np.asarray(spread_by_hour, np.float64)[hr] + comm + slip
        else: cost = np.full(len(t), cost_pts + comm + slip)
        self.cost = cost
        self.rows = Q.build_levels(t, self.h, self.l, self.lmin, self.lday, tick, 600, 1080, 3_000_000, self.mstep)
        self.sw = {}
        n = len(t); i1 = int(n * split[0]); i2 = int(n * (split[0] + split[1]))
        self.cut = (int(t[i1]), int(t[i2]))
        self.cache = {}

    def swings(self, nf):
        if nf not in self.sw: self.sw[nf] = Q.swings(self.h, self.l, int(nf))
        return self.sw[nf]

    def params(self, g):
        P = np.zeros(Q.NP)
        P[Q.P_MODEL] = g["model"]; P[Q.P_NF] = max(g["nf"], 1); P[Q.P_LEGN] = g["leg_n"]; P[Q.P_PRE] = g["pre"]; P[Q.P_DMODE] = g["disp_mode"]
        P[Q.P_DMULT] = g["disp_mult"]; P[Q.P_DN] = max(g["disp_n"], 1); P[Q.P_DK] = max(g["disp_k"], 1); P[Q.P_DRETR] = g["disp_retr"]
        P[Q.P_SLMODE] = g["sl_mode"]; P[Q.P_BUF] = g["buf"]; P[Q.P_FPRE] = g["fvg_pre"]; P[Q.P_FMIN] = g["fvg_min"]; P[Q.P_WIN] = g["win"]
        P[Q.P_ORD] = g["ord_mode"]; P[Q.P_START] = g["start"]; P[Q.P_END] = g["end"]; P[Q.P_MAXTR] = g["max_tr"]; P[Q.P_STOPWIN] = g["stop_win"]
        P[Q.P_TPR] = g["tp_r"]; P[Q.P_BER] = g["be_r"]; P[Q.P_SRC] = g["src_mask"]; P[Q.P_DIR] = g["dir"]; P[Q.P_ANCH] = g["anchor"]
        P[Q.P_MAXSL] = 6000 * self.tick; P[Q.P_TICK] = self.tick; P[Q.P_MAXF] = g["max_fvg"]; P[Q.P_SEL] = g["sel"]; P[Q.P_MINSL] = 0.0
        return P

    def run(self, g):
        k = key_of(g)
        if k in self.cache: return self.cache[k]
        nf = int(g["nf"]) if g["model"] == 0 else 1
        sh, sl = self.swings(max(nf, 1))
        out = np.zeros((6000, Q.NCOL)); cnt = np.zeros(8, np.int64)
        rb, rd, rp, rs, rt = self.rows
        nt = Q.simulate(self.t, self.o, self.h, self.l, self.c, self.lmin, self.lday, self.cost, rb, rd, rp, rs, sh, sl, self.params(g), out, cnt)
        res = (out[:nt].copy(), cnt)
        if len(self.cache) > 300: self.cache.clear()
        self.cache[k] = res; return res

    def seg_mask(self, tr, seg):
        te = tr[:, 0]
        if seg == "train": return te < self.cut[0]
        if seg == "val": return (te >= self.cut[0]) & (te < self.cut[1])
        if seg == "lock": return te >= self.cut[1]
        return np.ones(len(te), bool)


def trades(ctx, g, seg, cost_mult=1.0, extra=0.0):
    tr, _ = ctx.run(g); tr = tr[ctx.seg_mask(tr, seg)]
    if len(tr) == 0: return np.zeros(0), np.zeros(0, np.int64), tr
    r = tr[:, 4] - (cost_mult * tr[:, 20] + extra) / tr[:, 5]
    return r, tr[:, 18].astype(np.int64), tr


def fitness(r, min_n):
    if len(r) < min_n: return -9.0
    return float(np.mean(r)) if np.mean(r) <= 0 else tstat(r)


def search(ctx, budget_s, seed=1, min_n=80, pop=200, progress=None):
    rng = np.random.default_rng(seed); t0 = time.time(); seen = {}; top = []; phase = 0
    def ev(g):
        k = key_of(g)
        if k in seen: return
        r, _, _ = trades(ctx, g, "train"); seen[k] = (fitness(r, min_n), len(r), float(np.mean(r)) if len(r) else 0.0, _norm(g))
    def fresh(top):
        for _ in range(30):
            g = mutate(top[rng.integers(len(top))][3], rng, 1 + int(rng.integers(3)))
            if key_of(g) not in seen: return g
        return random_genome(rng)
    ev(BASE); ev({**BASE, "model": 1})
    while time.time() - t0 < budget_s:
        if phase % 20 == 0 and len(seen) >= pop: top = sorted(seen.values(), key=lambda x: -x[0])[:max(30, len(seen) // 50)]
        for _ in range(20):
            if not top or rng.random() < 0.3: ev(random_genome(rng))
            else: ev(fresh(top))
        phase += 1
        if progress and phase % 3 == 0: progress(len(seen), time.time() - t0, max((v[0] for v in seen.values()), default=0), seen)
    return sorted(seen.values(), key=lambda x: -x[0]), len(seen)


def validate(ctx, cands, k_final=100, alpha=0.05, use_lock=True, prior_k=0, prior_lock=0, cb=None, pass_min=0.70):
    out = []; finals = []; seen_p = set()
    for c_ in cands:
        if c_[0] <= 0: continue
        kp = key_of(c_[3])
        if kp in seen_p: continue
        seen_p.add(kp); finals.append(c_)
        if len(finals) >= k_final: break
    K = len(finals); zval = zcrit(alpha / max(K + prior_k, 1))
    for fit, n_tr, avg_tr, g in finals:
        if cb: cb("finalisti", len(out), K, "liq")
        rec = {"genome": _norm(g), "train": {"n": n_tr, "avg_r": avg_tr, "t": fit}, "stages": {}}; out.append(rec)
        rv, _, _ = trades(ctx, g, "val"); tv = tstat(rv)
        rec["val"] = {"n": len(rv), "avg_r": float(np.mean(rv)) if len(rv) else 0.0, "t": tv}
        if len(rv) < 30 or np.mean(rv) <= 0 or tv < zval:
            rec["fail"] = "validare: t=%.2f (n=%d) < prag %.2f" % (tv, len(rv), zval); continue
        rec["stages"]["val"] = True
        def tv_avg(gg, cm=1.0):
            a, _, _ = trades(ctx, gg, "train", cm); b, _, _ = trades(ctx, gg, "val", cm); r = np.concatenate([a, b]); return (float(np.mean(r)) if len(r) else -1.0)
        base = tv_avg(g); nbs = [tv_avg(x) for x in neighbours(g)]
        pos = float(np.mean([v > 0 for v in nbs])) if nbs else 0.0
        rec["robust"] = {"neighbours": len(nbs), "pos_share": pos, "median_avg_r": float(np.median(nbs)) if nbs else 0.0}
        if pos < 0.7 or np.median(nbs) < 0.4 * base: rec["fail"] = "robustete: doar %.0f%% din variantele vecine pozitive" % (pos * 100); continue
        rec["stages"]["robust"] = True
        s2 = tv_avg(g, 2.0); rec["cost_x2_avg_r"] = s2
        if s2 <= 0: rec["fail"] = "costuri x2: avg R=%.3f <= 0" % s2; continue
        rec["stages"]["cost"] = True
        a, da, _ = trades(ctx, g, "train"); b, db, _ = trades(ctx, g, "val")
        r = np.concatenate([a, b]); day = np.concatenate([da, db]); chunks = np.array_split(r, 4)
        posc = sum(1 for c in chunks if len(c) and np.mean(c) > 0); rec["blocks_pos"] = posc
        if posc < 3: rec["fail"] = "stabilitate: doar %d/4 perioade pozitive" % posc; continue
        rec["stages"]["time"] = True
        best = None
        for rk in (0.25, 0.5, 0.75, 1.0):
            ev = ftmo.evaluate(r, day, risk_pct=rk, stride=1); sc = ev["pass_both"] - ev["fail_daily"] - ev["fail_total"] * 0.5
            if best is None or sc > best[0]: best = (sc, rk, ev)
        rk, ev = best[1], best[2]; bs = ftmo.bootstrap(r, day, risk_pct=rk, n_days=120, reps=400)
        rec["ftmo"] = {"risk_pct": rk, "rolling_pass_both": ev["pass_both"], "rolling_fail_daily": ev["fail_daily"], "rolling_fail_total": ev["fail_total"], "bootstrap_pass_both": bs["pass_both"], "starts": ev["starts"]}
        rec["stages"]["ftmo"] = True
    surv = [x for x in out if x["stages"].get("ftmo")]
    if cb: cb("lockbox", len(surv), len(surv), "")
    if use_lock and surv:
        zl = zcrit(alpha / (len(surv) + prior_lock))
        for x in surv:
            rl, dl, _ = trades(ctx, x["genome"], "lock"); tl = tstat(rl)
            x["lock"] = {"n": len(rl), "avg_r": float(np.mean(rl)) if len(rl) else 0.0, "t": tl, "thr": zl}
            elk = ftmo.evaluate(rl, dl, risk_pct=x["ftmo"]["risk_pct"], stride=1) if len(rl) >= 20 else {"pass_both": 0.0, "starts": 0}
            x["lock"]["ftmo_pass_both"] = elk["pass_both"]; x["lock"]["ftmo_starts"] = elk["starts"]
            x["stages"]["lock"] = bool(len(rl) >= 20 and np.mean(rl) > 0 and tl >= zl)
            if not x["stages"]["lock"]: x["fail"] = "lockbox: t=%.2f (n=%d) < prag %.2f" % (tl, len(rl), zl)
            elif x["ftmo"]["bootstrap_pass_both"] < pass_min or elk["pass_both"] < pass_min:
                x["fail"] = "FTMO: trecere %.0f%% (bootstrap) / %.0f%% (lockbox) < %.0f%%" % (x["ftmo"]["bootstrap_pass_both"] * 100, elk["pass_both"] * 100, pass_min * 100)
            else: x["relevant"] = True
    return out


def describe(ctx, g):
    """Statistici pentru o varianta: pe segmente, brut (fara costuri) si net, rata de castig, contoare de evenimente."""
    tr, cnt = ctx.run(g); res = {"counters": {k: int(v) for k, v in zip(("evenimente", "fara_MSS", "MSS", "FVG_invalid(>max)", "fara_intrare", "SL_prea_mare", "tranzactii"), cnt[:7])}, "seg": {}}
    for seg in ("train", "val", "lock", "all"):
        m = ctx.seg_mask(tr, seg); x = tr[m]
        if len(x) == 0: res["seg"][seg] = {"n": 0}; continue
        net = x[:, 4] - x[:, 20] / x[:, 5]
        res["seg"][seg] = {"n": int(len(x)), "avg_net": float(np.mean(net)), "avg_gross": float(np.mean(x[:, 4])), "t_net": tstat(net), "win": float(np.mean(x[:, 4] > 0)),
                           "loss": float(np.mean(x[:, 4] < 0)), "be": float(np.mean(x[:, 4] == 0)), "avg_risk_pts": float(np.mean(x[:, 5])), "avg_cost_r": float(np.mean(x[:, 20] / x[:, 5]))}
    return res
