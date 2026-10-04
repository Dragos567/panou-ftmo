# Studiu 3 (M15): volum si proxy de order flow + order block, pe DAX / NIKKEI / EURUSD.
# Date disponibile = bare cu volum de ticuri (nu exista adancime de piata/delta real). Proxy-uri:
#  - volum relativ (RVOL): volumul primelor 30 min sau al barei vs media aceleiasi bare in ultimele 14 zile ("index in play", ca in ORB stocks-in-play);
#  - delta aproximativ: volum * (2*close - high - low) / (high - low) (locatia inchiderii in bara), cumulat pe sesiune;
#  - absorbtie: volum mare + bara cu range mic la un maxim/minim de sesiune;
#  - order block: ultima bara opusa inainte de un impuls cu spargere de structura, apoi revenirea in zona.
# Se masoara doar miscarea de dupa semnal (bps, dupa scaderea derivei orei), fara stop/tinta. Aceleasi segmente si aceleasi reguli de candidat ca la study2.
import os, sys, json, time, argparse
import numpy as np
from . import liq as Q
from .run import load_m1
from .data import resample
from .study2 import ctstat, MK, HB

VER = 1


def build3(m15, tz, a, b):
    t = m15["t"]; K = (b - a) // 15
    off = Q.ro_offset(t) if tz == "RO" else np.zeros(len(t), np.int64)
    loc = t + off; lmin = ((loc % 86400) // 60).astype(np.int64); lday = (loc // 86400).astype(np.int64)
    ins = (lmin >= a) & (lmin < b)
    days = np.unique(lday[ins]); di = {int(d): i for i, d in enumerate(days)}; D = len(days)
    M = {k: np.full((D, K), np.nan) for k in ("o", "h", "l", "c", "v")}
    T0 = np.zeros(D, np.int64); idx = np.flatnonzero(ins)
    rows = np.array([di[int(x)] for x in lday[idx]]); cols = ((lmin[idx] - a) // 15).astype(np.int64)
    for k in M: M[k][rows, cols] = m15[k][idx].astype(np.float64)
    T0[rows] = t[idx]
    ok = np.sum(~np.isnan(M["c"]), axis=1) >= int(0.9 * K)
    return {k: v[ok] for k, v in M.items()}, T0[ok], days[ok], K


def prev_mean(X, n=14, minp=7):
    """Media pe ultimele n zile (fara ziua curenta) pentru fiecare coloana; NaN daca sunt sub minp zile valide."""
    D = X.shape[0]; out = np.full(X.shape, np.nan)
    for d in range(minp, D):
        blk = X[max(0, d - n):d]
        cnt = np.sum(~np.isnan(blk), axis=0)
        with np.errstate(all="ignore"):
            m = np.nanmean(blk, axis=0)
        out[d] = np.where(cnt >= minp, m, np.nan)
    return out


def signals3(M, K, seg):
    o, h, l, c, v = M["o"], M["h"], M["l"], M["c"], M["v"]; D = c.shape[0]; out = []
    v0 = np.nan_to_num(v)
    # --- semnale de baza (ca la study2) ---
    r30 = np.nan_to_num(np.sign(c[:, 1] - o[:, 0]))
    def orb_first(r):
        hi = np.nanmax(h[:, :r], axis=1)[:, None]; lo = np.nanmin(l[:, :r], axis=1)[:, None]
        up = c > hi; dn = c < lo; up[:, :r] = False; dn[:, :r] = False
        S = np.zeros((D, K)); S[up] = 1; S[dn] = -1
        first = np.zeros((D, K)); an = S != 0; j0 = np.where(an.any(axis=1), an.argmax(axis=1), -1)
        for d in range(D):
            if j0[d] >= 0: first[d, j0[d]] = S[d, j0[d]]
        return first
    def donch(L):
        S = np.zeros((D, K))
        for j in range(L, K):
            hh = np.nanmax(h[:, j - L:j], axis=1); ll = np.nanmin(l[:, j - L:j], axis=1)
            S[:, j] = np.where(c[:, j] > hh, 1, np.where(c[:, j] < ll, -1, 0))
        F = np.zeros((D, K)); F[:, 1:] = np.where((S[:, 1:] != 0) & (S[:, :-1] == 0), S[:, 1:], 0)
        return F
    orb30, orb60, dc8, dc16 = orb_first(2), orb_first(4), donch(8), donch(16)
    mom = np.zeros((D, K)); mom[:, 1] = r30
    # --- A) RVOL al primelor 30 min (ziua "in play") ---
    V01 = v0[:, 0] + v0[:, 1]; rv = np.full(D, np.nan)
    pm = prev_mean(V01[:, None], 14, 7)[:, 0]; rv = V01 / pm
    trn = (seg == 0) & ~np.isnan(rv)
    if trn.sum() > 30:
        q1, q2 = np.nanquantile(rv[trn], [0.33, 0.67])
        hi_d = (rv >= q2)[:, None]; lo_d = (rv <= q1)[:, None]
        for nm, S in (("ORB 30m prima iesire", orb30), ("ORB 60m prima iesire", orb60), ("Momentum prima 1/2 ora -> restul zilei", mom), ("Donchian 2h", dc8)):
            out.append((nm + " | zi RVOL mare", "-", np.where(hi_d, S, 0)))
            out.append((nm + " | zi RVOL mica", "-", np.where(lo_d, S, 0)))
    # --- B) volumul barei de semnal vs media aceleiasi bare (14 zile) ---
    pmb = prev_mean(v0, 14, 7); rvb = v0 / pmb
    for nm, S in (("ORB 30m iesiri repetate", None), ("Donchian 2h", dc8), ("Donchian 4h", dc16)):
        if S is None:
            hi = np.nanmax(h[:, :2], axis=1)[:, None]; lo = np.nanmin(l[:, :2], axis=1)[:, None]
            up = c > hi; dn = c < lo; up[:, :2] = False; dn[:, :2] = False
            pu = np.zeros((D, K), bool); pd_ = np.zeros((D, K), bool); pu[:, 1:] = up[:, :-1]; pd_[:, 1:] = dn[:, :-1]
            S = np.zeros((D, K)); S[up & ~pu] = 1; S[dn & ~pd_] = -1
        out.append((nm + " | volum bara >=1.5x", "-", np.where(rvb >= 1.5, S, 0)))
        out.append((nm + " | volum bara <1.0x", "-", np.where(rvb < 1.0, S, 0)))
    # --- C) delta aproximativ (locatia inchiderii) ---
    rng = h - l
    with np.errstate(all="ignore"):
        clv = np.where(rng > 0, (2 * c - h - l) / rng, 0.0)
    dv = np.nan_to_num(v * clv); CD = np.cumsum(dv, axis=1)
    conf = np.zeros((D, K)); dive = np.zeros((D, K))
    for j in range(8, K):
        for side in (1, -1):
            sel = dc8[:, j] == side
            if not sel.any(): continue
            if side == 1: okc = CD[:, j] >= np.max(CD[:, :j], axis=1)
            else: okc = CD[:, j] <= np.min(CD[:, :j], axis=1)
            conf[sel & okc, j] = side; dive[sel & ~okc, j] = side
    out.append(("Donchian 2h | delta confirmat (cumul. la maxim/minim)", "-", conf))
    out.append(("Donchian 2h | delta divergent", "-", dive))
    # --- D) absorbtie: volum mare, range mic, la maxim/minim de sesiune -> reversare ---
    prg = prev_mean(np.where(np.isnan(rng), np.nan, rng), 14, 7)
    small = (rng <= 0.7 * prg)
    ab = np.zeros((D, K))
    for j in range(6, K - 1):
        nh = h[:, j] >= np.nanmax(h[:, :j], axis=1); nl = l[:, j] <= np.nanmin(l[:, :j], axis=1)
        base = (rvb[:, j] >= 2.0) & small[:, j]
        ab[base & nh, j] = -1; ab[base & nl, j] = 1
    out.append(("Absorbtie la maxim/minim de sesiune (semnal = reversare)", "-", ab))
    # --- E) order block: ultima bara opusa inainte de impuls cu spargere de structura; intrare la revenirea in zona ---
    ob = np.zeros((D, K)); body = np.abs(c - o)
    for d in range(D):
        for j in range(9, K - 2):
            mb = np.nanmedian(body[d, j - 8:j])
            if not mb or np.isnan(mb) or np.isnan(body[d, j]): continue
            if body[d, j] < 2 * mb: continue
            if c[d, j] > np.nanmax(h[d, j - 8:j]) and c[d, j] > o[d, j]: dr = 1
            elif c[d, j] < np.nanmin(l[d, j - 8:j]) and c[d, j] < o[d, j]: dr = -1
            else: continue
            i = None
            for q in range(j - 1, max(j - 6, 0), -1):
                if (dr == 1 and c[d, q] < o[d, q]) or (dr == -1 and c[d, q] > o[d, q]): i = q; break
            if i is None: continue
            zh, zl = h[d, i], l[d, i]
            for t_ in range(j + 1, min(j + 17, K - 1)):
                if dr == 1:
                    if c[d, t_] < zl: break
                    if l[d, t_] <= zh: ob[d, t_] = 1; break
                else:
                    if c[d, t_] > zh: break
                    if h[d, t_] >= zl: ob[d, t_] = -1; break
    out.append(("Order block (impuls 2x corp + spargere 8 bare, revenire in zona)", "-", ob))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt3"); os.makedirs(out, exist_ok=True)

    def save(d):
        json.dump(d, open(os.path.join(out, "result.json.tmp"), "w")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time()), "markets": {}, "rows": [], "candidates": [], "cross": []}
    save(res)
    try:
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        for sym, (tz, a0, b0) in MK.items():
            try: m1 = load_m1(a.data, sym)
            except Exception as e:
                res["markets"][sym] = {"eroare": repr(e)}; continue
            m15 = resample(m1, 900)
            M, T0, days, K = build3(m15, tz, a0, b0); D = len(days)
            sp = None
            try:
                means = [v["mean"] for v in spd.get(sym, {}).values() if v["n"] >= 30]
                if means: sp = float(np.median(means))
            except Exception: pass
            px = float(np.nanmedian(M["c"])); cost_bps = (1e4 * 1.75 * sp / px) if sp else None
            seg = np.where(np.arange(D) < D * 0.5, 0, np.where(np.arange(D) < D * 0.75, 1, 2))
            vv = M["v"][~np.isnan(M["v"])]
            res["markets"][sym] = {"zile": int(D), "cost_bps": round(cost_bps, 2) if cost_bps else None, "de": time.strftime("%Y-%m-%d", time.gmtime(int(T0[0]))), "pana": time.strftime("%Y-%m-%d", time.gmtime(int(T0[-1]))),
                                   "vol_mediana": float(np.median(vv)) if len(vv) else None, "vol_zero": round(float((vv == 0).mean()), 3) if len(vv) else None}
            c = M["c"]
            for sname, par, S in signals3(M, K, seg):
                for hn in ("30m", "60m", "120m", "EOD"):
                    if hn == "EOD":
                        fwd = np.full((D, K), np.nan); fwd[:, :K - 1] = (c[:, K - 1:K] - c[:, :K - 1]) / c[:, :K - 1] * 1e4
                    else:
                        hz = HB[("30m", "60m", "120m").index(hn)]; fwd = np.full((D, K), np.nan); fwd[:, :K - hz] = (c[:, hz:] - c[:, :K - hz]) / c[:, :K - hz] * 1e4
                    with np.errstate(all="ignore"):
                        mu = np.nan_to_num(np.nanmean(np.where((seg == 0)[:, None], fwd, np.nan), axis=0))
                    ex = S * (fwd - mu[None, :]); em = (S != 0) & ~np.isnan(ex)
                    if em.sum() < 40: continue
                    dd = np.broadcast_to(np.arange(D)[:, None], (D, K)); X = ex[em]; DY = dd[em]; SG = seg[DY]; SD = S[em]
                    row = {"mk": sym, "sig": sname, "par": par, "H": hn}
                    for gn, gm in (("train", SG == 0), ("val", SG == 1), ("lock", SG == 2), ("tot", np.ones(len(X), bool))):
                        x = X[gm]; tt = ctstat(x, DY[gm]) if len(x) else None
                        row[gn] = {"n": int(gm.sum()), "bps": round(float(x.mean()), 2) if len(x) else None, "t": round(tt, 2) if tt is not None else None}
                    row["long_bps"] = round(float(X[SD > 0].mean()), 2) if (SD > 0).any() else None
                    row["short_bps"] = round(float(X[SD < 0].mean()), 2) if (SD < 0).any() else None
                    res["rows"].append(row)
                    tr, va, lk, tot = row["train"], row["val"], row["lock"], row["tot"]
                    if all(z["bps"] is not None and z["n"] >= 40 for z in (tr, va, lk)) and tot["t"] is not None:
                        sg = np.sign(tr["bps"])
                        if sg != 0 and np.sign(va["bps"]) == sg and np.sign(lk["bps"]) == sg and abs(tot["t"]) >= 2.5:
                            res["candidates"].append({"mk": sym, "sig": sname, "H": hn, "tip": "continuare" if sg > 0 else "reversare", "tr": tr, "va": va, "lk": lk, "tot": tot,
                                                      "net_bps": round(abs(tot["bps"]) - (cost_bps or 0), 2), "acopera_costul": bool(cost_bps and abs(tot["bps"]) > cost_bps)})
            save(res)
        keys = {}
        for cnd in res["candidates"]: keys.setdefault((cnd["sig"], cnd["H"], cnd["tip"]), []).append(cnd["mk"])
        res["cross"] = [{"sig": k[0], "H": k[1], "tip": k[2], "piete": v} for k, v in keys.items() if len(v) >= 2]
        res["tests"] = len(res["rows"]); res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = "bps = puncte de baza; + = in sensul semnalului, dupa scaderea derivei. Candidat = acelasi semn pe train/val/lock, n>=40 fiecare, |t|>=2.5. Din %d teste, cateva ies bune din intamplare." % res["tests"]
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1500:]; save(res)


if __name__ == "__main__":
    main()
