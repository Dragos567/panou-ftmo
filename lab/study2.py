# Studiu pe timeframe mare (M15), mai multe intrari pe zi: ce semnale intraday (ORB, momentum, Donchian, gap) au un avantaj statistic,
# pe DAX / NIKKEI / EURUSD, fara strategie (fara stop/tinta): se masoara doar miscarea de dupa semnal, in puncte de baza (bps), dupa scaderea derivei orei.
# Costul mediu al unei tranzactii e raportat in bps ca sa se vada daca avantajul il acopera. Segmente: train 50% / validare 25% / lockbox 25% (pe zile).
import os, sys, json, time, argparse
import numpy as np
from . import liq as Q
from .run import load_m1
from .data import resample

VER = 1
MK = {"DAX": ("RO", 600, 1110), "EURUSD": ("RO", 600, 1110), "NIKKEI": ("UTC", 0, 360)}   # (fus, minut start, minut sfarsit) ale sesiunii
HB = [2, 4, 8]                                                                              # orizonturi in bare M15 (30, 60, 120 min); + "EOD"


def ctstat(x, day):
    n = len(x)
    if n < 20: return None
    m = float(x.mean()); u, inv = np.unique(day, return_inverse=True); G = len(u)
    if G < 5: return None
    s = np.bincount(inv, weights=x - m)
    var = float((s ** 2).sum()) * G / (G - 1) / (n ** 2)
    return m / np.sqrt(var) if var > 0 else None


def build(m15, tz, a, b):
    t = m15["t"]; K = (b - a) // 15
    off = Q.ro_offset(t) if tz == "RO" else np.zeros(len(t), np.int64)
    loc = t + off; lmin = ((loc % 86400) // 60).astype(np.int64); lday = (loc // 86400).astype(np.int64)
    ins = (lmin >= a) & (lmin < b)
    days = np.unique(lday[ins]); di = {int(d): i for i, d in enumerate(days)}
    D = len(days)
    M = {k: np.full((D, K), np.nan) for k in ("o", "h", "l", "c")}
    T0 = np.zeros(D, np.int64)
    idx = np.flatnonzero(ins)
    rows = np.array([di[int(x)] for x in lday[idx]]); cols = ((lmin[idx] - a) // 15).astype(np.int64)
    for k in M: M[k][rows, cols] = m15[k][idx]
    T0[rows] = t[idx]
    ok = np.sum(~np.isnan(M["c"]), axis=1) >= int(0.9 * K)
    return {k: v[ok] for k, v in M.items()}, T0[ok], days[ok], K


def signals(M, K):
    """Intoarce o lista de (nume, parametru, matrice D x K cu directia semnalului la inchiderea barei: +1 / -1 / 0)."""
    o, h, l, c = M["o"], M["h"], M["l"], M["c"]; D = c.shape[0]; out = []
    for r, nm in ((2, "30m"), (4, "60m")):
        hi = np.nanmax(h[:, :r], axis=1)[:, None]; lo = np.nanmin(l[:, :r], axis=1)[:, None]
        up = (c > hi); dn = (c < lo); up[:, :r] = False; dn[:, :r] = False
        S = np.zeros((D, K)); S[up] = 1; S[dn] = -1
        first = np.zeros((D, K))
        any_ = (S != 0)
        j0 = np.where(any_.any(axis=1), any_.argmax(axis=1), -1)
        for d in range(D):
            if j0[d] >= 0: first[d, j0[d]] = S[d, j0[d]]
        out.append(("ORB prima iesire", nm, first))
        fresh = np.zeros((D, K)); pu = np.zeros((D, K), bool); pd_ = np.zeros((D, K), bool)
        pu[:, 1:] = up[:, :-1]; pd_[:, 1:] = dn[:, :-1]
        fresh[up & ~pu] = 1; fresh[dn & ~pd_] = -1
        out.append(("ORB iesiri repetate", nm, fresh))
    for L, nm in ((8, "2h"), (16, "4h")):
        S = np.zeros((D, K))
        for j in range(L, K):
            hh = np.nanmax(h[:, j - L:j], axis=1); ll = np.nanmin(l[:, j - L:j], axis=1)
            S[:, j] = np.where(c[:, j] > hh, 1, np.where(c[:, j] < ll, -1, 0))
        F = np.zeros((D, K)); F[:, 1:] = np.where((S[:, 1:] != 0) & (S[:, :-1] == 0), S[:, 1:], 0)
        out.append(("Donchian (iesire din canal)", nm, F))
    # momentum: semnul primelor 30 min -> restul zilei / ultimele 30 min
    r30 = np.sign(c[:, 1] - o[:, 0]); r30 = np.nan_to_num(r30)
    S = np.zeros((D, K)); S[:, 1] = r30; out.append(("Momentum prima 1/2 ora -> restul zilei", "-", S))
    S = np.zeros((D, K)); S[:, K - 3] = r30; out.append(("Momentum prima 1/2 ora -> ultima 1/2 ora", "-", S))
    # gap: deschidere vs inchiderea zilei anterioare (continuare), semnal la inchiderea celei de-a doua bare
    gap = np.zeros(D); gap[1:] = o[1:, 0] - c[:-1, K - 1]; gs = np.nan_to_num(np.sign(gap)); gs[0] = 0
    S = np.zeros((D, K)); S[:, 1] = gs; out.append(("Gap (continuare)", "-", S))
    S = np.zeros((D, K)); S[:, 1] = np.where(gs == r30, gs, 0); out.append(("Gap + prima 1/2 ora in acelasi sens", "-", S))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt2"); os.makedirs(out, exist_ok=True)

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
            M, T0, days, K = build(m15, tz, a0, b0)
            D = len(days)
            sp = None
            try:
                means = [v["mean"] for v in spd.get(sym, {}).values() if v["n"] >= 30]
                if means: sp = float(np.median(means))
            except Exception: pass
            px = float(np.nanmedian(M["c"]))
            cost_bps = (1e4 * 1.75 * sp / px) if sp else None
            seg = np.where(np.arange(D) < D * 0.5, 0, np.where(np.arange(D) < D * 0.75, 1, 2))
            res["markets"][sym] = {"zile": int(D), "bare_sesiune": int(K), "cost_bps": round(cost_bps, 2) if cost_bps else None, "de": time.strftime("%Y-%m-%d", time.gmtime(int(T0[0]))), "pana": time.strftime("%Y-%m-%d", time.gmtime(int(T0[-1])))}
            c = M["c"]
            for sname, par, S in signals(M, K):
                for hn in ("30m", "60m", "120m", "EOD"):
                    if hn == "EOD":
                        fwd = np.full((D, K), np.nan); fwd[:, :K - 1] = (c[:, K - 1:K] - c[:, :K - 1]) / c[:, :K - 1] * 1e4
                    else:
                        hz = HB[("30m", "60m", "120m").index(hn)]; fwd = np.full((D, K), np.nan); fwd[:, :K - hz] = (c[:, hz:] - c[:, :K - hz]) / c[:, :K - hz] * 1e4
                    with np.errstate(all="ignore"):
                        mu = np.nanmean(np.where((seg == 0)[:, None], fwd, np.nan), axis=0)
                    mu = np.nan_to_num(mu)
                    ex = S * (fwd - mu[None, :])
                    em = (S != 0) & ~np.isnan(ex)
                    if em.sum() < 30: continue
                    dd = np.broadcast_to(np.arange(D)[:, None], (D, K))
                    X = ex[em]; DY = dd[em]; SG = seg[DY]; SD = S[em]
                    row = {"mk": sym, "sig": sname, "par": par, "H": hn}
                    for gn, gm in (("train", SG == 0), ("val", SG == 1), ("lock", SG == 2), ("tot", np.ones(len(X), bool))):
                        x = X[gm]; tt = ctstat(x, DY[gm]) if len(x) else None
                        row[gn] = {"n": int(gm.sum()), "bps": round(float(x.mean()), 2) if len(x) else None, "t": round(tt, 2) if tt is not None else None}
                    row["long_bps"] = round(float(X[SD > 0].mean()), 2) if (SD > 0).any() else None
                    row["short_bps"] = round(float(X[SD < 0].mean()), 2) if (SD < 0).any() else None
                    res["rows"].append(row)
                    tr, va, lk, tot = row["train"], row["val"], row["lock"], row["tot"]
                    if all(z["bps"] is not None and z["n"] >= 60 for z in (tr, va, lk)) and tot["t"] is not None:
                        sg = np.sign(tr["bps"])
                        if sg != 0 and np.sign(va["bps"]) == sg and np.sign(lk["bps"]) == sg and abs(tot["t"]) >= 2.5:
                            net = abs(tot["bps"]) - (cost_bps or 0)
                            res["candidates"].append({"mk": sym, "sig": sname, "par": par, "H": hn, "tip": "continuare" if sg > 0 else "reversare", "tr": tr, "va": va, "lk": lk, "tot": tot, "net_bps": round(net, 2), "acopera_costul": bool(cost_bps and abs(tot["bps"]) > cost_bps)})
            save(res)
        keys = {}
        for cnd in res["candidates"]: keys.setdefault((cnd["sig"], cnd["par"], cnd["H"], cnd["tip"]), []).append(cnd["mk"])
        res["cross"] = [{"sig": k[0], "par": k[1], "H": k[2], "tip": k[3], "piete": v} for k, v in keys.items() if len(v) >= 2]
        res["tests"] = len(res["rows"]); res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("bps = puncte de baza ale pretului; + = in sensul semnalului (continuare), dupa scaderea derivei. Candidat = acelasi semn pe train, validare si lockbox, n>=60 in fiecare, |t|>=2.5 total. "
                       "Din %d teste, cateva celule pot arata bine din intamplare; conteaza cele care se repeta pe mai multe piete." % res["tests"])
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1500:]; save(res)


if __name__ == "__main__":
    main()
