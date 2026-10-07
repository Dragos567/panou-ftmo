# Studiu 22 (PILOT, 2 luni): CARTEA DE ORDINE la momentul tranzactiilor (Databento TBBO: cel mai bun bid/ask cu cantitati), 6E, 6B, NKD.
# Intrebare: dezechilibrul din carte (cantitate bid vs ask) + microprice prezic miscarea urmatoare MAI BINE decat fluxul de tranzactii? Si cat din miscare ramane dupa costul CFD FTMO?
# Metode: (a) decile: miscarea medie a mid-ului la H secunde dupa, pe decila superioara vs inferioara a trasaturii (in ticks), esantion fara suprapunere; (b) regula simpla cu prag din luna 1, testata in luna 2, net de cost;
# (c) cat explica fiecare trasatura (corelatie) si daca cartea adauga peste flux. Cost = spread CFD FTMO masurat x1.75 (in ticks) - acelasi ca in studiile 17-21.
import os, json, time, argparse, glob
import numpy as np
from .study12 import cost_frac
from .study17 import tstat

VER = 1
TICK = {"6E": 0.00005, "6B": 0.0001, "NKD": 5.0}
CFD = {"6E": "EURUSD", "6B": "GBPUSD", "NKD": "NIKKEI"}


def loadb(root, sym):
    P = []
    for f in sorted(glob.glob(os.path.join(root, sym, "*.npz"))):
        z = np.load(f)
        if len(z["t"]): P.append({k: z[k] for k in z.files})
    if not P: return None
    d = {k: np.concatenate([p[k] for p in P]).astype(np.float64) for k in P[0]}
    o = np.argsort(d["t"], kind="stable"); return {k: v[o] for k, v in d.items()}


def rolling(x, t, W):
    cs = np.concatenate([[0.0], np.cumsum(x)]); j = np.searchsorted(t, t - W, side="right"); i = np.arange(len(t)); return cs[i + 1] - cs[j], (i + 1 - j)


def feats(d):
    t = d["t"]; mid = (d["bid"] + d["ask"]) / 2.0; v = np.maximum(d["v"], 1.0)
    q = (d["bsz"] - d["asz"]) / np.maximum(d["bsz"] + d["asz"], 1.0)                       # dezechilibru carte la ultima tranzactie din secunda
    micro = (d["ask"] * d["bsz"] + d["bid"] * d["asz"]) / np.maximum(d["bsz"] + d["asz"], 1.0) - mid        # microprice - mid (unitati de pret)
    F = {"carte_q": q, "microprice": micro}
    for W in (15, 60, 300):
        sv, _ = rolling(d["v"], t, W); sd, _ = rolling(d["b"] - d["s"], t, W); si, _ = rolling(d["imb"], t, W)
        F["flux%d" % W] = np.where(sv > 0, sd / np.maximum(sv, 1.0), 0.0)                  # dezechilibru agresor (flux de tranzactii)
        F["carteMed%d" % W] = np.where(sv > 0, si / np.maximum(sv, 1.0), 0.0)               # dezechilibru carte, mediat pe tranzactii
    return F, mid


def fwd(t, mid, H):
    j = np.searchsorted(t, t + H, side="right") - 1; ok = (t[j] - t >= H * 0.5) & (j > np.arange(len(t))) & (t[j] - t <= H * 3)
    return np.where(ok, mid[j] - mid, np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt22"); os.makedirs(out, exist_ok=True)

    def save(d):
        tmp = os.path.join(out, "result.json.tmp"); open(tmp, "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(tmp, os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time()), "instrumente": {}}; save(res)
    try:
        root = os.path.join(a.data, "databento_book")
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        for s in TICK:
            d = loadb(root, s)
            if d is None: continue
            t = d["t"]; F, mid = feats(d); tk = TICK[s]; px = float(np.median(d["c"])); cf, meas = cost_frac(spd, CFD[s], px); cost_t = cf * px / tk
            half = t[0] + 0.5 * (t[-1] - t[0]); r = {"secunde_cu_tranzactii": int(len(t)), "de_la": time.strftime("%Y-%m-%d", time.gmtime(t[0])), "pana_la": time.strftime("%Y-%m-%d", time.gmtime(t[-1])),
                                                      "cost_CFD_in_ticks": round(cost_t, 2), "spread_futures_mediu_ticks": round(float(np.mean(d["ask"] - d["bid"]) / tk), 2), "rezultate": {}}
            for H in (5, 15, 60, 300):
                y = fwd(t, mid, H) / tk; rr = {}
                for fn, x in F.items():
                    m = np.isfinite(y) & np.isfinite(x)
                    if m.sum() < 2000: continue
                    xs, ys, ts = x[m], y[m], t[m]
                    cor = float(np.corrcoef(xs, ys)[0, 1])
                    # esantion fara suprapunere (un punct la fiecare H secunde)
                    keep = np.zeros(len(ts), bool); last = -1e18
                    for i in range(len(ts)):
                        if ts[i] - last >= H: keep[i] = True; last = ts[i]
                    xs2, ys2 = xs[keep], ys[keep]
                    if len(xs2) < 500: continue
                    lo, hi = np.quantile(xs2, 0.1), np.quantile(xs2, 0.9)
                    top = ys2[xs2 >= hi]; bot = ys2[xs2 <= lo]
                    spread_ticks = float(top.mean() - bot.mean()) / 2.0                     # castig mediu per tranzactie (long top / short bottom), in ticks
                    # regula: prag (cuantila 95% din luna 1) -> in luna 2, directie = semnul trasaturii, tinere H
                    tr = ts < half; te = ~tr; thr = np.quantile(np.abs(xs[tr]), 0.95) if tr.sum() > 500 else np.inf
                    sel = te & (np.abs(xs) >= thr)
                    g = np.sign(xs[sel]) * ys[sel]; kept = np.zeros(len(g), bool); last = -1e18; tt = ts[sel]
                    for i in range(len(tt)):
                        if tt[i] - last >= H: kept[i] = True; last = tt[i]
                    g = g[kept]
                    rr[fn] = {"corel": round(cor, 4), "decila_sup_minus_inf_ticks_jumate": round(spread_ticks, 3), "n_decile": [int(len(top)), int(len(bot))],
                              "t_decile": round(float((top.mean() - bot.mean()) / np.sqrt(top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot))), 2),
                              "regula_luna2": {"n": int(len(g)), "brut_ticks": round(float(g.mean()), 3) if len(g) else None, "net_ticks": round(float(g.mean() - cost_t), 3) if len(g) else None, "t_brut": round(tstat(g), 2) if len(g) > 5 else None}}
                r["rezultate"]["H%ds" % H] = rr
            res["instrumente"][s] = r; save(res)
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("PILOT pe 2 luni. 'decila_sup_minus_inf_ticks_jumate' = cat castiga in medie (inainte de cost) o tranzactie in directia trasaturii, in ticks; de comparat cu 'cost_CFD_in_ticks'. "
                       "'regula_luna2' = prag din luna 1, testat in luna 2; net = brut minus cost. Doua luni = esantion mic: doar indicator de directie, nu dovada.")
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
