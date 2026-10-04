# Studiu de eveniment pe DAX: ce face pretul dupa un sweep de nivel Daily/Weekly/Asia, FARA strategie (fara intrare, SL, TP, MSS, FVG).
# Daca nici aici nu exista o deviere fata de zgomot, nicio regula de intrare construita peste sweep nu va crea avantaj.
# Evenimentul se cunoaste la inchiderea barei de sweep; se masoara miscarea de la acea inchidere pana la H minute mai tarziu, in puncte,
# "in directia reversarii" (dupa sweep de high: -(miscare); dupa sweep de low: +(miscare)), minus deriva obisnuita a orei respective (media pe train).
import os, sys, json, time, argparse
import numpy as np
from . import liqx as X
from .run import load_m1

VER = 1
SYM = "DAX"
HZ = [5, 15, 30, 60, 120]


def ctstat(x, day):
    """t cu erori standard grupate pe zi (evenimentele din aceeasi zi nu sunt independente)."""
    n = len(x)
    if n < 12: return None
    m = float(x.mean()); u, inv = np.unique(day, return_inverse=True); G = len(u)
    if G < 4: return None
    s = np.bincount(inv, weights=x - m)
    var = float((s ** 2).sum()) * G / (G - 1) / (n ** 2)
    return m / np.sqrt(var) if var > 0 else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt"); os.makedirs(out, exist_ok=True)

    def save(d):
        json.dump(d, open(os.path.join(out, "result.json.tmp"), "w")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    save({"ver": VER, "state": "ruleaza", "started": int(time.time())})
    try:
        m1 = load_m1(a.data, SYM)
        spread = None
        try:
            spd = json.load(open(os.path.join(a.data, "spread.json"))).get(SYM, {})
            means = [v["mean"] for v in spd.values() if v["n"] >= 30]
            if means:
                med = float(np.median(means)); spread = [spd[str(h)]["mean"] if str(h) in spd and spd[str(h)]["n"] >= 30 else med for h in range(24)]
        except Exception: pass
        ctx = X.LCtx(m1, tick=0.01, cost_pts=1.5, spread_by_hour=spread, spread_mult=1.5, slip=0.5, comm=0.0)
        t, c = ctx.t, ctx.c; n = len(t)
        rb, rd, rp, rs, rt = ctx.rows
        rb = np.asarray(rb, np.int64); rd = np.asarray(rd, np.int64); rs = np.asarray(rs, np.int64)
        lmin = ctx.lmin; lday = ctx.lday
        cost_pts = float(np.median(ctx.cost))
        c0, c1 = ctx.cut
        # miscarea fata de inchiderea barei i, la H minute dupa (bara cu timpul t[i]+H*60; toleranta 2 minute)
        def fwd_all(H):
            tgt = t + H * 60; idx = np.searchsorted(t, tgt); idc = np.minimum(idx, n - 1)
            ok = (idx < n) & (np.abs(t[idc] - tgt) <= 120)
            return c[idc] - c, ok
        evb = lmin[rb] // 15                      # bucket de 15 minute (ora RO) al evenimentului
        allb = lmin // 15
        inwin = (lmin >= 600) & (lmin < 1080)
        train_mask = inwin & (t < c0)
        res = {"ver": VER, "state": "gata", "finished": 0, "bars": int(n), "events": int(len(rb)), "cost_pts": round(cost_pts, 3),
               "segments": [time.strftime("%Y-%m-%d", time.gmtime(int(x))) for x in (t[0], c0, c1, t[-1])], "rows": [], "hours": [], "candidates": []}
        evseg = np.where(t[rb] < c0, 0, np.where(t[rb] < c1, 1, 2))
        evday = lday[rb]
        evhour = ((lmin[rb] - 600) // 120)          # 0: 10-12, 1: 12-14, 2: 14-16, 3: 16-18 (ora RO)
        srcs = {"toate": np.ones(len(rb), bool), "Daily": (rs & 1) > 0, "Weekly": (rs & 2) > 0, "Asia": (rs & 4) > 0}
        dirs = {"ambele": np.ones(len(rb), bool), "high_sweep": rd == 1, "low_sweep": rd == -1}
        segs = {"train": evseg == 0, "val": evseg == 1, "lock": evseg == 2, "tot": np.ones(len(rb), bool)}
        for H in HZ:
            F, ok = fwd_all(H)
            mu = np.zeros(96)
            for b in range(96):
                mk = train_mask & ok & (allb == b)
                if mk.sum() > 200: mu[b] = float(F[mk].mean())
            evok = ok[rb]
            ex = -rd * (F[rb] - mu[evb])         # + = reversare dupa sweep (in puncte, dupa scaderea derivei)
            sd = float(np.std(F[inwin & ok]))     # scara miscarii la acest orizont
            res.setdefault("scale", {})[str(H)] = round(sd, 2)
            for sn, sm in srcs.items():
                for dn, dm in dirs.items():
                    row = {"H": H, "src": sn, "dir": dn}
                    for gn, gm in segs.items():
                        mk = sm & dm & gm & evok; x = ex[mk]
                        row[gn] = {"n": int(mk.sum()), "mean": round(float(x.mean()), 3) if len(x) else None, "t": (round(ctstat(x, evday[mk]), 2) if ctstat(x, evday[mk]) is not None else None)}
                    res["rows"].append(row)
                    tr, va = row["train"], row["val"]
                    if tr["t"] is not None and va["t"] is not None and tr["n"] >= 80 and va["n"] >= 40:
                        if np.sign(tr["mean"]) == np.sign(va["mean"]) and abs(tr["t"]) >= 2.0 and abs(va["t"]) >= 1.3 and sn != "toate":
                            res["candidates"].append({"H": H, "src": sn, "dir": dn, "train": tr, "val": va, "lock": row["lock"], "tip": "reversare" if tr["mean"] > 0 else "continuare"})
            for hi, hn in enumerate(("10-12", "12-14", "14-16", "16-18")):
                row = {"H": H, "ora": hn}
                for gn, gm in segs.items():
                    mk = (evhour == hi) & gm & evok; x = ex[mk]
                    tt = ctstat(x, evday[mk]) if len(x) else None
                    row[gn] = {"n": int(mk.sum()), "mean": round(float(x.mean()), 3) if len(x) else None, "t": round(tt, 2) if tt is not None else None}
                res["hours"].append(row)
        res["tests"] = len(res["rows"]); res["finished"] = int(time.time())
        res["nota"] = ("Pozitiv = reversare dupa sweep (in puncte DAX, dupa scaderea derivei orei). Costul unei tranzactii e cost_pts. "
                       "Sunt %d celule testate: la |t|>=2 se asteapta cateva false pozitive doar din intamplare; conteaza celulele unde train si validare au acelasi semn." % res["tests"])
        save(res)
    except Exception as e:
        import traceback; save({"ver": VER, "state": "eroare", "error": repr(e), "tb": traceback.format_exc()[-1500:]})


if __name__ == "__main__":
    main()
