# Studiu 23: TEST OUT-OF-SAMPLE pe un instrument NOU (6S = CHF futures, ~USDCHF pe FTMO), cu reguli INGHETATE din studiile 20/21 (fara nicio reglare).
# Ipoteza primara (inregistrata inainte de a vedea datele 6S): trasatura div240 (miscarea pe 4 h contra fluxului agresor), prag = cuantila 98% (din 6S), directia +1 (in sensul fluxului),
# intrare la ore fixe, sesiunea Asia (00-07 UTC), tinere 8 h, SL 1.5 sigma_H, cost CFD FTMO. Variante inrudite inghetate: R2..R4. Control: aceeasi regula cu directia inversa.
# Intreg esantionul 6S (3 ani) e OOS: nu a participat la nicio selectie. Se raporteaza si rangul regulii R1 intre toate cele 2304 variante din studiul 21 rulate pe 6S (daca efectul e real, R1 trebuie sa fie sus).
import os, json, time, argparse, itertools
import numpy as np
from .study12 import cost_frac
from .study11 import ftmo_block
from .study17 import load, tstat, seg, RISK
from .study18 import bars1m
from .study19 import features, sim
from .study21 import SESS2, FEATS, WINS, QS, HS, SES, daily_series

VER = 1
HYP = [("div240", 0.98, 1, 480, "asia"), ("div240", 0.98, 1, 480, "asia2"), ("div480", 0.99, 1, 480, "asia"), ("div480", 0.98, 1, 480, "asia2")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt23"); os.makedirs(out, exist_ok=True)

    def save(d):
        tmp = os.path.join(out, "result.json.tmp"); open(tmp, "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(tmp, os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time())}; save(res)
    try:
        root = os.path.join(a.data, "databento")
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        d = load(root, "6S")
        if d is None: raise RuntimeError("nu exista date 6S")
        B = bars1m(d); F, sig = features(B); px = float(np.median(B["c"])); cf, meas = cost_frac(spd, "USDCHF", px); cost = cf * px
        t0, t1 = float(B["t"][0]), float(B["t"][-1]); hr = (B["t"] % 86400.0) / 3600.0; min0 = (B["t"] % 3600.0) == 0.0
        res["date"] = {"de_la": time.strftime("%Y-%m-%d", time.gmtime(t0)), "pana_la": time.strftime("%Y-%m-%d", time.gmtime(t1)), "bare_1min": int(len(B["t"])), "cost_masurat": bool(meas), "cost_in_sigma_1min": round(float(cost / np.nanmedian(sig)), 2)}
        save(res)
        buf = np.zeros((40000, 3))

        def run(f, q, dr, H, se):
            x = F[f]; ax = np.abs(x[np.isfinite(x) & min0]); thr = float(np.quantile(ax, q))
            m = np.isfinite(x) & (np.abs(np.nan_to_num(x)) >= thr) & min0; lo, hi = SESS2[se]; m &= (hr >= lo) & (hr < hi); sa = np.sign(np.nan_to_num(x)) * dr; allr = []
            for sd_ in (1, -1):
                ms = m & (sa == sd_)
                if not ms.any(): continue
                nt = sim(B["t"], B["o"], B["h"], B["l"], B["c"], B["iid"], sig, ms, sd_, H, 1.5, cost, buf)
                if nt: allr.append(np.column_stack([buf[:nt, 0], buf[:nt, 1], np.zeros(nt), buf[:nt, 2]]))
            if not allr: return np.zeros((0, 4))
            r = np.vstack(allr); return r[np.argsort(r[:, 0])]
        res["reguli"] = []
        for k in HYP:
            for dr_name, dr in (("INGHETATA", k[2]), ("CONTROL_inversat", -k[2])):
                r = run(k[0], k[1], dr, k[3], k[4]); qs = np.linspace(t0, t1, 7)
                e = {"regula": dict(zip(["trasatura", "q", "dir", "H", "sesiune"], (k[0], k[1], dr, k[3], k[4]))), "tip": dr_name, "total": seg(r[:, 1]) if len(r) else {"n": 0},
                     "brut_R": round(float(r[:, 3].mean()), 4) if len(r) else None, "pe_6_perioade": [seg(r[(r[:, 0] >= qs[i]) & (r[:, 0] < qs[i + 1] + (1 if i == 5 else 0)), 1]) for i in range(6)] if len(r) else []}
                if dr_name == "INGHETATA" and len(r) > 60:
                    ds = daily_series(r, t0, t1, RISK)
                    if ds.std() > 0: e["sharpe_anual"] = round(float(ds.mean() / ds.std() * np.sqrt(252)), 2); e["ftmo_vol7%"] = ftmo_block(ds, 0.07, np.random.default_rng(23))
                res["reguli"].append(e)
            save(res)
        # rangul R1 intre toate variantele din studiul 21, pe 6S
        grid = [(fk + str(w), q, dr, H, se) for fk in FEATS for w in WINS for q in QS for dr in (1, -1) for H in HS for se in SES]
        tv = {}
        for gi, k in enumerate(grid):
            r = run(*k)
            if len(r) >= 30: tv[k] = (tstat(r[:, 1]), float(r[:, 1].mean()), len(r))
            if gi % 400 == 0: res["progres"] = "%d/%d" % (gi, len(grid)); save(res)
        ts = np.array([v[0] for v in tv.values()]); k1 = ("div240", 0.98, 1, 480, "asia")
        res["rang_R1"] = {"variante_cu_trade": len(tv), "t_R1": round(tv[k1][0], 2) if k1 in tv else None, "R_net_R1": round(tv[k1][1], 4) if k1 in tv else None,
                          "fractie_variante_cu_t_mai_mare": round(float((ts > tv[k1][0]).mean()), 3) if k1 in tv else None, "medie_R_net_toate": round(float(np.mean([v[1] for v in tv.values()])), 4),
                          "top5": [{"param": dict(zip(["trasatura", "q", "dir", "H", "sesiune"], k)), "t": round(v[0], 2), "R_net": round(v[1], 4), "n": v[2]} for k, v in sorted(tv.items(), key=lambda kv: -kv[1][0])[:5]]}
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("6S nu a participat la nicio selectie. Daca R1 (inghetata) e pozitiva net, cu t>~2, si inversul e negativ, si R1 e in primele ~5% variante, exista suport pentru un efect real de flux pe monede; altfel R1 a fost noroc.")
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
