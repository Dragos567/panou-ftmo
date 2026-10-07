# Studiu 21: CAUTARE LARGA + ADEVAR STATISTIC. Familia de semnale de flux agresor/pret pe ferestre 2-8 h, tinere 4-16 h, sesiuni Asia/UE/SUA (6E, 6B, NKD, 3 ani, tranzactii reale).
# Trei teste care nu pot fi pacalite de numarul de variante:
#  (1) WALK-FORWARD: la fiecare trimestru din ultimii 2 ani alegem cea mai buna varianta pe TOT trecutul (t net) si o tranzactionam doar in trimestrul urmator. Seria OOS rezultata = performanta reala a 'procesului de cautare'.
#      Comparata cu media tuturor variantelor (selectia adauga ceva sau nu?).
#  (2) REALITY CHECK (sign-flip pe zile): distributia maximului t pe toate variantele sub ipoteza 'fara avantaj'; p-valoarea celei mai bune variante observate.
#  (3) Pentru candidatii cei mai buni: stabilitate pe instrument/perioada si probabilitatea FTMO la mai multe niveluri de risc.
import os, json, time, argparse, itertools
import numpy as np
from .study12 import cost_frac
from .study11 import ftmo_block
from .study17 import load, tstat, seg, INST, RISK
from .study18 import bars1m
from .study19 import features, sim, SESS

VER = 1
SESS2 = dict(SESS); SESS2["asia1"] = (0.0, 3.0); SESS2["asia2"] = (3.0, 7.0)
FEATS = ("div", "imb", "ret", "vwap")
WINS = (120, 240, 480)
QS = (0.90, 0.95, 0.98, 0.99)
HS = (240, 480, 720, 960)
SES = ("asia", "asia1", "asia2", "eu", "us", "all")


def daily_series(a_, t0, t1, scale):
    days = np.arange(int(t0 // 86400), int(t1 // 86400) + 1); dd = np.zeros(len(days))
    for tt_, rn in zip(a_[:, 0], a_[:, 1]): dd[int(tt_ // 86400) - days[0]] += rn * scale
    wk = np.array([(int(dv) + 4) % 7 < 5 for dv in days]); return dd[wk]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt21"); os.makedirs(out, exist_ok=True)

    def save(d):
        tmp = os.path.join(out, "result.json.tmp"); open(tmp, "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(tmp, os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time())}; save(res)
    try:
        root = os.path.join(a.data, "databento")
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        S = {}
        for s in INST:
            d = load(root, s)
            if d is None: continue
            B = bars1m(d); F, sig = features(B); px = float(np.median(B["c"])); cf, _ = cost_frac(spd, INST[s]["cfd"], px)
            S[s] = {"B": B, "F": F, "sig": sig, "cost": cf * px, "hr": (B["t"] % 86400.0) / 3600.0, "min0": (B["t"] % 3600.0) == 0.0}
        t0 = max(float(v["B"]["t"][0]) for v in S.values()); t1 = min(float(v["B"]["t"][-1]) for v in S.values())
        c1 = t0 + 0.5 * (t1 - t0)
        thr = {}
        for s, v in S.items():
            tr = v["B"]["t"] < c1
            for fk in FEATS:
                for w in WINS:
                    f = "%s%d" % (fk, w); x = np.abs(v["F"][f][tr & v["min0"]]); x = x[np.isfinite(x)]
                    thr[(s, f)] = {q: float(np.quantile(x, q)) for q in QS}
        grid = [(fk + str(w), q, dr, H, se) for fk in FEATS for w in WINS for q in QS for dr in (1, -1) for H in HS for se in SES]
        res["variante"] = len(grid); save(res)
        buf = np.zeros((40000, 3)); T = {}
        for gi, (f, q, dr, H, se) in enumerate(grid):
            allr = []
            for si, (s, v) in enumerate(S.items()):
                x = v["F"][f]; m = np.isfinite(x) & (np.abs(np.nan_to_num(x)) >= thr[(s, f)][q]); side_arr = np.sign(np.nan_to_num(x)) * dr
                lo, hi = SESS2[se]; m = m & v["min0"] & (v["hr"] >= lo) & (v["hr"] < hi)
                for sd_ in (1, -1):
                    ms = m & (side_arr == sd_)
                    if not ms.any(): continue
                    B = v["B"]; nt = sim(B["t"], B["o"], B["h"], B["l"], B["c"], B["iid"], v["sig"], ms, sd_, H, 1.5, v["cost"], buf)
                    if nt: allr.append(np.column_stack([buf[:nt, 0], buf[:nt, 1], np.full(nt, si), buf[:nt, 2]]))
            if allr:
                a_ = np.vstack(allr); T[grid[gi]] = a_[np.argsort(a_[:, 0])]
            if gi % 200 == 0: res["progres"] = "%d/%d" % (gi + 1, len(grid)); save(res)
        keys = [k for k, a_ in T.items() if len(a_) >= 120]; res["variante_cu_trade"] = len(keys)
        # (1) walk-forward
        edges = [t0 + (t1 - t0) * (12 + 3 * i) / 36.0 for i in range(9)]
        oos = []; wf = []; base = []
        for i in range(8):
            lo, hi = edges[i], edges[i + 1]; best = None; bt = -9
            for k in keys:
                a_ = T[k]; h_ = a_[a_[:, 0] < lo, 1]
                if len(h_) >= 60:
                    tt = tstat(h_)
                    if tt > bt: bt, best = tt, k
            te = [(k, T[k][(T[k][:, 0] >= lo) & (T[k][:, 0] < hi)]) for k in keys]
            allm = [x[:, 1].mean() for k, x in te if len(x) >= 5]
            base.append(float(np.mean(allm)) if allm else 0.0)
            if best is not None:
                x = T[best][(T[best][:, 0] >= lo) & (T[best][:, 0] < hi)]; oos.append(x)
                wf.append({"trimestru": i + 1, "aleasa": dict(zip(["trasatura", "q", "dir", "H", "sesiune"], best)), "t_trecut": round(bt, 2), "OOS": seg(x[:, 1])})
        res["walk_forward"] = wf
        if oos:
            o = np.vstack(oos); res["walk_forward_total"] = seg(o[:, 1]); res["walk_forward_brut_R"] = round(float(o[:, 3].mean()), 4)
            res["media_tuturor_variantelor_OOS"] = round(float(np.mean(base)), 4)
            ds = daily_series(o[o[:, 0].argsort()], edges[0], edges[-1], RISK)
            res["walk_forward_ftmo"] = {}
            if ds.std() > 0:
                res["walk_forward_sharpe"] = round(float(ds.mean() / ds.std() * np.sqrt(252)), 2); res["walk_forward_profit_pct_cont"] = round(float(ds.sum() * 100), 2)
                for vt in (0.04, 0.07, 0.10):
                    res["walk_forward_ftmo"]["vol%d%%" % int(vt * 100)] = ftmo_block(ds, vt, np.random.default_rng(21))
        save(res)
        # (2) reality check
        days_all = np.arange(int(t0 // 86400), int(t1 // 86400) + 1); rng = np.random.default_rng(210); NB = 400
        flips = rng.choice([-1.0, 1.0], size=(NB, len(days_all)))
        tobs = {}; nullmax = np.full(NB, -99.0)
        for k in keys:
            a_ = T[k]; x = a_[:, 1]; di = (a_[:, 0] // 86400).astype(int) - days_all[0]
            den = np.sqrt((x * x).sum()); tobs[k] = x.sum() / den if den > 0 else 0.0
            tf = (flips[:, di] @ x) / den; nullmax = np.maximum(nullmax, tf)
        top = sorted(keys, key=lambda k: -tobs[k])[:10]
        bt_ = tobs[top[0]]
        res["reality_check"] = {"t_cea_mai_buna": round(float(bt_), 2), "max_t_sub_zero_avantaj_p50": round(float(np.quantile(nullmax, .5)), 2), "p95": round(float(np.quantile(nullmax, .95)), 2), "p99": round(float(np.quantile(nullmax, .99)), 2),
                                "p_valoare": round(float((nullmax >= bt_).mean()), 3), "top10": []}
        for k in top:
            a_ = T[k]; res["reality_check"]["top10"].append({"param": dict(zip(["trasatura", "q", "dir", "H", "sesiune"], k)), "n": int(len(a_)), "R_net": round(float(a_[:, 1].mean()), 4), "R_brut": round(float(a_[:, 3].mean()), 4), "t": round(float(tobs[k]), 2),
                                                                         "per_instrument": {s: seg(a_[a_[:, 2] == i, 1]) for i, s in enumerate(S)}})
        # (3) pentru cea mai buna: perioade si FTMO
        k = top[0]; a_ = T[k]; qs = np.linspace(t0, t1, 7)
        res["cea_mai_buna"] = {"param": dict(zip(["trasatura", "q", "dir", "H", "sesiune"], k)), "pe_sase_perioade": [seg(a_[(a_[:, 0] >= qs[i]) & (a_[:, 0] < qs[i + 1] + (1 if i == 5 else 0)), 1]) for i in range(6)]}
        ds = daily_series(a_, t0, t1, RISK); res["cea_mai_buna"]["sharpe"] = round(float(ds.mean() / ds.std() * np.sqrt(252)), 2) if ds.std() > 0 else None
        res["cea_mai_buna"]["ftmo"] = {"vol%d%%" % int(vt * 100): ftmo_block(ds, vt, np.random.default_rng(22)) for vt in (0.04, 0.07, 0.10)} if ds.std() > 0 else None
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("%d variante. Walk-forward = performanta reala OOS a procesului de cautare. Reality check: daca t-ul celei mai bune nu depaseste p95 al maximului sub 'fara avantaj', cea mai buna variante e compatibila cu norocul." % len(grid))
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
