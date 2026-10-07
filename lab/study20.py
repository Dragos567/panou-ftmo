# Studiu 20: CONFIRMARE pe lockbox a 4 ipoteze alese in avans din studiul 19 (flux agresor pe 4 h, continuare, tinere 8-16 h). Lockbox = ultimele 25% din date, neatins la selectie.
# Nu se cauta nimic nou: 4 variante fixe. Raportam train/val/lock net+brut, pe instrument si pe trimestru, plus FTMO bootstrap pe toata perioada.
import os, json, time, argparse
import numpy as np
from .study12 import cost_frac
from .study11 import ftmo_block
from .study17 import load, tstat, seg, INST, RISK
from .study18 import bars1m
from .study19 import features, sim, SESS

VER = 1
HYP = [("imb240", 0.93, 1, 480, "eu"), ("imb240", 0.93, 1, 960, "eu"), ("div240", 0.98, 1, 480, "asia"), ("ret480", 0.93, -1, 480, "asia")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt20"); os.makedirs(out, exist_ok=True)

    def save(d):
        tmp = os.path.join(out, "result.json.tmp"); open(tmp, "w").write(json.dumps(d).replace("NaN", "null")); os.replace(tmp, os.path.join(out, "result.json"))
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
        c1 = t0 + 0.5 * (t1 - t0); c2 = t0 + 0.75 * (t1 - t0)
        thr = {}
        for s, v in S.items():
            tr = v["B"]["t"] < c1
            for f in {h[0] for h in HYP}:
                x = np.abs(v["F"][f][tr & v["min0"]]); x = x[np.isfinite(x)]
                thr[(s, f)] = {q: float(np.quantile(x, q)) for q in (0.93, 0.98)}
        buf = np.zeros((40000, 3)); res["hipoteze"] = []
        for (f, q, dr, H, se) in HYP:
            allr = []
            for si, (s, v) in enumerate(S.items()):
                x = v["F"][f]; m = np.isfinite(x) & (np.abs(np.nan_to_num(x)) >= thr[(s, f)][q]); side_arr = np.sign(np.nan_to_num(x)) * dr
                lo, hi = SESS[se]; m = m & v["min0"] & (v["hr"] >= lo) & (v["hr"] < hi)
                for sd_ in (1, -1):
                    ms = m & (side_arr == sd_)
                    if not ms.any(): continue
                    B = v["B"]; nt = sim(B["t"], B["o"], B["h"], B["l"], B["c"], B["iid"], v["sig"], ms, sd_, H, 1.5, v["cost"], buf)
                    if nt: allr.append(np.column_stack([buf[:nt, 0], buf[:nt, 1], np.full(nt, si), buf[:nt, 2]]))
            a_ = np.vstack(allr); a_ = a_[np.argsort(a_[:, 0])]
            P = lambda lo_, hi_, col=1: a_[(a_[:, 0] >= lo_) & (a_[:, 0] < hi_), col]
            r = {"param": {"trasatura": f, "cuantila": q, "dir": dr, "H_min": H, "sesiune": se}, "train": seg(P(0, c1)), "val": seg(P(c1, c2)), "lock": seg(P(c2, 9e12)), "toate": seg(P(0, 9e12)),
                 "brut_toate": round(float(P(0, 9e12, 3).mean()), 4), "brut_lock": round(float(P(c2, 9e12, 3).mean()), 4)}
            r["per_instrument"] = {s: seg(a_[a_[:, 2] == i, 1]) for i, s in enumerate(S)}
            qs = np.linspace(t0, t1, 7); r["per_sase_perioade"] = [seg(P(qs[i], qs[i + 1] + (1 if i == 5 else 0))) for i in range(6)]
            days = np.arange(int(t0 // 86400), int(t1 // 86400) + 1); dd = np.zeros(len(days))
            for tt_, rn, _, _ in a_: dd[int(tt_ // 86400) - days[0]] += rn * RISK
            wk_ = np.array([(int(dv) + 4) % 7 < 5 for dv in days]); dd = dd[wk_]
            r["sharpe_zilnic_anual"] = round(float(dd.mean() / dd.std() * np.sqrt(252)), 2) if dd.std() > 0 else None
            r["profit_total_pct_cont"] = round(float(dd.sum() * 100), 2)
            if dd.std() > 0: r["ftmo_toata_perioada"] = ftmo_block(dd, float(dd.std() * np.sqrt(252)), np.random.default_rng(20))
            res["hipoteze"].append(r)
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = "4 ipoteze alese in avans (din top-ul studiului 19, unde validarea nu trecuse pragul). Lockbox neatins la selectie. Daca lockbox nu confirma, nu exista nimic."
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
