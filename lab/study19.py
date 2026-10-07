# Studiu 19: ORDER FLOW REAL pe orizonturi LUNGI (1-16 ore), unde costul de spread e mic fata de miscare. Date: Databento 6E/6B/NKD, bare 1 min din tranzactii cu agresor.
# Doua familii: (A) trasaturi de flux/pret pe ferestre 1-24 h (dezechilibru agresor, miscare, divergenta, abatere de la VWAP, volum) cu continuare/reversie;
# (B) efect de ORA din zi: intrare la o ora fixa (UTC), directie fixa, tinere H.  Intrari doar la ore fixe (minutul 0), SL 1.5 sigma_H, iesire la timp.
# Raportam si R BRUT (fara cost) ca sa vedem daca exista informatie, apoi R NET (cost CFD FTMO masurat x1.75).
# Protocol: 50% train / 25% validare / 25% lockbox; praguri din TRAIN; top pe train -> validare -> lockbox (max 3). Parametri comuni pe 3 instrumente.
import os, json, time, argparse, itertools
import numpy as np
from .nb import njit
from .study12 import cost_frac
from .study11 import ftmo_block
from .study17 import load, tstat, seg, INST, RISK
from .study18 import bars1m, winidx

VER = 1
SESS = {"all": (0.0, 24.0), "asia": (0.0, 7.0), "eu": (7.0, 13.0), "us": (13.0, 21.0)}
WIN = (60, 120, 240, 480, 1440)
HOLD = (60, 120, 240, 480, 960)


def features(B):
    t, c, v, b, s, iid = B["t"], B["c"], B["v"], B["b"], B["s"], B["iid"]
    n = len(t); F = {}; idx = np.arange(n)
    cs_d = np.concatenate([[0], np.cumsum(b - s)]); cs_v = np.concatenate([[0], np.cumsum(v)]); cs_cv = np.concatenate([[0], np.cumsum(c * v)])
    dc = np.diff(c, prepend=c[0]); dc[1:][iid[1:] != iid[:-1]] = 0.0
    cs2 = np.concatenate([[0], np.cumsum(dc * dc)])
    sig = np.full(n, np.nan)
    if n > 121: sig[120:] = np.sqrt(np.maximum((cs2[121:] - cs2[1:n - 119]) / 120.0, 1e-18))
    for N in WIN:
        j = winidx(t, N)
        dl = cs_d[idx + 1] - cs_d[j]; vl = cs_v[idx + 1] - cs_v[j]
        imb = np.where(vl > 100, dl / np.maximum(vl, 1e-9), np.nan)
        jb = j - 1; jb2 = np.where(jb >= 0, jb, 0)
        ok = (jb >= 0) & (iid[jb2] == iid) & ((t - t[jb2]) <= 1.5 * N * 60 + 600)
        ret = np.where(ok, c - c[jb2], np.nan) / (sig * np.sqrt(N))
        vw = np.where(vl > 0, (cs_cv[idx + 1] - cs_cv[j]) / np.maximum(vl, 1e-9), np.nan)
        F["imb%d" % N] = imb
        F["ret%d" % N] = ret
        F["div%d" % N] = np.where(np.isfinite(ret) & np.isfinite(imb), -np.sign(ret) * imb * np.minimum(np.abs(ret), 4.0), np.nan)
        F["vwap%d" % N] = np.where(ok, (c - vw) / (sig * np.sqrt(N)), np.nan)
        jv = winidx(t, 5 * N); vprev = (cs_v[idx + 1] - cs_v[jv]) / np.maximum(5.0 * N, 1.0); vcur = vl / float(N)
        F["volx%d" % N] = np.where(vprev > 0, vcur / np.maximum(vprev, 1e-9), np.nan)
    return F, sig


@njit(cache=True)
def sim(t, o, h, l, c, iid, sig, mask, side, H, stopm, cost, out):
    n = len(t); nt = 0; free = -1.0; i = 0
    while i < n - 2:
        if mask[i] and t[i] >= free and t[i + 1] - t[i] <= 120.0 and sig[i] > 0:
            e = o[i + 1]; sd = stopm * sig[i] * np.sqrt(H)
            if sd >= 4.0 * cost:
                tex = t[i + 1] + H * 60.0; px = np.nan; j = i + 1; ok = True
                while j < n:
                    if iid[j] != iid[i]: ok = False; break
                    if t[j] - t[i + 1] > 1.5 * H * 60.0 + 600.0: ok = False; break
                    if side > 0 and l[j] <= e - sd: px = e - sd; break
                    if side < 0 and h[j] >= e + sd: px = e + sd; break
                    if t[j] >= tex - 60.0: px = c[j]; break
                    j += 1
                if ok and px == px and nt < out.shape[0]:
                    g = side * (px - e) / sd
                    out[nt, 0] = t[i + 1]; out[nt, 1] = g - cost / sd; out[nt, 2] = g; nt += 1
                    free = t[j] if j < n else t[n - 1]
        i += 1
    return nt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt19"); os.makedirs(out, exist_ok=True)

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
            B = bars1m(d); F, sig = features(B); px = float(np.median(B["c"])); cf, meas = cost_frac(spd, INST[s]["cfd"], px)
            S[s] = {"B": B, "F": F, "sig": sig, "cost": cf * px, "hr": (B["t"] % 86400.0) / 3600.0, "min0": (B["t"] % 3600.0) == 0.0}
        if not S: raise RuntimeError("nu exista date")
        t0 = max(float(v["B"]["t"][0]) for v in S.values()); t1 = min(float(v["B"]["t"][-1]) for v in S.values())
        c1 = t0 + 0.5 * (t1 - t0); c2 = t0 + 0.75 * (t1 - t0)
        fmt = lambda x: time.strftime("%Y-%m-%d", time.gmtime(x))
        res["split"] = {"train": [fmt(t0), fmt(c1)], "val": [fmt(c1), fmt(c2)], "lock": [fmt(c2), fmt(t1)]}
        save(res)
        feats = list(next(iter(S.values()))["F"].keys())
        thr = {}
        for s, v in S.items():
            tr = v["B"]["t"] < c1
            for f in feats:
                x = np.abs(v["F"][f][tr & v["min0"]]); x = x[np.isfinite(x)]
                thr[(s, f)] = {q: float(np.quantile(x, q)) if len(x) > 500 else np.inf for q in (0.80, 0.93, 0.98)}
        # familia A: trasatura x cuantila x directie x H x sesiune ; familia B: 'ora' fixa x directie x H
        gridA = [("A", f, q, dr, H, se) for f in feats for q in (0.80, 0.93, 0.98) for dr in (1, -1) for H in HOLD for se in ("all", "asia", "eu", "us")]
        gridB = [("B", "ora%02d" % hh, 0, dr, H, hh) for hh in range(24) for dr in (1, -1) for H in HOLD]
        grid = gridA + gridB; res["variante"] = len(grid); save(res)
        buf = np.zeros((40000, 3)); trades = {}
        for gi, (fam, f, q, dr, H, se) in enumerate(grid):
            allr = []
            for si, (s, v) in enumerate(S.items()):
                if fam == "A":
                    x = v["F"][f]
                    if f.startswith("volx"):
                        wn = f[4:]; r_ = v["F"]["ret" + wn]; m = np.isfinite(x) & (x >= max(thr[(s, f)][q], 1.0)) & np.isfinite(r_) & (r_ != 0); side_arr = np.sign(np.nan_to_num(r_)) * dr
                    else:
                        m = np.isfinite(x) & (np.abs(np.nan_to_num(x)) >= thr[(s, f)][q]); side_arr = np.sign(np.nan_to_num(x)) * dr
                    lo, hi = SESS[se]; m = m & v["min0"] & (v["hr"] >= lo) & (v["hr"] < hi)
                else:
                    m = v["min0"] & (np.abs(v["hr"] - se) < 0.01); side_arr = np.full(len(m), float(dr))
                for sd_ in (1, -1):
                    ms = m & (side_arr == sd_)
                    if not ms.any(): continue
                    B = v["B"]; nt = sim(B["t"], B["o"], B["h"], B["l"], B["c"], B["iid"], v["sig"], ms, sd_, H, 1.5, v["cost"], buf)
                    if nt: allr.append(np.column_stack([buf[:nt, 0], buf[:nt, 1], np.full(nt, si), buf[:nt, 2]]))
            trades[grid[gi]] = np.vstack(allr) if allr else np.zeros((0, 4))
            if gi % 100 == 0: res["progres"] = "%d/%d" % (gi + 1, len(grid)); save(res)

        def parts(a_, col=1): return a_[a_[:, 0] < c1, col], a_[(a_[:, 0] >= c1) & (a_[:, 0] < c2), col], a_[a_[:, 0] >= c2, col]
        rows = []
        for k, a_ in trades.items():
            tr_, va_, lk_ = parts(a_)
            if len(tr_) >= 150: rows.append((tstat(tr_), k, a_))
        rows.sort(key=lambda r: -r[0])
        res["n_variante_cu_trade_train"] = len(rows)
        for fam in ("A", "B"):
            rr = [(tt, k, a_) for tt, k, a_ in rows if k[0] == fam]
            g_tr = [parts(a_, 3)[0].mean() for _, _, a_ in rr]; n_tr = [parts(a_)[0].mean() for _, _, a_ in rr]
            g_va = [parts(a_, 3)[1].mean() for _, _, a_ in rr if len(parts(a_)[1]) >= 50]; n_va = [parts(a_)[1].mean() for _, _, a_ in rr if len(parts(a_)[1]) >= 50]
            res["familia_" + fam] = {"variante": len(rr), "R_brut_mediu_train": round(float(np.mean(g_tr)), 4) if g_tr else None, "R_net_mediu_train": round(float(np.mean(n_tr)), 4) if n_tr else None,
                                     "R_brut_mediu_val": round(float(np.mean(g_va)), 4) if g_va else None, "R_net_mediu_val": round(float(np.mean(n_va)), 4) if n_va else None,
                                     "train_t>=2": int(sum(1 for tt, _, _ in rr if tt >= 2.0)), "train_t>=2_si_val_net_t>=1.5": int(sum(1 for tt, k, a_ in rr if tt >= 2.0 and len(parts(a_)[1]) >= 50 and parts(a_)[1].mean() > 0 and tstat(parts(a_)[1]) >= 1.5))}
        # corelatie brut train vs brut val intre variante (daca exista structura reala, variantele bune in train raman bune in val)
        cc = [(parts(a_, 3)[0].mean(), parts(a_, 3)[1].mean()) for _, _, a_ in rows if len(parts(a_)[1]) >= 50]
        if len(cc) > 20: res["corel_R_brut_train_vs_val"] = round(float(np.corrcoef(np.array(cc).T)[0, 1]), 3)
        names = ["fam", "trasatura", "cuantila", "dir(+1 continuare,-1 reversie)", "H_min", "sesiune/ora"]
        res["top20_train"] = []; passers = []
        for tt, k, a_ in rows[:20]:
            tr_, va_, lk_ = parts(a_)
            e = {"param": dict(zip(names, k)), "train": seg(tr_), "val": seg(va_), "brut_train": round(float(parts(a_, 3)[0].mean()), 4), "brut_val": round(float(parts(a_, 3)[1].mean()), 4) if len(va_) else None}
            res["top20_train"].append(e)
            if len(va_) >= 50 and va_.mean() > 0 and tstat(va_) >= 1.5: passers.append((k, a_, e))
        res["trecut_validarea"] = len(passers); res["lockbox"] = []
        for k, a_, e in passers[:3]:
            tr_, va_, lk_ = parts(a_); r = {"param": e["param"], "train": e["train"], "val": e["val"], "lock": seg(lk_), "brut_lock": round(float(parts(a_, 3)[2].mean()), 4) if len(lk_) else None}
            per = {}
            for ii, s in enumerate(S):
                x = a_[a_[:, 2] == ii]
                per[s] = {"train": seg(x[x[:, 0] < c1, 1]), "val": seg(x[(x[:, 0] >= c1) & (x[:, 0] < c2), 1]), "lock": seg(x[x[:, 0] >= c2, 1])}
            r["per_instrument"] = per
            allv = a_[a_[:, 0] >= c1]
            days = np.arange(int(c1 // 86400), int(t1 // 86400) + 1); dd = np.zeros(len(days))
            for tt_, rn, _, _ in allv: dd[int(tt_ // 86400) - days[0]] += rn * RISK
            wk_ = np.array([(int(dv) + 4) % 7 < 5 for dv in days]); dd = dd[wk_]
            if len(dd) > 60 and dd.std() > 0:
                rng = np.random.default_rng(19); r["ftmo_val+lock"] = ftmo_block(dd, float(dd.std() * np.sqrt(252)), rng)
            res["lockbox"].append(r)
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("%d variante. R brut = fara cost (masoara informatia), R net = dupa cost CFD FTMO masurat x1.75. SL 1.5 sigma_H. Daca R brut mediu e ~0 in validare, nu exista informatie; "
                       "daca e pozitiv dar net negativ, problema e costul. 'corel_R_brut_train_vs_val' ~0 = fara structura stabila." % len(grid))
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
