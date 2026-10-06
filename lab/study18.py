# Studiu 18: ORDER FLOW REAL (Databento 1s -> bare de 1 minut). Cautare larga de semnale simple pe 6E, 6B, NKD, cu protocol strict.
# Trasaturi (la inchiderea barei de 1 min): dezechilibru agresor (delta/volum) pe N min, miscarea pretului pe N min (normata cu volatilitatea), divergenta (flux contra miscarii = absorbtie),
# explozie de volum (z fata de media zilei). Semnal = trasatura peste un prag (cuantila din TRAIN), directie +1 (continuare) sau -1 (reversie), tinere H minute, SL la 1.5 sigma_H.
# Intrare la deschiderea barei urmatoare; cost = spread CFD FTMO masurat x1.75 (+comision) in unitati de pret; fara suprapunere (un singur trade deschis pe instrument).
# Protocol: 50% train / 25% validare / 25% lockbox in timp; parametri comuni pe 3 instrumente; top pe train -> validare -> lockbox (max 3, o singura privire).
# Metrici in R (R = distanta SL). FTMO: bootstrap pe zile (ftmo_block) pe validare+lockbox, risc 0.4%/trade.
import os, json, time, argparse, itertools
import numpy as np
from .nb import njit
from .study12 import cost_frac
from .study11 import ftmo_block
from .study17 import load, tstat, seg, INST, RISK

VER = 1
SESS = {"all": (0.0, 24.0), "eu": (7.0, 16.0), "us": (13.5, 20.0), "asia": (0.0, 7.0)}


def bars1m(d):
    k = (d["t"] // 60).astype(np.int64)
    st = np.concatenate([[0], np.nonzero(np.diff(k))[0] + 1])
    en = np.concatenate([st[1:], [len(k)]]) - 1
    return {"t": (k[st] * 60).astype(np.float64), "o": d["o"][st], "h": np.maximum.reduceat(d["h"], st), "l": np.minimum.reduceat(d["l"], st), "c": d["c"][en],
            "v": np.add.reduceat(d["v"], st), "b": np.add.reduceat(d["b"], st), "s": np.add.reduceat(d["s"], st), "iid": d["iid"][en]}


def winidx(t, N):
    return np.searchsorted(t, t - N * 60.0, side="right")        # primul bar din fereastra (t-N*60, t]


def features(B):
    t, c, v, b, s, iid = B["t"], B["c"], B["v"], B["b"], B["s"], B["iid"]
    n = len(t); F = {}
    cs_d = np.concatenate([[0], np.cumsum(b - s)]); cs_v = np.concatenate([[0], np.cumsum(v)])
    dc = np.diff(c, prepend=c[0]); dc[1:][iid[1:] != iid[:-1]] = 0.0
    cs2 = np.concatenate([[0], np.cumsum(dc * dc)])
    sig = np.full(n, np.nan)
    if n > 121: sig[120:] = np.sqrt(np.maximum((cs2[121:] - cs2[1:n - 119]) / 120.0, 1e-18))
    idx = np.arange(n)
    for N in (5, 15, 30, 60):
        j = winidx(t, N)                                         # fereastra [j..i]
        dl = cs_d[idx + 1] - cs_d[j]; vl = cs_v[idx + 1] - cs_v[j]
        imb = np.where(vl > 0, dl / np.maximum(vl, 1e-9), 0.0)
        jb = j - 1; ok = (jb >= 0)
        jb2 = np.where(ok, jb, 0)
        ok &= (iid[jb2] == iid) & ((t - t[jb2]) <= 2.0 * N * 60 + 60)
        ret = np.where(ok, c - c[jb2], np.nan) / (sig * np.sqrt(N))
        F["imb%d" % N] = np.where(vl > 20, imb, np.nan)
        F["ret%d" % N] = ret
        F["div%d" % N] = np.where(np.isfinite(ret), -np.sign(ret) * imb * np.minimum(np.abs(ret), 4.0), np.nan)        # flux contra miscarii
    # explozie de volum: volumul pe 5 min fata de media ultimelor 600 min
    j5 = winidx(t, 5); v5 = (cs_v[idx + 1] - cs_v[j5]) / 5.0
    j600 = winidx(t, 600); v600 = (cs_v[idx + 1] - cs_v[j600]) / np.maximum(idx + 1 - j600, 1)
    zv = np.where(v600 > 0, v5 / np.maximum(v600, 1e-9), np.nan)
    F["volx"] = zv
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
                    if t[j] - t[i + 1] > 2.0 * H * 60.0 + 120.0: ok = False; break
                    if side > 0 and l[j] <= e - sd: px = e - sd; break
                    if side < 0 and h[j] >= e + sd: px = e + sd; break
                    if t[j] >= tex - 60.0: px = c[j]; break
                    j += 1
                if ok and px == px and nt < out.shape[0]:
                    out[nt, 0] = t[i + 1]; out[nt, 1] = side * (px - e) / sd - cost / sd; out[nt, 2] = t[j] if j < n else t[n - 1]; nt += 1
                    free = t[j] if j < n else t[n - 1]
        i += 1
    return nt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt18"); os.makedirs(out, exist_ok=True)

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
            S[s] = {"B": B, "F": F, "sig": sig, "cost": cf * px, "meas": bool(meas)}
        if not S: raise RuntimeError("nu exista date")
        t0 = max(float(v["B"]["t"][0]) for v in S.values()); t1 = min(float(v["B"]["t"][-1]) for v in S.values())
        c1 = t0 + 0.5 * (t1 - t0); c2 = t0 + 0.75 * (t1 - t0)
        fmt = lambda x: time.strftime("%Y-%m-%d", time.gmtime(x))
        res["split"] = {"train": [fmt(t0), fmt(c1)], "val": [fmt(c1), fmt(c2)], "lock": [fmt(c2), fmt(t1)]}
        res["bare_1min"] = {s: int(len(v["B"]["t"])) for s, v in S.items()}
        res["cost_in_sigma_1min"] = {s: round(float(v["cost"] / np.nanmedian(v["sig"])), 2) for s, v in S.items()}
        save(res)
        # praguri = cuantile pe TRAIN, per instrument si trasatura
        feats = list(next(iter(S.values()))["F"].keys())
        thr = {}
        for s, v in S.items():
            tr = v["B"]["t"] < c1
            for f in feats:
                x = np.abs(v["F"][f][tr]); x = x[np.isfinite(x)]
                thr[(s, f)] = {q: float(np.quantile(x, q)) if len(x) > 1000 else np.inf for q in (0.90, 0.97, 0.995)}
        grid = list(itertools.product(feats, (0.90, 0.97, 0.995), (1, -1), (5, 15, 30, 60), ("all", "eu", "us", "asia")))
        # 'vol*' sunt pozitive (explozie de volum) -> directia = a ultimei miscari de 5 min (ret5), nu un semn propriu
        res["variante"] = len(grid); save(res)
        buf = np.zeros((60000, 3)); trades = {}
        hrs = {s: ((v["B"]["t"] % 86400.0) / 3600.0) for s, v in S.items()}
        for gi, (f, q, dr, H, se) in enumerate(grid):
            allr = []
            for s, v in S.items():
                x = v["F"][f]
                if f == "volx":
                    r5 = v["F"]["ret5"]; sgn = np.sign(r5); m = (np.abs(np.nan_to_num(x)) >= thr[(s, f)][q]) & np.isfinite(r5) & (r5 != 0)
                    side_arr = sgn * dr
                else:
                    m = np.isfinite(x) & (np.abs(np.nan_to_num(x)) >= thr[(s, f)][q]); side_arr = np.sign(np.nan_to_num(x)) * dr
                lo, hi = SESS[se]; m &= (hrs[s] >= lo) & (hrs[s] < hi)
                for sd_ in (1, -1):
                    ms = m & (side_arr == sd_)
                    if ms.sum() == 0: continue
                    # un singur trade deschis pe instrument: rulam separat pe long si short si aplicam cooldown comun dupa aceea (aproximare: pe fiecare parte separat, cooldown propriu)
                    B = v["B"]; nt = sim(B["t"], B["o"], B["h"], B["l"], B["c"], B["iid"], v["sig"], ms, sd_, H, 1.5, v["cost"], buf)
                    if nt: allr.append(np.column_stack([buf[:nt, 0], buf[:nt, 1], np.full(nt, list(S).index(s))]))
            trades[grid[gi]] = np.vstack(allr) if allr else np.zeros((0, 3))
            if gi % 100 == 0: res["progres"] = "%d/%d" % (gi + 1, len(grid)); save(res)

        def parts(a_): return a_[a_[:, 0] < c1, 1], a_[(a_[:, 0] >= c1) & (a_[:, 0] < c2), 1], a_[a_[:, 0] >= c2, 1]
        rows = []
        for k, a_ in trades.items():
            tr_, va_, lk_ = parts(a_)
            if len(tr_) >= 200: rows.append((tstat(tr_), k, a_))
        rows.sort(key=lambda r: -r[0])
        res["n_variante_cu_trade_train"] = len(rows)
        # cate variante arata bine in train DAR si in validare (de comparat cu ce da norocul)
        n_tr_ok = n_both = 0; mm = []
        for tt, k, a_ in rows:
            tr_, va_, lk_ = parts(a_)
            if tt >= 2.0:
                n_tr_ok += 1
                if len(va_) >= 50 and va_.mean() > 0 and tstat(va_) >= 1.5: n_both += 1
            mm.append(tr_.mean())
        res["train_t>=2"] = n_tr_ok; res["train_t>=2_si_val_t>=1.5"] = n_both
        res["medie_R_net_train_toate"] = round(float(np.mean(mm)), 4) if mm else None
        res["medie_R_net_val_toate"] = round(float(np.mean([parts(a_)[1].mean() for _, _, a_ in rows if len(parts(a_)[1]) >= 50])), 4) if rows else None
        names = ["trasatura", "cuantila", "dir(+1 continuare,-1 reversie)", "H_min", "sesiune"]
        res["top15_train"] = []; passers = []
        for tt, k, a_ in rows[:15]:
            tr_, va_, lk_ = parts(a_)
            e = {"param": dict(zip(names, k)), "train": seg(tr_), "val": seg(va_)}
            res["top15_train"].append(e)
            if len(va_) >= 50 and va_.mean() > 0 and tstat(va_) >= 1.5: passers.append((k, a_, e))
        res["trecut_validarea"] = len(passers)
        res["lockbox"] = []
        for k, a_, e in passers[:3]:
            tr_, va_, lk_ = parts(a_); r = {"param": e["param"], "train": e["train"], "val": e["val"], "lock": seg(lk_)}
            per = {}
            for ii, s in enumerate(S):
                m_ = a_[:, 2] == ii; x = a_[m_]
                per[s] = {"train": seg(x[x[:, 0] < c1, 1]), "val": seg(x[(x[:, 0] >= c1) & (x[:, 0] < c2), 1]), "lock": seg(x[x[:, 0] >= c2, 1])}
            r["per_instrument"] = per
            allv = a_[a_[:, 0] >= c1]
            days = np.arange(int(c1 // 86400), int(t1 // 86400) + 1); dd = np.zeros(len(days))
            for tt_, rn, _ in allv: dd[int(tt_ // 86400) - days[0]] += rn * RISK
            wk_ = np.array([(int(dv) + 4) % 7 < 5 for dv in days]); dd = dd[wk_]
            if len(dd) > 60 and dd.std() > 0:
                rng = np.random.default_rng(18); r["ftmo_val+lock"] = ftmo_block(dd, float(dd.std() * np.sqrt(252)), rng)
            res["lockbox"].append(r)
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("Bare 1 min din tranzactii CME reale (agresor). Costuri CFD FTMO masurate x1.75. SL 1.5 sigma_H, iesire la timp H. %d variante: la atatea teste, t~3 in train apare din noroc; "
                       "conteaza validarea si lockbox-ul, iar 'train_t>=2_si_val_t>=1.5' trebuie comparat cu ce da norocul (~15-20%% din cele cu train t>=2 daca nu exista nimic real)." % len(grid))
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
