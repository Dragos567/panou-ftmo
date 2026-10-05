# Studiu 12: portofoliu = (A) trend zilnic pe pana la ~40 de piete FTMO (FX, metale, energie, agricole, indici, crypto) + (B) sleeve-uri intraday FX alese DOAR pe train (ferestre orare).
# Obiectiv: Sharpe net al portofoliului combinat >= ~0.8 pe train/val/lock; apoi probabilitatea FTMO cu bootstrap pe blocuri.
# Cost: 1.75 x spread (mediana peste ore din spread.json cand exista >= 3 esantioane, altfel ipoteza pe clasa de activ, marcat) + comision FX. FARA swap.
import os, json, time, glob, itertools, argparse
import numpy as np
from .study8 import load_tf, agg
from .data import DT
from .study11 import seg_stats, ftmo_block, ASSUMED as ASSUMED_PTS, COMM

VER = 1
OLD = ["EURUSD", "GBPUSD", "USDJPY", "GOLD", "US100", "US500", "US30", "UK100", "DAX", "NIKKEI"]
NEW = ["AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "SILVER", "COPPER", "PLAT", "PALL", "USOIL", "UKOIL", "NATGAS", "SOY", "WHEAT", "CORN", "SUGAR", "COFFEE", "COCOA", "COTTON",
       "EU50", "FRA40", "AUS200", "HK50", "SPN35", "N25", "US2K", "DXY", "BTC", "ETH"]
BPS = {"fx": 1.5, "metal": 6, "energy": 10, "agri": 20, "idx": 4, "crypto": 20}
CLS = {**{s: "fx" for s in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "DXY")}, **{s: "metal" for s in ("GOLD", "SILVER", "COPPER", "PLAT", "PALL")},
       **{s: "energy" for s in ("USOIL", "UKOIL", "NATGAS")}, **{s: "agri" for s in ("SOY", "WHEAT", "CORN", "SUGAR", "COFFEE", "COCOA", "COTTON")}, **{s: "crypto" for s in ("BTC", "ETH")}}


def daily_series(data, sym):
    if sym in OLD:
        m, _ = load_tf(data, sym); t = m["t"].astype(np.int64)
        d = agg(t, m["o"].astype(np.float64), m["h"].astype(np.float64), m["l"].astype(np.float64), m["c"].astype(np.float64), m["v"].astype(np.float64), 86400)
        return (d["t"] // 86400).astype(np.int64), d["c"]
    ps = sorted(glob.glob(os.path.join(data, "hist", "%s_1d" % sym, "p_*.bin")), key=lambda p: int(os.path.basename(p)[2:-4]))
    arrs = [np.fromfile(p, dtype=DT, count=os.path.getsize(p) // DT.itemsize) for p in ps]; arrs = [x for x in arrs if len(x)]
    if not arrs: raise RuntimeError("fara istoric 1d")
    a = np.concatenate(arrs); a = a[np.argsort(a["t"], kind="stable")]; _, ix = np.unique(a["t"], return_index=True); a = a[ix]
    k = (a["t"].astype(np.int64) + 3 * 3600) // 86400; _, ix = np.unique(k, return_index=True)
    return k[ix], a["c"][ix].astype(np.float64)


def cost_frac(spd, sym, px):
    sp = spd.get(sym, {}); vals = [v["mean"] for v in sp.values() if v["n"] >= (30 if sym in OLD else 3)]
    if vals: return 1.75 * float(np.median(vals)) / px + COMM.get(sym, 0.0) / px, True
    if sym in ASSUMED_PTS: return 1.75 * ASSUMED_PTS[sym] / px, False
    return 1.75 * BPS.get(CLS.get(sym, "idx"), 5) * 1e-4, False


def trend_sleeve(data, spd, syms_ok):
    D = {}; miss = {}
    for s in syms_ok:
        try: D[s] = daily_series(data, s)
        except Exception as e: miss[s] = repr(e)[:60]
    syms = [s for s in D if len(D[s][0]) > 400]; days = np.unique(np.concatenate([D[s][0] for s in syms])); nd = len(days); pos = {int(d): i for i, d in enumerate(days)}
    P = np.full((nd, len(syms)), np.nan)
    for j, s in enumerate(syms):
        for d, c in zip(*D[s]): P[pos[int(d)], j] = c
    R = np.zeros((nd, len(syms))); act = np.zeros((nd, len(syms)), bool); lastp = np.full(len(syms), np.nan)
    for i in range(nd):
        for j in range(len(syms)):
            if np.isfinite(P[i, j]):
                if np.isfinite(lastp[j]) and lastp[j] > 0: R[i, j] = P[i, j] / lastp[j] - 1; act[i, j] = True
                lastp[j] = P[i, j]
    V = np.full((nd, len(syms)), np.nan); lam = 1 - 2 / 61.0
    for j in range(len(syms)):
        v = np.nan
        for i in range(nd):
            if act[i, j]: v = R[i, j] ** 2 if not np.isfinite(v) else lam * v + (1 - lam) * R[i, j] ** 2
            V[i, j] = np.sqrt(v) if np.isfinite(v) else np.nan
    lp = np.log(np.where(P > 0, P, np.nan))
    for j in range(len(syms)):
        for i in range(1, nd):
            if not np.isfinite(lp[i, j]): lp[i, j] = lp[i - 1, j]
    flag = {}; cf = {}
    for j, s in enumerate(syms):
        px = float(np.nanmedian(P[:, j])); cf[s], flag[s] = cost_frac(spd, s, px)
    wk = (days + 3) // 7; reb = np.r_[False, wk[1:] != wk[:-1]]; LB = (60, 120, 250); i0 = 260
    out = {}
    for name, lbs in (("trend_60_120_250", LB), ("trend_120_250", (120, 250))):
        W = np.zeros(len(syms)); net = np.zeros(nd); turn = np.zeros(nd); nact = np.zeros(nd)
        for i in range(i0, nd):
            if reb[i]:
                sg = np.full(len(syms), np.nan)
                for j in range(len(syms)):
                    if np.isfinite(V[i - 1, j]) and V[i - 1, j] > 0 and i - 1 - max(lbs) >= 0 and act[max(0, i - 15):i, j].any():
                        v = [np.sign(lp[i - 1, j] - lp[i - 1 - L, j]) for L in lbs]
                        if all(np.isfinite(v)): sg[j] = float(np.mean(v))
                n_act = int(np.isfinite(sg).sum())
                for j in range(len(syms)):
                    wn = 0.0 if not np.isfinite(sg[j]) or n_act == 0 else sg[j] * min(4.0, 0.10 / (V[i - 1, j] * np.sqrt(252))) / n_act
                    net[i] -= abs(wn - W[j]) * cf[syms[j]]; turn[i] += abs(wn - W[j]); W[j] = wn
            net[i] += float((W * R[i]).sum()); nact[i] = float((W != 0).sum())
        out[name] = net
    return days, out, {"piete": syms, "lipsa": miss, "cost_ipoteza": [s for s in syms if not flag[s]], "n_activ_medie": round(float(nact[i0:].mean()), 1)}


def sleeves(data, spd, sym, days):
    m, _ = load_tf(data, sym); t = m["t"].astype(np.int64)
    m = agg(t, m["o"].astype(np.float64), m["h"].astype(np.float64), m["l"].astype(np.float64), m["c"].astype(np.float64), m["v"].astype(np.float64), 900)
    t, O, C = m["t"], m["o"], m["c"]; sp = spd.get(sym, {}); meas = {int(k): v["mean"] for k, v in sp.items() if v["n"] >= 30}
    fallback = float(np.median(list(meas.values()))) if meas else 0.0
    nd = len(days); res = {}
    for h in range(24):
        for k in (1, 2, 3):
            if h + k > 24: continue
            g = np.zeros(nd); ok = np.zeros(nd, bool)
            for di, d in enumerate(days):
                b = int(d) * 86400; i = int(np.searchsorted(t, b + h * 3600)); j = int(np.searchsorted(t, b + (h + k) * 3600))
                if i >= len(t) - 1 or j >= len(t) or t[i] != b + h * 3600 or t[j] != b + (h + k) * 3600: continue
                g[di] = O[j] / O[i] - 1; ok[di] = True
            cost = (1.75 * meas.get(h, fallback) + COMM.get(sym, 0.0)) / np.where(ok, np.maximum(1e-9, 1.0), 1.0) / float(np.nanmedian(O))
            res[(h, k)] = (g, ok, cost)
    return res


def tstat(x):
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 30 and x.std() > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt12"); os.makedirs(out, exist_ok=True)

    def save(d):
        open(os.path.join(out, "result.json.tmp"), "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time())}; save(res)
    try:
        rng = np.random.default_rng(12)
        have = 0
        for sy in NEW:
            try:
                if len(daily_series(a.data, sy)[0]) >= 1000: have += 1
            except Exception: pass
        if have < int(os.environ.get("S12_MIN", "20")) and not os.path.exists(os.path.join(out, "force")):
            res["state"] = "asteapta"; res["motiv"] = "istoric zilnic descarcat pentru %d/%d piete noi (minim 20)" % (have, len(NEW)); save(res); return
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        # ---------------- A: trend ----------------
        days, tr, info = trend_sleeve(a.data, spd, OLD + NEW); res["trend_info"] = info; nd = len(days); i0 = 260
        n = nd - i0; c1, c2 = int(n * 0.5), int(n * 0.75); d_ = days[i0:]
        res["perioada"] = [time.strftime("%Y-%m-%d", time.gmtime(int(d_[0]) * 86400)), time.strftime("%Y-%m-%d", time.gmtime(int(d_[-1]) * 86400))]
        ok_ = np.ones(n, bool); comp = {}
        for k, v in tr.items(): comp[k] = v[i0:]
        res["trend"] = {k: {"tot": seg_stats(v, ok_), "train": seg_stats(v[:c1], ok_[:c1]), "val": seg_stats(v[c1:c2], ok_[c1:c2]), "lock": seg_stats(v[c2:], ok_[c2:])} for k, v in comp.items()}; save(res)
        # ---------------- B: sleeve-uri intraday alese pe train ----------------
        sl = {}; chosen = []
        for sym in ("EURUSD", "GBPUSD", "USDJPY"):
            try: R = sleeves(a.data, spd, sym, d_)
            except Exception as e: sl[sym] = {"eroare": repr(e)[:100]}; continue
            rows = []
            for (h, k), (g, ok, cost) in R.items():
                for dr in (1, -1):
                    net = np.where(ok, dr * g - cost, 0.0)
                    tt = tstat(net[:c1][ok[:c1]]) if ok[:c1].sum() > 300 else 0.0
                    rows.append((tt, h, k, dr, net, ok))
            rows.sort(key=lambda r: -r[0]); top = []
            for tt, h, k, dr, net, ok in rows[:6]:
                top.append({"h": h, "k": k, "dir": dr, "t_train": round(tt, 2), "val_bps": round(float(net[c1:c2][ok[c1:c2]].mean() * 1e4), 2), "lock_bps": round(float(net[c2:][ok[c2:]].mean() * 1e4), 2), "n_train": int(ok[:c1].sum())})
            sl[sym] = top
            for want in (1, -1):
                best = next((r for r in rows if r[3] == want), None)
                if best and best[0] >= 2.5: chosen.append((sym, best)); comp["%s_%s_%02dh+%d" % (sym, "L" if want > 0 else "S", best[1], best[2])] = best[4]
        res["sleeves_top_pe_train"] = sl; res["sleeves_alese"] = [{"sym": s, "h": b[1], "k": b[2], "dir": b[3], "t_train": round(b[0], 2)} for s, b in chosen]; save(res)
        # ---------------- portofolii ----------------
        names_sl = [k for k in comp if not k.startswith("trend")]
        def scaled(k): v = comp[k]; sd = v[:c1].std(); return v * (0.10 / (sd * np.sqrt(252))) if sd > 0 else v * 0   # vol 10% pe train
        port = {}
        port["trend_120_250"] = comp["trend_120_250"]; port["trend_60_120_250"] = comp["trend_60_120_250"]
        if names_sl:
            sles = np.mean([scaled(k) for k in names_sl], axis=0); port["sleeves"] = sles
            tsc = scaled("trend_60_120_250"); port["trend+sleeves_50/50"] = 0.5 * tsc + 0.5 * sles
            port["corelatie_trend_vs_sleeves"] = float(np.corrcoef(comp["trend_60_120_250"], sles)[0, 1])
        pr = {}
        for k, v in port.items():
            if k.startswith("corel"): pr[k] = round(v, 3); continue
            pr[k] = {"tot": seg_stats(v, ok_), "train": seg_stats(v[:c1], ok_[:c1]), "val": seg_stats(v[c1:c2], ok_[c1:c2]), "lock": seg_stats(v[c2:], ok_[c2:]),
                     "ftmo_tot": [ftmo_block(v, vt, rng) for vt in (0.06, 0.10, 0.15)], "ftmo_val+lock": [ftmo_block(v[c1:], vt, rng) for vt in (0.06, 0.10, 0.15)]}
            yrs = {}
            for y in range(2018, 2027):
                m_ = np.array([time.gmtime(int(dd) * 86400).tm_year == y for dd in d_])
                if m_.sum() > 50: yrs[y] = round(float(v[m_].sum() / (v[m_].std() * np.sqrt(252) + 1e-12) * np.sqrt(252 / m_.sum())), 2)
            pr[k]["sharpe_pe_an"] = yrs
        res["portofolii"] = pr; res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("Sleeve-urile intraday au fost alese cu t >= 2.5 DOAR pe prima jumatate a datelor (144 de combinatii pe piata = risc de selectie; vezi val/lock). Trendul: ansamblu fara parametri alesi. Net de spread si comision, fara swap. "
                       "Portofoliile combina componente scalate la vol 10% pe train. FTMO = bootstrap pe blocuri de 10 zile la vol tinta 6/10/15%.")
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
