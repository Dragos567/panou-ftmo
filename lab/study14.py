# Studiu 14: factori PREDEFINITI (fara cautare de parametri), pe ~32 de piete zilnice, cu spread + SWAP real (din specificatiile FTMO, aplicat static pe tot istoricul = aproximare).
#  F1 TS_mom: semnul mediu 120/250z, long/short, vol targeting (ca EA-ul).
#  F2 XS_mom: in fiecare clasa de active (>=4 piete), long top 1/3 / short bottom 1/3 dupa randamentul pe 120z (neutru pe clasa).
#  F3 XS_rev: aceeasi structura, dupa randamentul pe 21z, pozitie inversa (long perdantii lunii, short castigatorii).
#  F4 TOM_idx: indici long din ultima zi de tranzactionare a lunii pana in a 3-a zi a lunii urmatoare.
# Rebalansare saptamanala (F1-F3), zilnica pentru F4. Pozitii scalate la vol 10% per piata (EWMA 60z, plafon 4x), impartite la nr. de piete active.
# Raport: Sharpe brut / net (spread+swap) pe train/val/lock, corelatii intre factori, portofoliu egal-risc (vol 10% pe train), FTMO bootstrap.
import os, json, time, argparse
import numpy as np
from .study12 import daily_series, cost_frac, OLD, NEW, CLS
from .study11 import seg_stats, ftmo_block

VER = 1
# (swapLong, swapShort, digits) in puncte pe zi, din /api/swaps (FTMO). USOIL lipsa -> exclus. Cripto: mod INTEREST (-30%/an) tratat separat.
SW = {"EURUSD": (-8.66, 0.34, 5), "GBPUSD": (-6.6, -4.13, 5), "USDJPY": (2.7, -17.91, 3), "GOLD": (-64.9, -4.2, 2), "US100": (-696.38, 34.17, 2), "US500": (-120.29, -45.78, 2),
      "US30": (-176.55, -923.92, 2), "UK100": (-236.98, 10.94, 2), "DAX": (-451.78, -4.56, 2), "NIKKEI": (-943.04, -314.35, 2), "AUDUSD": (-3.12, -6.37, 5), "USDCAD": (0.98, -11.86, 5),
      "SILVER": (-14.77, 0.4, 3), "COPPER": (-18.02, 2.55, 2), "PLAT": (-57.83, -22.3, 2), "UKOIL": (60.98, -275.78, 3), "NATGAS": (-5.83, 1.27, 3), "SOY": (-26.43, 2.96, 2),
      "WHEAT": (-22.99, 3.64, 2), "CORN": (-37.55, 7.47, 2), "SUGAR": (-0.87, 0.15, 2), "COFFEE": (4.57, -22.98, 2), "COCOA": (-18.13, 2.81, 1), "COTTON": (-4.07, 0.75, 2),
      "EU50": (-111.84, -1.13, 2), "FRA40": (-141.54, -1.43, 2), "AUS200": (-173.72, -34.34, 2), "HK50": (-463.41, 8.83, 2), "N25": (-10.7, 1.13, 2), "US2K": (-55.25, -5.66, 2), "DXY": (-23.05, 1.13, 3)}
CRYPTO_DAY = -0.30 / 365.0
IDXC = [s for s in ("US500", "US100", "US30", "DAX", "NIKKEI", "UK100", "EU50", "FRA40", "AUS200", "HK50", "US2K", "N25")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt14"); os.makedirs(out, exist_ok=True)

    def save(d):
        open(os.path.join(out, "result.json.tmp"), "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time())}; save(res)
    try:
        rng = np.random.default_rng(14)
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        D = {}
        for s in [x for x in OLD + NEW if x != "USOIL"]:
            try:
                d, c = daily_series(a.data, s)
                if len(d) > 400 and (s in SW or s in ("ETH", "BTC")): D[s] = (d, c)
            except Exception: pass
        syms = list(D); nS = len(syms); days = np.unique(np.concatenate([D[s][0] for s in syms])); nd = len(days); pos = {int(d): i for i, d in enumerate(days)}
        P = np.full((nd, nS), np.nan)
        for j, s in enumerate(syms):
            for d, c in zip(*D[s]): P[pos[int(d)], j] = c
        R = np.zeros((nd, nS)); act = np.zeros((nd, nS), bool); lastp = np.full(nS, np.nan); GAP = np.zeros((nd, nS)); lastd = np.full(nS, np.nan); PX = np.full((nd, nS), np.nan)
        for i in range(nd):
            for j in range(nS):
                if np.isfinite(P[i, j]):
                    if np.isfinite(lastp[j]) and lastp[j] > 0: R[i, j] = P[i, j] / lastp[j] - 1; act[i, j] = True; GAP[i, j] = days[i] - lastd[j]; PX[i, j] = lastp[j]
                    lastp[j] = P[i, j]; lastd[j] = days[i]
        V = np.full((nd, nS), np.nan); lam = 1 - 2 / 61.0
        for j in range(nS):
            v = np.nan
            for i in range(nd):
                if act[i, j]: v = R[i, j] ** 2 if not np.isfinite(v) else lam * v + (1 - lam) * R[i, j] ** 2
                V[i, j] = np.sqrt(v) if np.isfinite(v) else np.nan
        lp = np.log(np.where(P > 0, P, np.nan))
        for j in range(nS):
            for i in range(1, nd):
                if not np.isfinite(lp[i, j]): lp[i, j] = lp[i - 1, j]
        cf = {s: cost_frac(spd, s, float(np.nanmedian(P[:, j])))[0] for j, s in enumerate(syms)}
        cls = [CLS.get(s, "idx") for s in syms]
        def swapfrac(j, w, i):
            s = syms[j]
            if s in ("BTC", "ETH"): return CRYPTO_DAY
            pl, ps, dg = SW[s]; pts = pl if w > 0 else ps
            return pts * 10.0 ** (-dg) / PX[i, j]
        wk = (days + 3) // 7; reb = np.r_[False, wk[1:] != wk[:-1]]; i0 = 260
        # zi-in-luna pentru TOM (zile cu >= 5 piete active)
        nact_day = act.sum(1); dts = [time.gmtime(int(d) * 86400) for d in days]; dom = np.zeros(nd, int); cnt = 0; prevm = None
        for i in range(nd):
            m = (dts[i].tm_year, dts[i].tm_mon)
            if m != prevm: cnt = 0; prevm = m
            if nact_day[i] >= 5: cnt += 1; dom[i] = cnt
        okv = lambda i, j: bool(np.isfinite(V[i - 1, j]) and V[i - 1, j] > 0 and act[max(0, i - 15):i, j].any())
        lev = lambda i, j: min(4.0, 0.10 / (V[i - 1, j] * np.sqrt(252)))
        def f_ts(i):
            av = [j for j in range(nS) if okv(i, j) and i - 1 - 250 >= 0 and np.isfinite(lp[i - 1, j] - lp[i - 1 - 250, j])]; w = np.zeros(nS)
            for j in av: w[j] = 0.5 * (np.sign(lp[i - 1, j] - lp[i - 1 - 120, j]) + np.sign(lp[i - 1, j] - lp[i - 1 - 250, j])) * lev(i, j) / len(av)
            return w
        def f_xs(L, sgn):
            def f(i):
                av = [j for j in range(nS) if okv(i, j) and i - 1 - L >= 0 and np.isfinite(lp[i - 1, j] - lp[i - 1 - L, j])]; w = np.zeros(nS); tot = len(av)
                for c in set(cls[j] for j in av):
                    g = [j for j in av if cls[j] == c]
                    if len(g) < 4: continue
                    g.sort(key=lambda j: lp[i - 1, j] - lp[i - 1 - L, j]); k = max(1, len(g) // 3)
                    for j in g[-k:]: w[j] = sgn * lev(i, j) / (2 * k) * len(g) / tot
                    for j in g[:k]: w[j] = -sgn * lev(i, j) / (2 * k) * len(g) / tot
                return w
            return f
        def f_tom(i):
            w = np.zeros(nS)
            if not (1 <= dom[i] <= 3): return w
            av = [j for j in range(nS) if syms[j] in IDXC and okv(i, j)]
            for j in av: w[j] = lev(i, j) / max(1, len(av))
            return w
        facs = {"F1_TS_mom": (f_ts, False), "F2_XS_mom120": (f_xs(120, 1), False), "F3_XS_rev21": (f_xs(21, -1), False), "F4_TOM_indici": (f_tom, True)}
        n = nd - i0; c1, c2 = int(n * 0.5), int(n * 0.75); d_ = days[i0:]; ok_ = np.ones(n, bool)
        res["piete"] = syms; res["perioada"] = [time.strftime("%Y-%m-%d", time.gmtime(int(d_[0]) * 86400)), time.strftime("%Y-%m-%d", time.gmtime(int(d_[-1]) * 86400))]; res["factori"] = {}
        series = {}
        for name, (fn, daily) in facs.items():
            W = np.zeros(nS); net = np.zeros(nd); gross = np.zeros(nd); swp = np.zeros(nd); trn = np.zeros(nd)
            for i in range(i0, nd):
                if daily or reb[i]:
                    wn = fn(i)
                    for j in range(nS):
                        if wn[j] != W[j] or True:
                            c = abs(wn[j] - W[j]) * cf[syms[j]]; net[i] -= c; trn[i] += abs(wn[j] - W[j])
                    W = wn
                g = float((W * R[i]).sum()); gross[i] = g; net[i] += g
                for j in range(nS):
                    if W[j] != 0 and act[i, j]:
                        sw = abs(W[j]) * swapfrac(j, W[j], i) * GAP[i, j]; net[i] += sw; swp[i] += sw
            series[name] = net[i0:]
            x = net[i0:]; g = gross[i0:]
            res["factori"][name] = {"net": {"tot": seg_stats(x, ok_), "train": seg_stats(x[:c1], ok_[:c1]), "val": seg_stats(x[c1:c2], ok_[c1:c2]), "lock": seg_stats(x[c2:], ok_[c2:])},
                                    "brut_sharpe": seg_stats(g, ok_)["sharpe"] if seg_stats(g, ok_) else None, "swap_an_pct": round(float(swp[i0:].mean() * 252 * 100), 2), "turnover_an": round(float(trn[i0:].sum() / n * 252), 1)}
            save(res)
        names = list(series); M = np.array([series[k] for k in names])
        res["corelatii"] = {names[a_]: {names[b_]: round(float(np.corrcoef(M[a_], M[b_])[0, 1]), 2) for b_ in range(len(names))} for a_ in range(len(names))}
        sc = lambda v: v * (0.10 / (v[:c1].std() * np.sqrt(252))) if v[:c1].std() > 0 else v * 0
        def combo(keys):
            v = np.mean([sc(series[k]) for k in keys], axis=0)
            return {"keys": keys, "tot": seg_stats(v, ok_), "train": seg_stats(v[:c1], ok_[:c1]), "val": seg_stats(v[c1:c2], ok_[c1:c2]), "lock": seg_stats(v[c2:], ok_[c2:]),
                    "ftmo_tot": [ftmo_block(v, vt, rng) for vt in (0.06, 0.10, 0.15)], "ftmo_val+lock": [ftmo_block(v[c1:], vt, rng) for vt in (0.06, 0.10, 0.15)]}
        res["portofolii"] = {"toti_factorii": combo(names), "fara_F1": combo([k for k in names if not k.startswith("F1")]), "XS_si_TOM": combo([k for k in names if k.startswith(("F2", "F3", "F4"))])}
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("Factori definiti inainte de test (fara optimizare). Swap = valorile CURENTE ale FTMO aplicate pe tot istoricul (rate 2016-2021 erau mai mici -> swap-ul istoric real era mai mic; aproximatie pesimista pe long, optimista pe short). "
                       "Costuri: spread la fiecare schimbare de pozitie + swap pe fiecare noapte (calendaristica) detinuta. 'Portofolii' = factori scalati la vol 10% pe train si mediati.")
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
