# Studiu 13: indici pe partea lunga (beta de actiuni) cu vol targeting si filtru de trend. NU e alpha: e prima de risc a pietei de actiuni, depinde de regim.
# Cos: indici FTMO (US500, US100, US30, DAX, NIKKEI, UK100, EU50, FRA40, AUS200, HK50, US2K, N25). Variante: (a) mereu long, (b) long daca pretul > media 200 zile, altfel cash, (c) (b) cu 100 zile, (d) (b) 200z + iesire din 3 zile sub MA.
# Vol tinta 10% per piata (EWMA 60z, plafon levier 3x), egal ponderat pe piete active. Rebalansare saptamanala. Cost 1.75 x spread (masurat). Fara swap/dividende (CFD pe indici: swap long ~ cost de finantare, ATENTIE).
# Train 50% / val 25% / lock 25%; FTMO bootstrap pe blocuri de 10 zile; verificare si "fara 2020" (excludere martie-aprilie 2020) ca test de robustete.
import os, json, time, argparse
import numpy as np
from .study12 import daily_series, cost_frac
from .study11 import seg_stats, ftmo_block

VER = 1
IDX = ["US500", "US100", "US30", "DAX", "NIKKEI", "UK100", "EU50", "FRA40", "AUS200", "HK50", "US2K", "N25"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt13"); os.makedirs(out, exist_ok=True)

    def save(d):
        open(os.path.join(out, "result.json.tmp"), "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time())}; save(res)
    try:
        rng = np.random.default_rng(13)
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        D = {}
        for s in IDX:
            try:
                d, c = daily_series(a.data, s)
                if len(d) > 400: D[s] = (d, c)
            except Exception: pass
        syms = list(D); days = np.unique(np.concatenate([D[s][0] for s in syms])); nd = len(days); pos = {int(d): i for i, d in enumerate(days)}
        P = np.full((nd, len(syms)), np.nan)
        for j, s in enumerate(syms):
            for d, c in zip(*D[s]): P[pos[int(d)], j] = c
        R = np.zeros((nd, len(syms))); act = np.zeros((nd, len(syms)), bool); lastp = np.full(len(syms), np.nan)
        for i in range(nd):
            for j in range(len(syms)):
                if np.isfinite(P[i, j]):
                    if np.isfinite(lastp[j]) and lastp[j] > 0: R[i, j] = P[i, j] / lastp[j] - 1; act[i, j] = True
                    lastp[j] = P[i, j]
        V = np.full((nd, len(syms)), np.nan); lam = 1 - 2 / 61.0; M = {}
        ff = P.copy()
        for j in range(len(syms)):
            v = np.nan
            for i in range(nd):
                if act[i, j]: v = R[i, j] ** 2 if not np.isfinite(v) else lam * v + (1 - lam) * R[i, j] ** 2
                V[i, j] = np.sqrt(v) if np.isfinite(v) else np.nan
                if i and not np.isfinite(ff[i, j]): ff[i, j] = ff[i - 1, j]
        def ma(L):
            o = np.full((nd, len(syms)), np.nan)
            for j in range(len(syms)):
                c = np.nan_to_num(ff[:, j], nan=0.0); cs = np.cumsum(np.r_[0.0, c]); o[L:, j] = (cs[L + 1:] - cs[1:-L]) / L        # media L zile pana la i inclusiv
                o[:L, j] = np.nan
            return o
        MA200, MA100, MA50 = ma(200), ma(100), ma(50)
        cf = {s: cost_frac(spd, s, float(np.nanmedian(P[:, j])))[0] for j, s in enumerate(syms)}
        wk = (days + 3) // 7; reb = np.r_[False, wk[1:] != wk[:-1]]; i0 = 260; n = nd - i0; c1, c2 = int(n * 0.5), int(n * 0.75)
        d_ = days[i0:]; ok_ = np.ones(n, bool); yrs_of = np.array([time.gmtime(int(dd) * 86400).tm_year for dd in d_])
        covid = np.array([(time.gmtime(int(dd) * 86400).tm_year == 2020 and time.gmtime(int(dd) * 86400).tm_mon in (3, 4)) for dd in d_])
        variants = {"mereu_long": None, "long_peste_MA200": MA200, "long_peste_MA100": MA100, "long_MA200_si_MA50": "both"}
        res["piete"] = syms; res["perioada"] = [time.strftime("%Y-%m-%d", time.gmtime(int(d_[0]) * 86400)), time.strftime("%Y-%m-%d", time.gmtime(int(d_[-1]) * 86400))]; res["var"] = {}
        for name, flt in variants.items():
            W = np.zeros(len(syms)); net = np.zeros(nd)
            for i in range(i0, nd):
                if reb[i]:
                    on = np.zeros(len(syms), bool)
                    for j in range(len(syms)):
                        if not (np.isfinite(V[i - 1, j]) and V[i - 1, j] > 0 and act[max(0, i - 15):i, j].any()): continue
                        if flt is None: on[j] = True
                        elif isinstance(flt, str): on[j] = bool(np.isfinite(MA200[i - 1, j]) and ff[i - 1, j] > MA200[i - 1, j] and ff[i - 1, j] > MA50[i - 1, j])
                        else: on[j] = bool(np.isfinite(flt[i - 1, j]) and ff[i - 1, j] > flt[i - 1, j])
                    n_act = max(1, int(sum(1 for j in range(len(syms)) if np.isfinite(V[i - 1, j]) and act[max(0, i - 15):i, j].any())))
                    for j in range(len(syms)):
                        wn = min(3.0, 0.10 / (V[i - 1, j] * np.sqrt(252))) / n_act if on[j] else 0.0
                        net[i] -= abs(wn - W[j]) * cf[syms[j]]; W[j] = wn
                net[i] += float((W * R[i]).sum())
            x = net[i0:]
            r = {"tot": seg_stats(x, ok_), "train": seg_stats(x[:c1], ok_[:c1]), "val": seg_stats(x[c1:c2], ok_[c1:c2]), "lock": seg_stats(x[c2:], ok_[c2:]), "sem_mar_apr_2020": seg_stats(x, ~covid),
                 "ftmo_tot": [ftmo_block(x, v, rng) for v in (0.06, 0.10, 0.15)], "ftmo_val+lock": [ftmo_block(x[c1:], v, rng) for v in (0.06, 0.10, 0.15)]}
            r["ret_pe_an_pct"] = {int(y): round(float(x[yrs_of == y].sum() * 100), 1) for y in sorted(set(yrs_of)) if (yrs_of == y).sum() > 50}
            res["var"][name] = r; save(res)
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = "Beta de actiuni, nu alpha. Fara swap/finantare (CFD long pe indici platesc finantare ~ rata+marja: poate manca 2-5%/an din randament). Istoric 2018-2026 = in mare parte piata ascendenta. Vol nativa a portofoliului e sub 10% (diversificare)."
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
