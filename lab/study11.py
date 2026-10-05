# Studiu 11: trend following zilnic (time-series momentum) cu vol targeting, portofoliu pe piatele disponibile (FX, aur, indici). Rebalansare saptamanala (prima zi a saptamanii).
# Semnal: semnul randamentului pe L zile (L in 20/40/60/120/250) sau ansamblu (media semnelor, fara selectie). Pozitie = semn x (10% vol anuala tinta / vol EWMA 60z), plafon levier 4x per piata.
# Cost: 1.75 x spread (mediana orara masurata, altfel ipoteza) / pret x |schimbare pozitie| + comision FX. NU include swap (overnight) - la pozitii tinute saptamani conteaza.
# Train 50% / val 25% / lock 25% pe timp; FTMO: bootstrap pe blocuri de 10 zile, doua faze (+10%, +5%), pierdere zilnica 5%, DD total 10%, vol portofoliu scalata la 6/10/15% anual.
import os, json, time, argparse
import numpy as np
from .study8 import load_tf, agg

VER = 1
MARKETS = ["EURUSD", "GBPUSD", "USDJPY", "GOLD", "US100", "US500", "US30", "UK100", "DAX", "NIKKEI"]
ASSUMED = {"US100": 1.0, "US500": 0.5, "US30": 2.0, "UK100": 1.0}
COMM = {"EURUSD": 0.00005, "GBPUSD": 0.00005, "USDJPY": 0.005}
LBS = (20, 40, 60, 120, 250)


def daily(data, sym):
    m, _ = load_tf(data, sym); t = m["t"].astype(np.int64)
    d = agg(t, m["o"].astype(np.float64), m["h"].astype(np.float64), m["l"].astype(np.float64), m["c"].astype(np.float64), m["v"].astype(np.float64), 86400)
    return (d["t"] // 86400).astype(np.int64), d["c"]


def seg_stats(r, active):
    r = r[active]
    if len(r) < 60: return None
    ann = float(r.mean() * 252); vol = float(r.std() * np.sqrt(252)); cum = np.cumsum(r); dd = float((np.maximum.accumulate(cum) - cum).max())
    return {"dias": int(len(r)), "ret_an": round(ann, 4), "vol_an": round(vol, 4), "sharpe": round(ann / vol, 2) if vol > 0 else None, "maxDD": round(dd, 3)}


def ftmo_block(r, vol_target, rng, N=3000, T=520, B=10):
    r = np.asarray(r, float); s = r * (vol_target / (r.std() * np.sqrt(252))); n = len(s)
    nb = T // B + 1; st = rng.integers(0, n - B, (N, nb)); idx = (st[:, :, None] + np.arange(B)).reshape(N, -1)[:, :T]; x = s[idx]
    res = []
    for tgt in (0.10, 0.05):
        cum = np.cumsum(x, axis=1); br = (x <= -0.05) | (cum <= -0.10); up = cum >= tgt; up[:, :3] = False
        hu, hb = up.any(1), br.any(1); tu = np.where(hu, up.argmax(1), T + 1); tb = np.where(hb, br.argmax(1), T + 1); ok = hu & (tu < tb)
        res.append((float(ok.mean()), float(np.median(tu[ok]) + 1) if ok.any() else None))
    return {"vol_tinta": vol_target, "p1": round(res[0][0], 3), "p2": round(res[1][0], 3), "p_total": round(res[0][0] * res[1][0], 3), "zile_f1": res[0][1], "zile_f2": res[1][1]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt11"); os.makedirs(out, exist_ok=True)

    def save(d):
        open(os.path.join(out, "result.json.tmp"), "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time())}; save(res)
    try:
        rng = np.random.default_rng(11)
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        D = {}
        for s in MARKETS:
            try: D[s] = daily(a.data, s)
            except Exception as e: res.setdefault("lipsa", {})[s] = repr(e)[:80]
        syms = list(D); days = np.unique(np.concatenate([D[s][0] for s in syms])); nd = len(days); pos = {int(d): i for i, d in enumerate(days)}
        P = np.full((nd, len(syms)), np.nan)
        for j, s in enumerate(syms):
            for d, c in zip(*D[s]): P[pos[int(d)], j] = c
        R = np.zeros((nd, len(syms))); act = np.zeros((nd, len(syms)), bool); last = np.full(len(syms), np.nan); lastp = np.full(len(syms), np.nan)
        for i in range(nd):
            for j in range(len(syms)):
                if np.isfinite(P[i, j]):
                    if np.isfinite(lastp[j]): R[i, j] = P[i, j] / lastp[j] - 1; act[i, j] = True
                    lastp[j] = P[i, j]
        # volatilitate EWMA 60z (informatia pana la ziua i inclusiv)
        V = np.full((nd, len(syms)), np.nan); lam = 1 - 2 / 61.0
        for j in range(len(syms)):
            v = np.nan
            for i in range(nd):
                if act[i, j]: v = R[i, j] ** 2 if not np.isfinite(v) else lam * v + (1 - lam) * R[i, j] ** 2
                V[i, j] = np.sqrt(v) if np.isfinite(v) else np.nan
        lp = np.log(np.where(np.isfinite(P), P, np.nan)); lpf = lp.copy()
        for j in range(len(syms)):
            for i in range(1, nd):
                if not np.isfinite(lpf[i, j]): lpf[i, j] = lpf[i - 1, j]
        cstf = {}
        for j, s in enumerate(syms):
            sp = spd.get(s, {}); meas = [v["mean"] for v in sp.values() if v["n"] >= 30]
            sp_ = float(np.median(meas)) if meas else ASSUMED.get(s, 0.0); cstf[s] = (1.75 * sp_, COMM.get(s, 0.0), bool(meas))
        res["costuri_ipoteza"] = {s: (not cstf[s][2]) for s in syms}
        wk = (days + 3) // 7; reb = np.r_[False, wk[1:] != wk[:-1]]
        cfgs = {"L%d" % L: L for L in LBS}; cfgs["ansamblu"] = 0
        end = {}; series = {}
        i0 = 260                                                   # primele ~260 zile = istoric pentru semnal
        for name, L in cfgs.items():
            W = np.zeros(len(syms)); net = np.zeros(nd); gross = np.zeros(nd); turn = np.zeros(nd)
            for i in range(i0, nd):
                if reb[i]:                                         # decizie la ultima inchidere a saptamanii trecute, aplicata din prima zi a saptamanii
                    for j in range(len(syms)):
                        if not np.isfinite(V[i - 1, j]) or V[i - 1, j] <= 0: continue
                        if L: sg = np.sign(lpf[i - 1, j] - lpf[i - 1 - L, j])
                        else: sg = float(np.mean([np.sign(lpf[i - 1, j] - lpf[i - 1 - l, j]) for l in LBS]))
                        wn = sg * min(4.0, 0.10 / (V[i - 1, j] * np.sqrt(252))) / len(syms)
                        c = abs(wn - W[j]) * (cstf[syms[j]][0] + cstf[syms[j]][1]) / max(P[i - 1, j] if np.isfinite(P[i - 1, j]) else lastp[j], 1e-9)
                        gross[i] -= 0; net[i] -= c; turn[i] += abs(wn - W[j]); W[j] = wn
                gross[i] += float((W * R[i]).sum()); net[i] += float((W * R[i]).sum())
            series[name] = (net, gross, turn)
        n = nd - i0; d_ = days[i0:]; cut1, cut2 = int(n * 0.5), int(n * 0.75)
        tm = lambda a_: np.array([True] * len(a_))
        res["de"] = time.strftime("%Y-%m-%d", time.gmtime(int(d_[0]) * 86400)); res["pana"] = time.strftime("%Y-%m-%d", time.gmtime(int(d_[-1]) * 86400)); res["piete"] = syms
        rows = {}
        for name, (net, gross, turn) in series.items():
            x, g = net[i0:], gross[i0:]; ok = np.ones(n, bool)
            rows[name] = {"complet_net": seg_stats(x, ok), "complet_brut": seg_stats(g, ok), "train": seg_stats(x[:cut1], ok[:cut1]), "val": seg_stats(x[cut1:cut2], ok[cut1:cut2]), "lock": seg_stats(x[cut2:], ok[cut2:]),
                          "turnover_an": round(float(turn[i0:].sum() / n * 252), 1)}
            yrs = {}
            for y in range(2018, 2027):
                m_ = np.array([time.gmtime(int(dd) * 86400).tm_year == y for dd in d_])
                if m_.sum() > 50: yrs[y] = round(float(x[m_].sum() / 1 * 1), 3)
            rows[name]["net_pe_an_pct_la_vol_nativa"] = yrs
            rows[name]["ftmo_val+lock"] = [ftmo_block(x[cut1:], v, rng) for v in (0.06, 0.10, 0.15)]
            rows[name]["ftmo_complet"] = [ftmo_block(x, v, rng) for v in (0.06, 0.10, 0.15)]
        res["configuratii"] = rows
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("Randamentele sunt net de cost (1.75 x spread + comision), FARA swap. Vol nativa a portofoliului ~ vol tinta per piata / sqrt(nr piete efective). 'ftmo' reface seria la vol tinta 6/10/15% anual, bootstrap pe blocuri de 10 zile. "
                       "Ansamblul nu are parametri alesi. Nu includ obligatiuni/energie/agricole (fara istoric in laborator).")
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
