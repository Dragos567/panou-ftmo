# Studiu 9: robustete pentru ORB la deschiderea sesiunilor pe indici (US500 / US100 / US30 / DAX / UK100) si referinte (GOLD, EURUSD), pe M15.
#  1) Walk-forward pe ani: pentru fiecare an Y (din al 4-lea an), se alege combinatia (din 120: ORB 15/30/45/60/90 min x stop opp/mid x iesire end/1.5R/2R/3R x filtru none/trend/narrow)
#     cu cel mai bun net R pe anii ANTERIORI (n>=100), se aplica in anul Y; rezultatele OOS se concateneaza.
#  2) Decay: net R mediu pe ani (media tuturor combinatiilor, fara selectie) si pentru combinatia aleasa.
#  3) Sensibilitate la cost: net la spread x1 / x2 / x3, plus multiplul de cost la care avantajul ajunge la zero.
#  4) Simulare FTMO 2 faze pe seria OOS (zile fara tranzactie = 0), risc 0.5% / 1% pe tranzactie.
# Spread: masurat live pe FTMO unde exista (pe ora UTC), altfel ipoteza (US100 1.0, US500 0.5, US30 2.0, UK100 1.0 puncte) - marcat explicit.
import os, json, time, itertools, argparse, datetime as dt
import numpy as np
from .study8 import load_tf, agg, session_times, first, ct, S, ftmo_days, INF

VER = 2
MARKETS = ["US500", "US100", "US30", "DAX", "UK100", "GOLD", "EURUSD"]
ASSUMED = {"US100": 1.0, "US500": 0.5, "US30": 2.0, "UK100": 1.0}
CFGS = list(itertools.product((15, 30, 45, 60, 90), ("opp", "mid"), ("end", "1.5R", "2R", "3R"), ("none", "trend", "narrow")))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt9"); os.makedirs(out, exist_ok=True)

    def save(d):
        open(os.path.join(out, "result.json.tmp"), "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time()), "markets": {}}
    save(res)
    try:
        rng = np.random.default_rng(9)
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        for sym in MARKETS:
            try: m, is15 = load_tf(a.data, sym)
            except Exception as e: res["markets"][sym] = {"eroare": repr(e)}; continue
            if len(m["t"]) < (20000 if is15 else 100000): res["markets"][sym] = {"eroare": "putine date"}; continue
            t = m["t"].astype(np.int64)
            sp = spd.get(sym, {}); meas = {int(k): v["mean"] for k, v in sp.items() if v["n"] >= 30}
            def spread_at(hr):
                if hr in meas: return meas[hr], True
                if sym in ASSUMED: return ASSUMED[sym], False
                means = list(meas.values()); return (float(np.median(means)) if means else 0.0), bool(means)
            m15 = agg(t, m["o"].astype(np.float64), m["h"].astype(np.float64), m["l"].astype(np.float64), m["c"].astype(np.float64), m["v"].astype(np.float64), 900)
            t15, O, H, L, C = m15["t"], m15["o"], m15["h"], m15["l"], m15["c"]
            dd = agg(t, m["o"].astype(np.float64), m["h"].astype(np.float64), m["l"].astype(np.float64), m["c"].astype(np.float64), m["v"].astype(np.float64), 86400)
            trend = np.zeros(len(dd["t"]))
            for i in range(51, len(dd["t"])): trend[i] = np.sign(dd["c"][i - 1] - dd["c"][i - 51:i - 1].mean())
            dmap = {int(x // 86400): i for i, x in enumerate(dd["t"])}
            alldays = np.unique(t15 // 86400)
            years_of = {int(d): (dt.datetime(1970, 1, 1) + dt.timedelta(days=int(d))).year for d in alldays}
            mres = {"de": time.strftime("%Y-%m-%d", time.gmtime(int(t[0]))), "pana": time.strftime("%Y-%m-%d", time.gmtime(int(t[-1]))), "zile": int(len(alldays)),
                    "spread": {str(h): [round(spread_at(h)[0], 4), "masurat" if spread_at(h)[1] else "ipoteza"] for h in (7, 8, 13, 14, 15)}}
            res["markets"][sym] = mres
            for kind in ("Londra", "NY"):
                st = session_times(alldays, kind); pre = []
                for (T0, Te), d in zip(st, alldays):
                    i0 = int(np.searchsorted(t15, T0)); ie = int(np.searchsorted(t15, Te))
                    if i0 >= len(t15) or t15[i0] != T0 or ie - i0 < 0.8 * (Te - T0) / 900 or int(d) not in dmap or trend[dmap[int(d)]] == 0: continue
                    pre.append((int(d), i0, ie, trend[dmap[int(d)]]))
                if len(pre) < 500: continue
                allt = []                                               # per combinatie: array (zi, an, R, cost_R_x1)
                for (ORn, stop_t, ex, flt) in CFGS:
                    n = ORn // 15; widths = []; trs = []
                    for d, i0, ie, tr_d in pre:
                        orh, orl = float(H[i0:i0 + n].max()), float(L[i0:i0 + n].min()); w = orh - orl; prev = widths[-20:]; widths.append(w)
                        if flt == "narrow" and (len(prev) < 10 or w > np.median(prev)): continue
                        lim = min(ie - 2, i0 + 12); j = i0 + n; dirn = 0
                        while j <= lim:
                            if C[j] > orh: dirn = 1; break
                            if C[j] < orl: dirn = -1; break
                            j += 1
                        if dirn == 0 or (flt == "trend" and dirn != tr_d): continue
                        e = j + 1; entry = float(O[e]); stop = (orl if dirn > 0 else orh) if stop_t == "opp" else (orh + orl) / 2
                        risk = (entry - stop) * dirn
                        if risk <= 0: continue
                        if dirn > 0: hh, ll, oo, cc, en_, sp_ = H[e:ie], L[e:ie], O[e:ie], C[ie - 1], entry, stop
                        else: hh, ll, oo, cc, en_, sp_ = -L[e:ie], -H[e:ie], -O[e:ie], -C[ie - 1], -entry, -stop
                        iS = first(ll <= sp_)
                        Rs = (min(sp_, oo[iS]) - en_) / risk if iS < INF else -1.0        # stop cu gap: se executa la deschidere daca bara se deschide dincolo de stop
                        if ex == "end": R = Rs if iS < INF else (cc - en_) / risk
                        else:
                            mult = float(ex[:-1]); iT = first(hh >= en_ + mult * risk)
                            R = Rs if iS <= iT and iS < INF else (mult if iT < INF else (cc - en_) / risk)
                        hr = int((t15[e] // 3600) % 24)
                        trs.append((d, years_of[d], R, 1.75 * spread_at(hr)[0] / risk))
                    allt.append(np.array(trs, float) if trs else np.zeros((0, 4)))
                years = sorted(set(years_of.values())); y0 = years[0]
                # --- walk-forward ---
                oos = []; chosen = {}; avg_all = {}
                for y in years:
                    if y < y0 + 3: continue
                    best, bm = None, -9
                    for ci, arr in enumerate(allt):
                        tr = arr[arr[:, 1] < y]
                        if len(tr) < 100: continue
                        mm = float((tr[:, 2] - tr[:, 3]).mean())
                        if mm > bm: bm, best = mm, ci
                    if best is None: continue
                    arr = allt[best]; ty = arr[arr[:, 1] == y]
                    chosen[y] = {"cfg": list(CFGS[best]), "train_R": round(bm, 3), "oos_n": int(len(ty)), "oos_R": round(float((ty[:, 2] - ty[:, 3]).mean()), 3) if len(ty) else None}
                    oos.append(ty)
                    ms = [float((a_[a_[:, 1] == y][:, 2] - a_[a_[:, 1] == y][:, 3]).mean()) for a_ in allt if (a_[:, 1] == y).sum() >= 15]
                    avg_all[y] = round(float(np.mean(ms)), 3) if ms else None
                # decay pe ani fara selectie (media tuturor combinatiilor)
                decay = {}
                for y in years:
                    ms = [float((a_[a_[:, 1] == y][:, 2] - a_[a_[:, 1] == y][:, 3]).mean()) for a_ in allt if (a_[:, 1] == y).sum() >= 15]
                    decay[y] = round(float(np.mean(ms)), 3) if ms else None
                row = {"zile_sesiune": len(pre), "combinatii": len(allt), "decay_media_toate": decay, "alese_pe_ani": chosen}
                if oos:
                    o = np.concatenate(oos); dy = o[:, 0].astype(np.int64)
                    for mult in (1, 2, 3):
                        net = o[:, 2] - mult * o[:, 3]; row["oos_net_x%d" % mult] = S(net, dy)
                    row["oos_brut"] = S(o[:, 2], dy); row["cost_R_med"] = round(float(np.median(o[:, 3])), 3)
                    row["multiplu_cost_zero"] = round(float(o[:, 2].mean() / o[:, 3].mean()), 2) if o[:, 3].mean() > 0 else None
                    row["pe_an"] = round(len(o) / max(len(oos), 1), 1)
                    oy = {int(y) for y in chosen}; oday = np.array([d for d in alldays if years_of[int(d)] in oy])
                    dayR = {}
                    for r in o: dayR[int(r[0])] = dayR.get(int(r[0]), 0.0) + r[2] - r[3]
                    series = np.array([dayR.get(int(d), 0.0) for d in oday])
                    row["ftmo_oos"] = [ftmo_days(series, rp, rng) for rp in (0.5, 1.0)]
                    row["oos_ani"] = len(oos)
                res["markets"][sym][kind] = row; save(res)
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("OOS = anii in care combinatia a fost aleasa doar pe anii anteriori (cel putin 3 ani de istoric). R net dupa cost 1.75 x spread. 'multiplu_cost_zero' = de cate ori ar trebui sa creasca costul ca avantajul sa dispara. "
                       "decay_media_toate = media net R a tuturor celor 120 de combinatii pe an (fara selectie).")
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
