# Studiu 10: trei ipoteze de microstructura / anomalii statistice, cu aceeasi rigoare (cost 1.75 x spread + comision, train 50% / val 25% / lockbox 25%, hard-stop zilnic 3.5%).
#  H1  Absorbtie de volum (proxy CVD) pe EURUSD / GOLD, M15: extrem de pret >= k x ATR + dezechilibru de volum >= 3:1 (300%) pe ultimele 4 bare + confirmare. ATENTIE: nu exista tick-uri reale in istoric,
#      doar OHLC + tick volume; volumul se imparte cumparat/vandut dupa pozitia inchiderii in bara (proxy). CVD real pe tick-uri FTMO necesita un colector nou (date doar de acum inainte).
#  H2  Fereastra 15:30-16:30 UTC (US100 / GOLD / US500): intrare doar in fereastra, in directia (pret - Anchor VWAP de la deschiderea NY) si/sau a impulsului de volum; iesire la 16:30; zero tranzactii in afara ferestrei.
#      Variante "follow" (directia ceruta) si "fade" (invers) ca verificare.
#  H3  Pair trading cu Z-score (EURUSD/GBPUSD, US100/US500): regresie rulanta pe log-preturi, intrare la |Z| > 2.2, iesire la Z = 0, stop la |Z| = 4.5, limita de timp. Costuri pe ambele picioare.
# Cost: 1.75 x spread (pret) la ora intrarii (masurat live unde exista, altfel ipoteza marcata) + comision FX 0.5 pip dus-intors (indici/aur: 0, ipoteza). Hard-stop zilnic: R/zi limitat la -3.5% / risc%.
import os, json, time, itertools, argparse
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from .study8 import load_tf, agg, session_times, first, S, ftmo_days, INF

VER = 1
ASSUMED = {"US100": 1.0, "US500": 0.5, "US30": 2.0, "UK100": 1.0}
COMM = {"EURUSD": 0.00005, "GBPUSD": 0.00005}          # comision FX dus-intors in unitati de pret (ipoteza)
DAILY_STOP = 3.5                                       # % din cont


def load15(data, sym):
    m, _ = load_tf(data, sym)
    return agg(m["t"].astype(np.int64), m["o"].astype(np.float64), m["h"].astype(np.float64), m["l"].astype(np.float64), m["c"].astype(np.float64), m["v"].astype(np.float64), 900)


def atr20(H, L, C):
    pc = np.r_[C[0], C[:-1]]; tr = np.maximum(H - L, np.maximum(np.abs(H - pc), np.abs(L - pc)))
    cs = np.cumsum(np.r_[0.0, tr]); a = np.full(len(tr), np.nan); a[20:] = (cs[21:] - cs[1:-20]) / 20.0   # media TR a 20 bare, pana la bara i inclusiv
    return a


def trade_R(O, H, L, C, e, xe, dirn, risk):
    """Intrare la O[e], iesire la C[xe-1], stop la distanta 'risk' cu executie gap-aware. R brut."""
    en = O[e]
    if dirn > 0:
        st = en - risk; hit = first(L[e:xe] <= st)
        if hit < INF: return (min(st, O[e + hit]) - en) / risk
        return (C[xe - 1] - en) / risk
    st = en + risk; hit = first(H[e:xe] >= st)
    if hit < INF: return (en - max(st, O[e + hit])) / risk
    return (en - C[xe - 1]) / risk


class Cost:
    def __init__(self, spd, sym):
        sp = spd.get(sym, {}); self.meas = {int(k): v["mean"] for k, v in sp.items() if v["n"] >= 30}; self.sym = sym
    def at(self, hr):
        if hr in self.meas: return self.meas[hr], True
        if self.sym in ASSUMED: return ASSUMED[self.sym], False
        v = list(self.meas.values()); return (float(np.median(v)) if v else 0.0), bool(v)
    def price(self, hr): return 1.75 * self.at(hr)[0] + COMM.get(self.sym, 0.0)


def report(cfgs, alldays, rng):
    """cfgs: {nume: array (zi_abs, R_brut, cost_R)}; imparte pe zile 50/25/25, alege pe train, FTMO pe validare+lockbox."""
    D = len(alldays); rank = {int(d): i for i, d in enumerate(alldays)}
    seg = lambda days: np.array([0 if rank[int(d)] < D * 0.5 else 1 if rank[int(d)] < D * 0.75 else 2 for d in days]) if len(days) else np.array([], int)
    rows = {}; best, bm = None, -9
    for name, a in cfgs.items():
        if len(a) == 0: rows[name] = {"n": 0}; continue
        sg = seg(a[:, 0]); net = a[:, 1] - a[:, 2]; r = {"brut_medie": round(float(a[:, 1].mean()), 3), "cost_R_med": round(float(np.median(a[:, 2])), 3)}
        for k, nm in enumerate(("train", "val", "lock")):
            q = sg == k; r[nm] = S(net[q], a[q, 0]) if q.any() else {"n": 0}
        r["candidat"] = bool(all(r[nm].get("n", 0) >= 20 and r[nm].get("m", -1) > 0 for nm in ("train", "val", "lock")))
        rows[name] = r
        if r["train"].get("n", 0) >= 40 and r["train"]["m"] > bm: bm, best = r["train"]["m"], name
    out = {"configuratii": rows, "alesa_pe_train": best}
    for k, nm in enumerate(("train", "val", "lock")):
        ms = [r[nm]["m"] for r in rows.values() if r.get(nm, {}).get("n", 0) >= 20]; out["media_toate_" + nm] = round(float(np.mean(ms)), 3) if ms else None
    if best:
        a = cfgs[best]; sg = seg(a[:, 0]); net = a[:, 1] - a[:, 2]; fz = {}
        for lab, ok in (("val+lock", lambda i: i >= D * 0.5), ("lock", lambda i: i >= D * 0.75)):
            days = [d for d in alldays if ok(rank[int(d)])]; fz[lab] = []
            for rp in (0.5, 1.0):
                dayR = {}
                for d, x in zip(a[:, 0], net): dayR[int(d)] = dayR.get(int(d), 0.0) + x
                s = np.array([max(dayR.get(int(d), 0.0), -DAILY_STOP / rp) for d in days])      # hard-stop zilnic 3.5%
                fz[lab].append(ftmo_days(s, rp, rng) if len(s) > 30 else None)
        out["ftmo_oos"] = fz
    return out


# ============================================================ H2 ============================================================
def run_H2(data, sym, spd, rng):
    m = load15(data, sym); t, O, H, L, C, V = m["t"], m["o"], m["h"], m["l"], m["c"], m["v"]
    cost = Cost(spd, sym); at = atr20(H, L, C); alldays = np.unique(t // 86400)
    ny = session_times(alldays, "NY"); sgn = np.sign(C - O); tp = (H + L + C) / 3.0
    ev = []                                              # (zi, i0, ie, avwap, impuls, atr, dist)
    for (T0, _), d in zip(ny, alldays):
        W0 = int(d) * 86400 + 15 * 3600 + 1800; i0 = int(np.searchsorted(t, W0)); iA = int(np.searchsorted(t, T0))
        if i0 >= len(t) - 5 or t[i0] != W0 or i0 - iA < 2 or t[i0 + 3] != W0 + 3 * 900 or t[iA] - T0 > 900: continue
        w = V[iA:i0]; w = w if w.sum() > 0 else np.ones(len(w)); av = float((tp[iA:i0] * w).sum() / w.sum())
        vv = V[i0 - 4:i0]; imp = float((sgn[i0 - 4:i0] * vv).sum() / vv.sum()) if vv.sum() > 0 else 0.0
        if not np.isfinite(at[i0 - 1]): continue
        ev.append((int(d), i0, i0 + 4, av, imp, float(at[i0 - 1]), float(C[i0 - 1] - av)))
    cfgs = {}
    for sig, mind, side in itertools.product(("vwap", "impuls", "ambele"), (0.0, 0.5), ("follow", "fade")):
        trs = []
        for d, i0, ie, av, imp, a_, dist in ev:
            if abs(dist) < mind * a_: continue
            sv, si = np.sign(dist), np.sign(imp)
            dirn = sv if sig == "vwap" else si if sig == "impuls" else (sv if sv == si else 0)
            if dirn == 0: continue
            if side == "fade": dirn = -dirn
            risk = 3.0 * a_; R = trade_R(O, H, L, C, i0, ie, int(dirn), risk)
            trs.append((d, R, cost.price(15) / risk))
        cfgs["%s|min%.1f|%s" % (sig, mind, side)] = np.array(trs, float) if trs else np.zeros((0, 3))
    res = report(cfgs, alldays, rng); res["zile_cu_fereastra"] = len(ev)
    res["spread_15h"] = [round(cost.at(15)[0], 4), "masurat" if cost.at(15)[1] else "ipoteza"]; return res


# ============================================================ H1 ============================================================
def run_H1(data, sym, spd, rng):
    m = load15(data, sym); t, O, H, L, C, V = m["t"], m["o"], m["h"], m["l"], m["c"], m["v"]; N = len(t)
    cost = Cost(spd, sym); at = atr20(H, L, C); alldays = np.unique(t // 86400)
    rg = H - L; loc = np.where(rg > 0, (C - L) / np.where(rg > 0, rg, 1), 0.5); buy = V * loc; sell = V - buy
    cs = lambda x: np.cumsum(np.r_[0.0, x]); cb, cl = cs(buy), cs(sell)
    B4 = np.full(N, np.nan); S4 = np.full(N, np.nan); B4[3:] = cb[4:] - cb[:N - 3]; S4[3:] = cl[4:] - cl[:N - 3]
    with np.errstate(all="ignore"): dn = S4 / B4; up = B4 / S4
    cvd = np.cumsum(buy - sell)
    lmin = np.full(N, np.nan); hmax = np.full(N, np.nan); lmin[20:] = swv(L, 20).min(1)[:N - 20]; hmax[20:] = swv(H, 20).max(1)[:N - 20]       # extremul celor 20 de bare ANTERIOARE
    cand = []
    with np.errstate(all="ignore"):
        lo = np.flatnonzero((L <= lmin) & (dn >= 3.0) & np.isfinite(at)); hi = np.flatnonzero((H >= hmax) & (up >= 3.0) & np.isfinite(at))
    for j in lo: cand.append((int(j), 1))
    for j in hi: cand.append((int(j), -1))
    cand.sort(); cfgs = {}
    for rm, hold, dv in itertools.product((1.0, 1.5), (4, 8, 16), ("off", "on")):
        trs = []; free = 0
        for j, dirn in cand:
            if j < free or j + 2 + hold >= N or t[j + 2] - t[j] > 1800 or rg[j] < rm * at[j - 1]: continue
            if not (C[j + 1] > O[j + 1] if dirn > 0 else C[j + 1] < O[j + 1]): continue
            if dv == "on":                                    # divergenta: CVD mai bun la noul extrem decat la extremul anterior
                w = slice(j - 20, j); k = j - 20 + int(np.argmin(L[w]) if dirn > 0 else np.argmax(H[w]))
                if not (cvd[j] > cvd[k] if dirn > 0 else cvd[j] < cvd[k]): continue
            e = j + 2; en = O[e]; risk = max((en - min(L[j], L[j + 1])) if dirn > 0 else (max(H[j], H[j + 1]) - en), 0.5 * at[j])
            R = trade_R(O, H, L, C, e, e + hold, dirn, risk); hr = int((t[e] // 3600) % 24)
            trs.append((int(t[e] // 86400), R, cost.price(hr) / risk)); free = e + hold
        cfgs["ATRx%.1f|hold%d|div_%s" % (rm, hold, dv)] = np.array(trs, float) if trs else np.zeros((0, 3))
    res = report(cfgs, alldays, rng); res["nota_date"] = "proxy din OHLC + tick volume, nu CVD real pe tick-uri"; return res


# ============================================================ H3 ============================================================
def run_H3(data, sy, sx, spd, rng):
    a, b = load15(data, sy), load15(data, sx)
    _, ia, ib = np.intersect1d(a["t"], b["t"], return_indices=True)
    t = a["t"][ia]; Oy, Cy, Ox, Cx = a["o"][ia], a["c"][ia], b["o"][ib], b["c"][ib]; N = len(t)
    cy, cx = Cost(spd, sy), Cost(spd, sx); alldays = np.unique(t // 86400)
    ly, lx = np.log(Cy), np.log(Cx); lyo, lxo = np.log(Oy), np.log(Ox); my0, mx0 = ly.mean(), lx.mean(); ly0, lx0 = ly - my0, lx - mx0
    cfgs = {}
    for W in (192, 480, 960):
        c1 = lambda x: np.cumsum(np.r_[0.0, x]); sm = lambda x: (c1(x)[W:] - c1(x)[:-W]) / W
        mx, my = sm(lx0), sm(ly0); vx = sm(lx0 * lx0) - mx * mx; vy = sm(ly0 * ly0) - my * my; cv = sm(lx0 * ly0) - mx * my
        with np.errstate(all="ignore"):
            beta = cv / vx; sd = np.sqrt(np.maximum(vy - beta * beta * vx, 1e-18)); z = (ly0[W - 1:] - (my + beta * (lx0[W - 1:] - mx))) / sd
        Z = np.full(N, np.nan); Bt = np.full(N, np.nan); SD = np.full(N, np.nan); Z[W - 1:] = z; Bt[W - 1:] = beta; SD[W - 1:] = sd
        for T in (48, 96, 192):
            trs = []; pos = 0; e = 0; dy = 0; bb = sd0 = 0.0; Zl = Z.tolist(); tl = t.tolist(); Bl = Bt.tolist(); SDl = SD.tolist()
            for i in range(W, N - 1):
                gap = tl[i + 1] - tl[i] > 3600
                if pos == 0:
                    zi = Zl[i]
                    if zi == zi and abs(zi) > 2.2 and not gap and 0.2 < Bl[i] < 3.0:
                        pos = 1; e = i + 1; dy = -1 if zi > 0 else 1; bb = Bl[i]; sd0 = SDl[i]
                else:
                    zi = Zl[i]
                    if dy * zi >= 0 or dy * zi <= -4.5 or i - e + 1 >= T or gap:
                        x = i + 1; pnl = dy * ((lyo[x] - lyo[e]) - bb * (lxo[x] - lxo[e])); risk = (4.5 - 2.2) * sd0
                        hr = int((tl[e] // 3600) % 24)
                        cst = (cy.price(hr) / Oy[e] + abs(bb) * cx.price(hr) / Ox[e])
                        trs.append((tl[e] // 86400, pnl / risk, cst / risk)); pos = 0
            cfgs["W%d|T%d" % (W, T)] = np.array(trs, float) if trs else np.zeros((0, 3))
    res = report(cfgs, alldays, rng); res["bare"] = int(N); res["de"] = time.strftime("%Y-%m-%d", time.gmtime(int(t[0]))); res["pana"] = time.strftime("%Y-%m-%d", time.gmtime(int(t[-1])))
    res["spread_ipoteza"] = {s: (not bool(c.meas)) for s, c in ((sy, cy), (sx, cx))}; return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt10"); os.makedirs(out, exist_ok=True)

    def save(d):
        open(os.path.join(out, "result.json.tmp"), "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time()), "H1": {}, "H2": {}, "H3": {}}; save(res)
    try:
        rng = np.random.default_rng(10)
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        jobs = [("H2", s, lambda s=s: run_H2(a.data, s, spd, rng)) for s in ("US100", "GOLD", "US500")]
        jobs += [("H3", "%s/%s" % p, lambda p=p: run_H3(a.data, p[0], p[1], spd, rng)) for p in (("EURUSD", "GBPUSD"), ("US100", "US500"))]
        jobs += [("H1", s, lambda s=s: run_H1(a.data, s, spd, rng)) for s in ("EURUSD", "GOLD")]
        for h, nm, fn in jobs:
            try: res[h][nm] = fn()
            except Exception as e:
                import traceback; res[h][nm] = {"eroare": repr(e), "tb": traceback.format_exc()[-900:]}
            save(res)
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("R net = R brut - cost (1.75 x spread + comision) / risc. Alegerea configuratiei doar pe train; 'candidat' = medie net > 0 pe train, validare si lockbox (n>=20 fiecare). "
                       "FTMO: bootstrap pe zilele val+lock, hard-stop zilnic 3.5%. H1 foloseste un proxy de volum (fara tick-uri reale). Spreadurile US100/US500 sunt ipoteza pana se aduna esantioane. Multe configuratii = risc de selectie.")
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
