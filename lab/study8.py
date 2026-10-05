# Studiu 8 (timeframe-uri mari, M15 / H1), doua concepte:
#  A) Volatility breakout / expansion la deschiderea sesiunilor (Londra 08:00 Europe/London, New York 09:30 America/New_York): ORB (15/30/60 min) si expansiune ATR (Crabel), pe XAUUSD (GOLD), US100, plus EURUSD / DAX / US500 ca referinta.
#  B) Cross-asset lead-lag / momentum (US100->EURUSD, US500->US100, USDJPY->EURUSD, ...) pe M15 si H1, versiune bruta si reziduala.
# Costuri: 1.75 x spread-ul mediu al orei de intrare (masurat live pe FTMO pentru GOLD/EURUSD/DAX); US100/US500 nu au spread masurat -> ipoteza (US100 1.0 pct, US500 0.5 pct), raportat si la cost x2.
# Impartire pe zile: train 50% / validare 25% / lockbox 25%. Parametrii se aleg DOAR pe train. Candidat = acelasi semn net pozitiv pe train, validare si lockbox.
import os, json, time, glob, itertools, argparse, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
from .run import load_m1
from .data import DT

VER = 1
INF = 10 ** 9
UTC = dt.timezone.utc
LON, NYZ = ZoneInfo("Europe/London"), ZoneInfo("America/New_York")
ASSUMED_SPREAD = {"US100": 1.0, "US500": 0.5}                    # puncte de indice, ipoteza (CFD FTMO); se afiseaza explicit in rezultat
A_MARKETS = ["GOLD", "US100", "EURUSD", "DAX", "US500"]
PAIRS = [("US100", "EURUSD"), ("US500", "EURUSD"), ("US100", "GOLD"), ("GOLD", "EURUSD"), ("EURUSD", "GOLD"), ("US100", "US500"), ("US500", "US100"), ("DAX", "US100"), ("US100", "DAX"),
         ("GBPUSD", "EURUSD"), ("USDJPY", "EURUSD"), ("USDJPY", "GOLD"), ("US100", "USDJPY"), ("EURUSD", "EURUSD"), ("US100", "US100"), ("GOLD", "GOLD")]


def load_tf(data_dir, sym):
    """Intoarce (m1_sau_m15, este_m15). Prefera M1 (EURUSD/DAX/NIKKEI); pentru simbolurile descarcate direct pe M15 citeste fisierele 15m."""
    try:
        m = load_m1(data_dir, sym)
        if len(m["t"]) > 100000: return m, False
    except Exception: pass
    ps = sorted(glob.glob(os.path.join(data_dir, "hist", "%s_15m" % sym, "p_*.bin")), key=lambda p: int(os.path.basename(p)[2:-4]))
    arrs = [np.fromfile(p, dtype=DT, count=os.path.getsize(p) // DT.itemsize) for p in ps]
    arrs = [x for x in arrs if len(x)]
    if not arrs: raise RuntimeError("fara istoric " + sym)
    a = np.concatenate(arrs); a = a[np.argsort(a["t"], kind="stable")]; _, ix = np.unique(a["t"], return_index=True); a = a[ix]
    return {k: np.ascontiguousarray(a[k]) for k in ("t", "o", "h", "l", "c", "v")}, True


def agg(t, o, h, l, c, v, sec):
    k = (t // sec).astype(np.int64); u, st = np.unique(k, return_index=True); en = np.append(st[1:], len(k))
    return {"t": (u * sec).astype(np.int64), "o": o[st], "h": np.maximum.reduceat(h, st), "l": np.minimum.reduceat(l, st), "c": c[en - 1], "v": np.add.reduceat(v, st)}


def first(mask):
    i = np.flatnonzero(mask); return int(i[0]) if len(i) else INF


def ct(x, day):
    """Medie, t cu erori grupate pe zi."""
    x = np.asarray(x, float); n = len(x)
    if n < 20: return None
    u, inv = np.unique(day, return_inverse=True); G = len(u)
    if G < 8: return None
    m = x.mean(); s = np.bincount(inv, weights=x - m); var = float((s ** 2).sum()) * G / (G - 1) / n ** 2
    return float(m / np.sqrt(var)) if var > 0 else None


def S(x, day, nd=3):
    x = np.asarray(x, float)
    if len(x) == 0: return {"n": 0}
    t = ct(x, day)
    return {"n": int(len(x)), "m": round(float(x.mean()), nd), "t": round(t, 2) if t is not None else None, "win": round(float((x > 0).mean()), 3)}


def seg3(D):
    return lambda i: 0 if i < D * 0.5 else 1 if i < D * 0.75 else 2


def ftmo_days(day_r, risk_pct, rng, N=3000, T=520):
    """day_r: R net pe zi (include zilele fara tranzactie = 0). Doua faze (+10%, +5%), pierdere zilnica 5%, pierdere totala 10% din capitalul initial. Fara limita de timp in afara de T zile."""
    dr = np.asarray(day_r, float) * risk_pct / 100.0; res = []
    for tgt in (0.10, 0.05):
        r = dr[rng.integers(0, len(dr), (N, T))]; cum = np.cumsum(r, axis=1)
        br = (r <= -0.05) | (cum <= -0.10); up = cum >= tgt; up[:, :3] = False
        hu, hb = up.any(1), br.any(1); tu = np.where(hu, up.argmax(1), T + 1); tb = np.where(hb, br.argmax(1), T + 1)
        ok = hu & (tu < tb); res.append((float(ok.mean()), float(np.median(tu[ok]) + 1) if ok.any() else None))
    return {"risk_pct": risk_pct, "p1": round(res[0][0], 3), "p2": round(res[1][0], 3), "p_total": round(res[0][0] * res[1][0], 3), "zile_f1": res[0][1], "zile_f2": res[1][1]}


# ====================================================== A: sesiuni ======================================================
def session_times(days, kind):
    """Pentru fiecare zi UTC (zile de la epoca): (T0, Tend) in secunde UTC."""
    out = []
    for d in days:
        base = dt.datetime(1970, 1, 1, tzinfo=UTC) + dt.timedelta(days=int(d))
        if kind == "Londra":
            a = dt.datetime(base.year, base.month, base.day, 8, 0, tzinfo=LON); b = dt.datetime(base.year, base.month, base.day, 16, 30, tzinfo=LON)
        else:
            a = dt.datetime(base.year, base.month, base.day, 9, 30, tzinfo=NYZ); b = dt.datetime(base.year, base.month, base.day, 16, 0, tzinfo=NYZ)
        out.append((int(a.timestamp()), int(b.timestamp())))
    return out


def run_A(sym, m1, spread_fn, res_rows, rng):
    t, o, h, l, c, v = (m1[k] for k in ("t", "o", "h", "l", "c", "v")); t = t.astype(np.int64)
    m15 = agg(t, o, h, l, c, v, 900); t15, O, H, L, C = m15["t"], m15["o"], m15["h"], m15["l"], m15["c"]
    day15 = t15 // 86400
    dd = agg(t, o, h, l, c, v, 86400); atrd = np.full(len(dd["t"]), np.nan); tr = np.maximum(dd["h"] - dd["l"], np.maximum(abs(dd["h"] - np.roll(dd["c"], 1)), abs(dd["l"] - np.roll(dd["c"], 1))))
    for i in range(15, len(tr)): atrd[i] = tr[i - 14:i].mean()                  # ATR D1 pe zilele PRECEDENTE
    trend = np.zeros(len(dd["t"]))
    for i in range(51, len(dd["t"])): trend[i] = np.sign(dd["c"][i - 1] - dd["c"][i - 51:i - 1].mean())
    dday = dd["t"] // 86400; dmap = {int(x): i for i, x in enumerate(dday)}
    alldays = np.unique(day15); D = len(alldays); sg = seg3(D); dseg = {int(d): sg(i) for i, d in enumerate(alldays)}
    out = {}
    for kind in ("Londra", "NY"):
        st = session_times(alldays, kind)
        rows = {}; widths = []
        pre = []
        for (T0, Te), d in zip(st, alldays):
            i0 = int(np.searchsorted(t15, T0)); ie = int(np.searchsorted(t15, Te))
            if i0 >= len(t15) or t15[i0] != T0 or ie - i0 < 0.8 * (Te - T0) / 900 or int(d) not in dmap: continue
            di = dmap[int(d)]
            if not np.isfinite(atrd[di]) or trend[di] == 0: continue
            pre.append((int(d), i0, ie, atrd[di], trend[di]))
        # latimea OR istorica pentru filtrul "narrow"
        def orw(i0, n): return float(H[i0:i0 + n].max() - L[i0:i0 + n].min())
        def trade(sd, i0, ie, atr_d, tr_d, fam, par):
            ORn, stop_t, ex, flt = par[0], par[1], par[2], par[3]
            if fam == "ORB":
                n = ORn // 15
                if flt == "trend": pass
                orh, orl = float(H[i0:i0 + n].max()), float(L[i0:i0 + n].min()); jb = i0 + n; lim = min(ie - 2, i0 + 12)
                lvl_up, lvl_dn = orh, orl
            else:
                k = par[0]; op = float(O[i0]); jb = i0 + 1; lim = min(ie - 2, i0 + 12); lvl_up, lvl_dn = op + k * atr_d, op - k * atr_d
            dirn = 0; j = jb
            while j <= lim:
                if C[j] > lvl_up: dirn = 1; break
                if C[j] < lvl_dn: dirn = -1; break
                j += 1
            if dirn == 0: return None
            if flt == "trend" and dirn != tr_d: return None
            if fam == "ORB" and flt == "narrow":
                prev = [w for w in widths[-20:]]
                if len(prev) < 10 or orw(i0, ORn // 15) > np.median(prev): return None
            e = j + 1; entry = float(O[e])
            if fam == "ORB": stop = (orl if dirn > 0 else orh) if stop_t == "opp" else (orh + orl) / 2
            else: stop = float(O[i0])
            risk = (entry - stop) * dirn
            if risk <= 0: return None
            hh = H[e:ie] if dirn > 0 else -L[e:ie]; ll = L[e:ie] if dirn > 0 else -H[e:ie]; cc = C[ie - 1] if dirn > 0 else -C[ie - 1]; en_ = entry if dirn > 0 else -entry; sp_ = stop if dirn > 0 else -stop
            iS = first(ll <= sp_)
            if ex == "end": R = -1.0 if iS < INF else (cc - en_) / risk
            else:
                mult = float(ex[:-1]); iT = first(hh >= en_ + mult * risk)
                R = -1.0 if iS <= iT and iS < INF else (mult if iT < INF else (cc - en_) / risk)
            cost = 1.75 * spread_fn(int(((t15[e]) // 3600) % 24), entry)
            return (R, cost / risk, sd, risk)
        fams = [("ORB", p) for p in itertools.product((15, 30, 60), ("opp", "mid"), ("end", "2R", "3R"), ("none", "trend", "narrow"))] + \
               [("EXP", p) for p in itertools.product((0.25, 0.5, 0.75), ("open",), ("end", "2R"), ("none", "trend"))]
        for fam, par in fams:
            trs = []; widths.clear()
            for sd, i0, ie, atr_d, tr_d in pre:
                if fam == "ORB": widths.append(orw(i0, par[0] // 15))
                r = trade(sd, i0, ie, atr_d, tr_d, fam, par)
                if r: trs.append(r)
            if not trs: continue
            a = np.array(trs, float); rr = a[:, 0]; cr = a[:, 1]; dy = a[:, 2].astype(np.int64); sgm = np.array([dseg[int(x)] for x in dy])
            row = {"fam": fam, "par": list(par), "n": len(a), "brut": S(rr, dy), "net1": {}, "net2": {}}
            for nm, mk in (("train", sgm == 0), ("val", sgm == 1), ("lock", sgm == 2), ("tot", sgm >= 0)):
                row["net1"][nm] = S(rr[mk] - cr[mk], dy[mk]); row["net2"][nm] = S(rr[mk] - 2 * cr[mk], dy[mk])
            row["pe_an"] = round(len(a) / max(D / 252, 1), 1); row["cost_R_med"] = round(float(np.median(cr)), 3)
            rows[(fam, tuple(par))] = (row, a, sgm)
        lst = [r[0] for r in rows.values()]
        okr = [r for r in lst if r["net1"]["train"]["n"] >= 80]
        best = max(okr, key=lambda r: r["net1"]["train"]["m"]) if okr else None
        ftm = None
        if best:
            row, a, sgm = rows[(best["fam"], tuple(best["par"]))]
            dayR = {}
            for (R_, c_, d_, rk) in a: dayR[int(d_)] = dayR.get(int(d_), 0.0) + R_ - c_
            arr = np.array([dayR.get(int(d), 0.0) for d in alldays]); ftm = [ftmo_days(arr, rp, rng) for rp in (0.5, 1.0)]
            ftm_val = None
            vm = np.array([dseg[int(d)] >= 1 for d in alldays])
            if vm.sum() > 100: ftm_val = [ftmo_days(arr[vm], rp, rng) for rp in (0.5, 1.0)]
        mean_all = float(np.mean([r["net1"]["tot"]["m"] for r in lst if r["net1"]["tot"].get("m") is not None])) if lst else None
        cands = [r for r in lst if all(r["net1"][k].get("m") is not None and r["net1"][k]["m"] > 0 and r["net1"][k]["n"] >= 30 for k in ("train", "val", "lock"))]
        out[kind] = {"zile": D, "combinatii": len(lst), "net1_medie_toate": round(mean_all, 3) if mean_all is not None else None, "pozitive_tot": sum(1 for r in lst if (r["net1"]["tot"].get("m") or 0) > 0),
                     "aleasa_pe_train": best, "ftmo_aleasa": ftm, "ftmo_aleasa_val_lock": ftm_val if best else None, "candidati": cands[:10], "top_train": sorted(okr, key=lambda r: -r["net1"]["train"]["m"])[:5]}
    return out


# ====================================================== B: lead-lag ======================================================
def run_B(series, sptab, tfname, sec):
    res = []
    for L_, F_ in PAIRS:
        if L_ not in series or F_ not in series: continue
        A_, B_ = series[L_][tfname], series[F_][tfname]
        tt, ia, ib = np.intersect1d(A_["t"], B_["t"], return_indices=True)
        if len(tt) < 5000: continue
        cl, cf, of = A_["c"][ia], B_["c"][ib], B_["o"][ib]; n = len(tt); day = tt // 86400
        D = len(np.unique(day)); idx = np.arange(n); seg = np.where(idx < n * 0.5, 0, np.where(idx < n * 0.75, 1, 2))
        r1 = np.zeros(n); r1[1:] = cl[1:] / cl[:-1] - 1
        var = np.empty(n); v_ = float(np.var(r1[:1000])) if n > 1000 else 1e-8
        for i in range(n): var[i] = v_; v_ = 0.97 * v_ + 0.03 * r1[i] * r1[i]
        sd = np.sqrt(np.maximum(var, 1e-14)); hr = ((tt // 3600) % 24).astype(int)
        cost = 1e4 * 1.75 * np.array([sptab[F_][x] for x in range(24)])[hr] / np.where(of > 0, of, np.nan)       # bps pe tranzactie
        ofn = np.roll(of, -1)
        for k in (1, 2, 4):
            okk = np.zeros(n, bool); okk[k:] = (tt[k:] - tt[:-k]) == k * sec
            xl = np.full(n, np.nan); xf = np.full(n, np.nan); xl[k:] = cl[k:] / cl[:-k] - 1; xf[k:] = cf[k:] / cf[:-k] - 1; xl[~okk] = np.nan; xf[~okk] = np.nan
            for h_ in (1, 2, 4):
                yy = np.full(n, np.nan); yy[:n - h_] = cf[h_:] / ofn[:n - h_] - 1
                ok_ = np.zeros(n, bool); ok_[:n - h_] = (tt[h_:] - tt[:n - h_]) == h_ * sec; yy[~ok_] = np.nan; yy[n - 1] = np.nan
                ybps = yy * 1e4
                for ver in ("brut", "rez"):
                    if ver == "rez":
                        if L_ == F_: continue
                        trm = (seg == 0) & np.isfinite(xl) & np.isfinite(xf)
                        if trm.sum() < 1000: continue
                        x = xl - float(np.cov(xl[trm], xf[trm])[0, 1] / np.var(xf[trm])) * xf
                    else: x = xl
                    z = x / (sd * np.sqrt(k))
                    for thr in (1.0, 1.5):
                        m = np.isfinite(z) & np.isfinite(ybps) & np.isfinite(cost) & (np.abs(z) >= thr)
                        trm = m & (seg == 0)
                        if trm.sum() < 150: continue
                        s = np.sign(np.mean(np.sign(z[trm]) * ybps[trm]))
                        if s == 0: continue
                        pn = s * np.sign(z[m]) * ybps[m]; net = pn - cost[m]; sm = seg[m]; dm = day[m]
                        row = {"L": L_, "F": F_, "tf": tfname, "k": k, "h": h_, "thr": thr, "ver": ver, "dir": "continuare" if s > 0 else "reversare", "n": int(m.sum()),
                               "pe_an": int(m.sum() / max(D / 252, 1)), "cost_bps": round(float(np.mean(cost[m])), 2)}
                        for nm, mk in (("train", sm == 0), ("val", sm == 1), ("lock", sm == 2), ("tot", sm >= 0)):
                            row["brut_" + nm] = S(pn[mk], dm[mk], 2); row["net_" + nm] = S(net[mk], dm[mk], 2)
                        res.append(row)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt8"); os.makedirs(out, exist_ok=True)

    def save(d):
        open(os.path.join(out, "result.json.tmp"), "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time()), "date": {}, "A": {}, "B": []}
    save(res)
    try:
        rng = np.random.default_rng(5)
        need = {}
        for s_ in ("GOLD", "US100"):
            n_ = sum(os.path.getsize(p) // DT.itemsize for p in glob.glob(os.path.join(a.data, "hist", "%s_15m" % s_, "p_*.bin")))
            need[s_] = n_
        if any(v < int(os.environ.get("S8_MIN15", "150000")) for v in need.values()):                    # < ~6 ani de M15: asteapta descarcarea
            res["state"] = "asteapta"; res["bare_m15"] = need; save(res); return
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        def make_sp(sym):
            sp = spd.get(sym, {}); means = [x["mean"] for x in sp.values() if x["n"] >= 30]; med = float(np.median(means)) if means else None
            def f(hr, px):
                if sym in ASSUMED_SPREAD: return ASSUMED_SPREAD[sym]
                if sym in ("GBPUSD", "USDJPY") and not sp: return {"GBPUSD": 0.00008, "USDJPY": 0.008}[sym]
                e = sp.get(str(hr)); return e["mean"] if e and e["n"] >= 30 else (med or 0.0)
            return f
        syms = sorted({s for p in PAIRS for s in p} | set(A_MARKETS)); M1 = {}; IS15 = {}
        for s in syms:
            try:
                m, is15 = load_tf(a.data, s)
                if len(m["t"]) > (20000 if is15 else 100000): M1[s] = m; IS15[s] = is15
            except Exception: pass
        res["date"] = {s: {"bare_m1": int(len(m["t"])), "de": time.strftime("%Y-%m-%d", time.gmtime(int(m["t"][0]))), "pana": time.strftime("%Y-%m-%d", time.gmtime(int(m["t"][-1]))),
                           "spread": "ipoteza %.2f" % ASSUMED_SPREAD[s] if s in ASSUMED_SPREAD else ("masurat live" if spd.get(s) else "lipsa -> 0 (nesigur)")} for s, m in M1.items()}
        save(res)
        for s in A_MARKETS:
            if s not in M1: continue
            res["A"][s] = run_A(s, M1[s], make_sp(s), None, rng); save(res)
        series = {}
        for s, m in M1.items():
            t = m["t"].astype(np.int64); args = (t, m["o"].astype(np.float64), m["h"].astype(np.float64), m["l"].astype(np.float64), m["c"].astype(np.float64), m["v"].astype(np.float64))
            series[s] = {"M15": agg(*args, 900), "H1": agg(*args, 3600)}
        spf = {s: [make_sp(s)(x, 1.0) for x in range(24)] for s in M1}
        for tfn, sec in (("H1", 3600), ("M15", 900)):
            res["B"].extend(run_B(series, spf, tfn, sec)); save(res)
        B = res["B"]
        cand = [r for r in B if all(r["net_" + k].get("m") is not None and r["net_" + k]["m"] > 0 and r["net_" + k]["n"] >= 50 for k in ("train", "val", "lock")) and (r["net_tot"].get("t") or 0) >= 3.0]
        res["B_candidati"] = sorted(cand, key=lambda r: -(r["net_tot"].get("t") or 0))[:15]
        res["B_total"] = len(B); res["B_net_pozitiv_tot"] = sum(1 for r in B if (r["net_tot"].get("m") or 0) > 0)
        res["B_top_brut_t"] = sorted(B, key=lambda r: -abs(r["brut_tot"].get("t") or 0))[:10]
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("R = rezultat in multipli de risc dupa cost (1.75 x spread la ora de intrare). A: o tranzactie pe sesiune pe zi, SL inaintea TP pe aceeasi bara M15. B: bps dupa cost, semnal la inchiderea barei t, intrare la deschiderea t+1, iesire la inchiderea t+h; "
                       "directia (continuare/reversare) se alege pe train. Candidat = net pozitiv pe train, validare si lockbox (B si cu t>=3 total). Din sute de combinatii, cateva ies bine din intamplare.")
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
