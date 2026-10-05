# Studiu 7: modulele Londra (sweep Asia -> MSS M1 -> FVG/OB) si New York (displacement + delta proxy -> retest FVG) din "Tri-Session Liquidity Engine", testate pe istoric M1 real.
# Backtest complet: ordin limit, SL/TP1/TP2, SL inainte de TP pe aceeasi bara, cost = 1.75 x spread-ul orei de intrare, rezultate in R (dupa cost), train 50% / validare 25% / lockbox 25% pe zile.
# Pragurile in pips sunt exprimate in unitati u = 1 bp din pretul median (EURUSD ~ 1.1 pips), ca sa fie comparabile pe EURUSD / DAX / NIKKEI. Selectia parametrilor se face DOAR pe train.
import os, json, time, itertools, argparse, datetime as dt
import numpy as np
from .run import load_m1
from .tri_engine import atr, bars_to_short, detect_mss, latest_fvg, last_opposite_candle, ny_open_utc, htf_order_blocks, UTC

VER = 1
MARKETS = ["EURUSD", "DAX", "NIKKEI"]
INF = 10 ** 9


def agg(t, o, h, l, c, v, sec, d0):
    k = ((t - d0) // sec).astype(np.int64); u, st = np.unique(k, return_index=True); en = np.append(st[1:], len(k))
    return {"t": (d0 + u * sec).astype(np.float64), "o": o[st], "h": np.maximum.reduceat(h, st), "l": np.minimum.reduceat(l, st), "c": c[en - 1], "v": np.add.reduceat(v, st)}


def tstat(x):
    n = len(x)
    if n < 20: return None
    s = float(np.std(x, ddof=1)); return float(np.mean(x) / (s / np.sqrt(n))) if s > 0 else None


def first(mask):
    i = np.flatnonzero(mask); return int(i[0]) if len(i) else INF


def sim_short(h, l, c, k0, entry, sl, tps, ttl_end, end):
    """Spatiul de short. Ordin limit de vanzare la `entry` valabil pana la ttl_end; apoi SL/TP1/TP2 pana la `end`. SL inaintea TP pe aceeasi bara. Intoarce (R brut, bara de intrare) sau None."""
    if k0 >= len(h): return None
    iF = first(h[k0:ttl_end] >= entry)
    if iF == INF: return None
    kf = k0 + iF; risk = sl - entry
    hh, ll = h[kf:end], l[kf:end]
    if len(hh) == 0: return None
    iS = first(hh >= sl)
    # TP-uri: lista (pret, fractiune); fractiunile insumeaza 1
    R = 0.0; alive = 1.0
    for p, fr in tps:
        iT = first(ll <= p)
        if iT < iS:                       # TP atins strict inainte de SL
            R += fr * (entry - p) / risk; alive -= fr
        else: break
    if alive > 1e-9:
        if iS < INF: R -= alive * 1.0
        else: R += alive * (entry - c[min(end, len(c)) - 1]) / risk
    return R, kf


def stats(R, days):
    R = np.asarray(R, float)
    if len(R) == 0: return {"n": 0}
    u, inv = np.unique(days, return_inverse=True); sd = np.bincount(inv, weights=R - R.mean()); G = len(u)
    t = float(R.mean() / (np.sqrt((sd ** 2).sum() * G / max(G - 1, 1)) / len(R))) if G > 5 and (sd ** 2).sum() > 0 else None
    return {"n": int(len(R)), "R": round(float(R.mean()), 3), "tot": round(float(R.sum()), 1), "win": round(float((R > 0).mean()), 3), "t": round(t, 2) if t is not None else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt7"); os.makedirs(out, exist_ok=True)

    def save(d):
        open(os.path.join(out, "result.json.tmp"), "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time()), "markets": {}}
    save(res)
    try:
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        for sym in MARKETS:
            try: m1 = load_m1(a.data, sym)
            except Exception as e: res["markets"][sym] = {"eroare": repr(e)}; continue
            t, o, h, l, c, v = (m1[k] for k in ("t", "o", "h", "l", "c", "v")); t = t.astype(np.int64)
            sp = spd.get(sym, {}); means = [x["mean"] for x in sp.values() if x["n"] >= 30]; med = float(np.median(means)) if means else None
            sph = {hr: (sp[str(hr)]["mean"] if str(hr) in sp and sp[str(hr)]["n"] >= 30 else med) for hr in range(24)}
            u = 1e-4 * float(np.median(c))
            dayid = t // 86400; days = np.unique(dayid); D = len(days)
            seg_of = {int(d): (0 if i < D * 0.5 else 1 if i < D * 0.75 else 2) for i, d in enumerate(days)}
            lo_i = np.searchsorted(t, days * 86400); hi_i = np.searchsorted(t, days * 86400 + 86400)
            # ---------- context pe zi ----------
            ctx = []
            for di, d in enumerate(days):
                a0, b0 = lo_i[di], hi_i[di]
                if b0 - a0 < 600: continue
                d0 = int(d) * 86400
                tt = t[a0:b0]; mm = (tt - d0) // 60
                asia = (mm >= 0) & (mm < 480)
                if asia.sum() < 0.8 * 480: continue
                dh, dl = float(h[a0:b0][asia].max()), float(l[a0:b0][asia].min())
                end_m = 21 * 60
                endi = int(np.searchsorted(mm, end_m))
                ctx.append({"d": int(d), "seg": seg_of[int(d)], "a0": a0, "b0": b0, "d0": d0, "AH": dh, "AL": dl, "end": endi, "yr": dt.datetime.fromtimestamp(d0, UTC).year})
            mres = {"zile": len(ctx), "u_pret": round(u, 6), "cost_bps_med": round(1e4 * 1.75 * med / float(np.median(c)), 2) if med else None,
                    "de": time.strftime("%Y-%m-%d", time.gmtime(int(t[0]))), "pana": time.strftime("%Y-%m-%d", time.gmtime(int(t[-1])))}
            # ---------- LONDRA ----------
            def london(params):
                pen_u, body, min_rr, tpm = params
                trades = []
                for x in ctx:
                    a0, b0, d0 = x["a0"], x["b0"], x["d0"]
                    tt = t[a0:b0]; o_, h_, l_, c_, v_ = o[a0:b0], h[a0:b0], l[a0:b0], c[a0:b0], v[a0:b0]
                    # M5 din M1
                    mm5 = agg(tt, o_, h_, l_, c_, v_, 300, d0)
                    ntr = 0; cur_end_t = d0 + 8 * 3600
                    for i in range(len(mm5["t"])):
                        bt = mm5["t"][i]; bend = bt + 300
                        if bend < d0 + 8 * 3600 or bend > d0 + 12 * 3600 + 1800 or bend <= cur_end_t: continue
                        if ntr >= 2: break
                        side = 0
                        if mm5["h"][i] >= x["AH"] + pen_u * u and mm5["c"][i] < x["AH"]: side = -1; ext0 = float(mm5["h"][max(0, i - 5):i + 1].max())
                        elif mm5["l"][i] <= x["AL"] - pen_u * u and mm5["c"][i] > x["AL"]: side = 1; ext0 = float(mm5["l"][max(0, i - 5):i + 1].min())
                        else: continue
                        te = int(np.searchsorted(tt, bend + 90 * 60)); ts = int(np.searchsorted(tt, bend))
                        m1d = {"t": tt, "o": o_, "h": h_, "l": l_, "c": c_, "v": v_}
                        s = bars_to_short({k: vv[:te] for k, vv in m1d.items()}, side)
                        atr1 = atr(s, 14)
                        if not np.isfinite(atr1): continue
                        first_i = int(np.searchsorted(s["t"], bend - 600))
                        ms = detect_mss(s, first_i, body, atr1)
                        if ms is None: continue
                        j, _ = ms
                        sj = {k: vv[:j + 1] for k, vv in s.items()}
                        zone = None; fv = latest_fvg(sj, max(j - 1, 2))
                        if fv and fv[2] >= j: zone = (fv[0], fv[1])
                        else: zone = last_opposite_candle(sj, j)
                        if zone is None: continue
                        entry = zone[0] + 0.5 * (zone[1] - zone[0])
                        ext = max(ext0, float(s["h"][max(first_i - 1, 0):j + 1].max()))
                        # M5 ATR pana la momentul j
                        k5 = int(np.searchsorted(mm5["t"] + 300, tt[j] + 1))
                        a5 = atr({k: vv[:k5] for k, vv in mm5.items()}, 14)
                        if not np.isfinite(a5): continue
                        hr = int((tt[j] - d0) // 3600); spr = sph.get(hr) or 0.0
                        sl = ext + 0.15 * a5 + spr; risk = sl - entry
                        if risk < 4 * u or risk > 25 * u: continue
                        far = x["AL"] if side < 0 else -x["AH"]; eq = (x["AH"] + x["AL"]) / 2 if side < 0 else -(x["AH"] + x["AL"]) / 2
                        if (entry - far) / risk < min_rr: continue
                        tps = [(eq, 0.5), (far, 0.5)] if tpm == "split" else [(far, 1.0)]
                        sh = bars_to_short(m1d, side)
                        r = sim_short(sh["h"], sh["l"], sh["c"], j + 1, entry, sl, tps, min(j + 1 + 60, len(tt)), x["end"] - 0)
                        if r is None: continue
                        R, kf = r
                        cost = 1.75 * (sph.get(int((tt[kf] - d0) // 3600)) or 0.0)
                        trades.append((x["d"], x["seg"], x["yr"], side, R, R - cost / risk, risk / u))
                        ntr += 1; cur_end_t = tt[min(kf + 1, len(tt) - 1)]
                return trades
            grid = list(itertools.product((1.0, 1.5, 3.0), (0.8, 1.2), (1.0, 1.5), ("split", "far")))
            Lrows = []
            for g in grid:
                tr = london(g)
                row = {"pen_u": g[0], "body_atr": g[1], "min_rr": g[2], "tp": g[3], "n": len(tr)}
                if tr:
                    arr = np.array([(q[0], q[1], q[2], q[3], q[4], q[5], q[6]) for q in tr], float)
                    for nm, sg in (("train", 0), ("val", 1), ("lock", 2)):
                        mk = arr[:, 1] == sg; row[nm] = stats(arr[mk, 5], arr[mk, 0]) if mk.any() else {"n": 0}
                    row["tot"] = stats(arr[:, 5], arr[:, 0]); row["brut_R"] = round(float(arr[:, 4].mean()), 3); row["risc_u"] = round(float(np.median(arr[:, 6])), 1)
                    row["lung"] = round(float((arr[:, 3] > 0).mean()), 2); row["pe_an"] = len(arr) / max(D / 250, 1)
                    row["_tr"] = tr if g == grid[0] else None
                Lrows.append(row)
                res["markets"][sym] = {**mres, "londra": [{k: v_ for k, v_ in r_.items() if k != "_tr"} for r_ in Lrows]}; save(res)
            # ---------- NEW YORK ----------
            h1cache = {}
            nyday = []
            for x in ctx:
                a0, b0, d0 = x["a0"], x["b0"], x["d0"]
                tt = t[a0:b0]
                n0 = ny_open_utc(dt.datetime.fromtimestamp(d0 + 43200, UTC)).timestamp(); n1 = n0 + 900
                w0 = int(np.searchsorted(tt, n0)); w1 = int(np.searchsorted(tt, n1))
                if w1 - w0 < 12 or tt[min(w1, len(tt) - 1)] > n1 + 120: continue
                o_, h_, l_, c_, v_ = o[a0:b0], h[a0:b0], l[a0:b0], c[a0:b0], v[a0:b0]
                m5 = agg(tt[:w1], o_[:w1], h_[:w1], l_[:w1], c_[:w1], v_[:w1], 300, d0); a5 = atr(m5, 14)
                if not np.isfinite(a5): continue
                move = float(c_[w1 - 1] - o_[w0]); sg = np.sign(c_[w0:w1] - o_[w0:w1]); vv = v_[w0:w1]; delta = float((sg * vv).sum() / max(vv.sum(), 1e-9))
                side = 1 if move > 0 else -1
                sd = bars_to_short({"t": tt, "o": o_, "h": h_, "l": l_, "c": c_, "v": v_}, side)
                fv = latest_fvg({k: q[:w1] for k, q in sd.items()}, w0)
                lm = (tt >= d0 + 8 * 3600) & (tt < d0 + 12 * 3600 + 1800)
                lhl = (float(h_[lm].max()), float(l_[lm].min())) if lm.any() else None
                # H1 order blocks (din ultimele 200 de ore, fara viitor)
                hs = max(a0 - 200 * 60, 0); tt_h = t[hs:a0 + w1]; hr_ = agg(tt_h, o[hs:a0 + w1], h[hs:a0 + w1], l[hs:a0 + w1], c[hs:a0 + w1], v[hs:a0 + w1], 3600, int(tt_h[0] // 3600 * 3600))
                obs = htf_order_blocks(hr_)
                nyday.append({"x": x, "tt": tt, "sd": sd, "w0": w0, "w1": w1, "a5": a5, "move": move, "delta": delta, "side": side, "fvg": fv, "lhl": lhl, "obs": obs})
            def newyork(params):
                disp, dmin, veto, tpm = params
                trades = []
                for q in nyday:
                    x, sd, w0, w1, a5, side = q["x"], q["sd"], q["w0"], q["w1"], q["a5"], q["side"]
                    if abs(q["move"]) < disp * a5 or np.sign(q["delta"]) != side or abs(q["delta"]) < dmin or q["fvg"] is None: continue
                    zone = (q["fvg"][0], q["fvg"][1]); entry = zone[0] + 0.5 * (zone[1] - zone[0])
                    tt = q["tt"]; hr = int((tt[w1 - 1] - x["d0"]) // 3600); spr = sph.get(hr) or 0.0
                    sl = float(sd["h"][w0:w1].max()) + 0.1 * a5 + spr; risk = sl - entry
                    if risk < 4 * u or risk > 25 * u: continue
                    if veto and any(k == -side and lo <= (entry if side < 0 else -entry) <= hi for lo, hi, k in q["obs"]): continue
                    tp = entry - 2.0 * risk if tpm != "1.5R" else entry - 1.5 * risk
                    if tpm == "londra" and q["lhl"]:
                        lvl = q["lhl"][1] if side < 0 else -q["lhl"][0]
                        if lvl < entry and (entry - lvl) / risk >= 1.5: tp = lvl
                    r = sim_short(sd["h"], sd["l"], sd["c"], w1, entry, sl, [(tp, 1.0)], min(w1 + 75, len(tt)), x["end"])
                    if r is None: continue
                    R, kf = r
                    cost = 1.75 * (sph.get(int((tt[kf] - x["d0"]) // 3600)) or 0.0)
                    trades.append((x["d"], x["seg"], x["yr"], side, R, R - cost / risk, risk / u))
                return trades
            Nrows = []
            for g in itertools.product((0.8, 1.2, 1.6), (0.0, 0.15, 0.3), (True, False), ("londra", "2R", "1.5R")):
                tr = newyork(g); row = {"disp_atr": g[0], "delta_min": g[1], "veto_htf": g[2], "tp": g[3], "n": len(tr)}
                if tr:
                    arr = np.array(tr, float)
                    for nm, sg in (("train", 0), ("val", 1), ("lock", 2)):
                        mk = arr[:, 1] == sg; row[nm] = stats(arr[mk, 5], arr[mk, 0]) if mk.any() else {"n": 0}
                    row["tot"] = stats(arr[:, 5], arr[:, 0]); row["brut_R"] = round(float(arr[:, 4].mean()), 3); row["risc_u"] = round(float(np.median(arr[:, 6])), 1); row["pe_an"] = len(arr) / max(D / 250, 1)
                Nrows.append(row)
            res["markets"][sym] = {**mres, "londra": [{k: v_ for k, v_ in r_.items() if k != "_tr"} for r_ in Lrows], "ny": Nrows}; save(res)
        # selectie pe train (n>=60), raportare val/lock
        res["alese"] = {}
        for sym, m in res["markets"].items():
            for mod in ("londra", "ny"):
                rows = [r for r in m.get(mod, []) if r.get("train", {}).get("n", 0) >= 60]
                if not rows: continue
                best = max(rows, key=lambda r: r["train"]["R"]); res["alese"]["%s/%s" % (sym, mod)] = best
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = "R = rezultat net in multipli de risc (dupa cost 1.75 x spread-ul orei de intrare). 'alese' = combinatia cu cel mai bun R pe train (n>=60); val si lock sunt neatinse la alegere."
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
