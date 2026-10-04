# Studiu 6: backtest complet al driftului EURUSD de noapte (long, ~21-24 UTC), cu costuri REALE pe ora intrarii, swap la rollover, sensibilitate la spread (x1/x2/x3),
# split train 50% / validare 25% / lockbox 25%, impartire pe an / DST SUA / zi a saptamanii, grup de control (aceleasi ferestre pe toate orele) si simulare FTMO 2-step (bootstrap).
# Varianta "primara" se alege DOAR pe train (cel mai mare net la costul x1), apoi se raporteaza val/lock. O tranzactie pe zi, fara SL/TP: intrare la deschiderea minutului E, iesire la deschiderea minutului X (UTC).
import os, json, time, argparse, datetime as dt
import numpy as np
from .run import load_m1

VER = 3
SYM = "EURUSD"
ENTRY = (1260, 1290, 1320, 1350)       # 21:00, 21:30, 22:00, 22:30 UTC
EXIT = (1380, 1410, 1440)              # 23:00, 23:30, 24:00 UTC
SWAP_BPS = 0.7                         # ipoteza: cost de swap pe noapte pentru long EURUSD (diferential de dobanda + marja broker), miercuri x3
COMM_BPS = 0.0                         # comisionul e inclus in cost = 1.75 x spread (ca in studiile anterioare)


def us_dst(day):
    """True daca ziua (zile de la epoca) e in DST-ul SUA (a 2-a duminica din martie .. prima duminica din noiembrie)."""
    d = dt.date(1970, 1, 1) + dt.timedelta(days=int(day)); y = d.year
    m = dt.date(y, 3, 1); m += dt.timedelta(days=(6 - m.weekday()) % 7 + 7)
    n = dt.date(y, 11, 1); n += dt.timedelta(days=(6 - n.weekday()) % 7)
    return m <= d < n


def tstat(x):
    n = len(x)
    if n < 20: return None
    s = float(x.std(ddof=1))
    return float(x.mean() / (s / np.sqrt(n))) if s > 0 else None


def seg_stats(x):
    return {"n": int(len(x)), "bps": round(float(x.mean()), 3) if len(x) else None, "t": (round(tstat(x), 2) if tstat(x) is not None else None),
            "win": round(float((x > 0).mean()), 3) if len(x) else None}


def ftmo_sim(net, mae, L, rng, N=3000, T=300):
    """Bootstrap pe tranzactii: faza 1 (+10%), faza 2 (+5%), pierdere zilnica 5%, pierdere maxima 10% (inclusiv excursiunea nefavorabila din tranzactie)."""
    out = []
    for tgt in (0.10, 0.05):
        ix = rng.integers(0, len(net), (N, T))
        r = L * net[ix] / 1e4; m = L * mae[ix] / 1e4
        cum = np.cumsum(r, axis=1); low = cum - r - m
        br = (m >= 0.05) | (low <= -0.10)
        up = cum >= tgt; up[:, :3] = False                       # minim 4 zile de tranzactionare
        has_up = up.any(axis=1); t_up = np.where(has_up, up.argmax(axis=1), T + 1)
        has_br = br.any(axis=1); t_br = np.where(has_br, br.argmax(axis=1), T + 1)
        ok = has_up & (t_up < t_br)
        out.append((float(ok.mean()), float(np.median(t_up[ok]) + 1) if ok.any() else None, float((has_br & (t_br <= t_up)).mean())))
    return {"L": L, "p1": round(out[0][0], 3), "p2": round(out[1][0], 3), "p_total": round(out[0][0] * out[1][0], 3),
            "zile_f1": out[0][1], "zile_f2": out[1][1], "cade_f1": round(out[0][2], 3), "cade_f2": round(out[1][2], 3)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt6"); os.makedirs(out, exist_ok=True)

    def save(d):
        open(os.path.join(out, "result.json.tmp"), "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time())}
    save(res)
    try:
        spd = json.load(open(os.path.join(a.data, "spread.json"))).get(SYM, {})
        means = [v["mean"] for v in spd.values() if v["n"] >= 30]
        med = float(np.median(means)) if means else None
        sp_h = {}
        for h in range(24):
            e = spd.get(str(h)); sp_h[h] = (e["mean"] if e and e["n"] >= 30 else med)
        m1 = load_m1(a.data, SYM); t = m1["t"]; o = m1["o"].astype(np.float64); lo = m1["l"].astype(np.float64)
        res["spread_orar_UTC"] = [{"h": h, "spread_pret": sp_h[h], "n": (spd.get(str(h)) or {}).get("n", 0), "bps": round(1e4 * sp_h[h] / float(np.median(o)), 3) if sp_h[h] else None} for h in range(24)]
        days = np.unique(t // 86400)
        D = len(days)
        res["meta"] = {"de": time.strftime("%Y-%m-%d", time.gmtime(int(t[0]))), "pana": time.strftime("%Y-%m-%d", time.gmtime(int(t[-1]))), "zile": int(D), "swap_bps_noapte": SWAP_BPS,
                       "nota_cost": "cost = 1.75 x spread-ul mediu al orei de intrare (masurat live pe contul FTMO); x2/x3 simuleaza largirea spread-ului la rollover"}
        dst = np.array([us_dst(d) for d in days]); wd = ((days + 3) % 7).astype(int)          # 0 = luni
        yr = np.array([(dt.date(1970, 1, 1) + dt.timedelta(days=int(d))).year for d in days])
        seg = np.where(np.arange(D) < D * 0.5, 0, np.where(np.arange(D) < D * 0.75, 1, 2))

        def trades(E, X):
            te = days * 86400 + E * 60; tx = days * 86400 + X * 60
            ie = np.searchsorted(t, te); ix_ = np.searchsorted(t, tx)
            ok = (ie < len(t)) & (ix_ < len(t))
            ok &= (t[np.minimum(ie, len(t) - 1)] == te) & (t[np.minimum(ix_, len(t) - 1)] == tx)
            ie = np.minimum(ie, len(t) - 1); ix_ = np.minimum(ix_, len(t) - 1)
            gross = np.full(D, np.nan); mae = np.full(D, np.nan)
            for k in np.flatnonzero(ok):
                p0 = o[ie[k]]; gross[k] = (o[ix_[k]] - p0) / p0 * 1e4
                mae[k] = max(0.0, (p0 - lo[ie[k]:ix_[k]].min()) / p0 * 1e4)
            sp = sp_h[E // 60] or med
            cost = 1e4 * 1.75 * sp / np.where(ok, o[ie], 1.0)
            roll = np.where(dst, 1260, 1320)                          # rollover 17:00 NY = 21:00 UTC vara / 22:00 UTC iarna
            cross = (E < roll) & (roll <= X)
            swap = np.where(cross, SWAP_BPS * np.where(wd == 2, 3, 1), 0.0)
            return ok, gross, mae, cost, swap, cross

        variants = []; store = {}
        for E in ENTRY:
            for X in EXIT:
                if E >= X: continue
                ok, g, mae, cost, swap, cross = trades(E, X)
                row = {"E": "%02d:%02d" % (E // 60, E % 60), "X": "%02d:%02d" % (X // 60 % 24, X % 60), "zile_cu_date": int(ok.sum())}
                for nm, mk in (("tot", ok), ("train", ok & (seg == 0)), ("val", ok & (seg == 1)), ("lock", ok & (seg == 2))):
                    row[nm] = seg_stats(g[mk])
                for mult in (1, 2, 3):
                    nt = g - mult * cost - swap
                    for nm, mk in (("train", ok & (seg == 0)), ("val", ok & (seg == 1)), ("lock", ok & (seg == 2)), ("tot", ok)):
                        row.setdefault("net%d" % mult, {})[nm] = round(float(nt[mk].mean()), 3) if mk.any() else None
                row["cost_bps"] = round(float(np.nanmean(cost[ok])), 3) if ok.any() else None
                row["swap_medio"] = round(float(swap[ok].mean()), 3) if ok.any() else None
                row["breakeven_mult"] = round(float((np.nanmean(g[ok]) - swap[ok].mean()) / np.nanmean(cost[ok])), 2) if ok.any() else None
                variants.append(row); store[(E, X)] = (ok, g, mae, cost, swap, cross)
        res["variante"] = variants
        # --- ferestre de ZI / toate sesiunile: directia (long/short) se alege pe train, apoi se raporteaza val/lock ---
        WIN = [("Asia 00-07", 0, 420), ("Europa 07-13", 420, 780), ("Londra 07-12", 420, 720), ("Overlap 12-16", 720, 960), ("Londra+NY 08-16", 480, 960),
               ("SUA 13-21", 780, 1260), ("NY 16-21", 960, 1260), ("Europa+SUA 07-21", 420, 1260), ("Zi completa 00-21", 0, 1260)]
        sess = []; sstore = {}
        for nm, E, X in WIN:
            ok, g, mae, cost, swap, cross = trades(E, X)
            tr = ok & (seg == 0)
            if tr.sum() < 100: continue
            sg = 1.0 if g[tr].mean() >= 0 else -1.0
            gd = sg * g; sw = swap if sg > 0 else np.zeros(D)
            row = {"fereastra": nm, "directie": "long" if sg > 0 else "short", "n": int(ok.sum()), "cost_bps": round(float(np.nanmean(cost[ok])), 3)}
            for mult in (1, 2):
                nt = gd - mult * cost - sw
                row["net%d" % mult] = {k: (round(float(nt[mk].mean()), 3) if mk.any() else None) for k, mk in (("train", tr), ("val", ok & (seg == 1)), ("lock", ok & (seg == 2)), ("tot", ok))}
            for k, mk in (("train", tr), ("val", ok & (seg == 1)), ("lock", ok & (seg == 2)), ("tot", ok)): row[k] = seg_stats(gd[mk])
            row["pe_dst"] = [{"grup": lb, **seg_stats(gd[ok & (dst == v)])} for v, lb in ((True, "DST SUA"), (False, "iarna"))]
            row["pe_an"] = [{"grup": str(y), **seg_stats(gd[ok & (yr == y)])} for y in sorted(set(yr.tolist()))]
            sess.append(row); sstore[nm] = (ok, gd - cost - sw, np.where(sg > 0, mae, 0.0))
        res["sesiuni_zi"] = sess
        # --- toate ferestrele orare (start la fiecare ora UTC, 1h/2h/3h/4h), directia aleasa pe train; candidat = acelasi semn net pe val si lock ---
        ore = []
        for dur in (60, 120, 180, 240):
            for h in range(24):
                E = h * 60; X = E + dur
                ok2, g2, mae2, c2, s2, cr2 = trades(E, X)
                tr2 = ok2 & (seg == 0)
                if tr2.sum() < 200: continue
                sg = 1.0 if g2[tr2].mean() >= 0 else -1.0
                gd = sg * g2; sw = s2 if sg > 0 else np.zeros(D); nt = gd - c2 - sw
                r = {"ora_utc": h, "dur_h": dur // 60, "dir": "long" if sg > 0 else "short", "n": int(ok2.sum()), "cost": round(float(np.nanmean(c2[ok2])), 3), "tot": seg_stats(gd[ok2])}
                for k, mk in (("train", tr2), ("val", ok2 & (seg == 1)), ("lock", ok2 & (seg == 2))): r[k] = round(float(gd[mk].mean()), 3); r["net_" + k] = round(float(nt[mk].mean()), 3)
                r["net_tot"] = round(float(nt[ok2].mean()), 3)
                r["net2_tot"] = round(float((gd - 2 * c2 - sw)[ok2].mean()), 3)
                r["stabil"] = bool(r["net_train"] > 0 and r["net_val"] > 0 and r["net_lock"] > 0)
                ore.append(r)
        res["ferestre_orare"] = ore
        # simulare FTMO pentru cea mai buna fereastra de zi (alegere pe train, net x1), excluzand noaptea
        cand = [r for r in sess if r["net1"]["train"] is not None]
        if cand:
            bw = max(cand, key=lambda r: r["net1"]["train"]); okw, nw, mw = sstore[bw["fereastra"]]
            if bw["directie"] == "short":                       # mae pentru short = excursiunea in sus; o aproximam cu cea mai mare pierdere realizata (conservator)
                mw = np.maximum(0.0, -nw) * 1.5
            rng0 = np.random.default_rng(11)
            res["ftmo_zi"] = {"fereastra": bw["fereastra"], "directie": bw["directie"], "rezultate": [ftmo_sim(nw[okw], mw[okw], L, rng0) for L in (5, 10, 20)]}
        # --- varianta primara: aleasa DOAR pe train (net x1) ---
        best = max(variants, key=lambda r: (r["net1"]["train"] if r["net1"]["train"] is not None else -9))
        Eb = [e for e in ENTRY if "%02d:%02d" % (e // 60, e % 60) == best["E"]][0]
        Xb = [x for x in EXIT if "%02d:%02d" % (x // 60 % 24, x % 60) == best["X"]][0]
        ok, g, mae, cost, swap, cross = store[(Eb, Xb)]
        P = {"fereastra": "%s-%s UTC" % (best["E"], best["X"]), "alegere": "cea mai buna pe train la cost x1 (net dupa swap)", "stat": best}
        net1 = g - cost - swap
        grp = lambda keyarr, labels: [{"grup": lb, **seg_stats(g[ok & (keyarr == k)]), "net1": (round(float(net1[ok & (keyarr == k)].mean()), 3) if (ok & (keyarr == k)).any() else None)} for k, lb in labels]
        P["pe_an"] = grp(yr, [(y, str(y)) for y in sorted(set(yr.tolist()))])
        P["pe_dst"] = grp(dst.astype(int), [(1, "DST SUA (vara)"), (0, "ora de iarna")])
        P["pe_zi"] = grp(wd, [(0, "luni"), (1, "marti"), (2, "miercuri"), (3, "joi"), (4, "vineri")])
        P["traversare_rollover"] = grp(cross.astype(int), [(1, "tine peste rollover"), (0, "nu tine peste rollover")])
        x = net1[ok]; cum = np.cumsum(x)
        P["net1"] = {"medie_bps": round(float(x.mean()), 3), "sharpe_anual": round(float(x.mean() / x.std() * np.sqrt(252)), 2), "max_dd_bps": round(float((np.maximum.accumulate(cum) - cum).max()), 1),
                     "tranzactii": int(len(x)), "castig_total_bps": round(float(cum[-1]), 1), "std_bps": round(float(x.std()), 2)}
        # --- control: aceeasi durata de fereastra (1h si 3h) pornita la fiecare ora UTC ---
        plc = []
        for dur in (60, 180):
            for h in range(24):
                E = h * 60; X = E + dur
                ok2, g2, _, c2, s2, _ = trades(E, X)
                if ok2.sum() < 300: continue
                gg = g2[ok2]; plc.append({"dur_min": dur, "ora_utc": h, "n": int(ok2.sum()), "bps": round(float(gg.mean()), 3), "t": round(tstat(gg), 2), "net1": round(float((gg - c2[ok2]).mean()), 3)})
        res["control"] = plc
        # --- FTMO 2-step (bootstrap pe tranzactii, o tranzactie/zi) ---
        rng = np.random.default_rng(7)
        okv = ok & ~np.isnan(g)
        scen = {"toate zilele, cost x1": (net1[okv], mae[okv]),
                "doar val+lock, cost x1": (net1[okv & (seg >= 1)], mae[okv & (seg >= 1)]),
                "toate zilele, cost x2": ((g - 2 * cost - swap)[okv], mae[okv]),
                "control (fara avantaj: media scoasa)": (net1[okv] - net1[okv].mean(), mae[okv])}
        res["ftmo"] = {nm: [ftmo_sim(nv, mv, L, rng) for L in (5, 10, 15, 20, 30)] for nm, (nv, mv) in scen.items()}
        res["primara"] = P
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("Net = brut - cost - swap. Varianta primara e aleasa pe train, deci val/lock sunt neatinse. Simularea FTMO foloseste leverage L (valoare nominala / capital) si tranzactii extrase la intamplare din istoric; "
                       "nu are limita de timp (max 300 zile) si ignora gap-urile de weekend. Costul real al spread-ului la rollover poate depasi media masurata - vezi coloanele x2/x3.")
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1500:]; save(res)


if __name__ == "__main__":
    main()
