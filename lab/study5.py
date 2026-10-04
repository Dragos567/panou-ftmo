# Studiu 5: (A) efectul de sesiune / ora din zi (Breedon & Ranaldo: moneda locala se depreciaza in orele ei de tranzactionare), pe EURUSD, DAX, NIKKEI;
#           (B) continuare vs reversare dupa miscari de marimi diferite (z = randament / volatilitate recenta), pe bare H1 si M15, pe cele trei piete.
# Se masoara medii in bps, t cu erori grupate pe zi, pe train 50% / validare 25% / lockbox 25% (in timp), comparate cu costul (1.75 x spread) per tranzactie.
import os, sys, json, time, argparse
import numpy as np
from .run import load_m1
from .data import resample

VER = 1
MARKETS = ["EURUSD", "DAX", "NIKKEI"]
SESS = [("Asia 00-07", 0, 7), ("Europa 07-13", 7, 13), ("SUA 13-21", 13, 21), ("Europa+SUA 07-21", 7, 21), ("Noapte 21-24", 21, 24)]   # UTC


def ctstat(x, day):
    n = len(x)
    if n < 20: return None
    m = float(x.mean()); u, inv = np.unique(day, return_inverse=True); G = len(u)
    if G < 5: return None
    s = np.bincount(inv, weights=x - m)
    var = float((s ** 2).sum()) * G / (G - 1) / (n ** 2)
    return m / np.sqrt(var) if var > 0 else None


def stats(x, day, seg):
    out = {}
    for gn, gm in (("train", seg == 0), ("val", seg == 1), ("lock", seg == 2), ("tot", np.ones(len(x), bool))):
        v = x[gm]; tt = ctstat(v, day[gm]) if len(v) else None
        out[gn] = {"n": int(gm.sum()), "bps": round(float(v.mean()), 2) if len(v) else None, "t": round(tt, 2) if tt is not None else None}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt5"); os.makedirs(out, exist_ok=True)

    def save(d):
        json.dump(d, open(os.path.join(out, "result.json.tmp"), "w")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time()), "markets": {}, "hours": [], "sessions": [], "bins": [], "candidates": []}
    save(res)
    try:
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        for sym in MARKETS:
            try: m1 = load_m1(a.data, sym)
            except Exception as e:
                res["markets"][sym] = {"eroare": repr(e)}; continue
            sp = None
            try:
                means = [v["mean"] for v in spd.get(sym, {}).values() if v["n"] >= 30]
                if means: sp = float(np.median(means))
            except Exception: pass
            m = resample(m1, 3600); t = m["t"]; o, c = m["o"].astype(np.float64), m["c"].astype(np.float64); n = len(t)
            px = float(np.median(c)); cost_bps = 1e4 * 1.75 * sp / px if sp else None
            res["markets"][sym] = {"cost_bps": round(cost_bps, 2) if cost_bps else None, "de": time.strftime("%Y-%m-%d", time.gmtime(int(t[0]))), "pana": time.strftime("%Y-%m-%d", time.gmtime(int(t[-1]))), "bare_h1": int(n)}
            day = (t // 86400).astype(np.int64); hr = ((t // 3600) % 24).astype(np.int64)
            seg = np.where(np.arange(n) < n * 0.5, 0, np.where(np.arange(n) < n * 0.75, 1, 2))
            ret = (c - o) / o * 1e4
            # (A1) pe ore UTC
            for h in range(24):
                mk = hr == h
                if mk.sum() > 300: res["hours"].append({"mk": sym, "ora_utc": h, **stats(ret[mk], day[mk], seg[mk])})
            # (A2) pe sesiuni (un randament pe zi: de la prima deschidere la ultima inchidere din interval)
            for nm, a0, b0 in SESS:
                ins = (hr >= a0) & (hr < b0)
                idx = np.flatnonzero(ins)
                if len(idx) < 300: continue
                d_ = day[idx]; cut = np.flatnonzero(np.diff(d_)) + 1; st = np.concatenate(([0], cut)); en = np.concatenate((cut, [len(idx)])) - 1
                cnt = en - st + 1; good = cnt >= max(2, (b0 - a0) - 2)
                s0 = idx[st][good]; e0 = idx[en][good]
                sr = (c[e0] - o[s0]) / o[s0] * 1e4
                srow = {"mk": sym, "sesiune": nm, **stats(sr, day[s0], seg[s0])}
                res["sessions"].append(srow)
                tr, va, lk, tot = srow["train"], srow["val"], srow["lock"], srow["tot"]
                if all(z["bps"] is not None and z["n"] >= 100 for z in (tr, va, lk)) and tot["t"] is not None:
                    sg = np.sign(tr["bps"])
                    if sg != 0 and np.sign(va["bps"]) == sg and np.sign(lk["bps"]) == sg and abs(tot["t"]) >= 2.0:
                        res["candidates"].append({"tip": "sesiune", "mk": sym, "sesiune": nm, "directie": "long" if sg > 0 else "short", "tr": tr, "va": va, "lk": lk, "tot": tot, "net_bps": round(abs(tot["bps"]) - (cost_bps or 0), 2), "acopera_costul": bool(cost_bps and abs(tot["bps"]) > cost_bps)})
            # (B) bine dupa marimea miscarii, pe H1 si M15
            for tfn, sec in (("H1", 3600), ("M15", 900)):
                mm = resample(m1, sec); tt = mm["t"]; oo, cc = mm["o"].astype(np.float64), mm["c"].astype(np.float64); nn = len(tt)
                r = (cc - oo) / oo * 1e4
                lam = 0.97; var = np.empty(nn); v_ = np.var(r[:500]) if nn > 500 else 1.0
                for i in range(nn):
                    var[i] = v_; v_ = lam * v_ + (1 - lam) * r[i] * r[i]          # sigma la inceputul barei i (fara r[i])
                z = r / np.sqrt(var)
                dd = (tt // 86400).astype(np.int64); sg_ = np.where(np.arange(nn) < nn * 0.5, 0, np.where(np.arange(nn) < nn * 0.75, 1, 2))
                for hz in (1, 2):
                    fwd = np.full(nn, np.nan)
                    fwd[:nn - hz] = (cc[hz:] - cc[:nn - hz]) / cc[:nn - hz] * 1e4
                    # doar ferestre cu timp contiguu (fara goluri de weekend): hz bare inainte la distanta exacta
                    okt = np.zeros(nn, bool); okt[:nn - hz] = (tt[hz:] - tt[:nn - hz]) == hz * sec
                    for lo_, hi_ in ((0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 99)):
                        mk = okt & (np.abs(z) >= lo_) & (np.abs(z) < hi_) & ~np.isnan(fwd) & (var > 0)
                        if mk.sum() < 200: continue
                        x = np.sign(z[mk]) * fwd[mk]               # + = continuare in sensul miscarii
                        row = {"mk": sym, "tf": tfn, "H_bare": hz, "z": "%s-%s" % (lo_, hi_ if hi_ < 99 else "+"), **stats(x, dd[mk], sg_[mk])}
                        res["bins"].append(row)
                        tr, va, lk, tot = row["train"], row["val"], row["lock"], row["tot"]
                        if all(q["bps"] is not None and q["n"] >= 100 for q in (tr, va, lk)) and tot["t"] is not None:
                            sg = np.sign(tr["bps"])
                            if sg != 0 and np.sign(va["bps"]) == sg and np.sign(lk["bps"]) == sg and abs(tot["t"]) >= 2.5:
                                res["candidates"].append({"tip": "marime miscare", "mk": sym, "tf": tfn, "H_bare": hz, "z": row["z"], "efect": "continuare" if sg > 0 else "reversare", "tr": tr, "va": va, "lk": lk, "tot": tot,
                                                          "net_bps": round(abs(tot["bps"]) - (cost_bps or 0), 2), "acopera_costul": bool(cost_bps and abs(tot["bps"]) > cost_bps)})
            save(res)
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = "bps = puncte de baza; la bins + = continuare in sensul miscarii anterioare. Candidat = acelasi semn pe train/val/lock, n>=100, |t|>=2.0 (sesiuni) sau 2.5 (bins)."
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1500:]; save(res)


if __name__ == "__main__":
    main()
