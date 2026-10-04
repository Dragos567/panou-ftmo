# Strategie clasica: MA 50/100/200 (trend) + retragere Fibonacci + StochRSI/RSI (declansare), intrare pe M15 sau H1, pe DAX / NIKKEI / EURUSD.
# Reguli (fixate de mine, pentru ca nu au fost specificate; variantele sunt in grila):
#  - Trend long: MA50 > MA100 > MA200 (regula 0) sau close > MA200 si MA50 > MA200 (regula 1); short invers. MA = SMA sau EMA.
#  - Piciorul: ultimul swing low urmat de swing high (fractal 3 bare, confirmat dupa 3 bare); retragerea Fibonacci = (swing high - close) / picior,
#    trebuie sa fie in banda aleasa (0.382-0.618 sau 0.5-0.786), pretul sa nu fi stricat swing-ul low, piciorul recent (<= 80 bare) si >= 1 ATR.
#  - Declansator long: StochRSI(14,14,3,3) K trece peste D, cu K < 30 inainte (mod 0) sau doar K > D si K in crestere (mod 1); optional RSI(14) intre 30 si 55.
#  - Intrare la inchiderea barei. SL: sub swing low - 0.1 ATR (mod 0) sau 1.5 ATR (mod 1). TP: k x risc (1.5 / 2 / 3). Iesire fortata dupa 96 bare M15 / 48 bare H1.
#  - Ordinea SL/TP in aceeasi bara: mereu SL primul (pesimist). Cost = 1.75 x spread mediu (aceeasi regula ca la studiile anterioare), scazut din fiecare tranzactie. Risc minim = 2 x cost.
# O pozitie la un moment dat; mai multe tranzactii pe zi sunt permise. Segmente: train 50% / validare 25% / lockbox 25% (in timp).
import os, sys, json, time, argparse, itertools
import numpy as np
from .nb import njit
from .run import load_m1
from .data import resample

VER = 1
MARKETS = ["DAX", "NIKKEI", "EURUSD"]
TFS = {"M15": 900, "H1": 3600}


@njit(cache=True)
def indicators(c, h, l, kind):
    n = len(c)
    ma = np.full((3, n), np.nan)
    per = (50, 100, 200)
    for k in range(3):
        p = per[k]
        if kind == 0:
            s = 0.0
            for i in range(n):
                s += c[i]
                if i >= p: s -= c[i - p]
                if i >= p - 1: ma[k, i] = s / p
        else:
            a = 2.0 / (p + 1.0); e = c[0]
            for i in range(n):
                e = a * c[i] + (1 - a) * e
                if i >= p - 1: ma[k, i] = e
    rsi = np.full(n, np.nan); ag = 0.0; al = 0.0
    for i in range(1, n):
        d = c[i] - c[i - 1]; g = d if d > 0 else 0.0; ls = -d if d < 0 else 0.0
        if i <= 14:
            ag += g / 14.0; al += ls / 14.0
            if i == 14: rsi[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
        else:
            ag = (ag * 13 + g) / 14.0; al = (al * 13 + ls) / 14.0
            rsi[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    st = np.full(n, np.nan)
    for i in range(27, n):
        lo = 1e18; hi = -1e18
        for j in range(i - 13, i + 1):
            if rsi[j] < lo: lo = rsi[j]
            if rsi[j] > hi: hi = rsi[j]
        st[i] = (rsi[i] - lo) / (hi - lo) * 100.0 if hi > lo else 50.0
    K = np.full(n, np.nan); D = np.full(n, np.nan)
    for i in range(29, n):
        K[i] = (st[i] + st[i - 1] + st[i - 2]) / 3.0
    for i in range(31, n):
        D[i] = (K[i] + K[i - 1] + K[i - 2]) / 3.0
    atr = np.full(n, np.nan); a_ = 0.0
    for i in range(1, n):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        if i <= 14:
            a_ += tr / 14.0
            if i == 14: atr[i] = a_
        else:
            a_ = (a_ * 13 + tr) / 14.0; atr[i] = a_
    return ma, rsi, K, D, atr


@njit(cache=True)
def simulate(o, h, l, c, ma, rsi, K, D, atr, cost, maxhold, ma_rule, fmin, fmax, st_mode, rsi_f, sl_mode, tp_k, out):
    n = len(c); nt = 0
    sh_p = np.nan; sh_i = -1; sl_p = np.nan; sl_i = -1
    pos = 0; ent = 0.0; sl = 0.0; tp = 0.0; risk = 0.0; ei = 0
    for i in range(n):
        # swing fractal (3) confirmat la i
        k = i - 3
        if k >= 3:
            ishh = True; ishl = True
            for j in range(1, 4):
                if not (h[k] > h[k - j] and h[k] >= h[k + j]): ishh = False
                if not (l[k] < l[k - j] and l[k] <= l[k + j]): ishl = False
            if ishh: sh_p = h[k]; sh_i = k
            if ishl: sl_p = l[k]; sl_i = k
        if pos != 0:
            hit = 0; px = 0.0
            if pos == 1:
                if l[i] <= sl: hit = -1; px = sl
                elif h[i] >= tp: hit = 1; px = tp
            else:
                if h[i] >= sl: hit = -1; px = sl
                elif l[i] <= tp: hit = 1; px = tp
            if hit == 0 and i - ei >= maxhold: hit = 2; px = c[i]
            if hit != 0:
                g = (px - ent) * pos / risk
                if nt < out.shape[0]:
                    out[nt, 0] = ei; out[nt, 1] = i; out[nt, 2] = pos; out[nt, 3] = g; out[nt, 4] = g - cost / risk; out[nt, 5] = risk
                    nt += 1
                pos = 0
            continue
        if i < 230 or np.isnan(ma[2, i]) or np.isnan(K[i]) or np.isnan(D[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]): continue
        m50 = ma[0, i]; m100 = ma[1, i]; m200 = ma[2, i]
        if ma_rule == 0:
            up = m50 > m100 and m100 > m200; dn = m50 < m100 and m100 < m200
        else:
            up = c[i] > m200 and m50 > m200; dn = c[i] < m200 and m50 < m200
        if up and sl_i >= 0 and sh_i > sl_i and i - sh_i <= 80:
            leg = sh_p - sl_p
            if leg >= atr[i] and c[i] > sl_p:
                ret = (sh_p - c[i]) / leg
                if ret >= fmin and ret <= fmax:
                    if st_mode == 0: trig = K[i] > D[i] and K[i - 1] <= D[i - 1] and K[i - 1] < 30.0
                    else: trig = K[i] > D[i] and K[i] > K[i - 1]
                    if trig and (rsi_f == 0 or (rsi[i] >= 30.0 and rsi[i] <= 55.0)):
                        s_ = sl_p - 0.1 * atr[i] if sl_mode == 0 else c[i] - 1.5 * atr[i]
                        r_ = c[i] - s_
                        if r_ > 0 and r_ >= 2.0 * cost:
                            pos = 1; ent = c[i]; sl = s_; risk = r_; tp = ent + tp_k * r_; ei = i
                            continue
        if dn and sh_i >= 0 and sl_i > sh_i and i - sl_i <= 80:
            leg = sh_p - sl_p
            if leg >= atr[i] and c[i] < sh_p:
                ret = (c[i] - sl_p) / leg
                if ret >= fmin and ret <= fmax:
                    if st_mode == 0: trig = K[i] < D[i] and K[i - 1] >= D[i - 1] and K[i - 1] > 70.0
                    else: trig = K[i] < D[i] and K[i] < K[i - 1]
                    if trig and (rsi_f == 0 or (rsi[i] >= 45.0 and rsi[i] <= 70.0)):
                        s_ = sh_p + 0.1 * atr[i] if sl_mode == 0 else c[i] + 1.5 * atr[i]
                        r_ = s_ - c[i]
                        if r_ > 0 and r_ >= 2.0 * cost:
                            pos = -1; ent = c[i]; sl = s_; risk = r_; tp = ent - tp_k * r_; ei = i
    return nt


def tstat(x):
    n = len(x)
    if n < 15: return None
    s = x.std(ddof=1)
    return float(x.mean() / (s / np.sqrt(n))) if s > 0 else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt4"); os.makedirs(out, exist_ok=True)

    def save(d):
        json.dump(d, open(os.path.join(out, "result.json.tmp"), "w")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time()), "markets": {}, "rows": []}
    save(res)
    try:
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        grid = list(itertools.product((0, 1), (0, 1), ((0.382, 0.618), (0.5, 0.786)), (0, 1), (0, 1), (0, 1), (1.5, 2.0, 3.0)))   # kind, ma_rule, fib, st_mode, rsi_f, sl_mode, tp_k
        for sym in MARKETS:
            try: m1 = load_m1(a.data, sym)
            except Exception as e:
                res["markets"][sym] = {"eroare": repr(e)}; continue
            sp = None
            try:
                means = [v["mean"] for v in spd.get(sym, {}).values() if v["n"] >= 30]
                if means: sp = float(np.median(means))
            except Exception: pass
            cost = 1.75 * sp if sp else 0.0
            res["markets"][sym] = {"spread": sp, "cost": cost}
            for tfn, sec in TFS.items():
                m = resample(m1, sec); t = m["t"]; o, h, l, c = [np.ascontiguousarray(m[k], np.float64) for k in ("o", "h", "l", "c")]
                n = len(t); i1, i2 = int(n * 0.5), int(n * 0.75); years = (t[-1] - t[0]) / 31557600.0
                maxhold = 96 if tfn == "M15" else 48
                res["markets"][sym][tfn] = {"bare": int(n), "de": time.strftime("%Y-%m-%d", time.gmtime(int(t[0]))), "pana": time.strftime("%Y-%m-%d", time.gmtime(int(t[-1]))), "ani": round(years, 2)}
                ind = {kd: indicators(c, h, l, kd) for kd in (0, 1)}
                buf = np.zeros((40000, 6))
                for (kd, mr, fib, sm, rf, sl_m, tk) in grid:
                    ma, rsi, K, D, atr = ind[kd]
                    nt = simulate(o, h, l, c, ma, rsi, K, D, atr, cost, maxhold, mr, fib[0], fib[1], sm, rf, sl_m, tk, buf)
                    tr = buf[:nt]
                    row = {"mk": sym, "tf": tfn, "p": [kd, mr, fib[0], fib[1], sm, rf, sl_m, tk], "n": int(nt), "py": round(nt / years, 1)}
                    for gn, lo_, hi_ in (("train", 0, i1), ("val", i1, i2), ("lock", i2, n), ("tot", 0, n)):
                        sel = tr[(tr[:, 0] >= lo_) & (tr[:, 0] < hi_)]
                        if len(sel):
                            x = sel[:, 4]; ts = tstat(x)
                            row[gn] = {"n": int(len(sel)), "r": round(float(x.mean()), 3), "rg": round(float(sel[:, 3].mean()), 3), "t": round(ts, 2) if ts is not None else None, "w": round(float((x > 0).mean()), 3)}
                        else: row[gn] = {"n": 0, "r": None, "rg": None, "t": None, "w": None}
                    res["rows"].append(row)
                save(res)
        res["state"] = "gata"; res["finished"] = int(time.time()); res["tests"] = len(res["rows"])
        res["nota"] = "r = R mediu net de costuri; rg = brut. p = [mediu(0 SMA/1 EMA), regula trend, fib min, fib max, mod StochRSI, filtru RSI, mod SL, TP in R]. Din %d combinatii, unele ies pozitive din intamplare." % res["tests"]
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1500:]; save(res)


if __name__ == "__main__":
    main()
