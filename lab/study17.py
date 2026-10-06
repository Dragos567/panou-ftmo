# Studiu 17: ORDER FLOW REAL (Databento, CME trades pe tick, agregat la 1 secunda): sweep de lichiditate + absorbtie.
# Instrumente (futures CME): 6E (EURUSD), 6B (GBPUSD), NKD (Nikkei). Executie simulata pe pretul futures, cost = spread CFD FTMO masurat x1.75 (+ comision) in unitati de pret.
# Logica (aceeasi ca in motorul C# Quantower): pivoti pe bare de structura (5/15 min), sweep = pretul trece pivotul cu X ticks, absorbtie = volum in zona / normal >= prag,
# dominanta agresorului >= prag, bara de 1 minut inchide inapoi in range cu fitil >= prag; intrare la inchiderea barei, SL dincolo de extrem, TP = RR x risc.
# 'flip=1' = control: aceeasi intrare, directie inversa (continuare). Se numara ca ipoteza separata.
# Protocol: 50% train / 25% validare / 25% lockbox (in timp). Parametri comuni tuturor instrumentelor. Top 10 pe train (t net) -> validare -> lockbox o singura data.
import os, json, time, argparse, itertools, glob
import numpy as np
from .nb import njit
from .study12 import cost_frac
from .study11 import ftmo_block

VER = 1
INST = {"6E": {"cfd": "EURUSD", "tick": 0.00005}, "6B": {"cfd": "GBPUSD", "tick": 0.0001}, "NKD": {"cfd": "NIKKEI", "tick": 5.0}}
RISK = 0.004          # 0.4% din cont per trade (FTMO 50k: 200 USD)


def load(root, sym):
    fs = sorted(glob.glob(os.path.join(root, sym, "*.npz")))
    parts = []
    for f in fs:
        try:
            z = np.load(f)
            if len(z["t"]): parts.append({k: z[k] for k in ("t", "o", "h", "l", "c", "v", "b", "s", "iid")})
        except Exception: pass
    if not parts: return None
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    o = np.argsort(d["t"], kind="stable")
    return {k: (v[o].astype(np.float64) if k != "iid" else v[o].astype(np.int64)) for k, v in d.items()}


@njit(cache=True)
def core(t, o, h, l, c, v, b, s, iid, struct_s, sweep_pts, max_pts, rej_min, ratio_thr, dom_thr, wick_thr, rr, buf_pts, min_sl, max_sl, hf, ht, hold_s, cool_s, flip, out):
    n = len(t); nt = 0
    CAP = 64
    pv_lvl = np.zeros(CAP); pv_high = np.zeros(CAP, np.int8); pv_n = 0
    sb_h = np.zeros(8); sb_l = np.zeros(8)       # ultimele bare de structura (fereastra glisanta de 5)
    sb_n = 0; cur_st = -1; st_h = 0.0; st_l = 0.0
    cur_min = -1; m_o = 0.0; m_h = 0.0; m_l = 0.0; m_c = 0.0; m_v = 0.0
    avgv = 0.0; nvb = 0
    sw = 0; sw_lvl = 0.0; sw_hi = 0; sw_t0 = 0.0; sw_ext = 0.0; sw_v = 0.0; sw_b = 0.0; sw_s = 0.0
    tr = 0; tr_side = 0.0; tr_entry = 0.0; tr_sl = 0.0; tr_tp = 0.0; tr_t0 = 0.0; tr_dist = 0.0
    last_exit = -1e18; prev_iid = iid[0]
    for i in range(n):
        ti = t[i]
        if iid[i] != prev_iid:                     # rulare contract: reset (pozitia se inchide la ultimul pret)
            if tr == 1:
                g = (c[i - 1] - tr_entry) * tr_side / tr_dist
                out[nt, 0] = tr_t0; out[nt, 1] = tr_side; out[nt, 2] = tr_entry; out[nt, 3] = tr_dist; out[nt, 4] = g; out[nt, 5] = t[i - 1]; nt += 1; tr = 0
            pv_n = 0; sw = 0; sb_n = 0; cur_st = -1; cur_min = -1; prev_iid = iid[i]
        mi = int(ti // 60)
        # --- bara de 1 minut inchisa -> rezolva sweep
        if cur_min != -1 and mi != cur_min:
            mv = m_v
            if sw == 1:
                back = (m_c < sw_lvl) if sw_hi == 1 else (m_c > sw_lvl)
                if back and avgv > 0:
                    secs = (cur_min * 60 + 60) - sw_t0
                    if secs < 3: secs = 3.0
                    if secs > 60: secs = 60.0
                    ratio = sw_v / (avgv / 60.0 * secs)
                    tot = sw_b + sw_s
                    dom = 0.0
                    if tot > 0: dom = (sw_b / tot) if sw_hi == 1 else (sw_s / tot)
                    rg = m_h - m_l
                    wick = 0.0
                    if rg > 0: wick = ((m_h - m_c) / rg) if sw_hi == 1 else ((m_c - m_l) / rg)
                    sl_px = (sw_ext + buf_pts) if sw_hi == 1 else (sw_ext - buf_pts)
                    dist = abs(sl_px - m_c)
                    hr = (cur_min * 60 // 3600) % 24
                    ok = (ratio >= ratio_thr) and (dom >= dom_thr) and (wick >= wick_thr) and (dist >= min_sl) and (dist <= max_sl) and (hr >= hf) and (hr < ht) and (tr == 0) and (cur_min * 60 + 60 - last_exit >= cool_s)
                    if ok:
                        side = -1.0 if sw_hi == 1 else 1.0          # sweep la maxim => SHORT
                        if flip == 1: side = -side
                        tr = 1; tr_side = side; tr_entry = m_c; tr_dist = dist; tr_t0 = cur_min * 60 + 60.0
                        if side > 0: tr_sl = m_c - dist; tr_tp = m_c + rr * dist
                        else: tr_sl = m_c + dist; tr_tp = m_c - rr * dist
                    sw = 0
                elif ti - sw_t0 > (rej_min + 1) * 60: sw = 0
            if mv > 0:
                nvb += 1
                avgv = mv if nvb == 1 else avgv + (mv - avgv) * (2.0 / 31.0)
            cur_min = mi; m_o = o[i]; m_h = h[i]; m_l = l[i]; m_c = c[i]; m_v = 0.0
        elif cur_min == -1:
            cur_min = mi; m_o = o[i]; m_h = h[i]; m_l = l[i]; m_c = c[i]; m_v = 0.0
        else:
            if h[i] > m_h: m_h = h[i]
            if l[i] < m_l: m_l = l[i]
            m_c = c[i]
        m_v += v[i]
        # --- bara de structura
        si = int(ti // struct_s)
        if cur_st == -1: cur_st = si; st_h = h[i]; st_l = l[i]
        elif si != cur_st:
            # inchide bara: muta in fereastra de 5
            if sb_n < 5:
                sb_h[sb_n] = st_h; sb_l[sb_n] = st_l; sb_n += 1
            else:
                for k in range(4):
                    sb_h[k] = sb_h[k + 1]; sb_l[k] = sb_l[k + 1]
                sb_h[4] = st_h; sb_l[4] = st_l
            if sb_n == 5:                       # pivot cu 2 bare in stanga/dreapta (centrul = indexul 2)
                ph = True; pl = True
                for k in range(5):
                    if k == 2: continue
                    if sb_h[k] >= sb_h[2]: ph = False
                    if sb_l[k] <= sb_l[2]: pl = False
                if ph and pv_n < CAP: pv_lvl[pv_n] = sb_h[2]; pv_high[pv_n] = 1; pv_n += 1
                if pl and pv_n < CAP: pv_lvl[pv_n] = sb_l[2]; pv_high[pv_n] = 0; pv_n += 1
            cur_st = si; st_h = h[i]; st_l = l[i]
        else:
            if h[i] > st_h: st_h = h[i]
            if l[i] < st_l: st_l = l[i]
        # --- sweep activ: acumuleaza zona
        if sw == 1:
            if sw_hi == 1:
                if h[i] > sw_ext: sw_ext = h[i]
                if h[i] >= sw_lvl: sw_v += v[i]; sw_b += b[i]; sw_s += s[i]
            else:
                if l[i] < sw_ext: sw_ext = l[i]
                if l[i] <= sw_lvl: sw_v += v[i]; sw_b += b[i]; sw_s += s[i]
        elif pv_n > 0:
            k = 0
            while k < pv_n:
                lv = pv_lvl[k]; hi = pv_high[k]
                hit = False
                if hi == 1 and h[i] >= lv + sweep_pts:
                    hit = True
                    brk = h[i] > lv + max_pts
                elif hi == 0 and l[i] <= lv - sweep_pts:
                    hit = True
                    brk = l[i] < lv - max_pts
                if hit:
                    if not brk:
                        sw = 1; sw_lvl = lv; sw_hi = hi; sw_t0 = ti; sw_ext = h[i] if hi == 1 else l[i]
                        sw_v = v[i]; sw_b = b[i]; sw_s = s[i]
                    pv_lvl[k] = pv_lvl[pv_n - 1]; pv_high[k] = pv_high[pv_n - 1]; pv_n -= 1
                    if sw == 1: break
                else: k += 1
        # --- trade deschis (SL inaintea TP in aceeasi secunda = pesimist)
        if tr == 1 and ti >= tr_t0:
            ex = 0.0; done = False
            if tr_side > 0:
                if l[i] <= tr_sl: ex = tr_sl; done = True
                elif h[i] >= tr_tp: ex = tr_tp; done = True
            else:
                if h[i] >= tr_sl: ex = tr_sl; done = True
                elif l[i] <= tr_tp: ex = tr_tp; done = True
            if (not done) and ti - tr_t0 >= hold_s: ex = c[i]; done = True
            if done:
                g = (ex - tr_entry) * tr_side / tr_dist
                if nt < out.shape[0]:
                    out[nt, 0] = tr_t0; out[nt, 1] = tr_side; out[nt, 2] = tr_entry; out[nt, 3] = tr_dist; out[nt, 4] = g; out[nt, 5] = ti; nt += 1
                tr = 0; last_exit = ti
    return nt


def tstat(x):
    x = np.asarray(x, float)
    if len(x) < 5 or x.std() == 0: return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def seg(x):
    x = np.asarray(x, float)
    if len(x) == 0: return {"n": 0}
    w = x[x > 0].sum(); lo = -x[x < 0].sum()
    return {"n": int(len(x)), "R_net": round(float(x.mean()), 4), "t": round(tstat(x), 2), "win": round(float((x > 0).mean()), 3), "PF": round(float(w / lo), 2) if lo > 0 else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt17"); os.makedirs(out, exist_ok=True)

    def save(d):
        open(os.path.join(out, "result.json.tmp"), "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time())}; save(res)
    try:
        root = os.path.join(a.data, "databento")
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        D = {}
        for s in INST:
            d = load(root, s)
            if d is not None: D[s] = d
        if not D: raise RuntimeError("nu exista date Databento in " + root)
        t0 = max(float(d["t"][0]) for d in D.values()); t1 = min(float(d["t"][-1]) for d in D.values())
        c1 = t0 + 0.5 * (t1 - t0); c2 = t0 + 0.75 * (t1 - t0)
        res["date"] = {s: {"secunde_cu_tranzactii": int(len(d["t"])), "de_la": time.strftime("%Y-%m-%d", time.gmtime(d["t"][0])), "pana_la": time.strftime("%Y-%m-%d", time.gmtime(d["t"][-1]))} for s, d in D.items()}
        res["split"] = {"train": [time.strftime("%Y-%m-%d", time.gmtime(t0)), time.strftime("%Y-%m-%d", time.gmtime(c1))], "val": [time.strftime("%Y-%m-%d", time.gmtime(c1)), time.strftime("%Y-%m-%d", time.gmtime(c2))], "lock": [time.strftime("%Y-%m-%d", time.gmtime(c2)), time.strftime("%Y-%m-%d", time.gmtime(t1))]}
        cost = {}
        for s, d in D.items():
            px = float(np.median(d["c"])); cf, meas = cost_frac(spd, INST[s]["cfd"], px); cost[s] = cf * px; res.setdefault("cost", {})[s] = {"pret_unitati": cost[s], "in_ticks": round(cost[s] / INST[s]["tick"], 2), "masurat": bool(meas)}
        save(res)
        grid = list(itertools.product([300, 900], [1, 3, 6], [1.5, 2.5, 4.0], [0.5, 0.6, 0.7], [0.3, 0.6], [2.0, 3.0], [0, 1]))
        res["variante"] = len(grid); trades = {}
        buf = np.zeros((200000, 6))
        for gi, (st, swt, rat, dom, wk, rr, flip) in enumerate(grid):
            allr = []
            for s, d in D.items():
                tk = INST[s]["tick"]
                nt = core(d["t"], d["o"], d["h"], d["l"], d["c"], d["v"], d["b"], d["s"], d["iid"], st, swt * tk, 40 * tk, 3, rat, dom, wk, rr, 2 * tk, max(6 * tk, 6 * cost[s]), 80 * tk, 6, 20, 6 * 3600, 900, flip, buf)
                x = buf[:nt].copy()
                if nt:
                    net = x[:, 4] - cost[s] / x[:, 3]
                    allr.append(np.column_stack([x[:, 0], net, np.full(nt, list(INST).index(s))]))
            trades[grid[gi]] = np.vstack(allr) if allr else np.zeros((0, 3))
            if gi % 50 == 0: res["progres"] = "%d/%d" % (gi + 1, len(grid)); save(res)
        def parts(a_):
            return a_[a_[:, 0] < c1, 1], a_[(a_[:, 0] >= c1) & (a_[:, 0] < c2), 1], a_[a_[:, 0] >= c2, 1]
        rows = []
        for k, a_ in trades.items():
            tr_, va_, lk_ = parts(a_)
            if len(tr_) >= 100: rows.append((tstat(tr_), k, a_))
        rows.sort(key=lambda r: -r[0])
        res["n_variante_cu_trade_train"] = len(rows)
        names = ["structura_s", "sweep_ticks", "volum_x", "dominanta", "fitil", "RR", "flip"]
        res["top10_train"] = []
        passers = []
        for tt, k, a_ in rows[:10]:
            tr_, va_, lk_ = parts(a_)
            e = {"param": dict(zip(names, k)), "train": seg(tr_), "val": seg(va_)}
            res["top10_train"].append(e)
            if len(va_) >= 30 and va_.mean() > 0 and tstat(va_) >= 1.5: passers.append((k, a_, e))
        res["trecut_validarea"] = len(passers)
        res["control_flip"] = {"nota": "perechi (flip=0 vs flip=1) cu aceiasi parametri: daca absorbtia are sens, flip=0 trebuie sa fie mai bun decat flip=1"}
        pairs = []
        for k, a_ in list(trades.items()):
            if k[-1] == 0:
                kk = k[:-1] + (1,); tr0, _, _ = parts(a_); tr1, _, _ = parts(trades[kk])
                if len(tr0) >= 100 and len(tr1) >= 100: pairs.append((tr0.mean(), tr1.mean()))
        if pairs:
            p = np.array(pairs); res["control_flip"].update(perechi=len(p), medie_R_train_normal=round(float(p[:, 0].mean()), 4), medie_R_train_inversat=round(float(p[:, 1].mean()), 4))
        res["medie_R_net_train_toate"] = round(float(np.mean([parts(a_)[0].mean() for a_ in trades.values() if len(parts(a_)[0]) >= 100])), 4) if rows else None
        # lockbox: o singura privire, doar pentru cele care au trecut validarea (max 3)
        res["lockbox"] = []
        for k, a_, e in passers[:3]:
            tr_, va_, lk_ = parts(a_); r = {"param": e["param"], "lock": seg(lk_)}
            allv = a_[a_[:, 0] >= c1]
            days = np.arange(int(c1 // 86400), int(t1 // 86400) + 1); dd = np.zeros(len(days))
            for tt_, rn, _ in allv: dd[int(tt_ // 86400) - days[0]] += rn * RISK
            wk_ = np.array([(int(dv) + 4) % 7 < 5 for dv in days]); dd = dd[wk_]
            if len(dd) > 60 and dd.std() > 0:
                rng = np.random.default_rng(17); r["ftmo_val+lock"] = ftmo_block(dd, float(dd.std() * np.sqrt(252)), rng)
            res["lockbox"].append(r)
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("Date reale CME (trades, agresor), bare 1 s. Costuri CFD FTMO (spread masurat x1.75 + comision). Parametri comuni pe 3 instrumente. "
                       "Se testeaza %d variante; un 't' ~3-4 in train pe atatea variante e posibil din noroc: contează doar validarea si lockbox-ul. SL inainte de TP in aceeasi secunda (pesimist)." % len(grid))
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
