# Studiu 15: SURSE NOI de informatie (nu doar preturi): COT (pozitionarea speculatorilor CFTC) si CARRY FX cu rate reale (FRED).
# Datele se descarca de pe server (sandbox-ul de dezvoltare nu are retea). Cache in <data>/ext15/.
#  C1 COT_contra: contrarian pe pozitionarea neta a speculatorilor (non-commercial, % din open interest), z-score 156 saptamani, w = -clip(z/2,-1,1).
#  C2 COT_trend : oglinda (w = +clip(z/2,-1,1)); ambele semne testate, deci se numara ca 2 ipoteze.
#  C3 CARRY_FX  : pe 5 perechi (EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD) diferential de rata (baza - cota) din FRED (decalaj 2 luni); long top 2 / short bottom 2.
#                 Venit de carry = diferential real - 0.7%/an marja broker pe pozitie. Se raporteaza si doar-pret (fara carry).
# Cost: spread la schimbare + swap static (ca studiul 14) pentru COT; pentru CARRY swap-ul = diferentialul real - marja.
# COT: raport pentru marti, disponibil de luni urmatoare (as_of + 6 zile). Rebalansare saptamanala. Vol 10% per piata, plafon 4x.
import os, io, csv, json, time, zipfile, argparse, re, urllib.request
import numpy as np
from .study12 import daily_series, cost_frac, OLD, NEW, CLS
from .study11 import seg_stats, ftmo_block
from .study14 import SW, CRYPTO_DAY

VER = 1
UA = {"User-Agent": "Mozilla/5.0 (research)"}
COT_MAP = {  # sim: (pattern-uri, semn)  semn=-1 cand piata COT e inversa fata de simbolul CFD (ex. JPY vs USDJPY)
    "EURUSD": (["EURO FX"], 1), "GBPUSD": (["BRITISH POUND"], 1), "USDJPY": (["JAPANESE YEN"], -1), "AUDUSD": (["AUSTRALIAN DOLLAR"], 1), "USDCAD": (["CANADIAN DOLLAR"], -1),
    "GOLD": (["GOLD - COMMODITY"], 1), "SILVER": (["SILVER - COMMODITY"], 1), "COPPER": (["COPPER"], 1), "PLAT": (["PLATINUM"], 1),
    "US500": (["E-MINI S&P 500"], 1), "US100": (["NASDAQ-100", "NASDAQ MINI"], 1), "US30": (["DJIA", "DOW JONES"], 1), "NIKKEI": (["NIKKEI STOCK AVERAGE"], 1), "US2K": (["RUSSELL"], 1),
    "NATGAS": (["NAT GAS NYME", "NATURAL GAS"], 1), "SOY": (["SOYBEANS - CHICAGO"], 1), "WHEAT": (["WHEAT-SRW", "WHEAT - CHICAGO"], 1), "CORN": (["CORN - CHICAGO"], 1),
    "SUGAR": (["SUGAR NO. 11"], 1), "COFFEE": (["COFFEE C"], 1), "COCOA": (["COCOA - ICE"], 1), "COTTON": (["COTTON NO. 2"], 1), "BTC": (["BITCOIN - CHICAGO"], 1), "DXY": (["U.S. DOLLAR INDEX"], 1)}
RATE_IDS = {"USD": "USA", "EUR": "EZ", "GBP": "GBR", "JPY": "JPN", "AUD": "AUS", "CAD": "CAN"}
PAIRS = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "USDJPY": ("USD", "JPY"), "AUDUSD": ("AUD", "USD"), "USDCAD": ("USD", "CAD")}


def fetch(url, path, minsize=200):
    if os.path.exists(path) and os.path.getsize(path) > minsize: return open(path, "rb").read()
    req = urllib.request.Request(url, headers=UA); b = urllib.request.urlopen(req, timeout=60).read()
    os.makedirs(os.path.dirname(path), exist_ok=True); open(path, "wb").write(b); return b


def load_cot(ext, log):
    rows = {}  # nume -> {as_of_day: net%oi}
    cnt = {}
    now_y = time.gmtime().tm_year
    urls = [("deacot1986_2016.zip", None)] + [("deacot%d.zip" % y, y) for y in range(2017, now_y + 1)]
    for fn, y in urls:
        try:
            b = fetch("https://www.cftc.gov/files/dea/history/" + fn, os.path.join(ext, fn)); z = zipfile.ZipFile(io.BytesIO(b))
        except Exception as e:
            log.append("COT %s: %r" % (fn, e)); continue
        for nm in z.namelist():
            txt = z.read(nm).decode("latin-1")
            rd = csv.reader(io.StringIO(txt)); hdr = next(rd); low = [h.strip().lower() for h in hdr]
            def col(pred):
                for k, h in enumerate(low):
                    if pred(h): return k
                return None
            ci_n = 0; ci_d = col(lambda h: "yyyy-mm-dd" in h); ci_l = col(lambda h: h.startswith("noncommercial positions-long (all)")); ci_s = col(lambda h: h.startswith("noncommercial positions-short (all)")); ci_o = col(lambda h: h.startswith("open interest (all)"))
            if None in (ci_d, ci_l, ci_s, ci_o): log.append("COT %s/%s: coloane lipsa" % (fn, nm)); continue
            for r in rd:
                try:
                    d = r[ci_d].strip(); t = time.strptime(d, "%Y-%m-%d"); day = int(time.mktime(t) // 86400) if False else int(__import__("calendar").timegm(t) // 86400)
                    if day < 15300: continue  # ~2011-12
                    oi = float(r[ci_o]); net = (float(r[ci_l]) - float(r[ci_s])) / oi if oi > 0 else np.nan
                    name = r[ci_n].strip().upper(); rows.setdefault(name, {})[day] = net
                except Exception: continue
    return rows


def pick(rows, pats):
    c = [n for n in rows if any(p in n for p in pats) and "MICRO" not in n and "CONSOLIDATED" not in n and "E-MICRO" not in n]
    if not c: return None
    c.sort(key=lambda n: -len(rows[n])); return c[0]


def load_rates(ext, log):
    out = {}
    for ccy, cc in RATE_IDS.items():
        for pre in ("IR3TIB01", "IRSTCI01"):
            sid = "%s%sM156N" % (pre, cc)
            try:
                b = fetch("https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + sid, os.path.join(ext, sid + ".csv"), 100).decode()
                ser = {}
                for ln in b.splitlines()[1:]:
                    p = ln.split(",")
                    if len(p) == 2 and p[1] not in (".", ""):
                        t = time.strptime(p[0], "%Y-%m-%d"); ser[int(__import__("calendar").timegm(t) // 86400)] = float(p[1])
                if len(ser) > 100: out[ccy] = (sid, ser); break
            except Exception as e:
                log.append("FRED %s: %r" % (sid, e))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    out = os.path.join(a.data, "evt15"); os.makedirs(out, exist_ok=True); ext = os.path.join(a.data, "ext15")

    def save(d):
        open(os.path.join(out, "result.json.tmp"), "w").write(json.dumps(d).replace("NaN", "null").replace("Infinity", "null")); os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    res = {"ver": VER, "state": "ruleaza", "started": int(time.time())}; save(res)
    try:
        log = []; rng = np.random.default_rng(15)
        try: spd = json.load(open(os.path.join(a.data, "spread.json")))
        except Exception: spd = {}
        cot = load_cot(ext, log); rates = load_rates(ext, log)
        res["descarcare"] = {"cot_piete_cftc": len(cot), "rate": {k: [v[0], len(v[1]), time.strftime("%Y-%m", time.gmtime(max(v[1]) * 86400))] for k, v in rates.items()}, "log": log[:12]}; save(res)
        if len(cot) < 5 and len(rates) < 4: raise RuntimeError("fara date externe: " + "; ".join(log[:4]))
        D = {}
        for s in [x for x in OLD + NEW if x != "USOIL"]:
            try:
                d, c = daily_series(a.data, s)
                if len(d) > 400 and (s in SW or s in ("ETH", "BTC")): D[s] = (d, c)
            except Exception: pass
        syms = list(D); nS = len(syms); days = np.unique(np.concatenate([D[s][0] for s in syms])); nd = len(days); pos = {int(d): i for i, d in enumerate(days)}
        P = np.full((nd, nS), np.nan)
        for j, s in enumerate(syms):
            for d, c in zip(*D[s]): P[pos[int(d)], j] = c
        R = np.zeros((nd, nS)); act = np.zeros((nd, nS), bool); lastp = np.full(nS, np.nan); GAP = np.zeros((nd, nS)); lastd = np.full(nS, np.nan); PX = np.full((nd, nS), np.nan)
        for i in range(nd):
            for j in range(nS):
                if np.isfinite(P[i, j]):
                    if np.isfinite(lastp[j]) and lastp[j] > 0: R[i, j] = P[i, j] / lastp[j] - 1; act[i, j] = True; GAP[i, j] = days[i] - lastd[j]; PX[i, j] = lastp[j]
                    lastp[j] = P[i, j]; lastd[j] = days[i]
        V = np.full((nd, nS), np.nan); lam = 1 - 2 / 61.0
        for j in range(nS):
            v = np.nan
            for i in range(nd):
                if act[i, j]: v = R[i, j] ** 2 if not np.isfinite(v) else lam * v + (1 - lam) * R[i, j] ** 2
                V[i, j] = np.sqrt(v) if np.isfinite(v) else np.nan
        cf = {s: cost_frac(spd, s, float(np.nanmedian(P[:, j])))[0] for j, s in enumerate(syms)}
        ix = {s: j for j, s in enumerate(syms)}
        def swapfrac(j, w, i):
            s = syms[j]
            if s in ("BTC", "ETH"): return CRYPTO_DAY
            pl, ps, dg = SW[s]; pts = pl if w > 0 else ps
            return pts * 10.0 ** (-dg) / PX[i, j]
        okv = lambda i, j: bool(np.isfinite(V[i - 1, j]) and V[i - 1, j] > 0 and act[max(0, i - 15):i, j].any())
        lev = lambda i, j: min(4.0, 0.10 / (V[i - 1, j] * np.sqrt(252)))
        wk = (days + 3) // 7; reb = np.r_[False, wk[1:] != wk[:-1]]; i0 = 260
        # ---- COT: z-score pe saptamani, disponibil la as_of+6
        Z = np.full((nd, nS), np.nan); matched = {}
        for s, (pats, sg) in COT_MAP.items():
            if s not in ix: continue
            nm = pick(cot, pats)
            if not nm: continue
            ser = sorted(cot[nm].items()); matched[s] = [nm, len(ser)]
            ad = np.array([d + 6 for d, _ in ser]); nets = np.array([v for _, v in ser]); zs = np.full(len(ser), np.nan)
            for k in range(len(ser)):
                w = nets[max(0, k - 155):k + 1]
                w = w[np.isfinite(w)]
                if len(w) >= 52 and w.std() > 0 and np.isfinite(nets[k]): zs[k] = sg * (nets[k] - w.mean()) / w.std()
            kk = np.searchsorted(ad, days - 1, side="right") - 1  # ultima observatie disponibila pana la ziua i-1
            for i in range(nd):
                if kk[i] >= 0 and days[i] - ad[kk[i]] < 20: Z[i, ix[s]] = zs[kk[i]]
        res["cot_potriviri"] = matched
        def f_cot(sign):
            def f(i):
                av = [j for j in range(nS) if okv(i, j) and np.isfinite(Z[i, j])]; w = np.zeros(nS)
                for j in av: w[j] = sign * float(np.clip(Z[i, j] / 2.0, -1, 1)) * lev(i, j) / max(1, len(av))
                return w
            return f
        # ---- CARRY
        rd = np.full((nd, nS), np.nan)
        def rate_at(ccy, day):
            sid, ser = rates[ccy]; ks = [d for d in ser if d <= day - 60]
            return ser[max(ks)] if ks else np.nan
        carry_ok = all(c in rates for c in RATE_IDS) and all(p in ix for p in PAIRS)
        if carry_ok:
            for i in range(0, nd):
                for p, (b, q) in PAIRS.items(): rd[i, ix[p]] = rate_at(b, int(days[i])) - rate_at(q, int(days[i]))
        def f_carry(i):
            w = np.zeros(nS)
            if not carry_ok: return w
            av = [ix[p] for p in PAIRS if okv(i, ix[p]) and np.isfinite(rd[i, ix[p]])]
            if len(av) < 4: return w
            av.sort(key=lambda j: rd[i, j])
            for j in av[-2:]: w[j] = lev(i, j) / 4.0
            for j in av[:2]: w[j] = -lev(i, j) / 4.0
            return w
        facs = {"C1_COT_contra": (f_cot(-1), "swap"), "C2_COT_trend": (f_cot(1), "swap"), "C3_CARRY_FX": (f_carry, "carry"), "C3b_CARRY_doar_pret": (f_carry, "none")}
        n = nd - i0; c1, c2 = int(n * 0.5), int(n * 0.75); d_ = days[i0:]; ok_ = np.ones(n, bool)
        res["piete"] = syms; res["perioada"] = [time.strftime("%Y-%m-%d", time.gmtime(int(d_[0]) * 86400)), time.strftime("%Y-%m-%d", time.gmtime(int(d_[-1]) * 86400))]; res["factori"] = {}
        series = {}
        for name, (fn, mode) in facs.items():
            W = np.zeros(nS); net = np.zeros(nd); gross = np.zeros(nd); swp = np.zeros(nd); trn = np.zeros(nd)
            for i in range(i0, nd):
                if reb[i]:
                    wn = fn(i)
                    for j in range(nS):
                        net[i] -= abs(wn[j] - W[j]) * cf[syms[j]]; trn[i] += abs(wn[j] - W[j])
                    W = wn
                g = float((W * R[i]).sum()); gross[i] = g; net[i] += g
                for j in range(nS):
                    if W[j] != 0 and act[i, j]:
                        if mode == "swap": sw = abs(W[j]) * swapfrac(j, W[j], i) * GAP[i, j]
                        elif mode == "carry" and np.isfinite(rd[i, j]): sw = (W[j] * rd[i, j] / 100.0 - abs(W[j]) * 0.007) / 365.0 * GAP[i, j]
                        else: sw = 0.0
                        net[i] += sw; swp[i] += sw
            series[name] = net[i0:]; x = net[i0:]; g = gross[i0:]; gs = seg_stats(g, ok_)
            res["factori"][name] = {"net": {"tot": seg_stats(x, ok_), "train": seg_stats(x[:c1], ok_[:c1]), "val": seg_stats(x[c1:c2], ok_[c1:c2]), "lock": seg_stats(x[c2:], ok_[c2:])},
                                    "brut_sharpe": gs["sharpe"] if gs else None, "swap_carry_an_pct": round(float(swp[i0:].mean() * 252 * 100), 2), "turnover_an": round(float(trn[i0:].sum() / n * 252), 1)}
            save(res)
        names = list(series); M = np.array([series[k] for k in names])
        res["corelatii"] = {names[a_]: {names[b_]: round(float(np.corrcoef(M[a_], M[b_])[0, 1]), 2) if M[a_].std() > 0 and M[b_].std() > 0 else None for b_ in range(len(names))} for a_ in range(len(names))}
        sc = lambda v: v * (0.10 / (v[:c1].std() * np.sqrt(252))) if v[:c1].std() > 0 else v * 0
        def combo(keys):
            v = np.mean([sc(series[k]) for k in keys], axis=0)
            return {"keys": keys, "tot": seg_stats(v, ok_), "train": seg_stats(v[:c1], ok_[:c1]), "val": seg_stats(v[c1:c2], ok_[c1:c2]), "lock": seg_stats(v[c2:], ok_[c2:]),
                    "ftmo_tot": [ftmo_block(v, vt, rng) for vt in (0.06, 0.10, 0.15)], "ftmo_val+lock": [ftmo_block(v[c1:], vt, rng) for vt in (0.06, 0.10, 0.15)]}
        res["portofolii"] = {"COT_contra+CARRY": combo(["C1_COT_contra", "C3_CARRY_FX"]), "COT_trend+CARRY": combo(["C2_COT_trend", "C3_CARRY_FX"])}
        res["state"] = "gata"; res["finished"] = int(time.time())
        res["nota"] = ("Surse NOI: COT (CFTC legacy, non-commercial net/OI, z 156 sapt., disponibil la as_of+6 zile) si rate FRED (decalaj 60 zile). C1/C2 sunt oglinzi: se numara ca 2 ipoteze. "
                       "Carry: venit = diferential real - 0.7%/an marja (presupunere). Verifica 'rate' (ultima luna disponibila) - daca seria s-a oprit, ffill = risc.")
        save(res)
    except Exception as e:
        import traceback; res["state"] = "eroare"; res["error"] = repr(e); res["tb"] = traceback.format_exc()[-1800:]; save(res)


if __name__ == "__main__":
    main()
