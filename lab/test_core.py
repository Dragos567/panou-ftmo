# Teste de corectitudine pentru motor. Ruleaza: python3 -m lab.test_core
import numpy as np, math
from .bt import run_bt, EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_END
from . import ftmo

def mk(rows):
    a = np.array(rows, float); return a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(), a[:, 3].copy()  # o h l c

def run(rows, sig, sd, tp_r=2.0, mh=100, cost=0.0, slip=0.0):
    o, h, l, c = mk(rows); n = len(o)
    sg = np.zeros(n, np.int8)
    for i, d in sig.items(): sg[i] = d
    return run_bt(o, h, l, c, sg, np.full(n, float(sd)), tp_r, mh, np.full(n, float(cost)), slip, 0, n)

def ref(o, h, l, c, sig, sd, tp_r, mh, cost, slip):
    """Referinta independenta (masina de stari, bara cu bara)."""
    n = len(o); out = []; pos = None
    for j in range(n):
        if pos is not None:
            d, e, sl, tp, ent, sdist = pos; px = None; why = None
            if j > ent:
                if (d == 1 and o[j] <= sl) or (d == -1 and o[j] >= sl): px, why = o[j], 1
                elif (d == 1 and o[j] >= tp) or (d == -1 and o[j] <= tp): px, why = o[j], 2
            if px is None:
                hs = l[j] <= sl if d == 1 else h[j] >= sl
                ht = h[j] >= tp if d == 1 else l[j] <= tp
                if hs: px, why = sl - d * slip, 1
                elif ht: px, why = tp, 2
            if px is None and j - ent + 1 >= mh: px, why = c[j], 3
            if px is None and j == n - 1: px, why = c[j], 4
            if px is not None:
                out.append((ent, j, d, (d * (px - e) - cost[ent]) / sdist, why)); pos = None
        if pos is None and j + 1 < n and sig[j] != 0 and sd[j] > 0:
            # semnalul de la bara j; daca pozitia tocmai s-a inchis in bara j, intrarea e tot la j+1
            d = int(sig[j]); e = o[j + 1]
            pos = (d, e, e - d * sd[j], e + d * sd[j] * tp_r, j + 1, sd[j])
    return out

def case(name, got, exp_r, exp_why):
    ent, ext, d, r, why = got
    ok = len(r) == 1 and abs(r[0] - exp_r) < 1e-9 and why[0] == exp_why
    print(("OK   " if ok else "FAIL ") + name, [round(x, 6) for x in r], list(why))
    return ok

def main():
    ok = True
    base = [(100, 100, 100, 100)]  # bara 0: semnal
    # long, stop atins: intrare 100, sl 99
    ok &= case("long SL", run(base + [(100, 100.5, 98.5, 99.2)], {0: 1}, 1), -1.0, EXIT_SL)
    ok &= case("long TP (2R)", run(base + [(100, 102.5, 99.5, 102.1)], {0: 1}, 1), 2.0, EXIT_TP)
    ok &= case("ambele in aceeasi bara -> SL", run(base + [(100, 103, 98, 100)], {0: 1}, 1), -1.0, EXIT_SL)
    ok &= case("gap sub stop -> iesire la deschidere", run(base + [(100, 100.4, 99.5, 100.1), (97, 97.5, 96.5, 97)], {0: 1}, 1), -3.0, EXIT_SL)
    ok &= case("short TP", run(base + [(100, 100.5, 97.8, 98.2)], {0: -1}, 1), 2.0, EXIT_TP)
    ok &= case("short SL", run(base + [(100, 101.2, 99.5, 100.8)], {0: -1}, 1), -1.0, EXIT_SL)
    ok &= case("cost scade R", run(base + [(100, 102.5, 99.5, 102.1)], {0: 1}, 1, cost=0.1), 1.9, EXIT_TP)
    ok &= case("slippage pe stop", run(base + [(100, 100.2, 98.5, 99.2)], {0: 1}, 1, slip=0.25), -1.25, EXIT_SL)
    ok &= case("iesire pe timp", run(base + [(100, 100.4, 99.6, 100.3), (100.3, 100.5, 99.7, 100.5)], {0: 1}, 1, mh=2), 0.5, EXIT_TIME)
    ok &= case("sfarsit de date", run(base + [(100, 100.4, 99.6, 100.3)], {0: 1}, 1), 0.3, EXIT_END)
    # semnal in timpul pozitiei ignorat
    g = run(base + [(100, 100.4, 99.6, 100.0), (100, 100.4, 99.6, 100.0), (100, 102.5, 99.6, 102.0)], {0: 1, 1: 1, 2: -1}, 1)
    n_ok = len(g[3]) == 1; print(("OK   " if n_ok else "FAIL ") + "semnale ignorate in pozitie", list(g[3])); ok &= n_ok
    # diferential cu referinta pe date aleatoare
    rng = np.random.default_rng(7); bad = 0; tot = 0
    for t in range(300):
        n = 400; c = 100 + np.cumsum(rng.normal(0, 0.3, n)); o = np.roll(c, 1); o[0] = c[0]
        o = o + rng.normal(0, 0.05, n) * (rng.random(n) < 0.05) * 6   # gap-uri rare
        h = np.maximum(o, c) + np.abs(rng.normal(0, 0.2, n)); l = np.minimum(o, c) - np.abs(rng.normal(0, 0.2, n))
        sig = (rng.random(n) < 0.15) * rng.choice([-1, 1], n); sig = sig.astype(np.int8)
        sd = np.abs(rng.normal(0.8, 0.3, n)) + 0.05; cost = np.abs(rng.normal(0.03, 0.01, n))
        tp_r = float(rng.choice([1.0, 1.5, 2.0, 3.0])); mh = int(rng.choice([3, 10, 50])); slip = float(rng.choice([0, 0.1]))
        e1, x1, d1, r1, w1 = run_bt(o, h, l, c, sig, sd, tp_r, mh, cost, slip, 0, n)
        rf = ref(o, h, l, c, sig, sd, tp_r, mh, cost, slip)
        tot += 1
        same = len(rf) == len(r1) and all(rf[k][0] == e1[k] and rf[k][1] == x1[k] and rf[k][2] == d1[k] and abs(rf[k][3] - r1[k]) < 1e-9 and rf[k][4] == w1[k] for k in range(len(rf)))
        if not same: bad += 1
    print(("OK   " if bad == 0 else "FAIL ") + "diferential 300 serii aleatoare, diferente:", bad); ok &= bad == 0
    # FTMO: cazuri simple
    r = np.array([1.0] * 12); day = np.arange(12)           # +1% pe zi (risc 1%), tinta 10% -> trece in 10 zile
    res, used, days = ftmo.run_phase(r, day, 0, 1.0, 10.0, 5.0, 10.0, 4)
    k1 = res == ftmo.PASS and used == 10; print(("OK   " if k1 else "FAIL ") + "FTMO trece la +10%", res, used, days); ok &= k1
    r = np.array([-1.0] * 6); day = np.zeros(6, np.int64)    # -6% intr-o zi -> pierdere zilnica
    res, used, _ = ftmo.run_phase(r, day, 0, 1.0, 10.0, 5.0, 10.0, 4)
    k2 = res == ftmo.FAIL_DAILY and used == 5; print(("OK   " if k2 else "FAIL ") + "FTMO pierdere zilnica la -5%", res, used); ok &= k2
    r = np.array([-1.0] * 12); day = np.arange(12)           # -1%/zi, 10 zile -> pierdere totala
    res, used, _ = ftmo.run_phase(r, day, 0, 1.0, 10.0, 5.0, 10.0, 4)
    k3 = res == ftmo.FAIL_TOTAL and used == 10; print(("OK   " if k3 else "FAIL ") + "FTMO pierdere totala la -10%", res, used); ok &= k3
    r = np.array([5.0, 5.0]); day = np.array([0, 0])        # tinta atinsa in prima zi dar < 4 zile -> nu trece
    res, used, days = ftmo.run_phase(r, day, 0, 1.0, 10.0, 5.0, 10.0, 4)
    k4 = res == ftmo.UNFINISHED; print(("OK   " if k4 else "FAIL ") + "FTMO cere minim 4 zile", res, days); ok &= k4
    print("TOT:", "TOATE TESTELE TREC" if ok else "EXISTA ESECURI"); return ok

if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
