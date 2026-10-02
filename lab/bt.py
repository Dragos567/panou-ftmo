# Motor de backtest Altrix. Reguli (identice in nucleul rapid si in referinta de test):
#  * semnalul se decide la inchiderea barei i; intrarea e la deschiderea barei i+1;
#  * stop si tinta sunt verificate in interiorul barelor; daca o bara atinge ambele, se considera ca stopul a fost primul (conservator);
#  * daca o bara se deschide dincolo de stop (gap), iesirea e la deschidere, nu la pretul stopului;
#  * rezultatul fiecarei tranzactii e in R (1R = distanta stopului), dupa costuri; o singura pozitie la un moment dat.
import numpy as np
from .nb import njit

EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_END = 1, 2, 3, 4


@njit(cache=True)
def run_bt(o, h, l, c, sig, sl_dist, tp_r, max_hold, cost, slip, start, end):
    n = len(o)
    ent = np.empty(n, np.int64); ext = np.empty(n, np.int64); dirs = np.empty(n, np.int8)
    rs = np.empty(n, np.float64); why = np.empty(n, np.int8)
    nt = 0; i = start
    while i < end - 1:
        d = sig[i]; sd = sl_dist[i]
        if d == 0 or not (sd > 0):
            i += 1; continue
        e = o[i + 1]
        sl = e - d * sd; tp = e + d * sd * tp_r
        j = i + 1; px = e; reason = EXIT_END; bars = 0; done = False
        while j < end:
            if j > i + 1:  # gap la deschidere
                if (d == 1 and o[j] <= sl) or (d == -1 and o[j] >= sl):
                    px = o[j]; reason = EXIT_SL; done = True
                elif (d == 1 and o[j] >= tp) or (d == -1 and o[j] <= tp):
                    px = o[j]; reason = EXIT_TP; done = True
            if not done:
                hit_sl = (l[j] <= sl) if d == 1 else (h[j] >= sl)
                hit_tp = (h[j] >= tp) if d == 1 else (l[j] <= tp)
                if hit_sl:
                    px = sl - d * slip; reason = EXIT_SL; done = True
                elif hit_tp:
                    px = tp; reason = EXIT_TP; done = True
            bars += 1
            if done: break
            if bars >= max_hold:
                px = c[j]; reason = EXIT_TIME; done = True; break
            j += 1
        if not done:
            j = end - 1; px = c[j]; reason = EXIT_END
        pnl = d * (px - e) - cost[i + 1]
        ent[nt] = i + 1; ext[nt] = j; dirs[nt] = d; rs[nt] = pnl / sd; why[nt] = reason
        nt += 1
        i = j
    return ent[:nt], ext[:nt], dirs[:nt], rs[:nt], why[:nt]


def stats(r, bars_per_year=None, n_bars=None):
    """Metrici dintr-un vector de rezultate in R."""
    r = np.asarray(r, np.float64); n = len(r)
    if n == 0: return {"n": 0, "win": 0.0, "avg_r": 0.0, "pf": 0.0, "dd_r": 0.0, "total_r": 0.0}
    w = r[r > 0]; ls = r[r < 0]
    eq = np.cumsum(r); peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
    return {"n": int(n), "win": float(len(w) / n), "avg_r": float(r.mean()),
            "pf": float(w.sum() / -ls.sum()) if len(ls) and ls.sum() < 0 else float("inf") if len(w) else 0.0,
            "dd_r": float((peak - eq).max()), "total_r": float(eq[-1]), "sd_r": float(r.std())}
