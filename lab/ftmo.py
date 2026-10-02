# Simulator reguli FTMO (2-step) pe o secventa de tranzactii in R. Toate regulile sunt parametri.
# Simplificare declarata: pierderea zilnica se verifica pe echitatea de la inchiderea tranzactiilor (nu pe cea flotanta intrazilnica).
import numpy as np
from .nb import njit

PASS, FAIL_DAILY, FAIL_TOTAL, UNFINISHED = 1, 2, 3, 0


@njit(cache=True)
def run_phase(r, day, k0, risk, target, max_daily, max_total, min_days):
    """Porneste de la tranzactia k0 cu echitate 0 (in %). Intoarce (rezultat, tranzactii folosite, zile de tranzactionare)."""
    n = len(r); eq = 0.0; cur_day = -1; day_start = 0.0; days = 0
    k = k0
    while k < n:
        if day[k] != cur_day:
            cur_day = day[k]; day_start = eq; days += 1
        eq += r[k] * risk
        if eq - day_start <= -max_daily: return FAIL_DAILY, k - k0 + 1, days
        if eq <= -max_total: return FAIL_TOTAL, k - k0 + 1, days
        if eq >= target and days >= min_days: return PASS, k - k0 + 1, days
        k += 1
    return UNFINISHED, n - k0, days


@njit(cache=True)
def rolling(r, day, risk, t1, t2, max_daily, max_total, min_days, stride):
    """Pentru fiecare punct de start: Challenge, apoi Verification din tranzactia urmatoare. Intoarce contoare."""
    n = len(r); started = 0; p1 = 0; both = 0; fd = 0; ft = 0; unf = 0
    k0 = 0
    while k0 < n:
        res, used, _ = run_phase(r, day, k0, risk, t1, max_daily, max_total, min_days)
        if res == UNFINISHED: unf += 1; k0 += stride; continue   # fereastra prea scurta: nu se numara
        started += 1
        if res == FAIL_DAILY: fd += 1
        elif res == FAIL_TOTAL: ft += 1
        else:
            p1 += 1
            k1 = k0 + used
            if k1 < n:
                res2, _, _ = run_phase(r, day, k1, risk, t2, max_daily, max_total, min_days)
                if res2 == PASS: both += 1
        k0 += stride
    return started, p1, both, fd, ft, unf


def evaluate(r, day, risk_pct=1.0, t1=10.0, t2=5.0, max_daily=5.0, max_total=10.0, min_days=4, stride=1):
    r = np.ascontiguousarray(r, np.float64); day = np.ascontiguousarray(day, np.int64)
    s, p1, both, fd, ft, unf = rolling(r, day, float(risk_pct), t1, t2, max_daily, max_total, min_days, stride)
    s = max(s, 1)
    return {"starts": int(s), "pass_challenge": p1 / s, "pass_both": both / s, "fail_daily": fd / s, "fail_total": ft / s, "unfinished_starts": int(unf)}


def bootstrap(r, day, risk_pct=1.0, n_days=120, reps=2000, seed=1, t1=10.0, t2=5.0, max_daily=5.0, max_total=10.0, min_days=4):
    """Amesteca zilele de tranzactionare (blocuri, ca sa pastreze clusterele dintr-o zi) si masoara probabilitatea de trecere."""
    r = np.asarray(r, np.float64); day = np.asarray(day, np.int64)
    ud, first = np.unique(day, return_index=True); order = np.argsort(first); ud = ud[order]
    groups = [r[day == d] for d in ud]
    rng = np.random.default_rng(seed); ok = 0
    for _ in range(reps):
        idx = rng.integers(0, len(groups), n_days)
        rr = np.concatenate([groups[i] for i in idx]); dd = np.concatenate([np.full(len(groups[i]), j, np.int64) for j, i in enumerate(idx)])
        res, used, _ = run_phase(rr, dd, 0, float(risk_pct), t1, max_daily, max_total, min_days)
        if res == PASS:
            res2, _, _ = run_phase(rr, dd, used, float(risk_pct), t2, max_daily, max_total, min_days) if used < len(rr) else (UNFINISHED, 0, 0)
            if res2 == PASS: ok += 1
    return {"reps": reps, "pass_both": ok / reps}
