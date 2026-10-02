# Piese de strategie (generatoare de semnale). Reguli comune:
#  * semnalul de la indexul j se decide la INCHIDEREA barei j si foloseste doar bare <= j;
#  * un pivot (swing) cu n bare in fiecare parte e cunoscut abia dupa n bare (la j = p + n);
#  * intoarce (sig, sd): directia (+1/-1/0) si distanta stopului propusa, in unitati de pret, masurata de la inchiderea barei j.
import numpy as np
from .nb import njit
from . import feat


@njit(cache=True)
def swings(h, l, n):
    """Ultimul pivot confirmat (maxim/minim) cunoscut la inchiderea fiecarei bare: nivel si index."""
    N = len(h); sh = np.full(N, np.nan); sl = np.full(N, np.nan)
    shi = np.full(N, -1, np.int64); sli = np.full(N, -1, np.int64)
    ch = np.nan; cl = np.nan; ih = -1; il = -1
    for j in range(N):
        p = j - n
        if p >= n:
            ok = True
            for k in range(1, n + 1):
                if not (h[p] > h[p - k]) or not (h[p] >= h[p + k]): ok = False; break
            if ok: ch = h[p]; ih = p
            ok = True
            for k in range(1, n + 1):
                if not (l[p] < l[p - k]) or not (l[p] <= l[p + k]): ok = False; break
            if ok: cl = l[p]; il = p
        sh[j] = ch; sl[j] = cl; shi[j] = ih; sli[j] = il
    return sh, sl, shi, sli


@njit(cache=True)
def sig_sweep_mss(o, h, l, c, a, n_piv, mss_wait, buf_atr, min_sweep_atr):
    """Sweep de lichiditate + schimbare de structura (MSS).
    Sweep: un fitil trece de ultimul pivot confirmat si bara se inchide inapoi in interior. Apoi, in maximum `mss_wait` bare,
    o inchidere dincolo de ultimul pivot opus (cel cunoscut in momentul sweep-ului) confirma MSS si da semnalul in directia opusa sweep-ului.
    Stopul: dincolo de extremul sweep-ului + buf_atr * ATR."""
    N = len(h); sh, sl, shi, sli = swings(h, l, n_piv)
    sig = np.zeros(N, np.int8); sd = np.zeros(N)
    armed = 0; until = -1; extreme = 0.0; ref = 0.0
    for j in range(n_piv * 2 + 2, N):
        if np.isnan(a[j]): continue
        lh = sh[j - 1]; ll = sl[j - 1]
        # sweep nou (suprascrie armarea precedenta)
        if not np.isnan(lh) and h[j] > lh and c[j] < lh and (h[j] - lh) >= min_sweep_atr * a[j] and j - shi[j - 1] >= 3 and not np.isnan(ll):
            armed = -1; until = j + mss_wait; extreme = h[j]; ref = ll
        elif not np.isnan(ll) and l[j] < ll and c[j] > ll and (ll - l[j]) >= min_sweep_atr * a[j] and j - sli[j - 1] >= 3 and not np.isnan(lh):
            armed = 1; until = j + mss_wait; extreme = l[j]; ref = lh
        if armed != 0:
            if j > until: armed = 0
            elif armed == -1 and c[j] < ref:
                d = (extreme + buf_atr * a[j]) - c[j]
                if d >= 0.3 * a[j]: sig[j] = -1; sd[j] = d
                armed = 0
            elif armed == 1 and c[j] > ref:
                d = c[j] - (extreme - buf_atr * a[j])
                if d >= 0.3 * a[j]: sig[j] = 1; sd[j] = d
                armed = 0
    return sig, sd


@njit(cache=True)
def sig_ob(o, h, l, c, a, n_piv, disp_k, lookback, max_age, need_reject, buf_atr, full_range):
    """Order block: ultima lumanare opusa inainte de o deplasare puternica (corp >= disp_k * ATR) care sparge ultimul pivot.
    Semnal la prima atingere a zonei (retest) cat timp zona e valida (nu s-a inchis dincolo de ea) si nu e mai veche de max_age bare.
    need_reject=1: bara de retest trebuie sa se inchida in directia tranzactiei. Stopul: dincolo de zona + buf_atr * ATR."""
    N = len(h); sh, sl, shi, sli = swings(h, l, n_piv)
    sig = np.zeros(N, np.int8); sd = np.zeros(N)
    CAP = 8
    zb = np.zeros((2, CAP)); zt = np.zeros((2, CAP)); zbirth = np.full((2, CAP), -1, np.int64)   # 0 = bullish (long), 1 = bearish (short)
    for j in range(n_piv * 2 + 2, N):
        if np.isnan(a[j]): continue
        # 1) retest-uri pe zonele deja existente (nascute inainte de j)
        trig_l = False; trig_s = False; sd_l = 0.0; sd_s = 0.0
        for k in range(CAP):
            if zbirth[0, k] >= 0:
                if c[j] < zb[0, k] or j - zbirth[0, k] > max_age: zbirth[0, k] = -1
                elif l[j] <= zt[0, k] and (need_reject == 0 or c[j] > o[j]) and not trig_l:
                    d = (c[j] - zb[0, k]) + buf_atr * a[j]
                    if d >= 0.3 * a[j]: trig_l = True; sd_l = d
                    zbirth[0, k] = -1
            if zbirth[1, k] >= 0:
                if c[j] > zt[1, k] or j - zbirth[1, k] > max_age: zbirth[1, k] = -1
                elif h[j] >= zb[1, k] and (need_reject == 0 or c[j] < o[j]) and not trig_s:
                    d = (zt[1, k] - c[j]) + buf_atr * a[j]
                    if d >= 0.3 * a[j]: trig_s = True; sd_s = d
                    zbirth[1, k] = -1
        if trig_l and not trig_s: sig[j] = 1; sd[j] = sd_l
        elif trig_s and not trig_l: sig[j] = -1; sd[j] = sd_s
        # 2) deplasare noua la bara j -> zona noua (poate fi folosita de la j+1)
        body = c[j] - o[j]
        if body >= disp_k * a[j] and not np.isnan(sh[j - 1]) and c[j] > sh[j - 1]:
            for p in range(j - 1, max(j - 1 - lookback, 0) - 1, -1):
                if c[p] < o[p]:
                    bot = l[p] if full_range == 1 else c[p]; top = h[p] if full_range == 1 else o[p]
                    slot = 0; oldest = 1 << 60
                    for k in range(CAP):
                        if zbirth[0, k] < 0: slot = k; oldest = -1; break
                        if zbirth[0, k] < oldest: oldest = zbirth[0, k]; slot = k
                    zb[0, slot] = bot; zt[0, slot] = top; zbirth[0, slot] = j; break
        elif -body >= disp_k * a[j] and not np.isnan(sl[j - 1]) and c[j] < sl[j - 1]:
            for p in range(j - 1, max(j - 1 - lookback, 0) - 1, -1):
                if c[p] > o[p]:
                    bot = l[p] if full_range == 1 else o[p]; top = h[p] if full_range == 1 else c[p]
                    slot = 0; oldest = 1 << 60
                    for k in range(CAP):
                        if zbirth[1, k] < 0: slot = k; oldest = -1; break
                        if zbirth[1, k] < oldest: oldest = zbirth[1, k]; slot = k
                    zb[1, slot] = bot; zt[1, slot] = top; zbirth[1, slot] = j; break
    return sig, sd


@njit(cache=True)
def sig_fvg_retest(o, h, l, c, a, min_gap_atr, max_age, need_reject, buf_atr, n_piv, need_sweep, sweep_win):
    """FVG (golul dintre lumanarea j-2 si j) cu retest in zona. need_sweep=1: valid doar daca in ultimele `sweep_win` bare a existat
    un sweep de lichiditate in directia opusa gol-ului si inchiderea care a creat gol-ul sparge structura (MSS) - ca in strategia FVG a ta."""
    N = len(h); sh, sl, shi, sli = swings(h, l, n_piv)
    sig = np.zeros(N, np.int8); sd = np.zeros(N)
    CAP = 8
    zb = np.zeros((2, CAP)); zt = np.zeros((2, CAP)); zbirth = np.full((2, CAP), -1, np.int64)
    last_sweep_low = -10 ** 9; last_sweep_high = -10 ** 9
    for j in range(n_piv * 2 + 3, N):
        if np.isnan(a[j]): continue
        lh = sh[j - 1]; ll = sl[j - 1]
        if not np.isnan(ll) and l[j] < ll and c[j] > ll: last_sweep_low = j
        if not np.isnan(lh) and h[j] > lh and c[j] < lh: last_sweep_high = j
        trig_l = False; trig_s = False; sd_l = 0.0; sd_s = 0.0
        for k in range(CAP):
            if zbirth[0, k] >= 0:
                if c[j] < zb[0, k] or j - zbirth[0, k] > max_age: zbirth[0, k] = -1
                elif l[j] <= zt[0, k] and (need_reject == 0 or c[j] > o[j]) and not trig_l:
                    d = (c[j] - zb[0, k]) + buf_atr * a[j]
                    if d >= 0.3 * a[j]: trig_l = True; sd_l = d
                    zbirth[0, k] = -1
            if zbirth[1, k] >= 0:
                if c[j] > zt[1, k] or j - zbirth[1, k] > max_age: zbirth[1, k] = -1
                elif h[j] >= zb[1, k] and (need_reject == 0 or c[j] < o[j]) and not trig_s:
                    d = (zt[1, k] - c[j]) + buf_atr * a[j]
                    if d >= 0.3 * a[j]: trig_s = True; sd_s = d
                    zbirth[1, k] = -1
        if trig_l and not trig_s: sig[j] = 1; sd[j] = sd_l
        elif trig_s and not trig_l: sig[j] = -1; sd[j] = sd_s
        # gol nou
        if l[j] - h[j - 2] >= min_gap_atr * a[j] and c[j - 1] > o[j - 1]:       # FVG bullish
            ok = need_sweep == 0 or (j - last_sweep_low <= sweep_win and not np.isnan(lh) and c[j] > lh)
            if ok:
                slot = 0; oldest = 1 << 60
                for k in range(CAP):
                    if zbirth[0, k] < 0: slot = k; oldest = -1; break
                    if zbirth[0, k] < oldest: oldest = zbirth[0, k]; slot = k
                zb[0, slot] = h[j - 2]; zt[0, slot] = l[j]; zbirth[0, slot] = j
        elif l[j - 2] - h[j] >= min_gap_atr * a[j] and c[j - 1] < o[j - 1]:     # FVG bearish
            ok = need_sweep == 0 or (j - last_sweep_high <= sweep_win and not np.isnan(ll) and c[j] < ll)
            if ok:
                slot = 0; oldest = 1 << 60
                for k in range(CAP):
                    if zbirth[1, k] < 0: slot = k; oldest = -1; break
                    if zbirth[1, k] < oldest: oldest = zbirth[1, k]; slot = k
                zb[1, slot] = h[j]; zt[1, slot] = l[j - 2]; zbirth[1, slot] = j
    return sig, sd


@njit(cache=True)
def sig_breakout(h, l, c, a, n):
    """Spargere de canal Donchian: inchidere peste maximul/minimul ultimelor n bare ANTERIOARE."""
    N = len(c); sig = np.zeros(N, np.int8); sd = np.zeros(N)
    hi = feat.roll_max(h, n); lo = feat.roll_min(l, n)
    for j in range(n + 1, N):
        if np.isnan(a[j]): continue
        if c[j] > hi[j - 1]: sig[j] = 1; sd[j] = a[j]
        elif c[j] < lo[j - 1]: sig[j] = -1; sd[j] = a[j]
    return sig, sd


@njit(cache=True)
def sig_pullback(c, a, ema_f, ema_s, rsi_v, lvl):
    """Pullback in trend: trend = ema_f vs ema_s; semnal cand RSI iese din zona de corectie (trece de `lvl` de jos in sus pentru long,
    de sus in jos de 100-lvl pentru short)."""
    N = len(c); sig = np.zeros(N, np.int8); sd = np.zeros(N)
    for j in range(2, N):
        if np.isnan(a[j]) or np.isnan(ema_s[j]) or np.isnan(rsi_v[j]) or np.isnan(rsi_v[j - 1]): continue
        if ema_f[j] > ema_s[j] and c[j] > ema_s[j] and rsi_v[j - 1] < lvl <= rsi_v[j]: sig[j] = 1; sd[j] = a[j]
        elif ema_f[j] < ema_s[j] and c[j] < ema_s[j] and rsi_v[j - 1] > 100 - lvl >= rsi_v[j]: sig[j] = -1; sd[j] = a[j]
    return sig, sd


# ---------------- filtre ----------------
@njit(cache=True)
def hour_mask(t, h0, h1):
    """True daca ora UTC a INCHIDERII barei e in [h0, h1) (se admite h0 > h1 = trece peste miezul noptii)."""
    N = len(t); out = np.zeros(N, np.bool_)
    for i in range(N):
        hh = ((t[i] % 86400) // 3600)
        out[i] = (hh >= h0 and hh < h1) if h0 <= h1 else (hh >= h0 or hh < h1)
    return out


def apply_mask(sig, sd, mask):
    s = sig.copy(); s[~mask] = 0; return s, sd
