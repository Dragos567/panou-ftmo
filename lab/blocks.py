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


# ---------------- lichiditate si sesiuni ----------------
@njit(cache=True)
def sig_liq_sweep(t, o, h, l, c, a, mode, min_sweep_atr, buf_atr, asia_end):
    """Sweep de lichiditate pe nivele cunoscute: mode 0 = maximul/minimul zilei UTC precedente (PDH/PDL); mode 1 = intervalul sesiunii asiatice
    (00:00-asia_end UTC) din ziua curenta, tranzactionat doar dupa terminarea ei. Fitil dincolo de nivel + inchidere inapoi in interior -> semnal invers."""
    N = len(c); sig = np.zeros(N, np.int8); sd = np.zeros(N)
    cur = -1; dh = -1e300; dl = 1e300; ph = np.nan; pl = np.nan; ah = -1e300; al = 1e300
    for j in range(N):
        d = t[j] // 86400
        if d != cur:
            if cur >= 0: ph = dh; pl = dl
            cur = d; dh = -1e300; dl = 1e300; ah = -1e300; al = 1e300
        hr = (t[j] % 86400) // 3600
        if np.isnan(a[j]):
            dh = max(dh, h[j]); dl = min(dl, l[j]); continue
        lh = np.nan; ll = np.nan
        if mode == 0: lh = ph; ll = pl
        elif hr >= asia_end and ah > -1e299: lh = ah; ll = al
        if not np.isnan(lh):
            if h[j] > lh and c[j] < lh and (h[j] - lh) >= min_sweep_atr * a[j]:
                dd = (h[j] + buf_atr * a[j]) - c[j]
                if dd >= 0.3 * a[j]: sig[j] = -1; sd[j] = dd
            elif l[j] < ll and c[j] > ll and (ll - l[j]) >= min_sweep_atr * a[j]:
                dd = c[j] - (l[j] - buf_atr * a[j])
                if dd >= 0.3 * a[j]: sig[j] = 1; sd[j] = dd
        dh = max(dh, h[j]); dl = min(dl, l[j])
        if hr < asia_end: ah = max(ah, h[j]); al = min(al, l[j])
    return sig, sd


@njit(cache=True)
def equal_levels_sweep(h, l, c, a, n_piv, tol_atr, min_sweep_atr, buf_atr, win):
    """Maxime/minime egale (doua pivoti in tol_atr*ATR unul de altul = lichiditate acumulata) sparte printr-un fitil si respinse."""
    N = len(c); sig = np.zeros(N, np.int8); sd = np.zeros(N)
    ph = np.full(8, np.nan); pl = np.full(8, np.nan); phi = np.full(8, -1); pli = np.full(8, -1)
    nh = 0; nl = 0
    for j in range(2 * n_piv + 1, N):
        p = j - n_piv
        ok = True
        for k in range(1, n_piv + 1):
            if not (h[p] > h[p - k]) or not (h[p] >= h[p + k]): ok = False; break
        if ok:
            ph[nh % 8] = h[p]; phi[nh % 8] = p; nh += 1
        ok = True
        for k in range(1, n_piv + 1):
            if not (l[p] < l[p - k]) or not (l[p] <= l[p + k]): ok = False; break
        if ok:
            pl[nl % 8] = l[p]; pli[nl % 8] = p; nl += 1
        if np.isnan(a[j]): continue
        # nivel de maxime egale: cel mai recent maxim cu un altul din apropiere
        for x in range(8):
            if phi[x] < 0 or j - phi[x] > win: continue
            for y in range(8):
                if y != x and phi[y] >= 0 and phi[y] < phi[x] and abs(ph[y] - ph[x]) <= tol_atr * a[j]:
                    lv = max(ph[x], ph[y])
                    if h[j] > lv and c[j] < lv and (h[j] - lv) >= min_sweep_atr * a[j] and phi[x] < j - 1:
                        dd = (h[j] + buf_atr * a[j]) - c[j]
                        if dd >= 0.3 * a[j]: sig[j] = -1; sd[j] = dd
        for x in range(8):
            if pli[x] < 0 or j - pli[x] > win: continue
            for y in range(8):
                if y != x and pli[y] >= 0 and pli[y] < pli[x] and abs(pl[y] - pl[x]) <= tol_atr * a[j]:
                    lv = min(pl[x], pl[y])
                    if l[j] < lv and c[j] > lv and (lv - l[j]) >= min_sweep_atr * a[j] and pli[x] < j - 1 and sig[j] == 0:
                        dd = c[j] - (l[j] - buf_atr * a[j])
                        if dd >= 0.3 * a[j]: sig[j] = 1; sd[j] = dd
    return sig, sd


# ---------------- proxy-uri de order flow (din tick volume; NU sunt order flow real) ----------------
@njit(cache=True)
def sig_vol_spike(o, h, l, c, v, a, n, k, mode):
    """Spike de tick-volum (v > k * media ultimelor n bare). mode 0 = continuare in directia barei; mode 1 = climax: bara cu spike si fitil mare
    de respingere (inchidere in treimea opusa) -> semnal invers."""
    N = len(c); sig = np.zeros(N, np.int8); sd = np.zeros(N)
    for j in range(n + 1, N):
        if np.isnan(a[j]): continue
        s = 0.0
        for q in range(j - n, j): s += v[q]
        m = s / n
        if m <= 0 or v[j] < k * m: continue
        rng = h[j] - l[j]
        if rng <= 0: continue
        loc = (c[j] - l[j]) / rng
        if mode == 0:
            if c[j] > o[j] and loc > 0.7: sig[j] = 1; sd[j] = max(c[j] - l[j], 0.5 * a[j])
            elif c[j] < o[j] and loc < 0.3: sig[j] = -1; sd[j] = max(h[j] - c[j], 0.5 * a[j])
        else:
            if loc > 0.66 and l[j] < l[j - 1] and rng >= 0.8 * a[j]: sig[j] = 1; sd[j] = max(c[j] - l[j], 0.5 * a[j])
            elif loc < 0.34 and h[j] > h[j - 1] and rng >= 0.8 * a[j]: sig[j] = -1; sd[j] = max(h[j] - c[j], 0.5 * a[j])
    return sig, sd


@njit(cache=True)
def sig_delta_div(o, h, l, c, v, a, n, m):
    """Divergenta de volum semnat (proxy de delta: v * pozitia inchiderii in bara): pret la maxim/minim pe n bare, dar delta cumulata pe m bare
    e de semn opus -> semnal invers, cu confirmarea inchiderii barei."""
    N = len(c); sig = np.zeros(N, np.int8); sd = np.zeros(N)
    dl = np.zeros(N)
    for j in range(N):
        r = h[j] - l[j]; dl[j] = v[j] * (2.0 * (c[j] - l[j]) / r - 1.0) if r > 0 else 0.0
    for j in range(max(n, m) + 1, N):
        if np.isnan(a[j]): continue
        s = 0.0
        for q in range(j - m + 1, j + 1): s += dl[q]
        hi = h[j - n]; lo = l[j - n]
        for q in range(j - n, j):
            if h[q] > hi: hi = h[q]
            if l[q] < lo: lo = l[q]
        if h[j] > hi and s < 0 and c[j] < o[j]: sig[j] = -1; sd[j] = max(h[j] - c[j], 0.5 * a[j]) + 0.1 * a[j]
        elif l[j] < lo and s > 0 and c[j] > o[j]: sig[j] = 1; sd[j] = max(c[j] - l[j], 0.5 * a[j]) + 0.1 * a[j]
    return sig, sd
