# Strategia LIQ pe DAX (M1): lichiditate Daily/Weekly/Asia -> sweep -> MSS -> FVG -> intrare 50% -> SL pe wick -> TP 2R, BE 1.5R.
# Implementare dupa specificatia "Strategia Backtest LIQ" (capitolele indicate in comentarii). Tot ce nu e fixat de specificatie e PARAMETRU (SPACE).
# Convenții: preturile sunt BID; timpul barelor e UTC real; ora Romaniei = UTC+2/+3 (regula UE de vara). Fara privire in viitor:
#  - un nivel Daily/Weekly devine disponibil abia dupa inchiderea zilei/saptamanii (00:00 ora RO); Asia high/low dupa 09:30;
#  - swing-urile fractale se folosesc abia dupa nf bare de confirmare; MSS/FVG/intrare se evalueaza bara cu bara.
# Ordinea in interiorul unei bare M1 e necunoscuta -> mereu varianta pesimista (SL inainte de TP; BE inainte de TP; intrarea nu poate fi si iesire pe TP in aceeasi bara).
import calendar, datetime as dtm
import numpy as np
from .nb import njit

SRC_D, SRC_W, SRC_A = 1, 2, 4

# ---------------- ora Romaniei ----------------
def ro_offset(t):
    t = np.asarray(t, np.int64)
    yrs = (t.astype("datetime64[s]").astype("datetime64[Y]").astype(np.int64) + 1970)
    off = np.full(len(t), 7200, np.int64)
    for y in np.unique(yrs):
        def last_sun(m):
            d = dtm.date(int(y), m, 31); d -= dtm.timedelta(days=(d.weekday() + 1) % 7)
            return calendar.timegm(d.timetuple()) + 3600          # 01:00 UTC
        a, b = last_sun(3), last_sun(10)
        s = (yrs == y) & (t >= a) & (t < b); off[s] = 10800
    return off


# ---------------- nivelurile de lichiditate si sweep-urile lor (nu depind de variantele strategiei) ----------------
@njit(cache=True)
def build_levels(t, h, l, lmin, lday, tick, win0, win1, cap_rows, mstep):
    n = len(t); eps = tick * 1e-3; CAP = 20000
    hp = np.empty(CAP); hs = np.zeros(CAP, np.int64); ht = np.zeros(CAP, np.int64); nh = 0
    lp = np.empty(CAP); ls = np.zeros(CAP, np.int64); lt = np.zeros(CAP, np.int64); nl = 0
    rb = np.zeros(cap_rows, np.int64); rd = np.zeros(cap_rows, np.int8); rp = np.zeros(cap_rows); rs = np.zeros(cap_rows, np.int8); rt = np.zeros(cap_rows, np.int64)
    nr = 0
    cur_day = lday[0]; cur_wk = (lday[0] + 3) // 7
    dh = -1e18; dl = 1e18; dc = 0; wh = -1e18; wl = 1e18; wc = 0
    ah = -1e18; al = 1e18; ac = 0; asia_done = False
    minh = 1e18; maxl = -1e18
    tmp_p = np.empty(CAP); tmp_s = np.zeros(CAP, np.int64); tmp_t = np.zeros(CAP, np.int64)
    for j in range(n):
        d = lday[j]
        if d != cur_day:
            # inchidere zi: adauga nivelurile Daily (disponibile de acum)
            for side in range(2):
                if side == 0 and dc * mstep >= 30:
                    p = dh; merged = False
                    for i in range(nh):
                        if abs(hp[i] - p) < tick * 0.5: hs[i] |= SRC_D; merged = True; break
                    if not merged and nh < CAP: hp[nh] = p; hs[nh] = SRC_D; ht[nh] = t[j]; nh += 1
                if side == 1 and dc * mstep >= 30:
                    p = dl; merged = False
                    for i in range(nl):
                        if abs(lp[i] - p) < tick * 0.5: ls[i] |= SRC_D; merged = True; break
                    if not merged and nl < CAP: lp[nl] = p; ls[nl] = SRC_D; lt[nl] = t[j]; nl += 1
            wk = (d + 3) // 7
            if wk != cur_wk:
                if wc * mstep >= 100:
                    p = wh; merged = False
                    for i in range(nh):
                        if abs(hp[i] - p) < tick * 0.5: hs[i] |= SRC_W; merged = True; break
                    if not merged and nh < CAP: hp[nh] = p; hs[nh] = SRC_W; ht[nh] = t[j]; nh += 1
                    p = wl; merged = False
                    for i in range(nl):
                        if abs(lp[i] - p) < tick * 0.5: ls[i] |= SRC_W; merged = True; break
                    if not merged and nl < CAP: lp[nl] = p; ls[nl] = SRC_W; lt[nl] = t[j]; nl += 1
                cur_wk = wk; wh = -1e18; wl = 1e18; wc = 0
            # nivelurile Asia expira la schimbarea zilei
            k = 0
            for i in range(nh):
                s = hs[i] & ~SRC_A
                if s != 0: hp[k] = hp[i]; hs[k] = s; ht[k] = ht[i]; k += 1
            nh = k; k = 0
            for i in range(nl):
                s = ls[i] & ~SRC_A
                if s != 0: lp[k] = lp[i]; ls[k] = s; lt[k] = lt[i]; k += 1
            nl = k
            cur_day = d; dh = -1e18; dl = 1e18; dc = 0; ah = -1e18; al = 1e18; ac = 0; asia_done = False
            minh = 1e18
            for i in range(nh):
                if hp[i] < minh: minh = hp[i]
            maxl = -1e18
            for i in range(nl):
                if lp[i] > maxl: maxl = lp[i]
        # Asia 02:00-09:30 (120..569): nivelurile apar la prima bara >= 09:30
        if lmin[j] >= 570 and not asia_done:
            asia_done = True
            if ac * mstep >= 60:
                p = ah; merged = False
                for i in range(nh):
                    if abs(hp[i] - p) < tick * 0.5: hs[i] |= SRC_A; merged = True; break
                if not merged and nh < CAP: hp[nh] = p; hs[nh] = SRC_A; ht[nh] = t[j]; nh += 1
                p = al; merged = False
                for i in range(nl):
                    if abs(lp[i] - p) < tick * 0.5: ls[i] |= SRC_A; merged = True; break
                if not merged and nl < CAP: lp[nl] = p; ls[nl] = SRC_A; lt[nl] = t[j]; nl += 1
            minh = 1e18
            for i in range(nh):
                if hp[i] < minh: minh = hp[i]
            maxl = -1e18
            for i in range(nl):
                if lp[i] > maxl: maxl = lp[i]
        if d == cur_day:
            pass
        # sweep-uri la aceasta bara (inainte de a include bara in acumulatori; nivelurile sunt din trecut)
        inw = lmin[j] >= win0 and lmin[j] < win1
        if nh > 0 and h[j] >= minh + tick - eps:
            m = 0; i = 0
            while i < nh:
                if hp[i] + tick <= h[j] + eps:
                    tmp_p[m] = hp[i]; tmp_s[m] = hs[i]; tmp_t[m] = ht[i]; m += 1
                    hp[i] = hp[nh - 1]; hs[i] = hs[nh - 1]; ht[i] = ht[nh - 1]; nh -= 1
                else: i += 1
            if m > 1:
                o = np.argsort(tmp_p[:m])
            else: o = np.arange(m)
            if inw:
                for q in range(m):
                    if nr < cap_rows:
                        k = o[q]; rb[nr] = j; rd[nr] = 1; rp[nr] = tmp_p[k]; rs[nr] = tmp_s[k]; rt[nr] = tmp_t[k]; nr += 1
            minh = 1e18
            for i in range(nh):
                if hp[i] < minh: minh = hp[i]
        if nl > 0 and l[j] <= maxl - tick + eps:
            m = 0; i = 0
            while i < nl:
                if lp[i] - tick >= l[j] - eps:
                    tmp_p[m] = lp[i]; tmp_s[m] = ls[i]; tmp_t[m] = lt[i]; m += 1
                    lp[i] = lp[nl - 1]; ls[i] = ls[nl - 1]; lt[i] = lt[nl - 1]; nl -= 1
                else: i += 1
            if m > 1:
                o = np.argsort(-tmp_p[:m])
            else: o = np.arange(m)
            if inw:
                for q in range(m):
                    if nr < cap_rows:
                        k = o[q]; rb[nr] = j; rd[nr] = -1; rp[nr] = tmp_p[k]; rs[nr] = tmp_s[k]; rt[nr] = tmp_t[k]; nr += 1
            maxl = -1e18
            for i in range(nl):
                if lp[i] > maxl: maxl = lp[i]
        # acumulatori (dupa verificarea sweep-urilor)
        if h[j] > dh: dh = h[j]
        if l[j] < dl: dl = l[j]
        dc += 1
        if h[j] > wh: wh = h[j]
        if l[j] < wl: wl = l[j]
        wc += 1
        if lmin[j] >= 120 and lmin[j] < 570:
            if h[j] > ah: ah = h[j]
            if l[j] < al: al = l[j]
            ac += 1
    return rb[:nr], rd[:nr], rp[:nr], rs[:nr], rt[:nr]


@njit(cache=True)
def swings(h, l, nf):
    """Fractal: swing high la i daca h[i] >= h din stanga (nf bare) si > h din dreapta (nf bare). Folosit doar dupa i+nf (confirmare)."""
    n = len(h); sh = np.zeros(n, np.bool_); sl = np.zeros(n, np.bool_)
    for i in range(nf, n - nf):
        ok = True
        for k in range(1, nf + 1):
            if h[i - k] > h[i] or h[i + k] >= h[i]: ok = False; break
        sh[i] = ok
        ok = True
        for k in range(1, nf + 1):
            if l[i - k] < l[i] or l[i + k] <= l[i]: ok = False; break
        sl[i] = ok
    return sh, sl


# ---------------- simulatorul strategiei ----------------
# indicii vectorului de parametri
(P_MODEL, P_NF, P_LEGN, P_PRE, P_DMODE, P_DMULT, P_DN, P_DK, P_DRETR, P_SLMODE, P_BUF, P_FPRE, P_FMIN, P_WIN, P_ORD, P_START, P_END,
 P_MAXTR, P_STOPWIN, P_TPR, P_BER, P_SRC, P_DIR, P_ANCH, P_MAXSL, P_TICK, P_MAXF, P_SEL, P_MINSL) = range(29)
NP = 29
NCOL = 21


@njit(cache=True)
def HV(d, h, l, j):
    return h[j] if d > 0 else -l[j]


@njit(cache=True)
def LV(d, h, l, j):
    return l[j] if d > 0 else -h[j]


@njit(cache=True)
def fvg_update(j, d, h, l, zlo, zhi, zi, zal, cnt, fmin, i1min):
    for k in range(cnt):
        if zal[k] and zi[k] < j and LV(d, h, l, j) <= zlo[k]: zal[k] = 0
    if j >= 2 and j - 2 >= i1min and LV(d, h, l, j) > HV(d, h, l, j - 2):
        lo = HV(d, h, l, j - 2); hi = LV(d, h, l, j)
        if hi - lo >= fmin and cnt < 64:
            zlo[cnt] = lo; zhi[cnt] = hi; zi[cnt] = j; zal[cnt] = 1; cnt += 1
    return cnt


@njit(cache=True)
def fvg_select(cnt, zal, maxf, sel):
    """Intoarce (index FVG, nr FVG-uri vii). Index -1: niciunul; -2: prea multe (setup invalid)."""
    a = 0
    for k in range(cnt):
        if zal[k]: a += 1
    if a == 0: return -1, 0
    if a > maxf: return -2, a
    if sel == 0: rank = a // 2          # 1->singurul, 2->ultimul, 3->cel din mijloc (specificatia, cap. 14)
    elif sel == 1: rank = a - 1
    else: rank = 0
    c = 0
    for k in range(cnt):
        if zal[k]:
            if c == rank: return k, a
            c += 1
    return -1, a


@njit(cache=True)
def disp_ok(d, o, h, l, c, j, mode, mult, N, k, retr):
    if mode == 0: return True
    if j - N < 0: return False
    s = 0.0
    for i in range(j - N, j): s += h[i] - l[i]
    avg = s / N
    if mode == 1: return (h[j] - l[j]) > mult * avg
    if j - k + 1 < 0: return False
    net = d * (c[j] - o[j - k + 1]); opp = 0
    for i in range(j - k + 1, j + 1):
        if d * (c[i] - o[i]) < 0: opp += 1
    return net >= mult * avg * k and opp <= retr


@njit(cache=True)
def _row(out, nt, j, t, rg, cost, risk, kind, ej):
    if nt < out.shape[0]:
        out[nt, 1] = t[j]; out[nt, 4] = rg; out[nt, 3] = rg - cost[ej] / risk; out[nt, 17] = kind; out[nt, 20] = cost[ej]


@njit(cache=True)
def simulate(t, o, h, l, c, lmin, lday, cost, sw_bar, sw_dir, sw_price, sw_src, swH, swL, P, out, cnt):
    """Ruleaza strategia pe tot istoricul. out: matrice de tranzactii (NCOL coloane); cnt: contoare (0 evenimente,1 fara MSS,2 MSS,3 FVG invalid,4 fara intrare,5 SL invalid,6 tranzactii)."""
    n = len(t); nsw = len(sw_bar); p = 0; nt = 0
    tick = P[P_TICK]; eps = tick * 1e-3
    model = int(P[P_MODEL]); nf = int(P[P_NF]); legn = int(P[P_LEGN]); pre = int(P[P_PRE]); fpre = int(P[P_FPRE])
    start = P[P_START]; end = P[P_END]; win = P[P_WIN] * 60
    srcm = int(P[P_SRC]); dirm = int(P[P_DIR]); ordm = int(P[P_ORD]); slmode = int(P[P_SLMODE])
    maxtr = P[P_MAXTR]; tpr = P[P_TPR]; ber = P[P_BER]; buf = P[P_BUF]; maxsl = P[P_MAXSL] * 1.0
    be_en = ber > 0 and ber < tpr
    zlo = np.zeros(64); zhi = np.zeros(64); zi = np.zeros(64, np.int64); zal = np.zeros(64, np.int8); zc = 0
    state = 0; cur_day = -1; tr_today = 0; stop_day = False
    d = 1; s = 0; t_anchor = 0; n_lv = 0; first_src = 0; last_src = 0; lo_run = 0.0; ext = 0; minup = 1e18; mss_j = 0; sel_k = -1; nfv = 0
    E = 0.0; slp = 0.0; tpp = 0.0; bep = 0.0; risk = 1.0; be_on = False; ej = 0
    for j in range(n):
        if lday[j] != cur_day:
            if state == 3:      # pozitie ramasa deschisa peste o zi fara bare in afara ferestrei (date lipsa)
                _row(out, nt, j - 1, t, (d * c[j - 1] - E) / risk, cost, risk, 4, ej); nt += 1
            elif state == 1: cnt[1] += 1
            elif state == 2: cnt[4] += 1
            state = 0; cur_day = lday[j]; tr_today = 0; stop_day = False
        if not (lmin[j] >= start and lmin[j] < end):
            if state == 3:
                _row(out, nt, j, t, (d * o[j] - E) / risk, cost, risk, 4, ej); nt += 1
            elif state == 1: cnt[1] += 1
            elif state == 2: cnt[4] += 1
            state = 0
            while p < nsw and sw_bar[p] <= j: p += 1
            continue
        allow_new = state == 0
        # ---------- A. pozitie deschisa ----------
        if state == 3 and j > ej:
            hi = HV(d, h, l, j); lo = LV(d, h, l, j); done = False; rg = 0.0; kind = 0
            if be_on:
                if lo <= E: done = True; rg = 0.0; kind = 3
                elif hi >= tpp: done = True; rg = tpr; kind = 1
            else:
                if lo <= slp: done = True; rg = -1.0; kind = 2
                elif be_en and hi >= bep and lo <= E: done = True; rg = 0.0; kind = 3     # ordine pesimista in bara
                elif hi >= tpp: done = True; rg = tpr; kind = 1
                elif be_en and hi >= bep: be_on = True
            if done:
                _row(out, nt, j, t, rg, cost, risk, kind, ej); nt += 1
                tr_today += 1; state = 0
                if rg > 0 and P[P_STOPWIN] > 0: stop_day = True
                if tr_today >= maxtr: stop_day = True
        # ---------- B. eveniment: asteapta MSS (1) sau ordin limita la FVG (2) ----------
        elif state == 1 or state == 2:
            if state == 1: expired = (t[j] - t_anchor) > win
            elif ordm == 0: expired = (t[j] - t_anchor) > win
            elif ordm == 1: expired = (t[j] - t[mss_j]) > win
            else: expired = False
            if expired:
                if state == 1: cnt[1] += 1
                else: cnt[4] += 1
                state = 0; allow_new = True
            else:
                was2 = state == 2
                # intrare: doar dupa MSS (bara MSS nu conteaza), pe FVG-ul ales la sfarsitul barei precedente
                if was2 and sel_k >= 0:
                    Ep = 0.5 * (zlo[sel_k] + zhi[sel_k])
                    if LV(d, h, l, j) <= Ep + eps:
                        slp0 = lo_run
                        if slmode == 1:
                            for i in range(j - 1 - nf, s - 1, -1):
                                if i < 0: break
                                fl = swL[i] if d > 0 else swH[i]
                                if fl:
                                    cd = LV(d, h, l, i)
                                    if cd < Ep: slp0 = cd
                                    break
                        sl_ = slp0 - buf; rk = Ep - sl_
                        if rk <= P[P_MINSL] or rk > maxsl:
                            cnt[5] += 1; state = 0
                        else:
                            E = Ep; slp = sl_; risk = rk; tpp = E + tpr * risk; bep = E + ber * risk; be_on = False; ej = j; state = 3; cnt[6] += 1
                            out[nt, 0] = t[j]; out[nt, 2] = d; out[nt, 5] = rk; out[nt, 6] = n_lv; out[nt, 7] = first_src; out[nt, 8] = last_src
                            out[nt, 9] = nfv; out[nt, 10] = 0; out[nt, 11] = (t[mss_j] - t[s]) / 60.0; out[nt, 12] = d * E; out[nt, 13] = d * slp
                            out[nt, 14] = t[s]; out[nt, 15] = t[mss_j]; out[nt, 16] = zhi[sel_k] - zlo[sel_k]; out[nt, 18] = lday[j]; out[nt, 19] = model
                            if LV(d, h, l, j) <= slp:      # in bara intrarii: SL are prioritate (pesimist)
                                _row(out, nt, j, t, -1.0, cost, risk, 2, ej); nt += 1
                                tr_today += 1; state = 0
                                if tr_today >= maxtr: stop_day = True
                if state == 1:
                    ok = False
                    if model == 0:
                        i = j - 1 - nf
                        if i >= s - pre and i >= 0:
                            fl = swH[i] if d > 0 else swL[i]
                            if fl:
                                v = HV(d, h, l, i)
                                if v < minup: minup = v
                        if minup < 1e17 and HV(d, h, l, j) >= minup + tick - eps: ok = True
                    else:
                        a0 = ext - legn
                        if a0 < 0: a0 = 0
                        lg = -1e18
                        for i in range(a0, ext + 1):
                            v = HV(d, h, l, i)
                            if v > lg: lg = v
                        if HV(d, h, l, j) >= lg + tick - eps: ok = True
                    if ok and not disp_ok(d, o, h, l, c, j, int(P[P_DMODE]), P[P_DMULT], int(P[P_DN]), int(P[P_DK]), int(P[P_DRETR])): ok = False
                    if ok: mss_j = j; state = 2; cnt[2] += 1
                if state == 1 or state == 2:
                    zc = fvg_update(j, d, h, l, zlo, zhi, zi, zal, zc, P[P_FMIN], s - fpre)
                    if LV(d, h, l, j) <= lo_run:
                        lo_run = LV(d, h, l, j)
                        if state == 1: ext = j
                    if state == 2:
                        sel_k, nfv = fvg_select(zc, zal, int(P[P_MAXF]), int(P[P_SEL]))
                        if sel_k == -2: cnt[3] += 1; state = 0; sel_k = -1
        # ---------- C. sweep-uri la aceasta bara ----------
        while p < nsw and sw_bar[p] < j: p += 1
        while p < nsw and sw_bar[p] == j:
            dd = -int(sw_dir[p])            # nivel maxim sweepuit -> setup bearish (d=-1)
            if (int(sw_src[p]) & srcm) != 0 and (dirm & (1 if dd > 0 else 2)) != 0:
                if state == 0 and allow_new and (not stop_day) and tr_today < maxtr:
                    d = dd; s = j; t_anchor = t[j]; n_lv = 1; first_src = int(sw_src[p]); last_src = first_src
                    lo_run = LV(d, h, l, j); ext = j; minup = 1e18; zc = 0; sel_k = -1; nfv = 0; state = 1; cnt[0] += 1
                    if model == 0:
                        for i in range(max(0, s - pre), s - nf):
                            fl = swH[i] if d > 0 else swL[i]
                            if fl:
                                v = HV(d, h, l, i)
                                if v < minup: minup = v
                    for i3 in range(max(2, s - fpre + 2), s + 1):
                        zc = fvg_update(i3, d, h, l, zlo, zhi, zi, zal, zc, P[P_FMIN], s - fpre)
                elif state == 1 and dd == d:
                    n_lv += 1; last_src = int(sw_src[p])
                    if int(P[P_ANCH]) == 1: t_anchor = t[j]
            p += 1
    return nt
