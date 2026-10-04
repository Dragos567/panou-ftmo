# Teste simulator LIQ: scenariu facut de mana (long: sweep Asia low -> MSS -> FVG -> 50% -> TP 2R), mirror (short), fara privire in viitor.
import calendar, datetime as dtm, numpy as np
from . import liq as Q, liqx as X


def day_bars(date, scen, minute0=600):
    """Bare M1 pentru o zi (ora RO = UTC+2, iarna). Pana la 10:00: plat 105 cu Asia low 103 la 04:00 si high 107 la 05:00; apoi scenariul; apoi plat."""
    base = calendar.timegm(dtm.date(*date).timetuple()) - 7200     # 00:00 RO in UTC
    rows = []
    last = None
    for m in range(0, 1440):
        if m < minute0:
            p = 105.0; h = p + 0.3; l = p - 0.3
            if m == 4 * 60: l = 103.0
            if m == 5 * 60: h = 107.0
            rows.append((base + m * 60, p, h, l, p))
        elif m - minute0 < len(scen):
            rows.append((base + m * 60,) + scen[m - minute0])
        else:
            q = scen[-1][3]; rows.append((base + m * 60, q, q + 0.2, q - 0.2, q))
    return rows


def flat_day(date, p=100.0):
    base = calendar.timegm(dtm.date(*date).timetuple()) - 7200
    return [(base + m * 60, p, p + 0.5, p - 0.5, p) for m in range(1440)]


def to_m1(rows):
    a = np.array(rows, np.float64)
    return {"t": a[:, 0].astype(np.int64), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4], "v": np.ones(len(a), np.float32)}


LONG = [(105, 105.3, 104.7, 105)] * 5 + [
    (105, 105, 102.5, 102.8), (102.8, 103.5, 102.0, 103.3), (103.3, 104.0, 103.2, 103.9), (103.9, 103.9, 103.0, 103.2), (103.2, 103.6, 103.1, 103.5),
    (103.5, 106.0, 103.4, 105.8), (105.8, 106.5, 105.0, 106.2), (106.2, 106.3, 104.2, 104.5), (104.5, 107.0, 104.4, 106.9), (106.9, 108.0, 106.5, 107.9),
    (107.9, 109.0, 107.5, 108.8)]


def mirror(scen, c=210.0):
    return [(c - o, c - l, c - h, c - cl) for (o, h, l, cl) in scen]


def build(scen, asia_mirror=False):
    rows = flat_day((2024, 1, 8)) + day_bars((2024, 1, 9), scen)
    rows += flat_day((2024, 1, 10), 108.5)
    if asia_mirror: pass
    return to_m1(rows)


def mkctx(m1):
    c = X.LCtx(m1, tick=0.01, cost_pts=0.0, slip=0.0); return c


def test_long():
    c = mkctx(build(LONG)); g = dict(X.BASE, nf=1, pre=10)
    tr, cnt = c.run(g)
    assert len(tr) == 1, (len(tr), cnt)
    r = tr[0]
    assert r[2] == 1 and abs(r[12] - 104.3) < 1e-9 and abs(r[13] - 102.0) < 1e-9, r
    assert abs(r[4] - 2.0) < 1e-9 and r[17] == 1, r
    assert cnt[0] >= 1 and cnt[2] == 1


def test_model_b_same():
    c = mkctx(build(LONG)); g = dict(X.BASE, model=1, leg_n=5)
    tr, cnt = c.run(g)
    assert len(tr) == 1 and abs(tr[0][4] - 2.0) < 1e-9, (tr, cnt)


def test_short_mirror():
    # oglindire: preturile 210-p; Asia: scenariul ramane in jurul lui 105 -> pentru oglinda construim ziua cu Asia high la 107 swept in sus
    scen = mirror(LONG)
    base = calendar.timegm(dtm.date(2024, 1, 9).timetuple()) - 7200
    rows = flat_day((2024, 1, 8), 110.0)
    d = []
    for m in range(0, 1440):
        if m < 600:
            p = 105.0; h = p + 0.3; l = p - 0.3
            if m == 240: h = 107.0
            if m == 300: l = 103.0
            d.append((base + m * 60, p, h, l, p))
        elif m - 600 < len(scen): d.append((base + m * 60,) + scen[m - 600])
        else: q = scen[-1][3]; d.append((base + m * 60, q, q + 0.2, q - 0.2, q))
    # scenariul oglindit porneste de la 105 (210-105) si sweep-uieste maximul Asia (107) dupa care cade -> bearish; Asia high 107 < 107.5 (varful oglindit)
    m1 = to_m1(rows + d + flat_day((2024, 1, 10), 101.5))
    c = mkctx(m1); tr, cnt = c.run(dict(X.BASE, nf=1, pre=10, dir=3))
    assert len(tr) >= 1, cnt
    sh = [r for r in tr if r[2] == -1]
    assert sh and abs(sh[0][4] - 2.0) < 1e-9, tr


def test_no_lookahead():
    m1 = build(LONG); c = mkctx(m1); g = dict(X.BASE, nf=1)
    tr_full, _ = c.run(g)
    cutoff = len(m1["t"]) - 1700    # taie dupa tranzactie
    sub = {k: v[:cutoff] for k, v in m1.items()}
    tr_sub, _ = mkctx(sub).run(g)
    ex = tr_full[tr_full[:, 1] < sub["t"][-1] - 600]; es = tr_sub[tr_sub[:, 1] < sub["t"][-1] - 600]
    assert np.allclose(ex[:, :5], es[:, :5]), (ex, es)


def test_levels_dst_and_cost():
    t = np.array([1704067200, 1720000000], np.int64)   # 2024-01-01 00:00 UTC (iarna), 2024-07-03 (vara)
    off = Q.ro_offset(t)
    assert off[0] == 7200 and off[1] == 10800


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
