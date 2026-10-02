import numpy as np
from . import blocks as B, feat


def walk(N=3000, seed=1):
    r = np.random.default_rng(seed)
    c = 100 + np.cumsum(r.normal(0, 0.3, N)); o = np.r_[c[0], c[:-1]] + r.normal(0, 0.05, N)
    h = np.maximum(o, c) + np.abs(r.normal(0, 0.15, N)); l = np.minimum(o, c) - np.abs(r.normal(0, 0.15, N))
    return o, h, l, c


def test_swings_delay():
    h = np.array([1, 2, 3, 5, 3, 2, 1, 1, 1, 1], float); l = h - 0.5
    sh, sl, shi, sli = B.swings(h, l, 2)
    assert np.isnan(sh[4]) and sh[5] == 5 and shi[5] == 3     # pivotul de la 3 cunoscut abia la 3+2=5


def test_sweep_mss_handmade():
    # minim pivot la 10 (nivel 9.0), maxim pivot, apoi sweep sub 9 care se inchide inapoi, apoi inchidere peste ultimul maxim
    c = [10, 10.2, 10, 9.6, 9.3, 9.1, 9.3, 9.6, 10, 10.4, 10.2, 9.9, 9.6, 9.9, 10.3, 10.6, 10.9, 11.2, 11.0, 10.8]
    N = 60; base = np.array(c + [11.0] * (N - len(c)), float)
    o = np.r_[base[0], base[:-1]]; h = np.maximum(o, base) + 0.1; l = np.minimum(o, base) - 0.1
    # bara 25: sweep sub minim
    l = l.copy(); l[18:] = 10.8; h[18:] = 11.0; o[18:] = 10.9; base[18:] = 10.9
    a = np.full(N, 0.4)
    sig, sd = B.sig_sweep_mss(o, h, l, base, a, 2, 6, 0.1, 0.0)
    assert sig.dtype == np.int8 and len(sig) == N


def test_no_lookahead():
    o, h, l, c = walk()
    a = feat.atr(h, l, c, 14)
    full = {
        'sw': B.sig_sweep_mss(o, h, l, c, a, 3, 8, 0.1, 0.1),
        'ob': B.sig_ob(o, h, l, c, a, 3, 1.0, 12, 60, 1, 0.1, 1),
        'fvg': B.sig_fvg_retest(o, h, l, c, a, 0.2, 40, 1, 0.1, 3, 0, 10),
        'fvgs': B.sig_fvg_retest(o, h, l, c, a, 0.1, 40, 1, 0.1, 3, 1, 10),
        'bo': B.sig_breakout(h, l, c, a, 20),
    }
    for name, (s, d) in full.items():
        assert (s != 0).sum() > 3, name + ' prea putine semnale: ' + str((s != 0).sum())
    for cut in (700, 1500, 2300):
        o2, h2, l2, c2 = o[:cut], h[:cut], l[:cut], c[:cut]; a2 = feat.atr(h2, l2, c2, 14)
        part = {
            'sw': B.sig_sweep_mss(o2, h2, l2, c2, a2, 3, 8, 0.1, 0.1),
            'ob': B.sig_ob(o2, h2, l2, c2, a2, 3, 1.0, 12, 60, 1, 0.1, 1),
            'fvg': B.sig_fvg_retest(o2, h2, l2, c2, a2, 0.2, 40, 1, 0.1, 3, 0, 10),
            'fvgs': B.sig_fvg_retest(o2, h2, l2, c2, a2, 0.1, 40, 1, 0.1, 3, 1, 10),
            'bo': B.sig_breakout(h2, l2, c2, a2, 20),
        }
        for name in full:
            assert np.array_equal(full[name][0][:cut], part[name][0]), (name, cut)
            assert np.allclose(full[name][1][:cut], part[name][1]), (name, cut)


def test_ob_zone_logic():
    # ultima lumanare bearish inainte de o deplasare bullish puternica -> retest = long
    N = 80; r = np.random.default_rng(3)
    c = 100 + np.cumsum(r.normal(0, 0.1, N)); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 0.05; l = np.minimum(o, c) - 0.05
    a = np.full(N, 0.3)
    sig, sd = B.sig_ob(o, h, l, c, a, 3, 1.0, 10, 50, 0, 0.1, 1)
    assert (sd[sig != 0] > 0).all()


def test_pullback_hour():
    t = np.arange(48) * 3600
    m = B.hour_mask(t, 7, 11)
    assert m.sum() == 8 and m[7] and not m[11]
    m2 = B.hour_mask(t, 22, 3)
    assert m2[23] and m2[1] and not m2[5]


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_'): v(); print('ok', k)
