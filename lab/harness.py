# Test de sanatate al motorului: pe zgomot pur NU trebuie sa iasa nicio strategie validata; pe o piata cu un edge plantat trebuie sa iasa.
import time, sys
import numpy as np
from . import search as S


def make_m1(days=520, seed=0, drift=0.0, sigma=0.00008, start=1_600_000_000):
    """Piata sintetica, luni-vineri 24h. drift>0: regimuri de trend (persistente ~6h) cu deriva = drift*sigma pe bara."""
    r = np.random.default_rng(seed); t = []; d0 = start - start % 86400
    for d in range(days):
        day = d0 + d * 86400
        if ((day // 86400) + 4) % 7 in (5, 6): continue      # sambata/duminica (epoca = joi)
        t.append(np.arange(day, day + 86400, 60))
    t = np.concatenate(t).astype(np.int64); n = len(t)
    ret = r.normal(0, sigma, n)
    if drift:
        blk = 360; nb = n // blk + 2; sgn = r.choice([-1.0, 1.0], nb)
        ret += np.repeat(sgn, blk)[:n] * drift * sigma
    c = 1.1 * np.exp(np.cumsum(ret)); o = np.r_[c[0], c[:-1]]
    wick = np.abs(r.normal(0, sigma * 0.5, (2, n))) * c
    h = np.maximum(o, c) + wick[0]; l = np.minimum(o, c) - wick[1]
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": np.full(n, 100, np.float32)}


def run(label, m1, budget, seed, **kw):
    ctx = S.Ctx(m1, "5m", cost_price=0.00008, slip_price=0.00002)
    res, ntr = S.search(ctx, budget, seed=seed)
    val = S.validate(ctx, res, ntr, k_final=25)
    ok = [v for v in val if v["stages"].get("lock")]; ftmo = [v for v in val if v["stages"].get("ftmo")]
    stage = {k: sum(1 for v in val if v["stages"].get(k)) for k in ("val", "robust", "cost", "time", "ftmo", "lock")}
    print(label, "tried", ntr, "bestTrainT %.2f" % res[0][0], "stages", stage, flush=True)
    return ok, val


if __name__ == "__main__":
    bud = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    if len(sys.argv) < 4:
        for s in (1, 2):
            run("ZGOMOT s%d" % s, make_m1(seed=s), bud, s)
    ok, val = run("EDGE   ", make_m1(seed=9, drift=float(sys.argv[2]) if len(sys.argv) > 2 else 0.25), bud, 9)
    for v in ok[:3]: print(v["genome"], v["lock"], v["ftmo"])
