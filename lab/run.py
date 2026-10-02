# Alergator de cautare: ruleaza ca proces separat (nice), scrie progresul si rezultatul in DATA/lab/<job>/.
#   python3 -m lab.run --data DIR --sym EURUSD --tf 5m --minutes 20 [--seed 1]
import os, sys, json, time, glob, argparse, traceback
import numpy as np

# COSTURI PROVIZORII (pret pe tranzactie dus-intors, in unitati de pret) - se inlocuiesc cu valori masurate pe contul real (Faza C)
COSTS = {"EURUSD": (0.00012, 0.00003), "NIKKEI": (8.0, 2.0), "GOLD": (0.30, 0.10), "USDJPY": (0.012, 0.003), "GBPUSD": (0.00015, 0.00004), "DAX": (2.0, 0.5), "UK100": (2.0, 0.5)}


def load_m1(data_dir, sym):
    from .data import DT
    ps = sorted(glob.glob(os.path.join(data_dir, "hist", "%s_1m" % sym, "p_*.bin")), key=lambda p: int(os.path.basename(p)[2:-4]))
    arrs = []
    for p in ps:
        n = os.path.getsize(p) // DT.itemsize
        if n: arrs.append(np.fromfile(p, dtype=DT, count=n))
    if not arrs: raise RuntimeError("fara istoric pentru " + sym)
    a = np.concatenate(arrs); a = a[np.argsort(a["t"], kind="stable")]
    _, ix = np.unique(a["t"], return_index=True); a = a[ix]
    return {k: np.ascontiguousarray(a[k]) for k in ("t", "o", "h", "l", "c", "v")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--sym", default="EURUSD")
    ap.add_argument("--tf", default="5m"); ap.add_argument("--minutes", type=float, default=10); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--job", default=None); a = ap.parse_args()
    from . import search as S
    job = a.job or time.strftime("%Y%m%d_%H%M%S") + "_%s_%s" % (a.sym, a.tf)
    out = os.path.join(a.data, "lab", job); os.makedirs(out, exist_ok=True)
    st = {"job": job, "sym": a.sym, "tf": a.tf, "state": "incarc date", "started": int(time.time()), "costs_provizorii": True}
    def save(**kw):
        st.update(kw); st["updated"] = int(time.time())
        json.dump(st, open(os.path.join(out, "status.json.tmp"), "w")); os.replace(os.path.join(out, "status.json.tmp"), os.path.join(out, "status.json"))
    save()
    try:
        m1 = load_m1(a.data, a.sym); c, s = COSTS.get(a.sym, (0.0, 0.0))
        iso = lambda x: time.strftime("%Y-%m-%d", time.gmtime(int(x)))
        save(state="pregatire", bars_m1=int(len(m1["t"])), first=iso(m1["t"][0]), last=iso(m1["t"][-1]))
        ctx = S.Ctx(m1, a.tf, cost_price=c, slip_price=s)
        save(state="cautare", segments={k: [iso(ctx.t[x[0]]), iso(ctx.t[min(x[1], ctx.n) - 1])] for k, x in ctx.seg.items()})
        def prog(n, el, best): save(tried=int(n), elapsed=int(el), best_train_t=float(best))
        res, ntr = S.search(ctx, a.minutes * 60, seed=a.seed, progress=prog)
        save(state="validare", tried=int(ntr), candidates=len(res))
        val = S.validate(ctx, res, ntr, k_final=25)
        stage = {k: sum(1 for v in val if v["stages"].get(k)) for k in ("val", "robust", "cost", "time", "ftmo", "lock")}
        json.dump({"job": job, "sym": a.sym, "tf": a.tf, "tried": ntr, "stages": stage, "finalists": val, "costs": {"round_trip": c, "slip": s, "provizorii": True}},
                  open(os.path.join(out, "result.json"), "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
        save(state="gata", stages=stage, elapsed=int(time.time() - st["started"]))
    except Exception as e:
        save(state="eroare", error=repr(e), tb=traceback.format_exc()[-1500:])


if __name__ == "__main__": main()
