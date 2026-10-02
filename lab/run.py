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
    ap.add_argument("--job", default=None); ap.add_argument("--finish", default=None); a = ap.parse_args()
    from . import search as S
    job = a.finish or a.job or time.strftime("%Y%m%d_%H%M%S") + "_%s_%s" % (a.sym, a.tf)
    out = os.path.join(a.data, "lab", job); os.makedirs(out, exist_ok=True)
    st = json.load(open(os.path.join(out, "status.json"))) if a.finish else {"job": job, "sym": a.sym, "tf": a.tf, "state": "incarc date", "started": int(time.time()), "costs_provizorii": True, "minutes": a.minutes}
    def save(**kw):
        st.update(kw); st["updated"] = int(time.time())
        json.dump(st, open(os.path.join(out, "status.json.tmp"), "w")); os.replace(os.path.join(out, "status.json.tmp"), os.path.join(out, "status.json"))
    save()
    try:
        m1 = load_m1(a.data, a.sym); c, s = COSTS.get(a.sym, (0.0, 0.0))
        iso = lambda x: time.strftime("%Y-%m-%d", time.gmtime(int(x)))
        save(state="pregatire", bars_m1=int(len(m1["t"])), first=iso(m1["t"][0]), last=iso(m1["t"][-1]))
        ctx = S.Ctx(m1, a.tf, cost_price=c, slip_price=s)
        if a.finish:
            ck = json.load(open(os.path.join(out, "ckpt.json")))
            res = [(c["fit"], c["n"], c["avg"], c["g"]) for c in ck]; ntr = int(st.get("tried") or len(res))
            save(state="validare", candidates=len(res), validation={"stage": "pregatire", "i": 0, "K": 25}, recovered=True)
        else: save(state="cautare", segments={k: [iso(ctx.t[x[0]]), iso(ctx.t[min(x[1], ctx.n) - 1])] for k, x in ctx.seg.items()})
        live = {"t": 0.0}
        def prog(n, el, best, seen=None):
            kw = dict(tried=int(n), elapsed=int(el), best_train_t=float(best))
            if seen is not None and time.time() - live["t"] > 8:     # previzualizare live: cei mai buni 8 (unici pe bloc+parametri); validarea e DOAR afisata, nu intra in selectie
                live["t"] = time.time(); top = []; kp = set()
                for fit, n_tr, avg, g in sorted(seen.values(), key=lambda x: -x[0])[:200]:
                    k = (g["blk"], tuple(sorted(g["p"].items())))
                    if k in kp or fit <= -9: continue
                    kp.add(k); rv, _, _ = S.trades(ctx, g, "val")
                    top.append({"blk": str(g["blk"]), "htf": str(g["htf"]), "n_tr": int(n_tr), "avg_tr": float(avg), "t_tr": float(fit), "n_val": int(len(rv)),
                                "avg_val": float(np.mean(rv)) if len(rv) else 0.0, "t_val": float(S.tstat(rv))})
                    if len(top) >= 8: break
                kw["live_top"] = top
            if seen is not None and time.time() - live.get("ck", 0) > 45:      # punct de control: daca serverul reporneste, validarea se face din el
                live["ck"] = time.time()
                best = sorted(seen.values(), key=lambda x: -x[0])[:300]
                tmp = os.path.join(out, "ckpt.tmp"); json.dump([{"fit": float(f), "n": int(nn), "avg": float(av), "g": S._norm(g)} for f, nn, av, g in best], open(tmp, "w")); os.replace(tmp, os.path.join(out, "ckpt.json"))
            save(**kw)
        if not a.finish: res, ntr = S.search(ctx, a.minutes * 60, seed=a.seed, progress=prog)
        save(state="validare", tried=int(ntr), candidates=len(res), validation={"stage": "pregatire", "i": 0, "K": 25})
        cp = os.path.join(a.data, "lab", "cum_%s_%s.json" % (a.sym, a.tf))
        try: cum = json.load(open(cp))
        except Exception: cum = {"k": 0, "lock": 0, "runs": 0}
        def vcb(stage, i, K, blk): save(validation={"stage": stage, "i": int(i), "K": int(K), "blk": blk, "thr": float(S.zcrit(0.05 / max(K + cum["k"], 1)))})
        val = S.validate(ctx, res, ntr, k_final=25, prior_k=cum["k"], prior_lock=cum["lock"], cb=vcb)
        cum["k"] += len([v for v in val]); cum["lock"] += len([v for v in val if v["stages"].get("ftmo")]); cum["runs"] += 1
        json.dump(cum, open(cp, "w"))
        good = [v for v in val if v["stages"].get("lock") and v["ftmo"]["bootstrap_pass_both"] >= 0.4]
        stage = {k: sum(1 for v in val if v["stages"].get(k)) for k in ("val", "robust", "cost", "time", "ftmo", "lock")}
        json.dump({"job": job, "sym": a.sym, "tf": a.tf, "tried": ntr, "stages": stage, "finalists": val, "relevant": len(good), "costs": {"round_trip": c, "slip": s, "provizorii": True}},
                  open(os.path.join(out, "result.json"), "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
        save(state="gata", relevant=len(good), finalists=len(val), stages=stage, elapsed=int(time.time() - st["started"]))
    except Exception as e:
        save(state="eroare", error=repr(e), tb=traceback.format_exc()[-1500:])


if __name__ == "__main__": main()
