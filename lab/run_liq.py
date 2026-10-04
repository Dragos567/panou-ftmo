# Alergator pentru strategia LIQ pe DAX (M1). Aceeasi schema de status/result ca lab.run, ca sa o afiseze aceeasi pagina Laborator.
#   python3 -m lab.run_liq --data DIR --minutes 20 [--seed 1] [--job J] [--finish J] [--k 100]
import os, sys, json, time, argparse, traceback, datetime as dtm
import numpy as np
SYM = "DAX"
SRCN = {1: "DAILY", 2: "WEEKLY", 4: "ASIA", 3: "DAILY+WEEKLY", 5: "DAILY+ASIA", 6: "WEEKLY+ASIA", 7: "DAILY+WEEKLY+ASIA"}


def desc(g):
    m = ("A fractal nf=%d pre=%d" % (g["nf"], g["pre"])) if g["model"] == 0 else ("B leg=%d" % g["leg_n"])
    d = ["fara displ."] if g["disp_mode"] == 0 else (["displ.1 x%.2f/%d" % (g["disp_mult"], g["disp_n"])] if g["disp_mode"] == 1 else ["displ.%dc x%.2f r<=%d" % (g["disp_k"], g["disp_mult"], g["disp_retr"])])
    s = "MSS %s | %s | SL %s+%.2f | FVG min %.1f pre %d max %d sel %d | fereastra %dmin ord%d | %02d:%02d-%02d:%02d | TP %.1fR BE %s | surse %d dir %d tr/zi %d" % (
        m, d[0], "wick" if g["sl_mode"] == 0 else "swing", g["buf"], g["fvg_min"], g["fvg_pre"], g["max_fvg"], g["sel"], g["win"], g["ord_mode"],
        g["start"] // 60, g["start"] % 60, g["end"] // 60, g["end"] % 60, g["tp_r"], ("%.1fR" % g["be_r"]) if g["be_r"] > 0 else "off", g["src_mask"], g["dir"], g["max_tr"])
    return s


def journal(ctx, g, path):
    from . import liqx as X
    tr, _ = ctx.run(g)
    def ro(ts):
        off = 7200 if True else 0
        from . import liq as Q
        o = int(Q.ro_offset(np.array([int(ts)], np.int64))[0]); return dtm.datetime.utcfromtimestamp(int(ts) + o).strftime("%Y-%m-%d %H:%M")
    kinds = {1: "TP", 2: "SL", 3: "BE", 4: "EOD"}
    with open(path, "w") as f:
        f.write("Symbol,Date,TimeRomaniaEntry,TimeRomaniaSweep,TimeRomaniaMSS,Direction,LevelsSwept,FirstSource,LastSource,MSSDelayMin,FVGsAtEntry,FVGSize,Entry50,SLwick,SLTicks,TP,BE,Result,RGross,RNet\n")
        for r in tr:
            d = int(r[2]); E = r[12]; sl = r[13]; rk = r[5]
            tp = E + d * g["tp_r"] * rk; be = E + d * g["be_r"] * rk
            net = r[4] - r[20] / rk
            f.write("GER40.cash,%s,%s,%s,%s,%s,%d,%s,%s,%.1f,%d,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%s,%.2f,%.3f\n" % (ro(r[0])[:10], ro(r[0]), ro(r[14]), ro(r[15]), "BUY" if d > 0 else "SELL", int(r[6]),
                    SRCN.get(int(r[7]), "?"), SRCN.get(int(r[8]), "?"), r[11], int(r[9]), r[16], E, sl, round(rk / ctx.tick), tp, be, kinds.get(int(r[17]), "?"), r[4], net))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--minutes", type=float, default=10); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--job", default=None); ap.add_argument("--finish", default=None); ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--sym", default=SYM); ap.add_argument("--tf", default="1m"); a = ap.parse_args()
    TF = a.tf if a.tf in ("1m", "5m", "15m") else "1m"; TFM = {"1m": 1, "5m": 5, "15m": 15}[TF]
    from . import liqx as X
    from .run import load_m1
    job = a.finish or a.job or time.strftime("%Y%m%d_%H%M%S") + "_DAX_liq"
    out = os.path.join(a.data, "lab", job); os.makedirs(out, exist_ok=True)
    st = json.load(open(os.path.join(out, "status.json"))) if a.finish else {"job": job, "sym": SYM, "tf": TF, "strategy": "liq", "state": "incarc date", "started": int(time.time()), "costs_provizorii": True, "minutes": a.minutes, "source": "metaapi (FTMO)"}
    def save(**kw):
        st.update(kw); st["updated"] = int(time.time())
        json.dump(st, open(os.path.join(out, "status.json.tmp"), "w")); os.replace(os.path.join(out, "status.json.tmp"), os.path.join(out, "status.json"))
    save()
    try:
        m1 = load_m1(a.data, SYM)
        if TFM > 1:
            from .data import resample
            m1 = resample(m1, TFM * 60)   # bare de TFM minute; nivelurile, MSS, FVG si intrarea se evalueaza pe aceste bare
        iso = lambda x: time.strftime("%Y-%m-%d", time.gmtime(int(x)))
        spread = None; prov = True; ns = 0
        try:
            spd = json.load(open(os.path.join(a.data, "spread.json"))).get(SYM, {})
            means = [v["mean"] for v in spd.values() if v["n"] >= 30]
            if means:
                med = float(np.median(means)); spread = [spd[str(h)]["mean"] if str(h) in spd and spd[str(h)]["n"] >= 30 else med for h in range(24)]
                prov = False; ns = int(sum(v["n"] for v in spd.values()))
        except Exception: pass
        save(state="pregatire", bars_m1=int(len(m1["t"])), first=iso(m1["t"][0]), last=iso(m1["t"][-1]), costs_provizorii=prov, spread_samples=ns)
        ctx = X.LCtx(m1, tick=0.01, cost_pts=1.5, spread_by_hour=spread, spread_mult=1.5, slip=0.5, comm=0.0)
        cuts = [iso(m1["t"][0]), iso(ctx.cut[0]), iso(ctx.cut[1]), iso(m1["t"][-1])]
        # baseline: strategia exact ca in specificatie (ambele modele MSS, cateva latimi de fractal / leg), fara nicio optimizare
        base = []
        for lab, ov in (("A fractal nf=1", dict(model=0, nf=1)), ("A fractal nf=2", dict(model=0, nf=2)), ("A fractal nf=3", dict(model=0, nf=3)), ("A fractal nf=5", dict(model=0, nf=5)),
                        ("B leg=5", dict(model=1, leg_n=5)), ("B leg=8", dict(model=1, leg_n=8)), ("B leg=12", dict(model=1, leg_n=12)), ("B leg=20", dict(model=1, leg_n=20)),
                        ("A nf=2 + displacement 1 lum. x1.5", dict(model=0, nf=2, disp_mode=1, disp_mult=1.5)), ("B leg=8 + displacement 1 lum. x1.5", dict(model=1, leg_n=8, disp_mode=1, disp_mult=1.5))):
            g = {**X.BASE, **ov}; d = X.describe(ctx, g); d["name"] = lab; base.append(d)
        json.dump({"updated": int(time.time()), "segments": cuts, "bars": int(len(m1["t"])), "levels_sweeps": int(len(ctx.rows[0])), "cost_provizoriu": prov, "variants": base},
                  open(os.path.join(a.data, "lab", "liq_baseline.json"), "w"))
        if a.finish:
            ck = json.load(open(os.path.join(out, "ckpt.json"))); res = [(c["fit"], c["n"], c["avg"], c["g"]) for c in ck]; ntr = int(st.get("tried") or len(res))
            save(state="validare", candidates=len(res), validation={"stage": "pregatire", "i": 0, "K": a.k}, recovered=True)
        else:
            save(state="cautare", segments={"train": [cuts[0], cuts[1]], "val": [cuts[1], cuts[2]], "lock": [cuts[2], cuts[3]]})
            live = {"t": 0.0}
            def prog(n, el, best, seen=None):
                kw = dict(tried=int(n), elapsed=int(el), best_train_t=float(best))
                if seen is not None and time.time() - live["t"] > 8:
                    live["t"] = time.time(); top = []; kp = set()
                    for fit, n_tr, avg, g in sorted(seen.values(), key=lambda x: -x[0])[:200]:
                        k = X.key_of(g)
                        if k in kp or fit <= -9: continue
                        kp.add(k); rv, _, _ = X.trades(ctx, g, "val")
                        top.append({"blk": desc(g), "htf": "-", "n_tr": int(n_tr), "avg_tr": float(avg), "t_tr": float(fit), "n_val": int(len(rv)),
                                    "avg_val": float(np.mean(rv)) if len(rv) else 0.0, "t_val": float(X.tstat(rv))})
                        if len(top) >= 8: break
                    kw["live_top"] = top
                if seen is not None and time.time() - live.get("ck", 0) > 45:
                    live["ck"] = time.time(); best = sorted(seen.values(), key=lambda x: -x[0])[:300]
                    tmp = os.path.join(out, "ckpt.tmp"); json.dump([{"fit": float(f), "n": int(nn), "avg": float(av), "g": X._norm(g)} for f, nn, av, g in best], open(tmp, "w")); os.replace(tmp, os.path.join(out, "ckpt.json"))
                save(**kw)
            res, ntr = X.search(ctx, a.minutes * 60, seed=a.seed, progress=prog)
            save(state="validare", tried=int(ntr), candidates=len(res), validation={"stage": "pregatire", "i": 0, "K": a.k})
        cp = os.path.join(a.data, "lab", "cum_DAX_liq.json")
        try: cum = json.load(open(cp))
        except Exception: cum = {"k": 0, "lock": 0, "runs": 0, "keys": [], "lkeys": []}
        cum.setdefault("keys", []); cum.setdefault("lkeys", [])
        def vcb(stage, i, K, blk): save(validation={"stage": stage, "i": int(i), "K": int(K), "blk": blk, "thr": float(X.zcrit(0.05 / max(K + cum["k"], 1)))})
        val = X.validate(ctx, res, k_final=a.k, prior_k=cum["k"], prior_lock=cum["lock"], cb=vcb)
        # corectia cumulata numara doar variante UNICE (nu le recontorizeaza la fiecare rulare)
        ks = set(map(tuple, map(list, cum["keys"]))) if False else set(cum["keys"])
        for v in val:
            kk = json.dumps(X.key_of(v["genome"])); 
            if kk not in ks: ks.add(kk)
            if v["stages"].get("ftmo") and kk not in cum["lkeys"]: cum["lkeys"].append(kk)
        cum["keys"] = sorted(ks); cum["k"] = len(ks); cum["lock"] = len(cum["lkeys"]); cum["runs"] += 1
        json.dump(cum, open(cp, "w"))
        good = [v for v in val if v.get("relevant")]
        stage = {k: sum(1 for v in val if v["stages"].get(k)) for k in ("val", "robust", "cost", "time", "ftmo", "lock")}
        for v in val: v["desc"] = desc(v["genome"]); v["genome"]["blk"] = "liq"
        # jurnal CSV (cap. 25) pentru cea mai buna varianta: relevanta, altfel cea mai buna din validare
        pick = good[0] if good else (sorted([v for v in val if v.get("val")], key=lambda v: -v["val"]["t"]) or [None])[0]
        if pick:
            g0 = {k: v for k, v in pick["genome"].items() if k != "blk"}
            try: journal(ctx, g0, os.path.join(out, "journal.csv"))
            except Exception as e: save(journal_error=repr(e))
        json.dump({"job": job, "sym": SYM, "tf": TF, "strategy": "liq", "tried": ntr, "stages": stage, "finalists": val, "relevant": len(good), "costs": {"provizorii": prov}},
                  open(os.path.join(out, "result.json"), "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
        save(state="gata", relevant=len(good), finalists=len(val), stages=stage, elapsed=int(time.time() - st["started"]))
    except Exception as e:
        save(state="eroare", error=repr(e), tb=traceback.format_exc()[-1500:])


if __name__ == "__main__": main()
