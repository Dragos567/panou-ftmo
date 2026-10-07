"""dbbook.py - Databento schema TBBO (tranzactie + cel mai bun bid/ask in momentul tranzactiei), agregat la 1 secunda. Cost plafonat, estimare gratuita inainte.
Rezultat: DATA/databento_book/<SIM>/<AAAA-LL>.npz: t, c (ultimul pret), bid, ask (ultimele), bsz, asz (marimi la cel mai bun nivel, ultimele), v, b, s (volum total/cumparator/vanzator agresor),
imb (suma size*(bsz-asz)/(bsz+asz) pe tranzactiile din secunda), iid."""
import os, json, time, threading, urllib.parse
import numpy as np
from . import dbdl

REC2 = np.dtype([("length", "u1"), ("rtype", "u1"), ("pub", "<u2"), ("iid", "<u4"), ("ts", "<u8"), ("price", "<i8"), ("size", "<u4"),
                 ("action", "S1"), ("side", "S1"), ("flags", "u1"), ("depth", "u1"), ("ts_recv", "<u8"), ("delta", "<i4"), ("seq", "<u4"),
                 ("bid_px", "<i8"), ("ask_px", "<i8"), ("bid_sz", "<u4"), ("ask_sz", "<u4"), ("bid_ct", "<u4"), ("ask_ct", "<u4")])
assert REC2.itemsize == 80
_T = {"th": None}


def parse(buf):
    if buf[:3] != b"DBN": raise ValueError("raspuns neasteptat: " + buf[:200].decode("utf8", "ignore"))
    mlen = int.from_bytes(buf[4:8], "little"); body = memoryview(buf)[8 + mlen:]
    if len(body) == 0: return np.zeros(0, REC2)
    if len(body) % 80 != 0: raise ValueError("lungime body %d nu e multiplu de 80" % len(body))
    return np.frombuffer(body, dtype=REC2)


def to_seconds(a):
    UND = dbdl.UNDEF
    a = a[(a["price"] != UND) & (a["bid_px"] != UND) & (a["ask_px"] != UND) & (a["bid_sz"] + a["ask_sz"] > 0)]
    if len(a) == 0: return None
    sec = (a["ts"] // 1_000_000_000).astype(np.int64); o = np.argsort(sec, kind="stable"); a = a[o]; sec = sec[o]
    idx = np.flatnonzero(np.r_[True, sec[1:] != sec[:-1]]); end = np.r_[idx[1:] - 1, len(a) - 1]
    sz = a["size"].astype(np.float64); bs = a["bid_sz"].astype(np.float64); asz = a["ask_sz"].astype(np.float64)
    buy = np.where(a["side"] == b"B", sz, 0.0); sell = np.where(a["side"] == b"A", sz, 0.0); imb = sz * (bs - asz) / (bs + asz)
    f = lambda x: x.astype(np.float32)
    return {"t": sec[idx].astype(np.uint32), "c": f(a["price"][end] / 1e9), "bid": f(a["bid_px"][end] / 1e9), "ask": f(a["ask_px"][end] / 1e9), "bsz": f(bs[end]), "asz": f(asz[end]),
            "v": f(np.add.reduceat(sz, idx)), "b": f(np.add.reduceat(buy, idx)), "s": f(np.add.reduceat(sell, idx)), "imb": f(np.add.reduceat(imb, idx)), "iid": a["iid"][end].astype(np.uint32)}


def fetch(key, sym, a, b):
    form = urllib.parse.urlencode({"dataset": dbdl.DATASET, "symbols": dbdl.DEFAULT[sym], "stype_in": "continuous", "stype_out": "instrument_id",
                                   "schema": "tbbo", "start": a, "end": b, "encoding": "dbn", "compression": "none"}).encode()
    return dbdl._get(dbdl.HOST + "timeseries.get_range", key, data=form)


def _jp(root): return os.path.join(root, "job.json")


def status(root):
    try: return json.load(open(_jp(root)))
    except Exception: return {"active": False}


def _save(root, j):
    j["updated"] = int(time.time()); tmp = _jp(root) + ".tmp"; json.dump(j, open(tmp, "w")); os.replace(tmp, _jp(root))


def run(root, key, j):
    try:
        for sym in j["syms"]:
            os.makedirs(os.path.join(root, sym), exist_ok=True)
            for a, b in dbdl.months(j["start"], j["end"]):
                f = os.path.join(root, sym, a[:7] + ".npz")
                if os.path.exists(f): continue
                c = None
                for att in range(5):
                    try: c = dbdl.cost(key, sym, a, b, "tbbo"); break
                    except Exception as e: j["err"] = "cost: " + repr(e)[:100]; _save(root, j); time.sleep(10 * (att + 1))
                if c is None: raise RuntimeError("cost esuat")
                if j["spent"] + c > j["cap"]:
                    j.update(active=False, err="plafon atins: %.2f + %.2f > %.2f" % (j["spent"], c, j["cap"])); _save(root, j); return
                buf = None
                for att in range(5):
                    try: buf = fetch(key, sym, a, b); break
                    except Exception as e: j["err"] = "fetch: " + repr(e)[:100]; _save(root, j); time.sleep(10 * (att + 1))
                if buf is None: raise RuntimeError("fetch esuat %s %s" % (sym, a))
                d = to_seconds(parse(buf)); del buf
                if d is None: d = {k: np.zeros(0) for k in ("t", "c", "bid", "ask", "bsz", "asz", "v", "b", "s", "imb", "iid")}
                np.savez_compressed(f + ".tmp.npz", **d); os.replace(f + ".tmp.npz", f)
                j["spent"] = round(j["spent"] + c, 4); j["done"].append("%s %s (%d s-bare, %.2f$)" % (sym, a[:7], len(d["t"]), c)); j["err"] = ""; _save(root, j)
        j.update(active=False, finished=int(time.time())); _save(root, j)
    except Exception as e:
        j.update(active=False, err=repr(e)[:300]); _save(root, j)


def start(root, key, syms, a, b, cap):
    if _T["th"] and _T["th"].is_alive(): return False
    os.makedirs(root, exist_ok=True)
    j = {"active": True, "syms": syms, "start": a, "end": b, "cap": float(cap), "spent": 0.0, "done": [], "err": "", "started": int(time.time())}; _save(root, j)
    _T["th"] = threading.Thread(target=run, args=(root, key, j), daemon=True); _T["th"].start(); return True
