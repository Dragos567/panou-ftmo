"""dbdl.py - descarcare Databento (schema trades) pe luni, agregat direct in bare de 1 secunda. Cost plafonat.

Fiecare luna se cere intai la 'get_cost' (gratuit); daca suma cheltuita + costul lunii depaseste plafonul, se opreste.
Rezultat: DATA/databento/<SIMBOL>/<AAAA-LL>.npz cu: t (epoch s), o,h,l,c, v (volum), b (volum cumparator agresor), s (volum vanzator agresor), n (nr tranzactii), iid (id instrument; se schimba la rulare contract).
"""
import os, json, time, threading, base64, urllib.request, urllib.parse, urllib.error, datetime as dt
import numpy as np

HOST = "https://hist.databento.com/v0/"
REC = np.dtype([("length", "u1"), ("rtype", "u1"), ("pub", "<u2"), ("iid", "<u4"), ("ts", "<u8"), ("price", "<i8"), ("size", "<u4"),
                ("action", "S1"), ("side", "S1"), ("flags", "u1"), ("depth", "u1"), ("ts_recv", "<u8"), ("delta", "<i4"), ("seq", "<u4")])
assert REC.itemsize == 48
UNDEF = 9223372036854775807
DEFAULT = {"6E": "6E.v.0", "6B": "6B.v.0", "NKD": "NKD.v.0", "GC": "GC.v.0", "YM": "YM.v.0", "ES": "ES.v.0", "NQ": "NQ.v.0"}
DATASET = "GLBX.MDP3"
_LOCK = threading.Lock()
_T = {"th": None}


def _hdr(key):
    return {"User-Agent": "altrix", "Authorization": "Basic " + base64.b64encode((key + ":").encode()).decode()}


def parse_dbn(buf):
    """Bytes DBN (necomprimat) -> array structurat de tranzactii."""
    if buf[:3] != b"DBN":
        raise ValueError("raspuns neasteptat: " + buf[:200].decode("utf8", "ignore"))
    mlen = int.from_bytes(buf[4:8], "little")
    body = memoryview(buf)[8 + mlen:]
    if len(body) == 0:
        return np.zeros(0, REC)
    if len(body) % 48 == 0:
        a = np.frombuffer(body, dtype=REC)
        if (a["rtype"] == 0).all() and (a["length"] == 12).all():
            return a
    out, i, n = [], 0, len(body)                       # varianta lenta, pentru inregistrari amestecate
    raw = bytes(body)
    while i + 2 <= n:
        ln = raw[i] * 4
        if ln <= 0 or i + ln > n: break
        if raw[i + 1] == 0 and ln == 48: out.append(raw[i:i + 48])
        i += ln
    return np.frombuffer(b"".join(out), dtype=REC) if out else np.zeros(0, REC)


def to_seconds(a):
    a = a[(a["price"] != UNDEF) & (a["action"] == b"T")]
    if len(a) == 0: return None
    sec = (a["ts"] // 1_000_000_000).astype(np.int64)
    o = np.argsort(sec, kind="stable"); a = a[o]; sec = sec[o]
    idx = np.flatnonzero(np.r_[True, sec[1:] != sec[:-1]])
    px = a["price"].astype(np.float64) / 1e9; sz = a["size"].astype(np.float64)
    end = np.r_[idx[1:] - 1, len(a) - 1]
    buy = np.where(a["side"] == b"B", sz, 0.0); sell = np.where(a["side"] == b"A", sz, 0.0)
    return {"t": sec[idx].astype(np.uint32), "o": px[idx].astype(np.float32), "h": np.maximum.reduceat(px, idx).astype(np.float32),
            "l": np.minimum.reduceat(px, idx).astype(np.float32), "c": px[end].astype(np.float32),
            "v": np.add.reduceat(sz, idx).astype(np.uint32), "b": np.add.reduceat(buy, idx).astype(np.uint32),
            "s": np.add.reduceat(sell, idx).astype(np.uint32), "n": np.diff(np.r_[idx, len(a)]).astype(np.uint16),
            "iid": a["iid"][idx].astype(np.uint32)}


def months(start, end):
    d = dt.date.fromisoformat(start); e = dt.date.fromisoformat(end); out = []
    while d < e:
        nx = (d.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        out.append((d.isoformat(), min(nx, e).isoformat())); d = nx
    return out


def _get(url, key, data=None, timeout=900):
    rq = urllib.request.Request(url, data=data, headers=_hdr(key))
    with urllib.request.urlopen(rq, timeout=timeout) as r: return r.read()


def cost(key, sym, a, b):
    q = urllib.parse.urlencode({"dataset": DATASET, "symbols": DEFAULT[sym], "stype_in": "continuous", "schema": "trades", "start": a, "end": b})
    return float(json.loads(_get(HOST + "metadata.get_cost?" + q, key, timeout=60)))


def fetch_month(key, sym, a, b):
    form = urllib.parse.urlencode({"dataset": DATASET, "symbols": DEFAULT[sym], "stype_in": "continuous", "stype_out": "instrument_id",
                                   "schema": "trades", "start": a, "end": b, "encoding": "dbn", "compression": "none"}).encode()
    return _get(HOST + "timeseries.get_range", key, data=form)


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
            for a, b in months(j["start"], j["end"]):
                f = os.path.join(root, sym, a[:7] + ".npz")
                if os.path.exists(f): continue
                c = cost(key, sym, a, b)
                if j["spent"] + c > j["cap"]:
                    j.update(active=False, err="plafon atins: %.2f + %.2f > %.2f (oprit la %s %s)" % (j["spent"], c, j["cap"], sym, a)); _save(root, j); return
                buf = fetch_month(key, sym, a, b)
                d = to_seconds(parse_dbn(buf))
                del buf
                if d is None: d = {k: np.zeros(0) for k in ("t", "o", "h", "l", "c", "v", "b", "s", "n", "iid")}
                np.savez_compressed(f + ".tmp.npz", **d); os.replace(f + ".tmp.npz", f)
                j["spent"] = round(j["spent"] + c, 4); j["done"].append("%s %s (%d s-bare, %.2f$)" % (sym, a[:7], len(d["t"]), c)); j["err"] = ""
                _save(root, j)
        j.update(active=False, finished=int(time.time())); _save(root, j)
    except Exception as e:
        j.update(active=False, err=repr(e)[:300]); _save(root, j)


def start(root, key, syms, a, b, cap):
    with _LOCK:
        if _T["th"] and _T["th"].is_alive(): return False
        os.makedirs(root, exist_ok=True)
        old = status(root)
        j = {"active": True, "syms": syms, "start": a, "end": b, "cap": float(cap), "spent": old.get("spent", 0.0) if old.get("syms") == syms and old.get("start") == a and old.get("end") == b else 0.0,
             "done": [], "err": "", "started": int(time.time())}
        _save(root, j)
        _T["th"] = threading.Thread(target=run, args=(root, key, j), daemon=True); _T["th"].start(); return True


def resume(root, key):
    """La pornirea serverului: reia o descarcare intrerupta (fisierele deja facute se sar)."""
    j = status(root)
    if j.get("active") and key: start(root, key, j["syms"], j["start"], j["end"], j["cap"])
