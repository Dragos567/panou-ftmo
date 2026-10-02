# Istoric de ticks de la Dukascopy (date publice, fara cont) -> bare M1 (bid OHLC) cu spread mediu si numar de ticks.
# Timpul e UTC real. Fisier pe zi: duka/<SIM>/d_YYYYMMDD.bin, record "<qddddff" = t, o, h, l, c, ticks, spread. Descarcare reluabila (o zi gata = fisier existent).
import os, io, lzma, time, struct, threading, datetime as dt, urllib.request, urllib.error
import numpy as np
from concurrent.futures import ThreadPoolExecutor

REC = struct.Struct("<qddddff"); RS = REC.size
DTY = np.dtype([("t", "<i8"), ("o", "<f8"), ("h", "<f8"), ("l", "<f8"), ("c", "<f8"), ("v", "<f4"), ("sp", "<f4")])
TICK = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("av", ">f4"), ("bv", ">f4")])
# simbolul nostru -> (cod Dukascopy, divizor de pret)
MAP = {"EURUSD": ("EURUSD", 1e5), "NIKKEI": ("JPNIDXJPY", 1e3), "GOLD": ("XAUUSD", 1e3), "USDJPY": ("USDJPY", 1e3), "GBPUSD": ("GBPUSD", 1e5), "DAX": ("DEUIDXEUR", 1e3), "UK100": ("GBRIDXGBP", 1e3)}
URL = "https://datafeed.dukascopy.com/datafeed/%s/%04d/%02d/%02d/%02dh_ticks.bi5"


def decode_hour(raw, scale):
    """Un fisier .bi5 (LZMA) -> (minut in ora, bid, ask) pentru fiecare tick."""
    if not raw: return None
    a = np.frombuffer(lzma.decompress(raw), dtype=TICK)
    if len(a) == 0: return None
    return (a["ms"] // 60000).astype(np.int64), a["bid"].astype(np.float64) / scale, a["ask"].astype(np.float64) / scale


def hour_to_m1(raw, scale, hour_start):
    d = decode_hour(raw, scale)
    if d is None: return None
    mi, bid, ask = d
    st = np.concatenate(([0], np.flatnonzero(np.diff(mi)) + 1)); en = np.concatenate((st[1:], [len(mi)]))
    out = np.zeros(len(st), DTY)
    out["t"] = hour_start + mi[st] * 60
    out["o"] = bid[st]; out["c"] = bid[en - 1]
    out["h"] = np.maximum.reduceat(bid, st); out["l"] = np.minimum.reduceat(bid, st)
    out["v"] = (en - st); out["sp"] = np.add.reduceat(ask - bid, st) / (en - st)
    return out


def fetch(url, tries=5):
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=25) as r: return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404: return b""
            last = e; time.sleep(2 + 3 * k)
        except Exception as e: last = e; time.sleep(2 + 3 * k)
    raise RuntimeError("descarcare esuata: %s (%r)" % (url, last))


class Duka:
    def __init__(self, data_dir, log, syms, years=10.0, workers=6):
        self.dir = os.path.join(data_dir, "duka"); os.makedirs(self.dir, exist_ok=True)
        self.log, self.syms, self.years, self.workers = log, [s for s in syms if s in MAP], float(years), workers
        self.st = {s: {"state": "pornire", "days_done": 0, "days_total": 0, "newest": None, "oldest": None, "err": None, "updated": None, "sample_close": None} for s in self.syms}
        self.lock = threading.Lock()

    def sdir(self, s):
        d = os.path.join(self.dir, s); os.makedirs(d, exist_ok=True); return d

    def day_path(self, s, d): return os.path.join(self.sdir(s), "d_%s.bin" % d.strftime("%Y%m%d"))

    def _set(self, s, **kw):
        with self.lock: self.st[s].update(kw); self.st[s]["updated"] = int(time.time())

    def start(self):
        for s in self.syms: threading.Thread(target=self._run, args=(s,), daemon=True).start()

    def get_day(self, s, d):
        code, scale = MAP[s]; base = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())
        def hr(h):
            raw = fetch(URL % (code, d.year, d.month - 1, d.day, h))
            return hour_to_m1(raw, scale, base + h * 3600)
        with ThreadPoolExecutor(self.workers) as ex: parts = [p for p in ex.map(hr, range(24)) if p is not None]
        return np.concatenate(parts) if parts else np.zeros(0, DTY)

    def _run(self, s):
        while True:
            self._once(s)
            time.sleep(6 * 3600)          # dupa ce e complet, adauga zilele noi la fiecare 6 ore

    def _once(self, s):
        try:
            today = dt.datetime.now(dt.timezone.utc).date()
            first = today - dt.timedelta(days=int(self.years * 365.25))
            days = [today - dt.timedelta(days=i) for i in range(1, (today - first).days + 1)]   # de la ieri inapoi
            days = [d for d in days if d.weekday() != 5]                                          # sambata nu are ticks
            todo = [d for d in days if not os.path.exists(self.day_path(s, d))]
            self._set(s, state="descarc", days_total=len(days), days_done=len(days) - len(todo))
            done = len(days) - len(todo)
            for d in todo:
                a = self.get_day(s, d)
                p = self.day_path(s, d); tmp = p + ".tmp"
                with open(tmp, "wb") as f: f.write(a.tobytes())
                os.replace(tmp, p); done += 1
                kw = {"days_done": done, "oldest": d.isoformat()}
                if len(a): kw["sample_close"] = float(a["c"][-1]); kw["sample_day"] = d.isoformat()
                if self.st[s]["newest"] is None: kw["newest"] = d.isoformat()
                self._set(s, **kw)
            self._set(s, state="complet"); open(os.path.join(self.sdir(s), "state.txt"), "w").write(str(int(time.time())))
        except Exception as e:
            self.log("duka", s, repr(e)); self._set(s, state="eroare", err=repr(e)[:300])

    def status(self):
        with self.lock: return {s: dict(v) for s, v in self.st.items()}


def load(data_dir, s):
    """Tot istoricul M1 al unui simbol, sortat, ca dict de array-uri (+ 'sp' = spread mediu pe bara). None daca lipseste."""
    import glob
    fs = sorted(glob.glob(os.path.join(data_dir, "duka", s, "d_*.bin")))
    arrs = [np.fromfile(f, dtype=DTY) for f in fs if os.path.getsize(f) >= RS]
    if not arrs: return None
    a = np.concatenate(arrs); a = a[np.argsort(a["t"], kind="stable")]
    _, ix = np.unique(a["t"], return_index=True); a = a[ix]
    return {k: np.ascontiguousarray(a[k]) for k in ("t", "o", "h", "l", "c", "v", "sp")}


def complete(data_dir, s):
    p = os.path.join(data_dir, "duka", s, "state.txt")
    return os.path.exists(p)
