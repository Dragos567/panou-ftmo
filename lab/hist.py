# Magazia de istoric adanc (M1) pentru motorul Altrix. Doar biblioteca standard: merge pe orice server, fara pachete in plus.
# Format: fisiere binare cu inregistrari fixe de 44 de octeti: t(int64, secunde) o h l c (float64) v(float32), sortate crescator.
import os, struct, time, threading, glob, json, datetime as dt

REC = struct.Struct("<qddddf")
RS = REC.size  # 44


def _pack(bars):
    return b"".join(REC.pack(int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5] or 0)) for b in bars)


class Hist:
    def __init__(self, data_dir, fetch, log, syms, years=3.0, tf="1m"):
        self.dir = os.path.join(data_dir, "hist"); os.makedirs(self.dir, exist_ok=True)
        self.fetch, self.log, self.syms, self.years, self.tf = fetch, log, list(syms), float(years), tf
        self.st = {s: {"state": "pornire", "oldest": None, "newest": None, "bars": 0, "pages": 0, "done_back": False, "err": None, "updated": None} for s in self.syms}
        self.lock = threading.Lock()

    # ---------- disc ----------
    def sdir(self, sym):
        d = os.path.join(self.dir, "%s_%s" % (sym, self.tf)); os.makedirs(d, exist_ok=True); return d

    def parts(self, sym):
        out = []
        for p in glob.glob(os.path.join(self.sdir(sym), "p_*.bin")):
            try:
                n = os.path.getsize(p) // RS
                if n: out.append((int(os.path.basename(p)[2:-4]), p, n))
            except Exception: pass
        return sorted(out)

    def scan(self, sym):
        ps = self.parts(sym)
        if not ps: return None, None, 0
        oldest, newest, tot = None, None, 0
        for first, p, n in ps:
            with open(p, "rb") as f:
                f.seek((n - 1) * RS); last = REC.unpack(f.read(RS))[0]
            oldest = first if oldest is None else min(oldest, first)
            newest = last if newest is None else max(newest, last); tot += n
        return oldest, newest, tot

    def write_part(self, sym, bars):
        if not bars: return
        bars = sorted(bars, key=lambda b: b[0])
        p = os.path.join(self.sdir(sym), "p_%d.bin" % bars[0][0]); tmp = p + ".tmp"
        with open(tmp, "wb") as f: f.write(_pack(bars))
        os.replace(tmp, p)

    def compact(self, sym):
        ps = self.parts(sym)
        if len(ps) < 2: return
        out_t, buf, last = None, [], None
        tmp = os.path.join(self.sdir(sym), "compact.tmp"); first_t = None
        with open(tmp, "wb") as w:
            for first, p, n in ps:
                data = open(p, "rb").read()
                for i in range(0, len(data), RS):
                    t = struct.unpack_from("<q", data, i)[0]
                    if last is not None and t <= last: continue
                    if first_t is None: first_t = t
                    w.write(data[i:i + RS]); last = t
        final = os.path.join(self.sdir(sym), "p_%d.bin" % first_t)
        os.replace(tmp, final)
        for first, p, n in ps:
            if os.path.abspath(p) != os.path.abspath(final):
                try: os.remove(p)
                except Exception: pass

    def read_bytes(self, sym):
        """Tot istoricul, sortat si fara dubluri, ca bytes (pentru numpy.frombuffer)."""
        self.compact(sym)
        ps = self.parts(sym)
        return open(ps[0][1], "rb").read() if ps else b""

    # ---------- descarcare ----------
    def start(self):
        for s in self.syms:
            threading.Thread(target=self._run, args=(s,), daemon=True).start()

    def _set(self, sym, **kw):
        with self.lock: self.st[sym].update(kw); self.st[sym]["updated"] = int(time.time())

    def _run(self, sym):
        o, n, tot = self.scan(sym)
        self._set(sym, oldest=o, newest=n, bars=tot, state="pornit")
        target = time.time() - self.years * 365.25 * 86400
        while True:
            try:
                done = self._back(sym, target)
                self._fwd(sym)
                self.compact(sym)
                o, n, tot = self.scan(sym)
                self._set(sym, oldest=o, newest=n, bars=tot, done_back=done, err=None, state="complet, se actualizeaza" if done else "descarc")
            except Exception as e:
                self.log("hist", sym, repr(e)); self._set(sym, err=repr(e)[:160])
                time.sleep(20); continue
            time.sleep(300 if done else 1)

    def _back(self, sym, target):
        o, n, tot = self.scan(sym)
        before = o  # None la prima rulare
        buf, pages = [], 0
        def flush():
            nonlocal buf
            if buf: self.write_part(sym, buf); buf = []
        while True:
            if o is not None and o <= target: flush(); return True
            got = self.fetch(sym, self.tf, before, 1000, True)
            if before is not None: got = [b for b in got if b[0] < before]
            if not got: flush(); return True      # sfarsitul istoricului disponibil
            buf.extend(got); pages += 1
            new_o = min(b[0] for b in got)
            if before is not None and new_o >= before: flush(); return True
            before = o = new_o if o is None else min(o, new_o)
            self._set(sym, state="descarc", oldest=o, pages=self.st[sym]["pages"] + 1)
            if pages % 40 == 0: flush(); self._set(sym, bars=self.scan(sym)[2])

    def _fwd(self, sym):
        o, n, tot = self.scan(sym)
        if n is None: return
        before, pages, got_all = None, [], []
        for _ in range(60):
            got = self.fetch(sym, self.tf, before, 1000, True)
            if before is not None: got = [b for b in got if b[0] < before]
            if not got: break
            got_all.extend(b for b in got if b[0] > n)
            if min(b[0] for b in got) <= n: break
            before = min(b[0] for b in got)
        if got_all: self.write_part(sym, got_all)

    def status(self):
        with self.lock: return {s: dict(v) for s, v in self.st.items()} | {"years": self.years}

    # ---------- raport de calitate ----------
    def report(self, sym):
        data = self.read_bytes(sym); n = len(data) // RS
        if not n: return {"sym": sym, "bars": 0}
        recs = list(REC.iter_unpack(data))
        t = [r[0] for r in recs]; rep = {"sym": sym, "tf": self.tf, "bars": n}
        iso = lambda x: dt.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M")
        rep["first"], rep["last"] = iso(t[0]), iso(t[-1]); rep["days"] = round((t[-1] - t[0]) / 86400, 1)
        bad = zero = 0; rets = []
        for r in recs:
            _, o, h, l, c, v = r
            if h < max(o, c) - 1e-12 or l > min(o, c) + 1e-12 or l > h: bad += 1
            if h == l: zero += 1
        rep["ohlc_invalide"] = bad; rep["lumanari_fara_miscare"] = zero
        gaps = {"<=5min": 0, "5-60min": 0, "1-6h": 0, ">6h in saptamana": 0, "weekend": 0}; big = []; wk_open = {}; wk_close = {}
        for i in range(1, n):
            g = t[i] - t[i - 1]
            if g <= 60: continue
            a, b = dt.datetime.utcfromtimestamp(t[i - 1]), dt.datetime.utcfromtimestamp(t[i])
            span_weekend = g > 6 * 3600 and a.weekday() in (4, 5) and b.weekday() in (6, 0)
            if g <= 300: gaps["<=5min"] += 1
            elif g <= 3600: gaps["5-60min"] += 1
            elif g <= 6 * 3600: gaps["1-6h"] += 1
            elif span_weekend:
                gaps["weekend"] += 1; wk_open[b.hour] = wk_open.get(b.hour, 0) + 1; wk_close[a.hour] = wk_close.get(a.hour, 0) + 1
            else: gaps[">6h in saptamana"] += 1
            if g > 3600 and not span_weekend: big.append((g, iso(t[i - 1]), iso(t[i])))
        rep["goluri"] = gaps; big.sort(reverse=True)
        rep["cele_mai_mari_goluri_in_saptamana"] = [{"durata_ore": round(g / 3600, 1), "de_la": a, "pana_la": b} for g, a, b in big[:8]]
        rep["ora_UTC_deschidere_dupa_weekend"] = dict(sorted(wk_open.items())); rep["ora_UTC_inchidere_inainte_de_weekend"] = dict(sorted(wk_close.items()))
        absr = []
        for i in range(1, n):
            p0 = recs[i - 1][4]
            if p0: absr.append((abs(recs[i][4] / p0 - 1), t[i]))
        if absr:
            vals = sorted(x[0] for x in absr if x[0] > 0); med = vals[len(vals) // 2] if vals else 0
            absr.sort(reverse=True)
            rep["salturi_mari_1m"] = [{"la": iso(tt), "pct": round(r * 100, 3)} for r, tt in absr[:5]]
            rep["mediana_miscare_1m_pct"] = round(med * 100, 5)
            rep["miscari_peste_30x_mediana"] = sum(1 for r, _ in absr if med and r > 30 * med)
        return rep
