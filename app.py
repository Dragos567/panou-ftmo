#!/usr/bin/env python3
# Panou Trading FTMO - server independent (non-stop). Date de la FTMO prin MetaApi; fara TradingView, fara Mac.
import os, sys, json, time, threading, hmac, secrets, gzip, base64, subprocess, collections, io, tarfile, shutil, hashlib
import urllib.request, urllib.parse, urllib.error, datetime as dt
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PANOU_DATA") or ("/opt/panou/data" if os.path.isdir("/opt/panou/data") else os.path.join(HERE, "data")); os.makedirs(DATA, exist_ok=True)
ADMIN = os.environ["PANOU_KEY"]
TOKEN, AID = os.environ["METAAPI_TOKEN"], os.environ["METAAPI_ACCOUNT_ID"]
PROV = os.environ.get("MA_PROV", "https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai")
CLIENT = os.environ.get("MA_CLIENT", "https://mt-client-api-v1.{region}.agiliumtrade.ai")
MDATA = os.environ.get("MA_MDATA", "https://mt-market-data-client-api-v1.{region}.agiliumtrade.ai")
PORT = int(os.environ.get("PANOU_PORT", "8080"))
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

SYMS = {"NIKKEI": "JP225.cash", "GOLD": "XAUUSD", "EURUSD": "EURUSD", "USDJPY": "USDJPY", "DAX": "GER40.cash", "GBPUSD": "GBPUSD", "UK100": "UK100.cash"}
TFS = ["1m", "5m", "15m", "1h", "4h", "1d"]
WANT = {"1m": 3000, "5m": 4000, "15m": 4000, "1h": 4000, "4h": 3000, "1d": 2000}
TFSEC = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
NEWS_SYM = {"NIKKEI": "ICMARKETS:JP225", "GOLD": "OANDA:XAUUSD", "EURUSD": "OANDA:EURUSD", "USDJPY": "OANDA:USDJPY", "DAX": "OANDA:DE30EUR",
            "GBPUSD": "OANDA:GBPUSD", "UK100": "OANDA:UK100GBP", "MACRO": "TVC:DXY"}
NEWS_MULTI = {"MACRO": ["TVC:DXY", "TVC:US10Y", "TVC:GOLD", "TVC:US02Y"],
              "ALL": ["OANDA:XAUUSD", "OANDA:EURUSD", "OANDA:USDJPY", "OANDA:DE30EUR", "ICMARKETS:JP225", "OANDA:GBPUSD", "OANDA:UK100GBP", "TVC:DXY", "TVC:US10Y"]}
NEWS_TAG = {"OANDA:XAUUSD": "GOLD", "OANDA:EURUSD": "EURUSD", "OANDA:USDJPY": "USDJPY", "OANDA:DE30EUR": "DAX", "OANDA:GBPUSD": "GBPUSD", "OANDA:UK100GBP": "UK100",
            "ICMARKETS:JP225": "NIKKEI", "TVC:DXY": "USD", "TVC:US10Y": "USD", "TVC:US02Y": "USD", "TVC:GOLD": "GOLD"}

LOG = collections.deque(maxlen=300)
def log(*a):
    LOG.append(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a))
STAT = {"last_ok": None, "region": None, "started": time.time(), "calls": 0, "fails": 0}

# ---------------- MetaApi ----------------
SEM_LIVE = threading.Semaphore(2)   # cereri istorice 'live' (coada)
SEM_BULK = threading.Semaphore(2)   # descarcare istoric (max 5 concurente pe cont la MetaApi)
def http(url, tries=4, timeout=70, sem=None):
    last = None
    for i in range(tries):
        try:
            rq = urllib.request.Request(url, headers={"auth-token": TOKEN, "Accept": "application/json"})
            with (sem or _NOSEM), urllib.request.urlopen(rq, timeout=timeout) as r:
                b = r.read(); STAT["calls"] += 1
                if r.status == 202: last = "202"; time.sleep(2); continue
                STAT["last_ok"] = time.time()
                return json.loads(b) if b else None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf8", "ignore")[:200]; last = "HTTP %s %s" % (e.code, body); STAT["fails"] += 1
            if e.code == 404: break
            if e.code in (429, 504, 500, 503): time.sleep(6 + 3 * i)
            else: time.sleep(1 + i)
        except Exception as e:
            last = repr(e); STAT["fails"] += 1; time.sleep(1 + i)
    log("MetaApi eroare:", url.split("/users/current/")[-1][:80], last)
    raise RuntimeError(last)

import contextlib
_NOSEM = contextlib.nullcontext()
_base = {}
def bases():
    if not _base:
        a = http("%s/users/current/accounts/%s" % (PROV, AID), tries=6)
        r = a.get("region") or "london"; STAT["region"] = r
        _base["c"] = CLIENT.format(region=r); _base["m"] = MDATA.format(region=r)
    return _base["c"], _base["m"]

def iso(t): return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")
def parse_t(s): return int(dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())
def fetch_candles(sym, tf, before=None, limit=1000, bulk=False):
    _, mb = bases()
    u = "%s/users/current/accounts/%s/historical-market-data/symbols/%s/timeframes/%s/candles?limit=%d" % (mb, AID, urllib.parse.quote(SYMS[sym], safe=""), tf, limit)
    if before: u += "&startTime=" + urllib.parse.quote(iso(dt.datetime.utcfromtimestamp(before)))
    out = []
    for c in http(u, sem=SEM_BULK if bulk else SEM_LIVE) or []:
        try: out.append([parse_t(c["time"]), c["open"], c["high"], c["low"], c["close"], c.get("tickVolume", c.get("volume", 0)) or 0])
        except Exception: pass
    return out

# ---------------- stocare lumanari ----------------
STORE, SLOCK, LASTREF, INFL, FULL = {}, threading.Lock(), {}, {}, {}
def path(sym, tf): return os.path.join(DATA, "c_%s_%s.json" % (sym, tf))
def load_disk():
    for s in SYMS:
        for tf in TFS:
            try: STORE[(s, tf)] = json.load(open(path(s, tf)))
            except Exception: pass
def save_disk(sym, tf):
    try:
        p = path(sym, tf); tmp = p + ".tmp"
        with SLOCK: b = list(STORE.get((sym, tf), []))
        json.dump(b, open(tmp, "w"), separators=(",", ":")); os.replace(tmp, p)
    except Exception as e: log("disk:", e)
def merge(sym, tf, bars):
    if not bars: return
    with SLOCK:
        d = {b[0]: b for b in STORE.get((sym, tf), [])}
        for b in bars: d[b[0]] = b
        v = [d[k] for k in sorted(d)]
        STORE[(sym, tf)] = v[-(WANT[tf] + 500):]

def backfill(sym, tf):
    want = WANT[tf]
    with SLOCK: have = list(STORE.get((sym, tf), []))
    while True:
        with SLOCK: cur = STORE.get((sym, tf), [])
        if len(cur) >= want: break
        before = cur[0][0] if cur else None
        got = fetch_candles(sym, tf, before, 1000, bulk=True)
        if before: got = [b for b in got if b[0] < before]
        if not got: break
        merge(sym, tf, got)
    save_disk(sym, tf); FULL[(sym, tf)] = True

def backfill_all():
    order = [(s, tf) for tf in ["1h", "15m", "5m", "1d", "4h", "1m"] for s in SYMS]
    q = collections.deque(order)
    def worker():
        while True:
            try: s, tf = q.popleft()
            except IndexError: return
            for k in range(3):
                try: backfill(s, tf); break
                except Exception as e: log("backfill", s, tf, e); time.sleep(5)
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for t in ts: t.start()
    for t in ts: t.join()
    log("backfill gata")
    STAT["backfill_done"] = time.time()

def refresh(sym, tf, n=4):
    k = (sym, tf)
    with SLOCK:
        th = INFL.get(k)
        if th and th.is_alive(): return th
        def run():
            try: merge(sym, tf, fetch_candles(sym, tf, None, n)); LASTREF[k] = time.time()
            except Exception: pass
        th = threading.Thread(target=run, daemon=True); INFL[k] = th; th.start(); return th

def prev_24h(bars):
    t0 = bars[-1][0] - 86400; prev = bars[0][4]
    for b in bars:
        if b[0] <= t0: prev = b[4]
        else: break
    return prev

def ensure(sym, tf):
    with SLOCK: have = STORE.get((sym, tf))
    if not have:
        merge(sym, tf, fetch_candles(sym, tf, None, 1000))
        threading.Thread(target=lambda: (backfill(sym, tf)), daemon=True).start()

def get_candles(sym, tf, tail=False):
    ensure(sym, tf)
    if time.time() - LASTREF.get((sym, tf), 0) > (1.2 if tail else 4):
        th = refresh(sym, tf); th.join(3.0 if tail else 0.5)
    with SLOCK: bars = list(STORE[(sym, tf)])
    src = "FTMO · MetaApi · " + SYMS[sym]
    if tail:
        return {"sym": sym, "tf": tf, "tail": True, "bars": bars[-60:], "price": bars[-1][4], "time": bars[-1][0], "source": src}
    bars = bars[-WANT[tf]:]
    return {"sym": sym, "tf": tf, "bars": bars, "price": bars[-1][4], "prevClose": prev_24h(bars), "time": bars[-1][0], "source": src}

_qc = {"t": 0, "v": {}}; _ql = threading.Lock()
def _fallback(s):
    with SLOCK: h = STORE.get((s, "1m")) or STORE.get((s, "5m")) or STORE.get((s, "15m")) or STORE.get((s, "1h"))
    if h: return {"price": h[-1][4], "prev": prev_24h(h)}
def _qone(s):
    # un simbol, independent: un simbol lent nu mai blocheaza restul
    while True:
        try:
            cb, _ = bases()
            p = http("%s/users/current/accounts/%s/symbols/%s/current-price" % (cb, AID, urllib.parse.quote(SYMS[s], safe="")), tries=1, timeout=10)
            with SLOCK: h = STORE.get((s, "1h")) or STORE.get((s, "15m")) or STORE.get((s, "5m"))
            prev = prev_24h(h) if h else (_qc["v"].get(s) or {}).get("prev")
            with _ql: _qc["v"][s] = {"price": p["bid"], "prev": prev}; _qc["t"] = time.time()
        except Exception:
            if s not in _qc["v"]:
                f = _fallback(s)
                if f:
                    with _ql: _qc["v"][s] = f
            time.sleep(2)
        time.sleep(1.0)
def start_quotes():
    for s in SYMS: threading.Thread(target=_qone, args=(s,), daemon=True).start()
def get_quotes():
    # raspuns instant din memorie; completeaza lipsurile din STORE
    with _ql: out = dict(_qc["v"])
    for s in SYMS:
        if s not in out:
            f = _fallback(s)
            if f: out[s] = f
    return out

# ---------------- stiri + calendar + traducere ----------------
_cache, _cl = {}, threading.Lock()
def cached(key, ttl, fn):
    now = time.time()
    with _cl:
        v = _cache.get(key)
        if v and now - v[0] < ttl: return v[1]
    try: val = fn()
    except Exception as e:
        log("cache", key, e)
        if v: return v[1]
        raise
    with _cl: _cache[key] = (time.time(), val)
    return val
TR, _trl = {}, threading.Lock()
def get_json(url, timeout=10, hdr=None):
    h = {"User-Agent": UA, "Origin": "https://www.tradingview.com", "Referer": "https://www.tradingview.com/", "Accept": "application/json"}
    h.update(hdr or {})
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r: return json.loads(r.read())
def sentence_case(t):
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        t = t.lower(); t = t[:1].upper() + t[1:]
    return t.strip()
def translate(text):
    with _trl:
        if text in TR: return TR[text]
    try:
        j = get_json("https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=ro&dt=t&q=" + urllib.parse.quote(sentence_case(text)), 8)
        out = "".join(seg[0] for seg in j[0] if seg and seg[0]).strip() or text
    except Exception:
        return None
    with _trl: TR[text] = out
    return out
def par_translate(items):
    res = [None] * len(items)
    def w(i, it): res[i] = translate(it)
    ths = [threading.Thread(target=w, args=(i, it)) for i, it in enumerate(items)]
    for t in ths: t.start()
    for t in ths: t.join()
    return res
def real_news(key):
    syms = NEWS_MULTI.get(key) or [NEWS_SYM[key]]
    per = 14 if len(syms) == 1 else 10; raw = []
    def one(x):
        try:
            j = get_json("https://news-mediator.tradingview.com/public/view/v1/symbol?filter=lang%3Aen&filter=symbol%3A" + urllib.parse.quote(x, safe="") + "&client=web&streaming=false")
            for i in (j.get("items") or [])[:per]:
                raw.append({"t": (i.get("title") or "").strip(), "p": i.get("published"), "s": i.get("storyPath"), "src": (i.get("provider") or {}).get("name"), "g": NEWS_TAG.get(x, "")})
        except Exception as e: log("news", x, e)
    ths = [threading.Thread(target=one, args=(x,)) for x in syms]
    for t in ths: t.start()
    for t in ths: t.join()
    seen, items = {}, []
    for it in sorted(raw, key=lambda i: -(i.get("p") or 0)):
        k = it["t"].lower()
        if not k: continue
        if k in seen:
            if it.get("g") and it["g"] not in seen[k]["tags"]: seen[k]["tags"].append(it["g"])
            continue
        it["tags"] = [it["g"]] if it.get("g") else []; seen[k] = it; items.append(it)
    items = items[:14] if len(syms) == 1 else items[:36]
    res = par_translate([i["t"] for i in items])
    return [{"title": res[i] or it["t"], "translated": bool(res[i]), "title_en": it["t"], "tags": it["tags"],
             "link": "https://www.tradingview.com" + (it.get("s") or ""), "source": it.get("src") or "", "ts": it.get("p") or 0} for i, it in enumerate(items)]
def real_calendar():
    now = dt.datetime.utcnow()
    f = (now - dt.timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z"); t = (now + dt.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    j = get_json("https://economic-calendar.tradingview.com/events?from=%s&to=%s&countries=US,EU,JP,DE,GB" % (f, t), 12)
    items = [e for e in (j.get("result") or []) if (e.get("importance", -1) if e.get("importance") is not None else -1) >= 0]
    items.sort(key=lambda e: e.get("date") or ""); items = items[:80]
    res = par_translate([e["title"] for e in items]); out = []
    for i, e in enumerate(items):
        try: ts = dt.datetime.strptime(e["date"][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
        except Exception: continue
        out.append({"id": e.get("id"), "title": res[i] or e["title"], "title_en": e["title"], "country": e.get("country"), "currency": e.get("currency"),
                    "importance": e.get("importance", 0), "ts": ts, "actual": e.get("actual"), "forecast": e.get("forecast"), "previous": e.get("previous"), "unit": e.get("unit") or ""})
    return out


# ---------------- auto-actualizare din GitHub ----------------
BASE = os.environ.get("PANOU_BASE", "/opt/panou"); CFGP = os.path.join(DATA, "repo.json")
CODELOAD = os.environ.get("GH_CODELOAD", "https://codeload.github.com")
def sync():
    try: c = json.load(open(CFGP))
    except Exception: return False
    h = {"User-Agent": "panou-ftmo"}
    if c.get("token"): h["Authorization"] = "token " + c["token"]
    try: b = urllib.request.urlopen(urllib.request.Request("%s/%s/tar.gz/refs/heads/%s" % (CODELOAD, c["repo"], c.get("branch", "main")), headers=h), timeout=60).read()
    except Exception as e: log("sync", repr(e)); return False
    sha = hashlib.sha256(b).hexdigest(); shp = os.path.join(DATA, "repo.sha")
    try: old = open(shp).read().strip()
    except Exception: old = ""
    if sha == old: return False
    REPO = os.path.join(BASE, "repo"); tmp = REPO + ".new"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    with tarfile.open(fileobj=io.BytesIO(b), mode="r:gz") as t:
        for m in t.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1] or ".." in parts[1] or m.issym() or m.islnk(): continue
            m.name = parts[1]; t.extract(m, tmp)
    shutil.rmtree(REPO + ".old", ignore_errors=True)
    if os.path.exists(REPO): os.rename(REPO, REPO + ".old")
    os.rename(tmp, REPO); open(shp, "w").write(sha); log("cod nou din GitHub", sha[:8]); return True
def updater():
    while True:
        time.sleep(60)
        try:
            if sync(): time.sleep(1); os._exit(0)
        except Exception as e: log("updater", repr(e))


SITE_FIXES = [
 ("Panou Trading Personal","Panou FTMO"),
 ("Panou Trading","Panou FTMO"),
 ("Mac-ul e offline. Pornește Panou Trading și TradingView pe Mac; până atunci vezi ultimele date primite.","Sursa de date FTMO nu răspunde momentan; vezi ultimele date primite. Serverul reîncearcă singur."),
 ("Prețurile și lumânările vin din TradingView Desktop (feed OANDA, exact ce vezi în aplicație); știrile vin din fluxul de știri TradingView și sunt traduse automat. Prețul unui broker diferă puțin de OANDA: verifică-l în platforma FTMO înainte de a intra. Datele se citesc din fluxul TradingView fără să schimbe graficul tău din aplicație.",
  "Prețurile și lumânările vin direct de la contul tău FTMO (MetaApi, prețuri BID, ca în MT5); știrile vin din fluxul de știri TradingView și sunt traduse automat. Verifică totuși prețul în platforma FTMO înainte de a intra."),
 ("Prețurile vin din TradingView Desktop (feed OANDA); verifică","Prețurile vin direct de la contul tău FTMO (MetaApi, BID); verifică"),
 ("– Mac-ul nu trimite date acum.","– sursa nu trimite date acum."),
 ("Datele se citesc din fluxul TradingView fără să schimbe graficul tău din aplicație.",""),
 (" Scanarea nu atinge graficul tău din TradingView."," Scanarea rulează pe server, non-stop."),
 ("sunt în prețul OANDA din TradingView","sunt în prețul FTMO (MetaApi)"),
 ("Datele vin prin TradingView Desktop rulat pe calculatorul tău.","Prețurile vin direct de la contul tău FTMO prin MetaApi, serverul rulează non-stop."),
 ("Sursa datelor: calendarul și știrile TradingView","Sursa datelor: prețuri FTMO (MetaApi); calendarul și știrile TradingView"),
]
SITE_SRC = {"macro.html": "/", "index.html": "/ma", "fvg.html": "/fvg", "manifest.webmanifest": "/manifest.webmanifest", "sw.js": "/sw.js",
            "icon-180.png": "/icon-180.png", "icon-192.png": "/icon-192.png", "icon-512.png": "/icon-512.png", "favicon.png": "/favicon.png"}
def fetch_site(src, code):
    out = {}
    for name, p in SITE_SRC.items():
        rq = urllib.request.Request(src.rstrip("/") + p, headers={"Cookie": "pk=" + urllib.parse.quote(code), "User-Agent": UA})
        b = urllib.request.urlopen(rq, timeout=30).read()
        if name.endswith(".webmanifest"):
            b = b.decode("utf8").replace("Panou Trading Personal", "Altrix").replace("Panou Trading", "Altrix").replace("Panou FTMO", "Altrix").encode("utf8")
        if name.endswith(".html"):
            t = b.decode("utf8")
            if len(t) < 20000 or "<html" not in t[:2000].lower() and "<!doctype" not in t[:200].lower(): raise RuntimeError("pagina %s invalida (%d)" % (name, len(t)))
            for a, c in SITE_FIXES: t = t.replace(a, c)
            b = t.encode("utf8")
        out[name] = b
    for name, b in out.items():
        dst = os.path.join(HERE, name); open(dst + ".tmp", "wb").write(b); os.replace(dst + ".tmp", dst)
    return {k: len(v) for k, v in out.items()}

# ---------------- acces ----------------
def load_key():
    p = os.path.join(DATA, "site_key.txt")
    try:
        k = open(p).read().strip()
        if k: return k
    except Exception: pass
    k = secrets.token_urlsafe(12); open(p, "w").write(k + "\n"); return k
KEY = load_key()
LOGIN = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Altrix</title>
<body style="margin:0;min-height:100vh;display:grid;place-items:center;background:#090d12;color:#d6dde7;font:16px system-ui,sans-serif">
<form style="width:min(92vw,340px)"><h1 style="font-size:20px;margin:0 0 6px">Altrix</h1><p style="color:#7c8a9b;margin:0 0 16px">Acces privat. Introdu codul de acces.</p>
<input name=k type=password autofocus autocomplete=current-password placeholder="Cod de acces" style="width:100%;box-sizing:border-box;padding:12px;border-radius:8px;border:1px solid #1c2530;background:#0f151c;color:inherit;font-size:16px">
<button style="width:100%;margin-top:10px;padding:12px;border-radius:8px;border:0;background:#e0a93b;color:#15110a;font-weight:600;font-size:16px">Intră</button>@@ERR@@</form></body>"""
STATIC = ["manifest.webmanifest", "sw.js", "icon-180.png", "icon-192.png", "icon-512.png", "favicon.png"]
PG = {"/": "macro.html", "/macro": "macro.html", "/macro.html": "macro.html", "/ma": "index.html", "/index.html": "index.html", "/fvg": "fvg.html", "/fvg.html": "fvg.html"}
RUN = {"p": None}
def run_probe():
    if RUN["p"] and RUN["p"].poll() is None: return False
    env = dict(os.environ, PROBE_OUT=os.path.join(DATA, "report.json"))
    RUN["p"] = subprocess.Popen([sys.executable, os.path.join(HERE, "probe.py")], env=env); return True

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str): body = body.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def admin(self, qs): return hmac.compare_digest(qs.get("key", [""])[0].encode(), ADMIN.encode())
    def authed(self, u, qs):
        ck = self.headers.get("Cookie", ""); got = ""
        for part in ck.split(";"):
            n, _, v = part.strip().partition("=")
            if n == "pk": got = urllib.parse.unquote(v)
        if got and hmac.compare_digest(got.encode(), KEY.encode()): return True
        k = qs.get("k", [""])[0]
        if k:
            if hmac.compare_digest(k.encode(), KEY.encode()):
                self.send_response(302); self.send_header("Location", u.path)
                self.send_header("Set-Cookie", "pk=%s; Max-Age=31536000; Path=/; HttpOnly; Secure; SameSite=Lax" % urllib.parse.quote(KEY))
                self.send_header("Content-Length", "0"); self.end_headers(); return False
            time.sleep(1.0); self.send(401, LOGIN.replace("@@ERR@@", '<p style="color:#ef5b6b">Cod greșit.</p>'), "text/html; charset=utf-8"); return False
        if u.path.startswith("/api/"): self.send(401, {"error": "Acces interzis. Deschide pagina și introdu codul."}); return False
        self.send(401, LOGIN.replace("@@ERR@@", ""), "text/html; charset=utf-8"); return False
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); qs = urllib.parse.parse_qs(u.query)
        try:
            if u.path.startswith("/admin/"):
                if not self.admin(qs): return self.send(403, {"error": "cheie"})
                if u.path == "/admin/probe":
                    try: return self.send(200, open(os.path.join(DATA, "report.json")).read())
                    except Exception: return self.send(200, {"stage": "inca nu a pornit"})
                if u.path == "/admin/run": return self.send(200, {"started": run_probe()})
                if u.path == "/admin/fetchsite": return self.send(200, fetch_site(qs.get("src", [""])[0], qs.get("code", [""])[0]))
                if u.path == "/admin/status": return self.send(200, {"build": BUILD, "stat": STAT, "sha": (open(os.path.join(DATA, "repo.sha")).read()[:8] if os.path.exists(os.path.join(DATA, "repo.sha")) else None)})
                if u.path == "/admin/log": return self.send(200, {"log": list(LOG), "stat": STAT})
                if u.path == "/admin/info":
                    with SLOCK: n = {"%s/%s" % k: len(v) for k, v in STORE.items()}
                    return self.send(200, {"site_key": KEY, "stat": STAT, "bars": n, "full": ["%s/%s" % k for k in FULL]})
                return self.send(404, {})
            if not self.authed(u, qs): return
            if u.path in PG:
                with open(os.path.join(HERE, PG[u.path]), "rb") as f: return self.send(200, f.read(), "text/html; charset=utf-8")
            if u.path.lstrip("/") in STATIC:
                ext = os.path.splitext(u.path)[1]
                ct = {".png": "image/png", ".js": "application/javascript", ".webmanifest": "application/manifest+json"}.get(ext, "application/octet-stream")
                with open(os.path.join(HERE, u.path.lstrip("/")), "rb") as f: return self.send(200, f.read(), ct)
            if u.path == "/api/status":
                age = round(time.time() - STAT["last_ok"], 1) if STAT["last_ok"] else None
                return self.send(200, {"agent_age": age, "source": "FTMO/MetaApi"})
            if u.path == "/api/candles":
                sym, tf = qs.get("sym", [""])[0], qs.get("tf", [""])[0]
                if sym not in SYMS or tf not in TFS: return self.send(400, {"error": "parametri invalizi"})
                return self.send(200, get_candles(sym, tf, qs.get("tail", [""])[0] == "1"))
            if u.path == "/api/build": return self.send(200, {"build": _mtime()})
            if u.path == "/api/quotes": return self.send(200, get_quotes())
            if u.path == "/api/calendar": return self.send(200, cached(("cal",), 300, real_calendar))
            if u.path == "/api/news":
                key = qs.get("sym", [""])[0]
                if key not in NEWS_SYM and key not in NEWS_MULTI: return self.send(400, {"error": "parametri invalizi"})
                return self.send(200, cached(("n", key), 120, lambda: real_news(key)))
            return self.send(404, {"error": "negasit"})
        except Exception as e:
            log("GET", u.path, repr(e)); return self.send(502, {"error": "Eroare la citirea datelor FTMO: %s" % e})
    def do_POST(self):
        u = urllib.parse.urlparse(self.path); qs = urllib.parse.parse_qs(u.query)
        if not self.admin(qs): return self.send(403, {"error": "cheie"})
        self.send(404, {})

def _mtime():
    m = 0
    for f in list(PG.values()) + STATIC + ["app.py"]:
        try: m = max(m, os.path.getmtime(os.path.join(HERE, f)))
        except Exception: pass
    return str(int(m))

def migrate():
    OLD = 'window.fetch=function(){return of.apply(this,arguments).then(r=>{try{'
    NEW = "let qFirst=1;function qGet(){try{return localStorage.getItem('pq')}catch(e){return null}}\nwindow.fetch=function(){const a0=arguments;const u0=String(a0[0]);\n if(u0.indexOf('/api/quotes')>=0){const c=qFirst?qGet():null;qFirst=0;\n  const sv=r=>{if(r.ok)r.clone().text().then(t=>{try{localStorage.setItem('pq',t)}catch(e){}}).catch(()=>{});return r};\n  if(c){of.apply(this,a0).then(sv).catch(()=>{});return Promise.resolve(new Response(c,{status:200,headers:{'Content-Type':'application/json'}}))}\n  return of.apply(this,a0).then(sv)}\n return of.apply(this,arguments).then(r=>{try{"
    for f in ("index.html", "fvg.html", "macro.html", "manifest.webmanifest"):
        p = os.path.join(HERE, f)
        try:
            t = open(p, encoding="utf-8").read(); o = t
            if "let qFirst=1;" not in t and OLD in t: t = t.replace(OLD, NEW + "return of.apply(this,arguments).then(r=>{try{", 1) if False else t.replace(OLD, NEW, 1)
            t = t.replace("Panou Trading Personal", "Altrix").replace("Panou Trading", "Altrix").replace("Panou FTMO", "Altrix")
            t = t.replace('<div class="brand">Panou FTMO<small>', '<div class="brand">Altrix<small>')
            t = t.replace('<div class="brand">Trading<small>', '<div class="brand">Altrix<small>')
            if t != o: open(p, "w", encoding="utf-8").write(t)
        except Exception as e: log("migrate", f, e)
migrate()
BUILD = _mtime()

if __name__ == "__main__":
    load_disk()
    def boot():
        for i in range(30):
            try: bases(); break
            except Exception: time.sleep(5)
        backfill_all()
    threading.Thread(target=boot, daemon=True).start()
    start_quotes()
    if os.path.exists(CFGP): threading.Thread(target=updater, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H); srv.daemon_threads = True
    print("Panou FTMO pornit pe", PORT, flush=True)
    srv.serve_forever()
