"""
Tri-Session Liquidity Engine  (EURUSD / GBPUSD / XAUUSD / US100)
================================================================
Asia  : construieste range-ul de lichiditate (High/Low/EQ/volum/spread); optional mean-reversion 1:1.5 doar in compresie.
London: sweep al range-ului Asia (penetrare + inchidere M5 inapoi in range) -> MSS pe M1 cu displacement -> intrare pe FVG / Order Block.
NY    : referinte London H/L + HTF OB (H1); dupa deschiderea NY (09:30 America/New_York) displacement + delta proxy -> retest FVG.
Risc  : manage_ftmo_risk() - cap zilnic soft 3.5%, kill 4.5%, total 7% / 9%, buget de risc deschis, news +-10 min, sizing dinamic per sesiune.

Toate orele interne sunt UTC. NY open e calculat din America/New_York (13:30 UTC vara, 14:30 UTC iarna).
Ziua FTMO (reset al pierderii zilnice) = 00:00 Europe/Prague, NU 00:00 UTC.

ATENTIE: este un schelet testabil, nu o strategie validata. Fiecare regula trebuie trecuta prin lab (train/val/lockbox, costuri reale) inainte de cont real.
"""
from __future__ import annotations
import json, math, urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, date
from typing import Optional, List, Dict, Tuple
from zoneinfo import ZoneInfo
import numpy as np

UTC = timezone.utc
NYTZ = ZoneInfo("America/New_York")
CET = ZoneInfo("Europe/Prague")


# ============================================================ CONFIG ============================================================
@dataclass
class Cfg:
    symbol: str = "EURUSD"
    ccys: Tuple[str, ...] = ("EUR", "USD")          # monedele relevante pentru filtrul de stiri (XAUUSD/US100 -> ("USD",))
    pip: float = 0.0001                             # marimea unui pip in unitati de pret (XAUUSD: 0.1, US100: 1.0)
    tick_size: float = 0.00001
    tick_value: float = 1.0                         # valoarea unui tick pe 1 lot, in moneda contului
    lot_step: float = 0.01; lot_min: float = 0.01; lot_max: float = 50.0
    initial_balance: float = 100_000.0

    # sesiuni (minute UTC de la 00:00)
    asia: Tuple[int, int] = (0, 480)
    london: Tuple[int, int] = (480, 750)            # 08:00-12:30
    ny_end: int = 21 * 60
    ny_last_entry: int = 19 * 60 + 30
    friday_flat: int = 20 * 60 + 45                 # vineri: inchide tot (evita gap de weekend)

    # --- Asia ---
    asia_mr_enabled: bool = False                   # optional; implicit oprit (vezi concluziile din lab)
    asia_spread_max_pips: float = 0.8               # pragul X de spread
    asia_min_range_pips: float = 8.0
    asia_compress_frac: float = 0.45                # range-ul asiatic < frac x ATR(D1) = zona de compresie
    asia_mr_from_min: int = 180                     # mean-reversion doar dupa 03:00 (range deja format)
    asia_mr_edge: float = 0.85                      # intrare cand pretul inchide M5 peste 85% din range
    asia_mr_rr: float = 1.5

    # --- London ---
    sweep_pen_pips: float = 1.5                     # penetrare minima X pips
    mss_body_atr: float = 1.0                       # corpul candelei de displacement >= k x ATR(M1,14)
    sl_buffer_atr: float = 0.15                     # buffer SL peste extremul sweep-ului (x ATR M5)
    min_sl_pips: float = 4.0; max_sl_pips: float = 25.0
    min_rr: float = 1.5
    ldn_setup_ttl_min: int = 60                     # ordinul limit expira
    ldn_sweep_window_min: int = 90                  # dupa sweep, MSS trebuie sa apara in acest timp

    # --- New York ---
    ny_disp_window_min: int = 15
    ny_disp_atr: float = 1.2                        # miscarea in primele 15 min >= k x ATR(M5)
    ny_delta_min: float = 0.15                      # |delta proxy| minim (tick volume semnat, normalizat)
    ny_htf_veto: bool = True                        # nu intra in HTF OB opus
    ny_setup_ttl_min: int = 75

    # --- Risc FTMO ---
    risk_pct: Dict[str, float] = field(default_factory=lambda: {"asia": 0.30, "london": 0.50, "ny": 0.75})
    max_trades: Dict[str, int] = field(default_factory=lambda: {"asia": 1, "london": 2, "ny": 2})
    daily_soft: float = 0.035                       # opreste sesiuni NOI
    daily_kill: float = 0.045                       # inchide tot + blocheaza ziua
    total_soft: float = 0.07                        # fata de balanta initiala (FTMO: pierdere maxima statica 10%)
    total_kill: float = 0.09
    news_before_min: int = 10; news_after_min: int = 10
    loss_streak_scale: float = 0.5                  # dupa 2 pierderi consecutive in zi
    profit_lock_pct: float = 0.02                   # peste +2% in zi, riscul se injumatateste


# ============================================================ UTILITARE ============================================================
def bars_to_short(b: Dict[str, np.ndarray], side: int) -> Dict[str, np.ndarray]:
    """Logica de setup e scrisa o singura data pentru SHORT (side=-1). Pentru LONG oglindim preturile (p -> -p)."""
    if side < 0: return b
    return {"t": b["t"], "o": -b["o"], "h": -b["l"], "l": -b["h"], "c": -b["c"], "v": b["v"]}

def px(p: float, side: int) -> float:
    return p if side < 0 else -p

def atr(b: Dict[str, np.ndarray], n: int = 14) -> float:
    h, l, c = b["h"], b["l"], b["c"]
    if len(c) < n + 1: return float("nan")
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(tr[-n:].mean())

def fractal_lows(l: np.ndarray) -> List[int]:
    # stanga <= (egalitatile sunt frecvente: deschiderea = inchiderea precedenta), dreapta strict <
    return [i for i in range(2, len(l) - 2) if l[i] <= l[i - 1] and l[i] <= l[i - 2] and l[i] < l[i + 1] and l[i] < l[i + 2]]

def utc_day_start(now: datetime) -> datetime:
    return datetime(now.year, now.month, now.day, tzinfo=UTC)

def ny_open_utc(now: datetime) -> datetime:
    d = now.astimezone(NYTZ).date()
    return datetime(d.year, d.month, d.day, 9, 30, tzinfo=NYTZ).astimezone(UTC)

def session_of(cfg: Cfg, now: datetime) -> Optional[str]:
    m = now.hour * 60 + now.minute
    if cfg.asia[0] <= m < cfg.asia[1]: return "asia"
    if cfg.london[0] <= m < cfg.london[1]: return "london"
    nyo = ny_open_utc(now); mo = nyo.hour * 60 + nyo.minute
    if mo <= m < cfg.ny_end: return "ny"
    return None                                     # 12:30-NY open si dupa 21:00: fara intrari


def latest_fvg(b: Dict[str, np.ndarray], start: int) -> Optional[Tuple[float, float, int]]:
    """FVG bearish (spatiul de short): low[a] > high[c] pentru 3 lumanari consecutive. Intoarce (lo, hi, idx_c) pentru cel mai recent, cautat de la `start`."""
    h, l = b["h"], b["l"]
    for c_ in range(len(h) - 1, max(start, 2) - 1, -1):
        a_ = c_ - 2
        if l[a_] > h[c_]: return float(h[c_]), float(l[a_]), c_
    return None

def last_opposite_candle(b: Dict[str, np.ndarray], j: int, lookback: int = 10) -> Optional[Tuple[float, float]]:
    """Order block bearish: ultima lumanare ascendenta inainte de displacement-ul j. Intoarce (lo, hi)."""
    for i in range(j - 1, max(j - lookback, 0) - 1, -1):
        if b["c"][i] > b["o"][i]: return float(b["l"][i]), float(b["h"][i])
    return None

def detect_mss(b: Dict[str, np.ndarray], first: int, k_body: float, atr1: float) -> Optional[Tuple[int, float]]:
    """Market Structure Shift (spatiul de short): o lumanara de displacement (corp >= k x ATR) care inchide sub ultimul swing low confirmat.
    `first` = indexul primei bare de dupa sweep. Intoarce (idx_j, nivel_swing)."""
    o, l, c = b["o"], b["l"], b["c"]
    sw = fractal_lows(l)
    for j in range(max(first, 5), len(c)):
        ok = [i for i in sw if i + 2 < j]
        if not ok: continue
        ref = float(l[ok[-1]])
        if c[j] < ref and c[j] < o[j] and abs(c[j] - o[j]) >= k_body * atr1: return j, ref
    return None

def htf_order_blocks(h1: Dict[str, np.ndarray], k_body: float = 1.5, max_age: int = 72) -> List[Tuple[float, float, int]]:
    """Order block-uri H1 nemitigate: (lo, hi, kind) kind=+1 cerere (ultima lumanara descendenta inainte de impuls ascendent), -1 oferta."""
    out = []; a = atr(h1, 14)
    if not np.isfinite(a): return out
    n = len(h1["c"])
    for j in range(max(n - max_age, 2), n):
        body = h1["c"][j] - h1["o"][j]
        if abs(body) < k_body * a: continue
        kind = 1 if body > 0 else -1
        for i in range(j - 1, max(j - 6, 0) - 1, -1):
            if (h1["c"][i] < h1["o"][i]) == (kind > 0):
                lo, hi = float(h1["l"][i]), float(h1["h"][i])
                after = h1["c"][j + 1:]
                mitigated = bool(len(after) and ((kind > 0 and after.min() < lo) or (kind < 0 and after.max() > hi)))
                if not mitigated: out.append((lo, hi, kind))
                break
    return out


# ============================================================ STRUCTURI ============================================================
@dataclass
class AsiaRange:
    high: float; low: float; eq: float; volume: float; spread_pips: float; width_pips: float; compressed: bool; valid: bool = True

@dataclass
class Setup:
    session: str; side: int; entry: float; sl: float; tps: List[Tuple[float, float]]    # [(pret, fractiune din pozitie)]
    expire: datetime; risk_pct: float; note: str = ""

@dataclass
class DayState:
    day: date
    asia: Optional[AsiaRange] = None
    london_hl: Optional[Tuple[float, float]] = None
    ldn_phase: str = "idle"; ldn_side: int = 0; ldn_extreme: float = 0.0; ldn_t: Optional[datetime] = None
    ny_done: bool = False
    trades: Dict[str, int] = field(default_factory=lambda: {"asia": 0, "london": 0, "ny": 0})
    pending: List[Setup] = field(default_factory=list)


# ============================================================ ASIA ============================================================
def process_asia(eng: "Engine", now: datetime) -> Optional[Setup]:
    """00:00-08:00 UTC. Construieste si salveaza range-ul; optional o singura tranzactie mean-reversion in compresie."""
    cfg, st, feed = eng.cfg, eng.st, eng.feed
    m5 = feed.bars(300, 200)
    day0 = utc_day_start(now).timestamp()
    sel = (m5["t"] >= day0) & (m5["t"] < day0 + cfg.asia[1] * 60)
    if sel.sum() < 6: return None
    hi, lo = float(m5["h"][sel].max()), float(m5["l"][sel].min())
    d1 = feed.bars(86400, 30); a_d1 = atr(d1, 14)
    sp = feed.spread() / cfg.pip
    w = (hi - lo) / cfg.pip
    st.asia = AsiaRange(hi, lo, (hi + lo) / 2, float(m5["v"][sel].sum()), sp, w,
                        compressed=bool(np.isfinite(a_d1) and (hi - lo) < cfg.asia_compress_frac * a_d1), valid=w >= cfg.asia_min_range_pips)
    mins = now.hour * 60 + now.minute
    if not (cfg.asia_mr_enabled and mins >= cfg.asia_mr_from_min and mins < cfg.asia[1] - 30): return None
    if not st.asia.valid or not st.asia.compressed or sp > cfg.asia_spread_max_pips or st.trades["asia"] >= cfg.max_trades["asia"]: return None
    last = m5["c"][-1]; pos = (last - lo) / (hi - lo)
    for side, hit in ((-1, pos >= cfg.asia_mr_edge), (1, pos <= 1 - cfg.asia_mr_edge)):
        if not hit: continue
        ext = hi if side < 0 else lo; buf = 0.15 * atr(m5, 14)
        sl = ext + buf if side < 0 else ext - buf
        risk = abs(sl - last)
        if risk < cfg.min_sl_pips * cfg.pip or risk > cfg.max_sl_pips * cfg.pip: continue
        tp = last + side * cfg.asia_mr_rr * risk
        return Setup("asia", side, last, sl, [(tp, 1.0)], now + timedelta(minutes=30), cfg.risk_pct["asia"], "asia_mr")
    return None


# ============================================================ LONDON ============================================================
def process_london(eng: "Engine", now: datetime) -> Optional[Setup]:
    """08:00-12:30 UTC. Judas swing: sweep al range-ului Asia -> MSS M1 -> limit pe FVG/OB. Un singur sweep activ la un moment dat."""
    cfg, st, feed = eng.cfg, eng.st, eng.feed
    A = st.asia
    if A is None or not A.valid or st.trades["london"] >= cfg.max_trades["london"]: return None
    m5 = feed.bars(300, 60); m1 = feed.bars(60, 240)
    if len(m5["c"]) < 20 or len(m1["c"]) < 40: return None
    pen = cfg.sweep_pen_pips * cfg.pip
    if st.ldn_phase == "idle":
        b = {k: v[-1] for k, v in m5.items()}                                   # ultima bara M5 INCHISA
        if b["h"] >= A.high + pen and b["c"] < A.high: st.ldn_side, st.ldn_extreme = -1, float(m5["h"][-6:].max())     # extremul sweep-ului = maximul ultimelor 30 min
        elif b["l"] <= A.low - pen and b["c"] > A.low: st.ldn_side, st.ldn_extreme = 1, float(m5["l"][-6:].min())
        else: return None
        st.ldn_phase, st.ldn_t = "swept", datetime.fromtimestamp(float(b["t"]), UTC)
        return None
    if st.ldn_phase != "swept": return None
    side = st.ldn_side
    ext = st.ldn_extreme
    if side < 0: ext = max(ext, float(m5["h"][-1]))                              # extremul sweep-ului continua sa creasca pana la MSS
    else: ext = min(ext, float(m5["l"][-1]))
    st.ldn_extreme = ext
    if now - st.ldn_t > timedelta(minutes=cfg.ldn_sweep_window_min): st.ldn_phase = "idle"; return None
    s = bars_to_short(m1, side)
    first = int(np.searchsorted(s["t"], st.ldn_t.timestamp() - 600))
    atr1 = atr(s, 14)
    mss = detect_mss(s, first, cfg.mss_body_atr, atr1)
    if mss is None: return None
    j, _ref = mss
    zone = None; tag = ""
    fvg = latest_fvg(s, max(j - 1, 2))
    if fvg and fvg[2] >= j: zone, tag = (fvg[0], fvg[1]), "fvg"
    else:
        ob = last_opposite_candle(s, j)
        if ob: zone, tag = ob, "ob"
    if zone is None: return None
    entry_s = zone[0] + 0.5 * (zone[1] - zone[0])
    sl_s = (ext if side < 0 else -ext) + cfg.sl_buffer_atr * atr(m5, 14) + feed.spread()   # in spatiul de short, SL e peste extrem
    risk = sl_s - entry_s
    if risk < cfg.min_sl_pips * cfg.pip or risk > cfg.max_sl_pips * cfg.pip: st.ldn_phase = "idle"; return None
    far_s = (A.low if side < 0 else -A.high); eq_s = (A.eq if side < 0 else -A.eq)
    if (entry_s - far_s) / risk < cfg.min_rr: st.ldn_phase = "idle"; return None
    st.ldn_phase = "armed"
    entry, sl = px(entry_s, side), px(sl_s, side)
    return Setup("london", side, entry, sl, [(px(eq_s, side), 0.5), (px(far_s, side), 0.5)], now + timedelta(minutes=cfg.ldn_setup_ttl_min), cfg.risk_pct["london"], "sweep_mss_" + tag)


# ============================================================ NEW YORK ============================================================
def process_new_york(eng: "Engine", now: datetime) -> Optional[Setup]:
    """De la 09:30 America/New_York. Referinte: London H/L + HTF OB. Displacement + delta proxy in primele 15 min, apoi retest FVG in directia displacement-ului."""
    cfg, st, feed = eng.cfg, eng.st, eng.feed
    if st.ny_done or st.trades["ny"] >= cfg.max_trades["ny"]: return None
    if now.hour * 60 + now.minute > cfg.ny_last_entry: return None
    t0 = ny_open_utc(now); t1 = t0 + timedelta(minutes=cfg.ny_disp_window_min)
    if now < t1: return None
    m1 = feed.bars(60, 400); m5 = feed.bars(300, 120)
    w = (m1["t"] >= t0.timestamp()) & (m1["t"] < t1.timestamp())
    if w.sum() < cfg.ny_disp_window_min - 3: st.ny_done = True; return None
    # London H/L (referinta)
    d0 = utc_day_start(now).timestamp(); lm = (m5["t"] >= d0 + cfg.london[0] * 60) & (m5["t"] < d0 + cfg.london[1] * 60)
    if lm.any(): st.london_hl = (float(m5["h"][lm].max()), float(m5["l"][lm].min()))
    move = float(m1["c"][w][-1] - m1["o"][w][0])
    a5 = atr(m5, 14); st.ny_done = True                                           # o singura evaluare pe zi (primul semnal)
    if not np.isfinite(a5) or abs(move) < cfg.ny_disp_atr * a5: return None
    side = 1 if move > 0 else -1
    sg = np.sign(m1["c"][w] - m1["o"][w]); vv = m1["v"][w]
    delta = float((sg * vv).sum() / max(vv.sum(), 1e-9))                           # delta PROXY (tick volume semnat) - nu e order flow real
    if np.sign(delta) != side or abs(delta) < cfg.ny_delta_min: return None
    s = bars_to_short(m1, side)
    first = int(np.flatnonzero(w)[0])
    fvg = latest_fvg({k: v[:first + int(w.sum())] for k, v in s.items()}, first)
    if not fvg: return None
    zone = (fvg[0], fvg[1]); entry_s = zone[0] + 0.5 * (zone[1] - zone[0])
    origin_s = float(s["h"][w].max())                                              # originea displacement-ului (spatiul de short)
    sl_s = origin_s + 0.1 * a5 + feed.spread(); risk = sl_s - entry_s
    if risk < cfg.min_sl_pips * cfg.pip or risk > cfg.max_sl_pips * cfg.pip: return None
    if cfg.ny_htf_veto:
        for lo, hi, kind in htf_order_blocks(feed.bars(3600, 200)):
            if kind == -side and lo <= px(entry_s, side) <= hi: return None         # intrare in HTF OB opus (cerere pentru short / oferta pentru long)
    cands = []
    if st.london_hl:
        lvl = st.london_hl[1] if side < 0 else st.london_hl[0]
        if (side < 0 and lvl < px(entry_s, side)) or (side > 0 and lvl > px(entry_s, side)): cands.append(lvl)
    tp = px(entry_s, side)
    tp = tp + side * 2.0 * risk                                                    # fallback 2R
    for lvl in cands:
        if abs(lvl - px(entry_s, side)) / risk >= cfg.min_rr: tp = lvl
    return Setup("ny", side, px(entry_s, side), px(sl_s, side), [(tp, 1.0)], now + timedelta(minutes=cfg.ny_setup_ttl_min), cfg.risk_pct["ny"], "ny_fvg_retest d=%.2f" % delta)


# ============================================================ RISC FTMO + STIRI ============================================================
@dataclass
class RiskState:
    allow_new: bool; kill: bool; scale: float; day_loss: float; total_loss: float; reason: str = ""

class NewsFilter:
    """events: [(datetime_utc, 'USD', 'high', 'CPI')]. Sursa recomandata: calendar economic (ex. FMP /economic_calendar) reimprospatat la 6h."""
    def __init__(self, events: List[Tuple[datetime, str, str, str]]): self.events = events
    def blocked(self, cfg: Cfg, now: datetime) -> Optional[str]:
        for t, ccy, imp, name in self.events:
            if imp.lower() != "high" or ccy not in cfg.ccys: continue
            if t - timedelta(minutes=cfg.news_before_min) <= now <= t + timedelta(minutes=cfg.news_after_min): return "%s %s %s" % (ccy, name, t.strftime("%H:%M"))
        return None

def load_calendar_fmp(api_key: str, d0: date, d1: date) -> List[Tuple[datetime, str, str, str]]:
    """Adaptor de verificat pe contul tau FMP (numele campurilor poate diferi)."""
    url = "https://financialmodelingprep.com/api/v3/economic_calendar?from=%s&to=%s&apikey=%s" % (d0, d1, api_key)
    out = []
    for e in json.load(urllib.request.urlopen(url, timeout=20)):
        try: out.append((datetime.strptime(e["date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC), str(e.get("currency", "")).upper(), str(e.get("impact", "")), str(e.get("event", ""))))
        except Exception: continue
    return out

def size_lots(cfg: Cfg, equity_ref: float, risk_pct: float, sl_dist: float) -> float:
    """Lot-ul se rotunjeste IN JOS; daca iese sub lotul minim, tranzactia se sare (nu se rotunjeste in sus)."""
    if sl_dist <= 0: return 0.0
    money = equity_ref * risk_pct / 100.0
    lots = money / (sl_dist / cfg.tick_size * cfg.tick_value)
    lots = math.floor(lots / cfg.lot_step + 1e-9) * cfg.lot_step
    return 0.0 if lots < cfg.lot_min else min(lots, cfg.lot_max)

def manage_ftmo_risk(eng: "Engine", now: datetime) -> RiskState:
    """Se apeleaza la fiecare pas, INAINTE de orice strategie. Latch-uri: odata declansat, cap-ul zilnic ramane activ pana la 00:00 CE(S)T."""
    cfg, R = eng.cfg, eng.risk_mem
    bal, eq = eng.broker.account()
    day_cet = now.astimezone(CET).date()
    if R.get("day") != day_cet:                                                    # ziua FTMO noua
        R.update(day=day_cet, ref=max(bal, eq), soft=False, kill=False, streak=0, closed_pnl=0.0)
    day_loss = (R["ref"] - eq) / cfg.initial_balance                               # fractiune din balanta initiala (inclusiv flotant)
    total_loss = (cfg.initial_balance - eq) / cfg.initial_balance
    reason = ""
    if day_loss >= cfg.daily_kill or total_loss >= cfg.total_kill: R["kill"] = True; reason = "KILL: day %.2f%% / total %.2f%%" % (day_loss * 100, total_loss * 100)
    if day_loss >= cfg.daily_soft or total_loss >= cfg.total_soft: R["soft"] = True; reason = reason or "SOFT: day %.2f%% / total %.2f%%" % (day_loss * 100, total_loss * 100)
    news = eng.news.blocked(cfg, now) if eng.news else None
    if news: reason = reason or "STIRI: " + news
    scale = 1.0
    if R["streak"] >= 2: scale *= cfg.loss_streak_scale
    if (eq - R["ref"]) / cfg.initial_balance >= cfg.profit_lock_pct: scale *= 0.5
    if now.weekday() == 4 and now.hour * 60 + now.minute >= cfg.friday_flat: R["kill"] = True; reason = reason or "vineri: inchidere pentru weekend"
    return RiskState(allow_new=not (R["soft"] or R["kill"] or news), kill=R["kill"], scale=scale, day_loss=day_loss, total_loss=total_loss, reason=reason)

def risk_budget_ok(eng: "Engine", rs: RiskState, new_risk_pct: float) -> bool:
    """Pierderea zilnica deja realizata + riscul deschis + riscul noii tranzactii trebuie sa incapa sub plafonul soft (cel mai rau caz = toate SL-urile lovite)."""
    return rs.day_loss * 100 + eng.broker.open_risk_pct() + new_risk_pct <= eng.cfg.daily_soft * 100


# ============================================================ MOTOR ============================================================
class Engine:
    """feed.bars(tf_sec, n) -> dict t,o,h,l,c,v (DOAR bare inchise) ; feed.spread() -> pret
       broker.account() -> (balance, equity) ; broker.open_risk_pct() ; broker.place_limit(setup, lots) ; broker.cancel_pending() ; broker.close_all()"""
    def __init__(self, cfg: Cfg, feed, broker, news: Optional[NewsFilter] = None):
        self.cfg, self.feed, self.broker, self.news = cfg, feed, broker, news
        self.st = DayState(day=date.min); self.risk_mem: Dict = {}; self.log: List[str] = []

    def step(self, now: datetime):
        """Se apeleaza la inchiderea fiecarui minut."""
        cfg = self.cfg
        if self.st.day != now.date(): self.st = DayState(day=now.date())
        rs = manage_ftmo_risk(self, now)
        if rs.kill: self.broker.cancel_pending(); self.broker.close_all(); self._log(now, rs.reason); return
        if not rs.allow_new:
            self.broker.cancel_pending()
            if rs.reason: self._log(now, rs.reason)
            return
        sess = session_of(cfg, now)
        setup = None
        if sess == "asia": setup = process_asia(self, now)
        elif sess == "london": setup = process_london(self, now)
        elif sess == "ny": setup = process_new_york(self, now)
        if setup is None: return
        pct = setup.risk_pct * rs.scale
        if not risk_budget_ok(self, rs, pct): self._log(now, "buget de risc depasit, setup sarit"); return
        lots = size_lots(cfg, self.broker.account()[0], pct, abs(setup.entry - setup.sl))
        if lots <= 0: self._log(now, "lot sub minim, setup sarit"); return
        self.broker.place_limit(setup, lots); self.st.trades[setup.session] += 1; self.st.pending.append(setup)
        self._log(now, "SETUP %s %s entry=%.5f sl=%.5f lots=%.2f risk=%.2f%% %s" % (setup.session, "LONG" if setup.side > 0 else "SHORT", setup.entry, setup.sl, lots, pct, setup.note))

    def on_trade_closed(self, pnl: float):
        """Apelat de adaptorul de broker la fiecare inchidere: tine seria de pierderi."""
        R = self.risk_mem; R["streak"] = R.get("streak", 0) + 1 if pnl < 0 else 0

    def _log(self, now, msg): self.log.append("%s %s" % (now.strftime("%Y-%m-%d %H:%M"), msg))
