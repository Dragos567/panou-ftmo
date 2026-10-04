# Diagnostic de fus orar: cum se leaga timpul barelor brokerului de ora Romaniei si de sesiunea indicelui.
import numpy as np, datetime as dt
from .run import load_m1


def report(data_dir, sym):
    m = load_m1(data_dir, sym); t = m["t"]; v = m["v"].astype(np.float64)
    day = t // 86400; hr = (t % 86400) // 3600; mo = (t % 86400) // 60 % 60
    dts = [dt.datetime.utcfromtimestamp(int(x)) for x in t[::2000]]
    out = {"sym": sym, "bars": int(len(t)), "first": dts[0].isoformat(), "last": dts[-1].isoformat()}
    months = np.array([dt.datetime.utcfromtimestamp(int(x)).month for x in t[::600]])
    # pe luni reprezentative: ore cu bare (ora serverului), numar mediu de bare pe ora si tick-volum mediu
    prof = {}
    mon_of = np.array([dt.datetime.utcfromtimestamp(int(x)).month for x in t[::1]]) if len(t) < 3_000_000 else None
    ym = (t // 86400 // 30).astype(int)
    for label, sel in (("iarna(ian-feb)", None), ("vara(iul-aug)", None)):
        pass
    # luam ultimele 2 ierni/veri complete din date
    tm = np.array([(dt.datetime.utcfromtimestamp(int(x)).year, dt.datetime.utcfromtimestamp(int(x)).month) for x in t[::1440]])
    res = {}
    for (y, mth, lab) in ((dt.datetime.utcfromtimestamp(int(t[-1])).year - 1, 1, "ianuarie"), (dt.datetime.utcfromtimestamp(int(t[-1])).year - 1, 7, "iulie")):
        a = int(dt.datetime(y, mth, 1).timestamp()); b = int(dt.datetime(y, mth + 1, 1).timestamp())
        s = (t >= a) & (t < b) & (((t // 86400 + 4) % 7) < 5)    # luni-vineri (epoca = joi)
        h = hr[s]; vv = v[s]
        cnt = np.bincount(h, minlength=24); vol = np.bincount(h, weights=vv, minlength=24)
        res[lab] = {"bare_pe_ora": cnt.tolist(), "tickvol_pe_ora": [int(x) for x in vol]}
    out["profil"] = res
    # trecerea de ora (DST): in ziua cu cea mai lunga/scurta pauza din martie/octombrie? raportam deschiderea de luni
    mon = (t // 86400 + 4) % 7 == 0
    first_mon = {}
    for d in np.unique(day[mon])[-60:]:
        s = day == d; first_mon[dt.datetime.utcfromtimestamp(int(d) * 86400).strftime("%Y-%m-%d")] = float(t[s][0] % 86400) / 3600
    out["prima_bara_luni_ore"] = first_mon
    return out
