# Incarcare istoric si resampling. Timpul e in secunde (cum il da MetaApi/FTMO). Nicio functie nu "se uita in viitor".
import numpy as np
from .hist import REC, RS

DT = np.dtype([("t", "<i8"), ("o", "<f8"), ("h", "<f8"), ("l", "<f8"), ("c", "<f8"), ("v", "<f4")])
TFSEC = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def from_bytes(b):
    a = np.frombuffer(b, dtype=DT)
    return {k: np.ascontiguousarray(a[k]) for k in ("t", "o", "h", "l", "c", "v")}


def resample(m, sec, drop_last=True):
    """M1 -> bare de `sec` secunde, aliniate la multipli de `sec` (UTC). Bara = prima deschidere, maxim, minim, ultima inchidere, suma volumului."""
    t = m["t"]; b = t // sec
    if len(t) == 0: return {k: m[k][:0] for k in m}
    cut = np.flatnonzero(np.diff(b)) + 1; st = np.concatenate(([0], cut))
    out = {"t": (b[st] * sec).astype(np.int64), "o": m["o"][st], "h": np.maximum.reduceat(m["h"], st),
           "l": np.minimum.reduceat(m["l"], st), "c": m["c"][np.concatenate((cut, [len(t)])) - 1],
           "v": np.add.reduceat(m["v"].astype(np.float64), st).astype(np.float32)}
    if drop_last and len(out["t"]) > 1:  # ultima bara poate fi incompleta
        out = {k: v[:-1] for k, v in out.items()}
    return out


def htf_index(base_t, base_sec, htf_t, htf_sec):
    """Pentru fiecare bara de baza: indexul ultimei bare HTF COMPLET inchise la inchiderea barei de baza (-1 daca nu exista).
    Folosit pentru ca o decizie luata la inchiderea barei i sa vada doar bare HTF deja inchise (fara privire in viitor)."""
    close_t = base_t + base_sec
    return np.searchsorted(htf_t + htf_sec, close_t, side="right") - 1


def map_htf(values, idx):
    """Valorile HTF la momentul fiecarei bare de baza (NaN unde nu exista)."""
    out = np.full(len(idx), np.nan)
    ok = idx >= 0; out[ok] = values[idx[ok]]; return out
