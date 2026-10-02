# Indicatori. Fiecare valoare la indexul i foloseste doar barele <= i (inchise).
import numpy as np
from .nb import njit


@njit(cache=True)
def ema(x, n):
    out = np.full(len(x), np.nan); k = 2.0 / (n + 1.0); s = np.nan
    for i in range(len(x)):
        if np.isnan(x[i]): out[i] = s; continue
        s = x[i] if np.isnan(s) else s + k * (x[i] - s)
        out[i] = s if i >= n - 1 else np.nan
    return out


@njit(cache=True)
def sma(x, n):
    out = np.full(len(x), np.nan); s = 0.0
    for i in range(len(x)):
        s += x[i]
        if i >= n: s -= x[i - n]
        if i >= n - 1: out[i] = s / n
    return out


@njit(cache=True)
def true_range(h, l, c):
    out = np.empty(len(h)); out[0] = h[0] - l[0]
    for i in range(1, len(h)):
        out[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return out


@njit(cache=True)
def atr(h, l, c, n):
    """ATR cu medie Wilder."""
    tr = true_range(h, l, c); out = np.full(len(h), np.nan); s = 0.0
    for i in range(len(h)):
        if i < n: s += tr[i]
        if i == n - 1: out[i] = s / n
        elif i >= n: out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


@njit(cache=True)
def rsi(c, n):
    out = np.full(len(c), np.nan); ag = 0.0; al = 0.0
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]; g = d if d > 0 else 0.0; ls = -d if d < 0 else 0.0
        if i <= n: ag += g / n; al += ls / n
        else: ag = (ag * (n - 1) + g) / n; al = (al * (n - 1) + ls) / n
        if i >= n: out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


@njit(cache=True)
def roll_max(x, n):
    """Maximul ultimelor n bare, INCLUSIV bara curenta."""
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        m = x[i]
        for j in range(i - n + 1, i): m = max(m, x[j])
        out[i] = m
    return out


@njit(cache=True)
def roll_min(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        m = x[i]
        for j in range(i - n + 1, i): m = min(m, x[j])
        out[i] = m
    return out


@njit(cache=True)
def shift(x, k):
    """x[i-k] (valoarea de acum k bare)."""
    out = np.full(len(x), np.nan)
    for i in range(k, len(x)): out[i] = x[i - k]
    return out
