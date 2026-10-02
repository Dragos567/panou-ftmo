# numba daca exista, altfel Python simplu (aceleasi rezultate, mai lent) - serverul instaleaza numba singur.
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k: return a[0]
        return lambda f: f
