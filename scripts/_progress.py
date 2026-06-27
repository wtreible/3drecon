"""Consistent progress UI for project scripts.

tqdm-backed (installed in both images) with a block-style bar, plus a `stage()`
banner for discrete pipeline steps. Falls back gracefully if tqdm is missing, so
importing this never breaks a script.

Usage:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _progress import track, stage

    stage(1, 3, "Detecting markers")
    for f in track(files, "detect", total=len(files)):
        ...
"""
try:
    from tqdm import tqdm as _tqdm
except Exception:                       # pragma: no cover
    _tqdm = None

_BAR = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


def track(iterable, desc="working", total=None):
    """Wrap an iterable with a block-style progress bar (no-op if tqdm absent)."""
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, total=total, bar_format=_BAR,
                 ascii=" ░▒▓█", leave=True, dynamic_ncols=True)


def counter(total=None, desc="working"):
    """Manual-update bar for while-loops: call .update(n) then .close(). No-op-safe."""
    if _tqdm is None:
        class _Noop:
            def update(self, n=1): pass
            def close(self): pass
            def set_postfix_str(self, *a, **k): pass
        return _Noop()
    return _tqdm(total=total, desc=desc, bar_format=_BAR,
                 ascii=" ░▒▓█", leave=True, dynamic_ncols=True)


def stage(i, n, msg):
    """Print a discrete pipeline-stage banner, e.g.  [2/3] Matching."""
    print(f"\n[{i}/{n}] {msg}", flush=True)
