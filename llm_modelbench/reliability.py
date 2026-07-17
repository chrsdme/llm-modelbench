"""Pure interpretation helpers for repeated benchmark rows; never recompute scores."""
from __future__ import annotations
from collections import defaultdict

def cell_summary(rows):
    present = [r for r in rows if r is not None]
    if len(present) < len(rows): return {"status":"missing", "range":None, "reason_changed":None}
    if len(present) < 2: return {"status":"insufficient-repeats", "range":None, "reason_changed":False}
    scores = [r.get("score") for r in present]
    numeric = [float(x) for x in scores if isinstance(x, (int,float))]
    if len(numeric) != len(present): return {"status":"missing", "range":None, "reason_changed":None}
    rng = round(max(numeric)-min(numeric),2); changed=len({str(r.get("reason") or "") for r in present})>1
    return {"status":"moving" if rng else ("reason-moving" if changed else "stable"), "range":rng, "reason_changed":changed}

def empirical_noise_band(cells):
    ranges=[s["range"] for s in cells if s.get("range") is not None]
    return max(ranges) if ranges else None

def category_summary(row_groups):
    out=defaultdict(list)
    for category, rows in row_groups:
        s=cell_summary(rows)
        if s.get("range") is not None: out[category].append(s["range"])
    return {c:{"cells":len(v),"max_range":max(v)} for c,v in out.items()}
