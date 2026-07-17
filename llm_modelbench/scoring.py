"""Deterministic scorers. Every function returns (score_0_to_100, reason).

Quality here is pure task correctness. Speed, VRAM, and value are handled elsewhere and never
folded into these numbers, so a slow model is never penalised on a task it got right.
"""
from __future__ import annotations

import json
import re
import statistics
import shutil
from typing import Any, Dict, List, Optional, Tuple

from . import sandbox


# ---- text helpers
def strip_thinking(text: str) -> str:
    """Remove model-internal reasoning blocks before deterministic scoring."""
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:thinking|reasoning|thoughts?)\s*\n.*?```", "", text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _lang_aliases(lang: Optional[str]) -> set[str]:
    if not lang:
        return set()
    language = lang.lower()
    aliases = {
        "python": {"python", "py"},
        "javascript": {"javascript", "js", "jsx", "ts", "typescript"},
        "html": {"html", "xml"},
        "css": {"css"},
        "bash": {"bash", "sh", "shell"},
    }
    return aliases.get(language, {language, language[:2]})


def extract_blocks(text: str, lang: Optional[str] = None, *, include_raw: bool = False) -> List[str]:
    """Return all candidate fenced blocks, with optional raw-text fallback.

    V9.5.8 returned the *last* fenced block only. That corrupted common answers that contain
    separate HTML and CSS blocks, or a correct Python block followed by an example block. V9.5.9
    scores all plausible candidates and lets each scorer select the best one.
    """
    clean = strip_thinking(text)
    pat = re.compile(r"```([a-zA-Z0-9_+\-.]*)\s*\n(.*?)```", re.DOTALL)
    blocks = [(tag.lower().strip(), body.strip()) for tag, body in pat.findall(clean or "")]
    if lang and blocks:
        aliases = _lang_aliases(lang)
        preferred = [body for tag, body in blocks if tag in aliases or any(tag.startswith(a) for a in aliases if a)]
        if preferred:
            out = preferred
        else:
            # Bare blocks are still valid candidates when no language-specific block exists.
            bare = [body for tag, body in blocks if not tag]
            out = bare or [body for _, body in blocks]
    else:
        out = [body for _, body in blocks]
    if include_raw or not out:
        raw = (clean or "").strip()
        if raw and raw not in out:
            out.append(raw)
    return out or [(clean or "").strip()]


def extract_code(text: str, lang: Optional[str] = None) -> str:
    """Backward-compatible first candidate extractor.

    New scorers should prefer extract_blocks() or best_over_blocks().
    """
    return extract_blocks(text, lang, include_raw=True)[0]


def _dedupe_candidates(candidates: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for cand in candidates:
        key = cand.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def _target_symbols_from_checks(checks: List[str]) -> List[str]:
    symbols: List[str] = []
    for chk in checks or []:
        for name in re.findall(r"\b([A-Za-z_]\w*)\s*\(", chk or ""):
            if name not in {"assert", "sorted", "list", "dict", "set", "len", "range", "print"} and name not in symbols:
                symbols.append(name)
    return symbols


def best_over_blocks(response: str, lang: Optional[str], fn, *, include_raw: bool = True,
                     max_candidates: int = 3, prefer_regex: Optional[str] = None) -> Tuple[float, str]:
    """Score bounded candidate blocks and return the best-scoring result.

    V9.5.10 caps execution fan-out and short-circuits on the first perfect score.
    Raw prose fallback is opt-in because executor scorers should never run narrative text.
    """
    candidates = _dedupe_candidates(extract_blocks(response, lang, include_raw=include_raw))
    if prefer_regex:
        rx = re.compile(prefer_regex, re.MULTILINE)
        preferred = [c for c in candidates if rx.search(c or "")]
        if preferred:
            rest = [c for c in candidates if c not in preferred]
            candidates = preferred + rest
    candidates = candidates[:max(1, int(max_candidates or 1))]
    # A score is a number in [0, 100]. `-1.0` is not a score, and an empty fenced block
    # (`\`\`\`python\n\`\`\``) leaves zero candidates and reaches the leaderboard as -1.0.
    best: Optional[Tuple[float, str]] = None
    for cand in candidates:
        result = fn(cand)
        score = result[0] if isinstance(result[0], (int, float)) else -1.0
        if best is None or score > (best[0] if isinstance(best[0], (int, float)) else -1.0):
            best = result
        if score >= 100.0:
            return result
    return best if best is not None else (0.0, "no scorable candidate")


def normalize(s: str, lower: bool = True, collapse_ws: bool = True) -> str:
    s = strip_thinking(s or "")
    if lower:
        s = s.lower()
    if collapse_ws:
        s = re.sub(r"\s+", " ", s).strip()
    return s


def edit_distance(a, b) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def char_accuracy(ref: str, hyp: str) -> float:
    if not ref:
        return 1.0 if not hyp else 0.0
    return max(0.0, 1.0 - edit_distance(ref, hyp) / max(len(ref), 1))


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


def median(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(vals), 2) if vals else None


# ---- scorers
def score_python(response: str, meta: Dict[str, Any], timeout: int = 10) -> Tuple[float, str]:
    symbols = _target_symbols_from_checks(meta.get("checks", []))
    prefer = r"^\s*def\s+(?:" + "|".join(re.escape(x) for x in symbols) + r")\s*\(" if symbols else None
    return best_over_blocks(
        response,
        "python",
        lambda code: sandbox.run_python_checks(code, meta.get("checks", []), timeout),
        include_raw=False,
        max_candidates=3,
        prefer_regex=prefer,
    )


def _missing_detail(label: str, missing: List[str], limit: int = 6) -> str:
    if not missing:
        return ""
    shown = ", ".join(str(x) for x in missing[:limit])
    if len(missing) > limit:
        shown += f", +{len(missing)-limit} more"
    return f", missing: {shown}"


def _contains_all(text: str, tokens: List[str]) -> bool:
    return all(t.lower() in text for t in tokens)


def _required_hits(text: str, required: List[str], required_any: List[List[str]],
                   required_any_re: Optional[List[List[str]]] = None) -> tuple[int, int, List[str]]:
    hits = 0
    total = 0
    missing: List[str] = []
    for token in required:
        total += 1
        if token.lower() in text:
            hits += 1
        else:
            missing.append(token)
    for group in required_any:
        total += 1
        if any(tok.lower() in text for tok in group):
            hits += 1
        else:
            missing.append(" OR ".join(group))
    for group in required_any_re or []:
        total += 1
        if any(re.search(rx, text, re.I) for rx in group):
            hits += 1
        else:
            missing.append(" OR ".join(group))
    return hits, total, missing


def score_tokens_in_code(response: str, meta: Dict[str, Any]) -> Tuple[float, str]:
    lang = meta.get("lang")

    def one(code: str) -> Tuple[float, str]:
        code_l = (code or "").lower()
        hits, total, missing = _required_hits(code_l, meta.get("required", []), meta.get("required_any", []), meta.get("required_any_re", []))
        return (100.0 * hits / total if total else 0.0), f"{hits}/{total} tokens{_missing_detail('tokens', missing)}"

    return best_over_blocks(response, lang, one)


def _needle_group_hit(text: str, group: List[str]) -> bool:
    return any(normalize(n) in text for n in group)


def score_contains(response: str, meta: Dict[str, Any]) -> Tuple[float, str]:
    text = normalize(response)
    groups: List[List[str]] = [[n] for n in meta.get("needles", [])] + [list(g) for g in meta.get("needles_any", [])]
    hits = [g for g in groups if _needle_group_hit(text, g)]
    missing = [" OR ".join(g) for g in groups if not _needle_group_hit(text, g)]
    frac = len(hits) / len(groups) if groups else 0.0
    reason = f"{len(hits)}/{len(groups)}{_missing_detail('needles', missing)}"
    if meta.get("all_required", True):
        return (100.0 if frac == 1.0 else 0.0), reason
    return 100.0 * frac, reason


def score_regex(response: str, meta: Dict[str, Any]) -> Tuple[float, str]:
    return (100.0 if re.search(meta["pattern"], response or "", re.I) else 0.0), "regex"


_TRAILING_SENTENCE_PUNCTUATION_RE = re.compile(r"[.!?…]+$")


def _strip_trailing_sentence_punctuation(s: str) -> str:
    """Remove harmless terminal sentence marks for natural-language exact answers.

    This is deliberately scoped to ``score_exact``. Commas, colons, semicolons,
    internal punctuation, and the punctuation-sensitive ``score_exact_code`` scorer
    remain unchanged.
    """
    return _TRAILING_SENTENCE_PUNCTUATION_RE.sub("", s)


def score_exact(response: str, meta: Dict[str, Any]) -> Tuple[float, str]:
    got = _strip_trailing_sentence_punctuation(normalize(response))
    want = _strip_trailing_sentence_punctuation(normalize(meta["expected"]))
    if got == want:
        return 100.0, "exact"
    return 0.0, f"exact mismatch, expected: {want[:80]!r}, found: {got[:80]!r}"


def normalize_code(s: str) -> str:
    """Normalize spacing around code punctuation without relaxing code equality."""
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s*([\-/.])\s*", r"\1", s)


def score_exact_code(response: str, meta: Dict[str, Any]) -> Tuple[float, str]:
    """Exact code scorer tolerant only of spacing around code punctuation."""
    got = normalize_code(response)
    want = normalize_code(meta["expected"])
    if got == want:
        return 100.0, "exact code"
    return 0.0, f"exact code mismatch, expected: {want[:80]!r}, found: {got[:80]!r}"


def score_lineset(response: str, meta: Dict[str, Any]) -> Tuple[float, str]:
    got = {normalize(line) for line in (response or "").splitlines() if line.strip()}
    want = {normalize(line) for line in meta.get("expected_lines", []) if line.strip()}
    if not want:
        return 0.0, "no expected"
    inter = got & want
    missing = sorted(want - inter)
    return 100.0 * len(inter) / len(want), f"{len(inter)}/{len(want)} lines{_missing_detail('lines', missing)}"


def _first_balanced_json_object(text: str) -> Optional[str]:
    """Return the first balanced JSON object from text, ignoring surrounding prose/fences."""
    text = strip_thinking(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def score_json_schema(response: str, meta: Dict[str, Any]) -> Tuple[float, str]:
    """Constraint adherence: valid JSON with required keys."""
    obj = _first_balanced_json_object(response or "")
    if not obj:
        return 0.0, "no json object"
    try:
        data = json.loads(obj)
    except json.JSONDecodeError as e:
        return 25.0, f"invalid json: {e}"
    missing = [k for k in meta.get("required_keys", []) if k not in data]
    if missing:
        return 50.0, f"missing {missing}"
    fenced = "```" in (response or "")
    return 100.0, "ok" if not fenced else "ok, fenced"


def score_ocr(response: str, meta: Dict[str, Any]) -> Tuple[float, str]:
    ref = normalize(meta["reference"])
    hyp = normalize(response)
    ca = char_accuracy(ref, hyp)
    wd = edit_distance(ref.split(), hyp.split())
    wa = max(0.0, 1.0 - wd / max(len(ref.split()), 1))
    return 100.0 * ca, f"char={ca:.3f} word={wa:.3f}"


def score_filesort(response: str, meta: Dict[str, Any], timeout: int = 10) -> Tuple[float, str]:
    def one(code: str) -> Tuple[float, str]:
        ran, layout = sandbox.run_script_in_fixture(code, meta["fixture_files"], timeout)
        if not ran:
            return 0.0, "did not run / unsafe"
        expected = meta["expected_layout"]
        total = sum(len(v) for v in expected.values()) or 1
        correct = sum(len(set(v) & layout.get(d, set())) for d, v in expected.items())
        return 100.0 * correct / total, f"{correct}/{total} placed"

    return best_over_blocks(response, "python", one, include_raw=False, max_candidates=3)



# ---- responsive web_nav cascade scorer (V9.5.12)
_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
HORIZONTAL, VERTICAL, NEITHER = "horizontal", "vertical", "neither"


def _web_stylesheets(text: str) -> str:
    css = " ".join(re.findall(r"<style[^>]*>(.*?)</style>", text or "", re.S | re.I))
    for tag, body in re.findall(r"```([a-zA-Z]*)\s*\n(.*?)```", text or "", re.S):
        if tag.lower() == "css" or ("{" in body and "<" not in body[:80]):
            css += "\n" + body
    return re.sub(r"/\*.*?\*/", " ", css, flags=re.S)


def _web_media_blocks(css: str) -> List[Tuple[str, str]]:
    """Brace-matched @media bodies. Regex-only parsing drops nested closing braces."""
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r"@media([^{}]*)\{", css or ""):
        i, depth = m.end(), 1
        while i < len(css) and depth:
            depth += (css[i] == "{") - (css[i] == "}")
            i += 1
        out.append((m.group(1), css[m.end():i - 1]))
    return out


def _web_outside_media(css: str) -> str:
    out: List[str] = []
    last = 0
    for m in re.finditer(r"@media[^{}]*\{", css or ""):
        i, depth = m.end(), 1
        while i < len(css) and depth:
            depth += (css[i] == "{") - (css[i] == "}")
            i += 1
        out.append(css[last:m.start()])
        last = i
    out.append((css or "")[last:])
    return " ".join(out)


def _web_decls(body: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for part in (body or "").split(";"):
        if ":" in part:
            k, _, v = part.partition(":")
            # Do not use rstrip("!important"): rstrip removes a character set, not a suffix.
            d[k.strip().lower()] = re.sub(r"\s*!\s*important\s*$", "", v.strip().lower())
    return d


def _web_rules(css: str) -> List[Tuple[str, Dict[str, str]]]:
    return [(s.strip().lower(), _web_decls(b)) for s, b in _RULE.findall(css or "")]


def _web_px(val: str, unit: str) -> float:
    return float(val) * (16.0 if unit.lower() in {"em", "rem"} else 1.0)


def _web_media_matches_width(condition: str, width: float) -> bool:
    """Return whether a width-limited media query applies at the supplied viewport width."""
    matches = re.findall(r"(max|min)-width\s*:\s*([\d.]+)\s*(px|em|rem)", condition or "", re.I)
    if not matches:
        return False
    for kind, val, unit in matches:
        px = _web_px(val, unit)
        if kind.lower() == "max" and width > px:
            return False
        if kind.lower() == "min" and width < px:
            return False
    return True


def _web_nav_selectors(html: str) -> Tuple[bool, set[str]]:
    m = re.search(r"<\s*nav\b[^>]*>(.*?)</\s*nav\s*>", html or "", re.S | re.I)
    if m:
        opening, inner = m.group(0).split(">", 1)[0] + ">", m.group(1)
    else:
        m2 = re.search(r"<\s*nav\b[^>]*/?>", html or "", re.S | re.I)
        if not m2:
            return False, set()
        opening, inner = m2.group(0), ""
    scope = opening + inner
    selectors: set[str] = {"nav"}
    for a, b in re.findall(r'class\s*=\s*(?:["\']([^"\']+)["\']|([\w-]+))', scope, re.I):
        selectors.update("." + c.lower() for c in (a or b).split())
    for tid in re.findall(r'id\s*=\s*["\']([^"\']+)["\']', scope, re.I):
        selectors.add("#" + tid.lower())
    selectors.update(t.lower() for t in re.findall(r"<\s*(ul|ol|li|a|div|span)\b", inner, re.I))
    return True, selectors


def _web_key_selectors(selector: str) -> set[str]:
    """Rightmost simple selector for each comma-separated selector.

    `nav a {display:block}` styles the link, not the nav. Matching every compound would let a
    descendant rule clobber the nav container's computed layout.
    """
    out: set[str] = set()
    for part in (selector or "").split(","):
        compounds = [c for c in re.split(r"[\s>+~]+", part.strip()) if c]
        if compounds:
            out.add(compounds[-1].split(":")[0])
    return out


def _web_computed(css: str, target: str, width: float) -> Dict[str, str]:
    """Small last-wins cascade for benchmark answers. Specificity puzzles are out of scope."""
    props: Dict[str, str] = {}
    for sel, d in _web_rules(_web_outside_media(css)):
        if target in _web_key_selectors(sel):
            props.update(d)
    for cond, body in _web_media_blocks(css):
        if _web_media_matches_width(cond, width):
            for sel, d in _web_rules(body):
                if target in _web_key_selectors(sel):
                    props.update(d)
    return props


def _web_layout(props: Dict[str, str]) -> str:
    disp = (props.get("display") or "").replace("-webkit-", "").replace("-ms-", "")
    if disp in {"flex", "inline-flex"}:
        return VERTICAL if (props.get("flex-direction", "row") or "row").startswith("column") else HORIZONTAL
    if disp == "grid":
        return HORIZONTAL if (props.get("grid-auto-flow", "row") or "row").startswith("column") else VERTICAL
    if disp in {"block", "list-item", "table-row"}:
        return VERTICAL
    if disp in {"inline", "inline-block", "table-cell"}:
        return HORIZONTAL
    return NEITHER


def _score_web_nav_candidate(candidate: str) -> Tuple[float, str]:
    html = candidate or ""
    css = _web_stylesheets(candidate)
    has_nav, nav_sels = _web_nav_selectors(html)
    if not has_nav:
        return 0.0, "0/100, missing: no semantic <nav> element"
    if not css.strip():
        return 20.0, "20/100: <nav> present, no CSS"

    styled = {t for sel, _ in _web_rules(css) for t in _web_key_selectors(sel)} & nav_sels
    if not styled:
        return 20.0, "20/100: no CSS rule targets any element in the <nav> subtree"

    best, why = 20.0, "no element in the <nav> subtree changes layout across 600px"
    for target in sorted(styled):
        narrow = _web_layout(_web_computed(css, target, 599.0))
        wide = _web_layout(_web_computed(css, target, 601.0))
        pts = 20.0
        notes: List[str] = []
        if wide == HORIZONTAL:
            pts += 30.0
        else:
            notes.append(f"`{target}` is not horizontal at 601px (got {wide})")
        if narrow == VERTICAL and narrow != wide:
            pts += 50.0
        elif narrow != VERTICAL:
            notes.append(f"`{target}` does not stack at 599px (got {narrow})")
        else:
            notes.append(f"`{target}` renders identically at 599px and 601px: not responsive")
        if pts > best:
            best, why = pts, "; ".join(notes)
        if pts == 100.0:
            return 100.0, "100/100: <nav> subtree is horizontal at 601px and vertical at 599px"
    return best, f"{int(best)}/100, missing: {why}"



def score_web_nav(response: str, meta: Dict[str, Any]) -> Tuple[float, str]:
    """Responsive-nav scorer using a tiny cascade at 599px and 601px.

    It scores the rendered layout transition, not scattered CSS tokens.
    """
    blocks = extract_blocks(response, None, include_raw=False)
    raw = strip_thinking(response)
    # Score the full answer first. A separate HTML fence plus CSS fence is one artifact;
    # scoring the HTML block alone can tie at 20 and produce the false reason "no CSS".
    candidates = [raw] if raw else []
    candidates.extend(blocks)
    if len(blocks) > 1:
        candidates.append("\n".join(blocks))
    candidates = _dedupe_candidates(candidates)
    results = [_score_web_nav_candidate(c) for c in candidates]
    if not results:
        # `<think>...</think>` truncated before the answer strips to empty. max() would raise.
        return 0.0, "0/100: no scorable content"
    return max(results, key=lambda r: (r[0], "no CSS" not in r[1]))

# ---- executable js_debounce scorer (V9.5.12)
def _safe_js_source(code: str) -> Optional[str]:
    lowered = (code or "").lower()
    forbidden = ["require(", "import ", "process", "child_process", "fs.", "eval(", "new function", "constructor"]
    if any(tok in lowered for tok in forbidden):
        return None
    return code or ""


def _run_js_debounce_check(code: str, timeout: int = 3) -> Tuple[Optional[float], str]:
    safe = _safe_js_source(code)
    if safe is None:
        return 0.0, "unsafe javascript"
    harness = """
const vm = require('vm');
const code = process.env.LLM_MODELBENCH_JS_CODE || '';
const sandbox = { setTimeout, clearTimeout, console: { log(){} } };
let debounce;
try {
  debounce = vm.runInNewContext(code + "\\n; typeof debounce !== 'undefined' ? debounce : undefined", sandbox, { timeout: 500 });
} catch (err) {
  console.log(JSON.stringify({ ok:false, reason:'eval_error:' + err.message }));
  process.exit(0);
}
if (typeof debounce !== 'function') {
  console.log(JSON.stringify({ ok:false, reason:'debounce not defined' }));
  process.exit(0);
}
let calls = [];
try {
  const fn = function(...args) { calls.push({ args }); };
  const debounced = debounce(fn, 30);
  if (typeof debounced !== 'function') {
    console.log(JSON.stringify({ ok:false, reason:'debounce did not return a function' }));
    process.exit(0);
  }
  for (let i = 0; i < 5; i++) debounced(i, 'x');
  setTimeout(() => {
    const ok = calls.length === 1 && calls[0].args[0] === 4 && calls[0].args[1] === 'x';
    console.log(JSON.stringify({ ok, calls: calls.length, last: calls[0] ? calls[0].args : null }));
  }, 90);
  setTimeout(() => process.exit(0), 150);
} catch (err) {
  console.log(JSON.stringify({ ok:false, reason:'runtime_error:' + err.message }));
}
"""
    cp = sandbox.run_node_harness(harness, safe, timeout=timeout)
    if cp is None:
        return None, "HARNESS_ERROR: node unavailable"
    if cp.returncode == 124:
        return None, "HARNESS_ERROR: debounce check timed out"
    if cp.returncode != 0:
        stderr_lines = (cp.stderr or "").strip().splitlines()
        detail = " | ".join(stderr_lines[-3:]) if stderr_lines else "no stderr"
        return None, f"HARNESS_ERROR: node exited {cp.returncode}: {detail[:300]}"
    out = (cp.stdout or "").strip().splitlines()[-1:]
    if not out:
        return 0.0, "no debounce check output"
    try:
        data = json.loads(out[0])
    except Exception:
        return 0.0, "invalid debounce check output"
    if data.get("ok") is True:
        return 100.0, "debounce executed: exactly one trailing invocation with latest args"
    return 0.0, str(data.get("reason") or f"calls={data.get('calls')} last={data.get('last')}")


def score_js_debounce(response: str, meta: Dict[str, Any]) -> Tuple[Optional[float], str]:
    if shutil.which("node") is None:
        return None, "HARNESS_ERROR: node unavailable"
    candidates = _dedupe_candidates(extract_blocks(response, "javascript", include_raw=False))
    rx = re.compile(r"\bdebounce\s*[=(]", re.MULTILINE)
    preferred = [c for c in candidates if rx.search(c or "")]
    if preferred:
        rest = [c for c in candidates if c not in preferred]
        candidates = preferred + rest
    candidates = candidates[:3] or [""]
    best: Tuple[Optional[float], str] = (0.0, "no executable debounce candidate")
    for code in candidates:
        result = _run_js_debounce_check(code)
        if result[0] is None:
            return result
        if float(result[0]) > float(best[0] or 0.0):
            best = result
        if result[0] >= 100.0:
            return result
    return best


def _strict_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = strip_thinking(text or "").strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _loose_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = _first_balanced_json_object(strip_thinking(text or ""))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _agentic_json_candidates(text: str) -> List[Tuple[Dict[str, Any], str, float]]:
    """Return parsed JSON action candidates with format multipliers.

    Agentic validity rule: decision correctness is scored first, then output format
    shades the result. Format can reduce a correct decision; it can never lift a
    wrong decision into correctness.
    """
    raw = strip_thinking(text or "").strip()
    out: List[Tuple[Dict[str, Any], str, float]] = []

    def add(obj_text: str, kind: str, multiplier: float) -> None:
        try:
            data = json.loads((obj_text or "").strip())
        except Exception:
            return
        if isinstance(data, dict):
            item = (data, kind, multiplier)
            if item not in out:
                out.append(item)

    if raw.startswith("{") and raw.endswith("}"):
        add(raw, "strict_json", 1.0)

    fence_pat = re.compile(r"```([a-zA-Z0-9_+\-.]*)\s*\n(.*?)```", re.DOTALL)
    fences = [(tag.lower().strip(), body.strip()) for tag, body in fence_pat.findall(raw)]
    if fences:
        outside = fence_pat.sub("", raw).strip()
        for tag, body in fences:
            json_obj = _first_balanced_json_object(body)
            if not json_obj:
                continue
            if not outside and tag in {"", "json"}:
                add(json_obj, "fenced_json", 0.90)
            else:
                add(json_obj, "prose_plus_json", 0.75)

    loose = _first_balanced_json_object(raw)
    if loose:
        if raw == loose:
            add(loose, "strict_json", 1.0)
        else:
            add(loose, "prose_plus_json", 0.75)

    return out


def _coerce_agentic_args(args: Any) -> Tuple[Dict[str, Any], str]:
    """Return canonical args and shape: object, wrapped_object, or invalid."""
    if isinstance(args, dict):
        return dict(args), "object"
    if isinstance(args, list):
        if len(args) == 1 and isinstance(args[0], dict):
            return dict(args[0]), "wrapped_object"
    return {}, "invalid"



def _agentic_allowed_top_level_keys(meta: Dict[str, Any]) -> set[str]:
    """Allowed envelope keys for an agentic action.

    The task contract is an allowlist, not a denylist. `reason` is only
    permitted for refusal tasks; otherwise it is just another extra key.
    """
    keys = {"tool", "args"}
    if meta.get("require_reason") or meta.get("expected_tool") is None:
        keys.add("reason")
    return keys


def _is_null_tool(value: Any) -> bool:
    return value is None or str(value).lower() in {"", "none", "null", "no_call", "refuse"}


def _agentic_caps_from_notes(notes: List[str]) -> List[str]:
    """Machine-readable decision/envelope notes for agentic diagnostics.

    These are the scorer's own notes, not a parser over the human reason string.
    Formatting-only notes and decision_cap suffixes are intentionally excluded.
    """
    ignore_prefixes = (
        "format=",
        "format_multiplier=",
        "decision_cap=",
        "tool_alias:",
    )
    ignore_exact = {"arg_aliases_applied", "format=wrapped_args"}
    out: List[str] = []
    for note in notes or []:
        text = str(note)
        if text in ignore_exact or text.startswith(ignore_prefixes):
            continue
        out.append(text)
    return out


def score_agentic_action_details(response: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Detailed agentic score decomposition.

    `score` is the historical blended persisted value. V9.5.17 also exposes
    `decision_score` and `format_multiplier` so aggregate/report layers can rank
    tool-call decisions separately from JSON hygiene.
    """
    strict = bool(meta.get("strict_json", True))
    if strict:
        candidates = _agentic_json_candidates(response)
    else:
        data = _loose_json_object(response)
        candidates = [(data, "loose_json", 1.0)] if data is not None else []
    if not candidates:
        raw = strip_thinking(response or "").strip()
        if raw.startswith("```"):
            reason = "invalid_json:fenced_non_json"
        elif "```" in raw:
            reason = "invalid_json:prose_or_code_fence_no_action_json"
        elif raw:
            reason = "invalid_json:no_action_object"
        else:
            reason = "invalid_json:empty"
        return {
            "score": 0.0,
            "reason": reason,
            "decision_score": 0.0,
            "format_multiplier": 1.0,
            "format_deviation": "invalid_json",
            "caps_fired": [reason],
            "notes": [reason],
        }

    def score_candidate(data: Dict[str, Any], fmt: str, multiplier: float) -> Dict[str, Any]:
        notes: List[str] = []
        caps: List[float] = []
        format_deviation = "strict_json" if fmt == "strict_json" else fmt

        if fmt != "strict_json":
            notes.append(f"format={fmt}")

        forbidden_keys = [str(k) for k in (meta.get("forbidden_top_level_keys") or []) if k in data]
        extra_keys = sorted(set(data) - _agentic_allowed_top_level_keys(meta))
        if forbidden_keys:
            notes.extend(f"forbidden_key={k}" for k in forbidden_keys[:4])
            caps.append(50.0)
        elif extra_keys:
            notes.extend(f"extra_top_level_key={k}" for k in extra_keys[:4])
            caps.append(50.0)

        has_tool_key = "tool" in data
        has_args_key = "args" in data
        if not has_tool_key:
            notes.append("tool")
            caps.append(50.0)
        if not has_args_key:
            notes.append("args object")
            caps.append(50.0)

        raw_args = data.get("args") if has_args_key else None
        args, args_shape = _coerce_agentic_args(raw_args)
        if has_args_key and args_shape == "wrapped_object":
            notes.append("format=wrapped_args")
            format_deviation = "wrapped_args"
            # Format deviations do not compound. A correct answer in the worst
            # accepted format must still outrank every wrong decision cap.
            multiplier = min(multiplier, 0.90)
        elif has_args_key and args_shape == "invalid":
            notes.append("args object")
            caps.append(50.0)

        expected_tool = meta.get("expected_tool", "__absent__")
        actual_tool = data.get("tool") if has_tool_key else None
        aliases = meta.get("tool_aliases") or {}
        canonical_tool = aliases.get(str(actual_tool), actual_tool) if actual_tool is not None else None
        if canonical_tool != actual_tool:
            notes.append(f"tool_alias:{actual_tool}->{canonical_tool}")

        allowed = set(str(x) for x in (meta.get("allowed_tools") or []))
        if canonical_tool is not None and not _is_null_tool(canonical_tool):
            if allowed and str(canonical_tool) not in allowed:
                notes.append("tool not allowed")
                caps.append(20.0)

        if expected_tool is None:
            tool_ok = _is_null_tool(canonical_tool)
            if not tool_ok:
                notes.append("no tool call")
                caps.append(40.0)
        else:
            tool_ok = str(canonical_tool) == str(expected_tool)
            if not tool_ok:
                if _is_null_tool(canonical_tool):
                    notes.append("over_refusal")
                notes.append(f"tool={expected_tool}")
                caps.append(40.0)

        expected_args = meta.get("expected_args") or {}
        arg_aliases = meta.get("arg_aliases") or {}
        if arg_aliases and args:
            remapped: Dict[str, Any] = {}
            for key, val in args.items():
                remapped[str(arg_aliases.get(key, key))] = val
            if remapped != args:
                notes.append("arg_aliases_applied")
            args = remapped

        args_ok = True
        if expected_args:
            for key, want in expected_args.items():
                if args.get(key) != want:
                    args_ok = False
                    notes.append(f"args.{key}={want!r}")
            # Extra args can be harmless metadata for future tasks unless explicitly forbidden.
        else:
            if args != {}:
                args_ok = False
                notes.append("empty args")
        if not args_ok:
            caps.append(65.0)

        if meta.get("require_reason"):
            reason_text = str(data.get("reason") or "").strip()
            if len(reason_text) < 8:
                notes.append("refusal reason")
                caps.append(70.0)

        decision_score = 100.0
        if caps:
            decision_score = min(caps)

        score = max(0.0, decision_score * multiplier)
        score = min(100.0, score)
        if multiplier != 1.0:
            notes.append(f"format_multiplier={multiplier:.2f}")
        if caps:
            notes.append(f"decision_cap={min(caps):g}")

        score = round(score, 2)
        if score >= 100.0:
            if not notes:
                reason = "agentic action ok"
            else:
                reason = f"agentic action ok ({', '.join(notes[:10])})"
        else:
            reason = f"agentic action {score:.1f}/100, missing: {', '.join(notes[:10])}"
        return {
            "score": score,
            "reason": reason,
            "decision_score": round(float(decision_score), 2),
            "format_multiplier": round(float(multiplier), 4),
            "format_deviation": format_deviation if multiplier != 1.0 else "strict_json",
            "caps_fired": _agentic_caps_from_notes(notes),
            "notes": list(notes),
        }

    scored = [score_candidate(*cand) for cand in candidates]
    return max(scored, key=lambda x: x["score"])


def score_agentic_action(response: str, meta: Dict[str, Any]) -> Tuple[float, str]:
    """Score deterministic local tool/action decisions, preserving the legacy tuple API."""
    detail = score_agentic_action_details(response, meta)
    return detail["score"], detail["reason"]



def _split_agentic_reason_notes(text: str) -> List[str]:
    """Split a reason note list on top-level commas only.

    Nested expected-arg reprs may contain commas, for example
    `args.ticket={'title': 'Disk alert', 'labels': ['infra', 'disk']}`.
    """
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    quote = ""
    esc = False
    for ch in str(text or ""):
        if quote:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
            continue
        if ch in "[{(":
            depth += 1
            buf.append(ch)
            continue
        if ch in "]})" and depth > 0:
            depth -= 1
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                out.append(part)
            buf = []
            continue
        buf.append(ch)
    part = "".join(buf).strip()
    if part:
        out.append(part)
    return out

def agentic_caps_fired_from_reason(reason: str) -> List[str]:
    """Extract machine-readable agentic failure notes from an existing reason string.

    V9.5.16 is a freeze/instrumentation release: this parser deliberately derives
    `caps_fired` from the already-emitted reason text instead of changing scorer
    logic or the score tuple. It filters formatting/diagnostic suffixes so the
    remaining values describe decision/envelope gates such as `tool not allowed`,
    `tool=calculator.add`, `empty args`, or `extra_top_level_key=reason`.
    """
    text = str(reason or "").strip()
    if not text:
        return []
    if text.startswith("invalid_json:"):
        return [text]
    marker = "missing: "
    if marker not in text:
        return []
    tail = text.split(marker, 1)[1]
    parts = _split_agentic_reason_notes(tail)
    ignore_prefixes = (
        "format=",
        "format_multiplier=",
        "decision_cap=",
        "tool_alias:",
    )
    ignore_exact = {"arg_aliases_applied", "format=wrapped_args"}
    out: List[str] = []
    for p in parts:
        if p in ignore_exact or p.startswith(ignore_prefixes):
            continue
        out.append(p)
    return out

def score_retrieval(embed_fn, meta: Dict[str, Any]) -> Tuple[float, str]:
    docs = meta["docs"]
    ids = list(docs.keys())
    texts = list(docs.values()) + [q for q, _ in meta["queries"]]
    emb = embed_fn(texts)
    if len(emb) != len(texts) or not all(emb):
        return 0.0, "embed failed"
    dvec, qvec = emb[:len(ids)], emb[len(ids):]
    hit1, rr = 0, 0.0
    for i, (_, gold) in enumerate(meta["queries"]):
        ranked = sorted(range(len(ids)), key=lambda j: cosine(qvec[i], dvec[j]), reverse=True)
        pos = ranked.index(ids.index(gold)) + 1
        hit1 += 1 if pos == 1 else 0
        rr += 1.0 / pos
    n = len(meta["queries"])
    r1, mrr = hit1 / n, rr / n
    return 100.0 * (0.7 * r1 + 0.3 * mrr), f"recall@1={r1:.2f} mrr={mrr:.2f}"


DETERMINISTIC = {
    "python": score_python,
    "tokens_in_code": score_tokens_in_code,
    "js_debounce": score_js_debounce,
    "contains": score_contains,
    "regex": score_regex,
    "exact": score_exact,
    "exact_code": score_exact_code,
    "lineset": score_lineset,
    "json_schema": score_json_schema,
    "ocr": score_ocr,
    "filesort": score_filesort,
    "web_nav": score_web_nav,
    "agentic_action": score_agentic_action,
}
