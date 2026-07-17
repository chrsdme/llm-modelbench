"""Clone and redundancy detection, done conservatively.

There are two levels of clone evidence:

1. Digest clones: the same Ollama model digest / ID. This is strong evidence and is safe
   enough for prune recommendations.
2. Probe clones: byte-identical normalised outputs to harder deterministic probes. This is
   only advisory. Empty or near-empty probe outputs are ignored and never count as matches.

The empty-output guard matters. Earlier builds treated two empty probe answer sets as a
perfect match, which created false clone clusters among unrelated reasoning models.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Tuple

from .scoring import normalize

PROBES = [
    "Write exactly two sentences. First sentence: describe why a model benchmark can overfit. Second sentence: give one mitigation. Do not use bullet points.",
    "Return only a compact JSON object with keys risk, cause, fix for: a Python benchmark serializes an argparse Namespace containing a function.",
    "Write a five-line Python function named stable_slug(text) that lowercases, replaces non-alphanumerics with single hyphens, and strips edge hyphens. Code only.",
    "Explain, in one paragraph under 70 words, why a long-context model may be slower even when its weight file is the same size.",
    "Produce exactly three comma-separated tags for a local Ollama model used in an AI workbench.",
    "Write one bash command that lists the 10 largest files under /srv/ai-workdesk without crossing filesystem boundaries.",
]

# If an answer becomes shorter than this after normalisation / thinking-strip, it is not
# discriminating evidence. It is usually a blank, a clipped thinking trace, or a refusal stub.
MIN_CANON_CHARS = 20
MIN_VALID_PROBES = 4


def _canonical(output: str) -> str:
    """Normalise while keeping enough text to distinguish models."""
    return normalize(output or "")[:1200]


def _is_valid_probe(canonical_output: str) -> bool:
    return len((canonical_output or "").strip()) >= MIN_CANON_CHARS


def probe_health(outputs: List[str]) -> Dict[str, int]:
    canon = [_canonical(o) for o in (outputs or [])]
    valid = sum(1 for c in canon if _is_valid_probe(c))
    return {
        "total": len(canon),
        "valid": valid,
        "invalid": len(canon) - valid,
        "min_required": MIN_VALID_PROBES,
    }


def fingerprint_outputs(outputs: List[str]) -> str:
    """Hash valid canonical probe outputs.

    Invalid outputs are represented explicitly so the digest is reproducible, but clone
    matching still refuses to use invalid/empty probe sets as evidence.
    """
    canon = [_canonical(o) for o in (outputs or [])]
    parts = [c if _is_valid_probe(c) else "<INVALID_PROBE_OUTPUT>" for c in canon]
    joined = "\u241f".join(parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def similarity(a: List[str], b: List[str]) -> float:
    """Fraction of valid discriminating probes where two answers match exactly.

    Empty or near-empty outputs never match. If either side has too few valid probes, the
    pair is non-actionable and similarity is 0.0. This prevents false-positive clone
    clusters from clipped <think> traces or blank probe answers.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    ca = [_canonical(x) for x in a]
    cb = [_canonical(y) for y in b]
    if sum(_is_valid_probe(x) for x in ca) < MIN_VALID_PROBES:
        return 0.0
    if sum(_is_valid_probe(y) for y in cb) < MIN_VALID_PROBES:
        return 0.0
    valid_pairs = [(x, y) for x, y in zip(ca, cb) if _is_valid_probe(x) and _is_valid_probe(y)]
    if len(valid_pairs) < MIN_VALID_PROBES:
        return 0.0
    same = sum(1 for x, y in valid_pairs if x == y)
    return same / len(valid_pairs)


def find_clones(fingerprints: Dict[str, List[str]], threshold: float = 1.0) -> List[Tuple[str, str, float]]:
    """Return advisory probe-clone pairs above threshold.

    The empty-output guard in similarity() makes this intentionally conservative.
    """
    names = list(fingerprints.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = similarity(fingerprints[names[i]], fingerprints[names[j]])
            if sim >= threshold:
                pairs.append((names[i], names[j], round(sim, 2)))
    return sorted(pairs, key=lambda x: x[2], reverse=True)


def invalid_probe_models(fingerprints: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """Models whose probe output is too sparse for clone detection."""
    out: Dict[str, Dict[str, int]] = {}
    for model, outputs in (fingerprints or {}).items():
        h = probe_health(outputs)
        if h["valid"] < h["min_required"]:
            out[model] = h
    return out


def _first_string(mapping: Dict[str, Any], keys: Iterable[str]) -> str:
    for k in keys:
        v = mapping.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def model_identity(model: str, tag_row: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Extract stable identity fields from an Ollama /api/tags row.

    Ollama commonly exposes the CLI `ollama list` ID as `digest` in /api/tags. Some
    builds/clients use `id` or `model_id`, so we check all common names.
    """
    row = tag_row or {}
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    digest = _first_string(row, ("digest", "id", "model_id"))
    if not digest:
        digest = _first_string(details, ("digest", "id", "model_id"))
    return {
        "model": model,
        "digest": digest,
        "size": row.get("size"),
        "modified_at": row.get("modified_at"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "family": details.get("family"),
        "families": details.get("families"),
    }


def _normal_digest(digest: Any) -> str:
    if not isinstance(digest, str):
        return ""
    d = digest.strip()
    # Avoid treating absent, tiny, or placeholder values as real identities.
    if len(d) < 8 or d.lower() in {"unknown", "none", "null", "n/a"}:
        return ""
    return d


def find_digest_clones(identities: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, float]]:
    """Certain clone pairs based on identical Ollama digest / ID."""
    groups: Dict[str, List[str]] = {}
    for model, ident in (identities or {}).items():
        d = _normal_digest((ident or {}).get("digest"))
        if d:
            groups.setdefault(d, []).append(model)
    pairs: List[Tuple[str, str, float]] = []
    for models in groups.values():
        models = sorted(models)
        if len(models) < 2:
            continue
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                pairs.append((models[i], models[j], 1.0))
    return pairs
