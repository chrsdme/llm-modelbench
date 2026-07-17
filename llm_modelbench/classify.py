"""Model classification and capability-aware task-family routing.

Routing evidence is intentionally layered:

1. explicit ``MODEL_PROFILES`` operator overrides;
2. capabilities reported by Ollama ``/api/show``;
3. functional probe results supplied by the planner/wizard;
4. conservative name hints when stronger evidence is unavailable.

An explicit profile is never made unreachable merely because Ollama returned a
non-empty capability list. Hybrid models retain every supported family rather
than being collapsed to either ``text`` or ``vision``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

# Override map: substring in model name -> forced class and/or base families.
# Keys are matched case-insensitively as substrings. Families are a base route,
# not an exclusive allow-list: declared/probed extra capabilities are unioned.
MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "medgemma": {"class": "medical", "families": ["vision", "text"]},
    "deepseek-r1": {"class": "reasoning"},
    "ornith": {"class": "reasoning"},
    "qwythos": {"class": "reasoning", "families": ["vision", "text"]},
    # Known fleet VLM conversions whose Ollama metadata has historically been
    # incomplete (often reporting only ``completion``).  These narrow profile
    # keys preserve their real vision route even without ``--auto``.
    "internvl3-8b": {"class": "vision", "families": ["vision", "text"]},
    "garnet-ocr-7b": {"class": "vision", "families": ["vision", "text"]},
    "vl-1-coder": {"class": "coding", "families": ["vision", "text"]},
    "abliterat": {"class": "experimental"},
    "heretic": {"class": "experimental"},
    "dolphin": {"class": "experimental"},
}

EMBED_HINTS = ("embed", "bge-", "bge:", "nomic", "mxbai", "rerank")
VISION_HINTS = (
    "-vl", "vl:", "vl-", "minicpm-v", "llava", "vision", "medgemma",
    "gemma3", "gemma-4", "qwen3.5", "qwopus3.5", "qwythos",
    "internvl", "garnet-ocr", "ocr-", "-ocr",
)
CODE_HINTS = ("coder", "codestral", "devstral", "deepseek-coder", "codegemma")

# Some community GGUF conversions of embedding/reranking architectures are
# reported by Ollama as completion-only. These are explicit, evidence-backed
# exceptions. Keep this list narrow.
FORCE_EMBEDDING_HINTS = (
    "llama-embed",
    "kalm-reranker",
)

FAMILY_ORDER = ("vision", "text", "embedding", "tools", "insert")
_CAPABILITY_FAMILIES = {
    "completion": ("text",),
    "vision": ("vision", "text"),
    "embedding": ("embedding",),
    "tools": ("tools", "text"),
    "tool": ("tools", "text"),
    "tool_calling": ("tools", "text"),
    "insert": ("insert",),
    "fim": ("insert",),
}


def _unique_families(values: Iterable[str]) -> List[str]:
    seen = {str(v).lower() for v in values if str(v).lower() in FAMILY_ORDER}
    return [family for family in FAMILY_ORDER if family in seen]


def profile_for(name: str) -> Dict[str, Any]:
    n = name.lower()
    merged: Dict[str, Any] = {}
    for key, profile in MODEL_PROFILES.items():
        if key in n:
            if "class" in profile:
                merged["class"] = profile["class"]
            if "families" in profile:
                merged["families"] = list(profile["families"])
    return merged


def _force_embedding_by_name(name: str) -> bool:
    n = name.lower()
    return any(hint in n for hint in FORCE_EMBEDDING_HINTS)


def _normalise_capabilities(capabilities: Optional[Iterable[str]]) -> List[str]:
    return [str(c).strip().lower() for c in capabilities or [] if str(c).strip()]


def families_from_capabilities(capabilities: Optional[Iterable[str]]) -> List[str]:
    out: List[str] = []
    for capability in _normalise_capabilities(capabilities):
        out.extend(_CAPABILITY_FAMILIES.get(capability, ()))
    return _unique_families(out)


def hinted_families(name: str) -> List[str]:
    n = name.lower()
    if _force_embedding_by_name(name) or any(hint in n for hint in EMBED_HINTS):
        return ["embedding"]
    out: List[str] = []
    if any(hint in n for hint in VISION_HINTS):
        out.extend(("vision", "text"))
    else:
        out.append("text")
    return _unique_families(out)


def families_for(
    name: str,
    capabilities: Optional[List[str]] = None,
    probed_families: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return every task family supported by the available evidence.

    ``probed_families`` is additive. A failed probe is handled by the capability
    interrogation layer, which decides whether a weak name hint should be
    removed while retaining explicit/declared capabilities for visible testing.
    """
    profile = profile_for(name)
    declared_caps = _normalise_capabilities(capabilities)
    declared = families_from_capabilities(declared_caps)
    n = name.lower()
    if _force_embedding_by_name(name):
        return ["embedding"]
    # Known embedding/reranking model names remain embedding-only even when
    # community metadata loosely co-declares ``completion`` or ``tools``.  This
    # exact pattern produced dozens of HTTP 400 rows for qwen3-embedding and
    # nomic-embed-code in a real fleet run.  Keep the rule name-scoped rather
    # than declaring every future model with an ``embedding`` capability to be
    # exclusive: a genuinely hybrid runtime can still be represented when its
    # name is not an embedding/reranker hint and it declares completion too.
    if any(hint in n for hint in EMBED_HINTS) and (
        "completion" not in declared_caps or "embedding" in declared_caps
    ):
        return ["embedding"]
    if "embedding" in declared_caps and "completion" not in declared_caps:
        return ["embedding"]
    out: List[str] = list(profile.get("families") or [])
    out.extend(declared)
    out.extend(probed_families or [])

    # Only use generic name hints when no stronger family evidence exists. An
    # explicit embedding exception above remains valid even with completion-only
    # metadata because that is precisely the metadata defect it documents.
    if not out:
        out.extend(hinted_families(name))

    return _unique_families(out) or ["text"]


def classify_model(
    name: str,
    capabilities: Optional[List[str]] = None,
    probed_families: Optional[Iterable[str]] = None,
) -> str:
    """Assign a broad display class without discarding hybrid capabilities.

    ``class`` and ``families`` are related but not identical: a Qwythos VLM can
    intentionally keep the display class ``reasoning`` while routing both text
    and vision tasks.  An embedding-only runtime route is different: allowing a
    name profile such as ``ornith`` to label that model ``reasoning`` would
    contradict its only executable family.  Resolve families first and let the
    embedding-only route win before applying a compatible profile class.
    """
    families = families_for(name, capabilities, probed_families)
    if families == ["embedding"] or ("embedding" in families and "text" not in families):
        return "embedding"

    profile = profile_for(name)
    if profile.get("class"):
        return str(profile["class"])

    n = name.lower()
    if "vision" in families:
        return "vision"
    if any(hint in n for hint in CODE_HINTS) or "insert" in families:
        return "coding"
    if "tools" in families:
        return "agentic"
    return "general"


def size_gb(model_row: Mapping[str, Any]) -> float:
    return round((model_row.get("size") or 0) / 1e9, 2)
