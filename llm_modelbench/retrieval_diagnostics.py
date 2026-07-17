"""Case-level diagnostics for unchanged retrieval scoring."""
from typing import Any, Dict, List
from .scoring import cosine

def diagnostics(embed_fn, meta: Dict[str, Any], embed_model: str) -> List[Dict[str, Any]]:
    docs, queries = meta["docs"], meta["queries"]
    ids, vectors = list(docs), embed_fn(list(docs.values()) + [q for q, _ in queries])
    dvec, qvec = vectors[:len(ids)], vectors[len(ids):]
    out = []
    for i, (query, gold) in enumerate(queries):
        ranked = sorted(range(len(ids)), key=lambda j: cosine(qvec[i], dvec[j]), reverse=True)
        gi, rank = ids.index(gold), ranked.index(ids.index(gold)) + 1
        near = next(j for j in ranked if j != gi)
        target, distractor = cosine(qvec[i], dvec[gi]), cosine(qvec[i], dvec[near])
        out.append({"query_index": i, "target_doc_id": gold, "top1_doc_id": ids[ranked[0]], "top3_doc_ids": [ids[j] for j in ranked[:3]], "target_rank": rank, "target_similarity": target, "nearest_distractor_doc_id": ids[near], "nearest_distractor_similarity": distractor, "margin": target - distractor, "pass_at_1": rank == 1, "embed_model": embed_model})
    return out
