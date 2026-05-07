"""BLEU-cluster duplication penalty for the Challenger reward (Phase 1).

Agglomeratively clusters all generated questions in a Phase 1 step's batch by
sentence-BLEU distance. Each rollout is assigned the fraction of the batch
that lies in its cluster: a unique question gets ~1/N (small penalty), a
question shared by many rollouts gets a large one. This is the diversity
hammer that prevents Challenger collapse to a single high-δ template.
"""
from collections import Counter

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.cluster import AgglomerativeClustering


def _bleu_distance_matrix(sentences: list[str]) -> np.ndarray:
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist


def cluster_share(
    questions: list[str],
    distance_threshold: float = 0.5,
    linkage: str = "average",
) -> list[float]:
    """Return per-question fraction of the batch in its BLEU cluster."""
    if not questions:
        return []
    dmat = _bleu_distance_matrix(questions)
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage,
    ).fit_predict(dmat)
    total = len(questions)
    sizes = Counter(labels)
    ratio = {lab: sz / total for lab, sz in sizes.items()}
    return [ratio[lab] for lab in labels]
