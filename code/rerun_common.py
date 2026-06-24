"""
Shared feature-extraction pipeline.

split_sentences + MiniLM embedding + cosine epsilon-proximity graph +
5 spectral + 4 Ollivier-Ricci features per trace.

Run all scripts with cwd = repo root.
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

# geom/ lives at repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geom.graph import cosine_threshold_graph, knn_graph, EPS_DEFAULT  # noqa: E402
from geom.spectral import all_spectral_features  # noqa: E402
from geom.ricci import all_curvatures, curvature_features  # noqa: E402

SPECTRAL_KEYS = ["spectral_entropy", "fiedler_value", "grs", "shs", "hfer"]
RICCI_KEYS = ["mean_kappa", "std_kappa", "frac_negative", "min_kappa"]

_MODEL = None
_CACHE = Path("data/embeddings_cache")


def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n+', text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def _cache_key(sentences):
    h = hashlib.sha256()
    for s in sentences:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embed(sentences, use_cache=True):
    """(n, 384) L2-normalized float32."""
    if not sentences:
        return np.zeros((0, 384), dtype=np.float32)
    if use_cache:
        _CACHE.mkdir(parents=True, exist_ok=True)
        cf = _CACHE / (_cache_key(sentences) + ".npy")
        if cf.exists():
            return np.load(cf)
    vecs = _model().encode(sentences, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    vecs = vecs.astype(np.float32)
    if use_cache:
        np.save(cf, vecs)
    return vecs


def embed_trace(cot_text, use_cache=True):
    return embed(split_sentences(cot_text), use_cache=use_cache)


def extract_features(cot_text, eps=EPS_DEFAULT, use_cache=True, graph_fn=None):
    """
    Return (feat_dict, status). feat_dict has n_sentences, n_edges,
    spectral{5}, ricci{4}. status 'ok' or a reason.
    """
    X = embed_trace(cot_text, use_cache=use_cache)
    if X.shape[0] < 3:
        return None, "too_few_sentences"
    G = cosine_threshold_graph(X, eps=eps) if graph_fn is None else graph_fn(X)
    if G.number_of_edges() == 0:
        return None, "no_edges"
    spectral = all_spectral_features(G, X)
    kappas = all_curvatures(G)
    ricci = curvature_features(kappas)
    return {
        "n_sentences": int(X.shape[0]),
        "n_edges": int(G.number_of_edges()),
        "spectral": spectral,
        "ricci": ricci,
    }, "ok"


def log(phase, msg):
    import time
    p = Path("RERUN_LOG.md")
    with p.open("a") as f:
        f.write("- %s | %s | %s\n" % (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), phase, msg))
