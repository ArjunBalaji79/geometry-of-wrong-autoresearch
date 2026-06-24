"""
Feature extraction for the autoresearch D1/D2 analyses.

Design constraints (from AUTORESEARCH.md / QUESTION.md):
  - Reuse the repo's existing graph + spectral feature utilities (geom/).
  - Use ricci-numpy (pure-numpy, bit-exact network simplex) for ALL curvature.
    We deliberately do NOT call geom.ricci (scipy linprog) anywhere: the task
    states the argmin edge and the frac-negative threshold are sensitive to
    solver error, so curvature must come from the exact reference solver.

This module produces, per trace:
  - the 9-dim geometric signature (5 spectral + 4 Ollivier-Ricci),
  - the underlying sentence list, embedding graph, and per-EDGE curvature
    (needed for D2 argmin-edge localization),
  - surface features (length, repetition) for the residualization control.

Run with cwd = repo root.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ricci-numpy"))

from geom.graph import cosine_threshold_graph, EPS_DEFAULT  # noqa: E402
from geom.spectral import all_spectral_features  # noqa: E402
import ricci_numpy as rn  # noqa: E402  (exact reference curvature — required)
import ricci_numpy.fast as rnf  # noqa: E402  (numba JIT; bit-exact to rn, used for bulk)

# Curvature backend. rnf (numba) is the SAME exact network-simplex algorithm as
# the pure-numpy reference rn; verified bit-exact (max|diff| = 0) on the trace
# graphs in this corpus (see src/verify_curvature_backends.py / results/00_survey.md).
# We never use an approximate (Sinkhorn/POT) solver.
_RN = rnf

SPECTRAL_KEYS = ["spectral_entropy", "fiedler_value", "grs", "shs", "hfer"]
RICCI_KEYS = ["mean_kappa", "std_kappa", "frac_negative", "min_kappa"]
SIG_KEYS = SPECTRAL_KEYS + RICCI_KEYS  # the 9-dim signature, fixed order

FRAC_NEG_THRESHOLD = -0.05  # matches geom/ricci.curvature_features default

_MODEL = None
_CACHE = ROOT / "data" / "embeddings_cache"  # reuse the repo's shared cache


# ---------------------------------------------------------------------------
# sentence splitting + embedding (identical to code/rerun_common.py)
# ---------------------------------------------------------------------------
def split_sentences(text):
    text = (text or "").strip()
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
    import hashlib
    h = hashlib.sha256()
    for s in sentences:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embed(sentences, use_cache=True):
    """(n, 384) L2-normalized float32. Shares data/embeddings_cache with the repo."""
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


# ---------------------------------------------------------------------------
# curvature via ricci-numpy (exact)
# ---------------------------------------------------------------------------
def edge_curvatures_exact(G: nx.Graph):
    """
    Exact Ollivier-Ricci curvature for every edge of G, using ricci-numpy.

    Returns (edges, kappas) where edges is a list of (u, v) tuples in the SAME
    order as kappas. ricci-numpy returns curvatures in lexicographic edge order
    np.argwhere(triu(A,1)>0); we reproduce that ordering for the edge list so
    that kappa[k] corresponds to edges[k] exactly (needed for D2 argmin).
    """
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    assert nodes == list(range(n)), "graph nodes must be 0..n-1"
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
    A = (A > 0).astype(int)
    np.fill_diagonal(A, 0)
    kappas = _RN.all_curvatures(A, alpha=0.5)
    edges = [tuple(int(x) for x in e) for e in np.argwhere(np.triu(A, 1) > 0)]
    return edges, np.asarray(kappas, dtype=float)


def curvature_features_from_kappas(kappas):
    k = np.asarray(kappas, dtype=float)
    if k.size == 0:
        return {"mean_kappa": 0.0, "std_kappa": 0.0,
                "frac_negative": 0.0, "min_kappa": 0.0}
    return {
        "mean_kappa": float(k.mean()),
        "std_kappa": float(k.std()),
        "frac_negative": float((k < FRAC_NEG_THRESHOLD).mean()),
        "min_kappa": float(k.min()),
    }


# ---------------------------------------------------------------------------
# surface features (for the residualization control)
# ---------------------------------------------------------------------------
def surface_features(sentences, cot_text):
    """
    Surface-form features that must be residualized symmetrically across tasks:
      - log token length of the full trace
      - n_sentences
      - repetition rate: 1 - (unique sentences / total sentences)  (exact dup loops)
      - bigram repetition: fraction of word-bigrams that are repeats
    """
    toks = re.findall(r"\w+", (cot_text or "").lower())
    n_tok = len(toks)
    n_sent = len(sentences)
    uniq = len(set(s.strip().lower() for s in sentences))
    rep_rate = 0.0 if n_sent == 0 else 1.0 - uniq / n_sent
    if n_tok >= 2:
        bigrams = list(zip(toks[:-1], toks[1:]))
        bg_rep = 1.0 - len(set(bigrams)) / len(bigrams)
    else:
        bg_rep = 0.0
    return {
        "log_n_tokens": float(np.log1p(n_tok)),
        "n_tokens": int(n_tok),
        "n_sentences": int(n_sent),
        "repetition_rate": float(rep_rate),
        "bigram_repetition": float(bg_rep),
    }


SURFACE_KEYS = ["log_n_tokens", "repetition_rate", "bigram_repetition"]


# ---------------------------------------------------------------------------
# full per-trace extraction
# ---------------------------------------------------------------------------
def extract_full(cot_text, eps=EPS_DEFAULT, use_cache=True):
    """
    Return (record, status). record has:
      sentences, n_sentences, n_edges, signature{9}, spectral{5}, ricci{4},
      surface{...}, edges[list of (u,v)], kappas[list].
    status 'ok' or a reason string.
    """
    sentences = split_sentences(cot_text)
    if len(sentences) < 3:
        return None, "too_few_sentences"
    X = embed(sentences, use_cache=use_cache)
    G = cosine_threshold_graph(X, eps=eps)
    if G.number_of_edges() == 0:
        return None, "no_edges"
    spectral = all_spectral_features(G, X)
    edges, kappas = edge_curvatures_exact(G)
    ricci = curvature_features_from_kappas(kappas)
    sig = {**spectral, **ricci}
    surf = surface_features(sentences, cot_text)
    rec = {
        "sentences": sentences,
        "n_sentences": int(X.shape[0]),
        "n_edges": int(G.number_of_edges()),
        "signature": {k: float(sig[k]) for k in SIG_KEYS},
        "spectral": spectral,
        "ricci": ricci,
        "surface": surf,
        "edges": edges,
        "kappas": [float(x) for x in kappas],
    }
    return rec, "ok"


def signature_vector(rec):
    return np.array([rec["signature"][k] for k in SIG_KEYS], dtype=float)
