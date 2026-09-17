"""NLI contradiction scoring (distance ablation D4, prereg §8).

roberta-large-mnli on CPU, lazy singleton, batched. Label order for
roberta-large-mnli is [contradiction, neutral, entailment].
"""
import numpy as np

_MODEL = None
_TOK = None


def _load():
    global _MODEL, _TOK
    if _MODEL is None:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        name = "roberta-large-mnli"
        _TOK = AutoTokenizer.from_pretrained(name)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(name)
        _MODEL.eval()
        _MODEL.to("cpu")
        _TORCH = torch
    return _TOK, _MODEL


def contradiction_probs(pairs, batch_size=8):
    """pairs: list of (premise, hypothesis) justification strings.
    Returns list of contradiction probabilities."""
    import torch
    tok, model = _load()
    out = []
    for s in range(0, len(pairs), batch_size):
        chunk = pairs[s:s + batch_size]
        prems = [p for p, _ in chunk]
        hyps = [h for _, h in chunk]
        enc = tok(prems, hyps, return_tensors="pt", truncation=True,
                  max_length=512, padding=True)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 0]  # contradiction
        out.extend(float(p) for p in probs)
    return out


def contradiction_matrix(justifications):
    """Symmetric NxN matrix: mean of both directional contradiction probs."""
    n = len(justifications)
    pairs, idx = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append((justifications[i], justifications[j]))
                idx.append((i, j))
    probs = contradiction_probs(pairs)
    mat = np.zeros((n, n))
    for (i, j), p in zip(idx, probs):
        mat[i][j] = p
    sym = (mat + mat.T) / 2.0
    return sym
