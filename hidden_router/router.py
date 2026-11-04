"""Closed-form Mahalanobis router (Gaussian classes with one shared, shrunk covariance per layer).

score_e(h) = log p(e) - 0.5 * (h - mu_e)^T S^-1 (h - mu_e)   (+ a term constant across experts)
           = w_e . h + b_e     with  w_e = S^-1 mu_e,  b_e = -0.5 mu_e . w_e + log p(e)
p(e | h)   = softmax_e(score_e(h))
S          = (1 - alpha) * C + alpha * (tr(C) / d) * I      (C = pooled within-expert covariance)

Layer and alpha are selected by top-1 accuracy on a stratified validation slice of the training
questions; the router is then refit on all training questions with the chosen (layer, alpha).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

DEFAULT_ALPHAS = (0.05, 0.2, 0.5)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_split(labels: Sequence[str], val_fraction: float, seed: int):
    """Per expert, hold out round(n * val_fraction) questions (none if the expert has < 3)."""
    by_label = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)
    rng = random.Random(seed)
    fit_idx, val_idx = [], []
    for label in sorted(by_label):
        idx = by_label[label][:]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction)) if len(idx) >= 3 else 0
        val_idx += idx[:n_val]
        fit_idx += idx[n_val:]
    return sorted(fit_idx), sorted(val_idx)


class _LayerStats:
    """Class means and pooled within-class covariance of one layer (computed once, reused for every alpha)."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, n_classes: int):
        self.center = X.mean(0)
        Xc = X - self.center
        counts = torch.bincount(y, minlength=n_classes).to(X.dtype)
        sums = torch.zeros(n_classes, X.shape[1], dtype=X.dtype, device=X.device).index_add_(0, y, Xc)
        self.means = sums / counts.clamp(min=1).unsqueeze(1)
        resid = Xc - self.means[y]
        self.cov = resid.T @ resid / max(1, X.shape[0] - n_classes)
        self.log_prior = torch.log(counts / counts.sum())

    def weights(self, alpha: float):
        d = self.cov.shape[0]
        shrunk = (1 - alpha) * self.cov + alpha * (torch.trace(self.cov) / d) * torch.eye(d, dtype=self.cov.dtype, device=self.cov.device)
        chol = torch.linalg.cholesky(shrunk)
        W = torch.cholesky_solve(self.means.T, chol).T  # [C, d] = (S^-1 mu_e)^T
        b = -0.5 * (self.means * W).sum(1) + self.log_prior
        return W, b


def _topk_hits(scores: torch.Tensor, y: torch.Tensor, ks=(1, 3, 5)) -> Dict[int, float]:
    ranks = (scores > scores.gather(1, y.unsqueeze(1))).sum(1)  # 0-based rank of the gold expert
    return {k: float((ranks < k).float().mean()) for k in ks}


class MahalanobisRouter:
    def __init__(self, labels: List[str], layer: int, alpha: float, center, W, b, selection=None):
        self.labels = list(labels)
        self.layer = int(layer)
        self.alpha = float(alpha)
        self.center = torch.as_tensor(center, dtype=torch.float32)
        self.W = torch.as_tensor(W, dtype=torch.float32)
        self.b = torch.as_tensor(b, dtype=torch.float32)
        self.selection = selection or {}

    # ---------- fitting ----------
    @classmethod
    def fit(cls, features: np.ndarray, labels: Sequence[str], *, layers=None, alphas=DEFAULT_ALPHAS,
            val_fraction: float = 0.2, seed: int = 42) -> "MahalanobisRouter":
        """features: [N, L+1, H] pooled hidden states of the TRAINING questions; labels: expert ids."""
        names = sorted(set(labels))
        label_id = {name: i for i, name in enumerate(names)}
        y_all = torch.tensor([label_id[l] for l in labels], device=_device())
        layers = list(range(features.shape[1])) if layers is None else list(layers)
        fit_idx, val_idx = stratified_split(labels, val_fraction, seed)
        if not val_idx:
            raise ValueError("Validation slice is empty; every expert has fewer than 3 training questions.")
        grid = []
        for layer in layers:
            X = torch.as_tensor(features[:, layer, :], dtype=torch.float32, device=_device())
            stats = _LayerStats(X[fit_idx], y_all[fit_idx], len(names))
            Xv = X[val_idx] - stats.center
            for alpha in alphas:
                W, b = stats.weights(alpha)
                hits = _topk_hits(Xv @ W.T + b, y_all[val_idx])
                grid.append({"layer": layer, "alpha": alpha, "val_top1": hits[1], "val_top3": hits[3], "val_top5": hits[5]})
            del X, stats
        best = max(grid, key=lambda g: (g["val_top1"], g["val_top5"], -g["layer"]))
        router = cls.fit_fixed(features, labels, layer=best["layer"], alpha=best["alpha"], names=names)
        router.selection = {"chosen": best, "grid": grid, "n_fit": len(fit_idx), "n_val": len(val_idx),
                            "val_fraction": val_fraction, "seed": seed}
        return router

    @classmethod
    def fit_fixed(cls, features: np.ndarray, labels: Sequence[str], *, layer: int, alpha: float, names=None):
        names = names or sorted(set(labels))
        label_id = {name: i for i, name in enumerate(names)}
        y = torch.tensor([label_id[l] for l in labels], device=_device())
        X = torch.as_tensor(features[:, layer, :], dtype=torch.float32, device=_device())
        stats = _LayerStats(X, y, len(names))
        W, b = stats.weights(alpha)
        return cls(names, layer, alpha, stats.center.cpu(), W.cpu(), b.cpu())

    # ---------- inference ----------
    def scores(self, features: np.ndarray) -> torch.Tensor:
        """features: [N, L+1, H] (all layers) or [N, H] (this router's layer). Returns log-posteriors [N, C]."""
        X = torch.as_tensor(features, dtype=torch.float32)
        if X.dim() == 3:
            X = X[:, self.layer, :]
        return torch.log_softmax((X - self.center) @ self.W.T + self.b, dim=1)

    def rank(self, features: np.ndarray, top: int | None = None) -> List[List[Dict]]:
        """Per question: experts sorted by posterior probability."""
        logp = self.scores(features)
        order = torch.argsort(logp, dim=1, descending=True)
        out = []
        for row, idx in zip(logp, order):
            idx = idx[:top] if top else idx
            out.append([{"rank": r + 1, "expert": self.labels[int(j)], "probability": float(row[j].exp()),
                         "score": float(row[j].exp()), "score_type": "mahalanobis_posterior"}
                        for r, j in enumerate(idx.tolist())])
        return out

    # ---------- persistence ----------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"labels": self.labels, "layer": self.layer, "alpha": self.alpha,
                    "center": self.center, "W": self.W, "b": self.b}, directory / "router.pt")
        (directory / "selection.json").write_text(json.dumps(self.selection, indent=2), encoding="utf-8")
        (directory / ".done").write_text("done\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "MahalanobisRouter":
        directory = Path(directory)
        state = torch.load(directory / "router.pt", map_location="cpu", weights_only=True)
        selection_path = directory / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else {}
        return cls(state["labels"], state["layer"], state["alpha"], state["center"], state["W"], state["b"], selection)

    @staticmethod
    def exists(directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / "router.pt").is_file() and (directory / ".done").is_file()
