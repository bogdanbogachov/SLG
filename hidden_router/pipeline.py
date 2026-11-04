"""Full-pipeline hook: fit the Mahalanobis router for the cell currently configured in CONFIG."""
from __future__ import annotations

import json
import os

from config import CONFIG
from logging_config import logger


def fit_kwargs() -> dict:
    cfg = CONFIG.get("hidden_router", {})
    return {"val_fraction": float(cfg.get("val_fraction", 0.2)),
            "alphas": tuple(float(a) for a in cfg.get("shrinkage_grid", (0.05, 0.2, 0.5)))}


def fit_current_cell(cell_dir: str, force: bool = False) -> str:
    """Fit on CONFIG['files']['qa_train'] and save to <cell_dir>/hidden_router (skipped if already fitted)."""
    from hidden_router.cells import expert_id_from_title
    from hidden_router.features import FeatureExtractor
    from hidden_router.router import MahalanobisRouter

    router_dir = os.path.join(cell_dir, "hidden_router")
    if MahalanobisRouter.exists(router_dir) and not force:
        logger.info("Mahalanobis router already fitted: %s", router_dir)
        return router_dir
    with open(CONFIG["files"]["qa_train"], encoding="utf-8") as f:
        train = json.load(f)
    extractor = FeatureExtractor()
    try:
        features = extractor.encode([r["question"] for r in train])
    finally:
        extractor.close()
    router = MahalanobisRouter.fit(features, [expert_id_from_title(r["title"]) for r in train],
                                   seed=int(CONFIG.get("seed", 42)), **fit_kwargs())
    router.save(router_dir)
    logger.info("Mahalanobis router: layer=%s alpha=%s val_top1=%.4f -> %s", router.layer, router.alpha,
                router.selection["chosen"]["val_top1"], router_dir)
    return router_dir
