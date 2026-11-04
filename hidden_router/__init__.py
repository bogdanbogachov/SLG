"""Hidden-state (Mahalanobis / shared-covariance Gaussian) router for SLG.

The router reads the base Llama-3.2-1B's hidden states of a question, mean-pooled per layer, and assigns
it to the expert whose training questions lie closest in Mahalanobis distance. It is closed-form (class
means + one shrunk shared covariance per layer); the layer and shrinkage are chosen on a validation slice
of the training questions, never on the test set.

Modules
- features.py   question -> per-layer pooled hidden states (+ an on-disk cache keyed by qid)
- router.py     MahalanobisRouter: fit / select layer / save / load / rank
- cells.py      dataset x seed cell helpers (splits, expert ids, scores, routing reports)
- probe.py      plan steps 1-4: extract, fit + test per cell, compare with the old router, estimate scores
- assemble.py   plan steps 5-6: check oracle reuse, build answers/slg_mahalanobis_router.jsonl
                (correct route -> reuse an existing answer of that expert; otherwise generate)

The full pipeline does not need this shortcut: `routing.method`/`--router mahalanobis` in inference/slg.py
uses the same router, and run_training fits it per cell (training_components.train_hidden_router).

CLI: python -m hidden_router {probe,assemble} [--datasets ...] [--seeds ...]
"""
