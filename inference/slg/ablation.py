"""Ablation switches for the three online mechanisms (A, B, C).

Each leave-one-out condition turns exactly one mechanism off so its
contribution can be measured against the full system:

* ``competence`` (A) off — the cosine ranking is no longer signed by online
  reliability; routing is static cosine similarity.
* ``deterministic`` (B) off — the domain-grounded engineering checks (numeric
  sanity, units, format, vetoes) are skipped; verification falls back to the
  generic 8B critic alone.
* ``abstention`` (C) off — every critic-passed answer is returned; the
  calibrated confidence bar never withholds.

``base`` turns all three off, leaving the bare combination (reasoning router +
critic + reroute) that the mechanisms are meant to improve on.

Selecting a non-full preset routes the run's answers/diagnostics to
``answers/<experiment>__<label>/`` so ablation runs never clobber the full run.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    """Which online mechanisms are active for a run."""

    competence: bool = True       # (A)
    deterministic: bool = True    # (B) deterministic engineering checks
    abstention: bool = True       # (C)
    label: str = "full"

    @property
    def suffix(self) -> str:
        """Output-directory suffix; empty for the full system."""
        return "" if self.label == "full" else f"__{self.label}"


PRESETS = {
    "full": AblationConfig(True, True, True, "full"),
    "no_competence": AblationConfig(False, True, True, "no_competence"),   # -A
    "no_verifier": AblationConfig(True, False, True, "no_verifier"),       # -B
    "no_abstention": AblationConfig(True, True, False, "no_abstention"),   # -C
    "base": AblationConfig(False, False, False, "base"),                   # none
}


def get_ablation(name: str) -> AblationConfig:
    """Look up an ablation preset by name (defaults to the full system)."""
    if not name:
        return PRESETS["full"]
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown ablation '{name}'. Choose one of: {', '.join(PRESETS)}."
        )
