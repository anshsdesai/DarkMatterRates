"""QCDark2 dielectric-screening policy helpers."""

from __future__ import annotations

import numpy as np

VALID_DIELECTRIC_SCREENING = {"rpa", "none"}


def normalize_dielectric_screening(screening: str | None) -> str:
    """Validate and normalize the QCDark2 dielectric screening selector."""
    if screening is None:
        raise ValueError(
            "QCDark2 calculations require an explicit screening choice. "
            "Pass screening='rpa' or screening='none'."
        )
    if not isinstance(screening, str):
        raise ValueError(
            f"screening={screening!r} not recognized. "
            f"Use one of: {sorted(VALID_DIELECTRIC_SCREENING)}."
        )
    screening_norm = screening.lower()
    if screening_norm not in VALID_DIELECTRIC_SCREENING:
        raise ValueError(
            f"screening={screening!r} not recognized. "
            f"Use one of: {sorted(VALID_DIELECTRIC_SCREENING)}."
        )
    return screening_norm


def dielectric_screening_epsilon(epsilon: np.ndarray, screening: str | None) -> np.ndarray:
    """Return epsilon_screen(q,E) for the requested QCDark2 screening mode."""
    screening_norm = normalize_dielectric_screening(screening)
    if screening_norm == "rpa":
        return epsilon
    # screening_norm == "none": epsilon_screen = 1
    return np.ones_like(epsilon, dtype=np.complex128)


def dielectric_screening_ratio(epsilon: np.ndarray, screening: str | None) -> np.ndarray:
    """Return |epsilon|^2 / |epsilon_screen|^2 for the requested mode."""
    eps_screen = dielectric_screening_epsilon(epsilon, screening)
    num = np.imag(epsilon) ** 2 + np.real(epsilon) ** 2
    den = np.imag(eps_screen) ** 2 + np.real(eps_screen) ** 2
    return num / den
