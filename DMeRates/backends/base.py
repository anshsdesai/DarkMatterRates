"""Backend protocol definitions for ``DMeRate`` source families."""

from __future__ import annotations


class RateBackend:
    """Small interface shared by source-family backends.

    The facade keeps backward compatibility, so this is intentionally lightweight
    rather than a strict abstract base class.
    """

    source_family = None
    rate_units = "implicit"

    @classmethod
    def supports_material(cls, material):
        return material in getattr(cls, "supported_materials", ())

    def attach(self, rate):
        raise NotImplementedError

    def energy_grid(self):
        raise NotImplementedError

    def differential_rate(self, *args, **kwargs):
        raise NotImplementedError

    def fold_to_ne(self, *args, **kwargs):
        return None
