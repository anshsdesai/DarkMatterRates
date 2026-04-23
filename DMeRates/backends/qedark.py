"""QEdark source-family backend."""

from __future__ import annotations

from .semiconductor_common import LegacySemiconductorBackend


class QEDarkBackend(LegacySemiconductorBackend):
    source_family = "qedark"
    is_qedark = True

    def form_factor_path(self):
        return f"../form_factors/QEDark/{self.material}_f2.txt"
