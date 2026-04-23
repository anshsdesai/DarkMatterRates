"""QCDark source-family backend."""

from __future__ import annotations

from .semiconductor_common import LegacySemiconductorBackend


class QCDarkBackend(LegacySemiconductorBackend):
    source_family = "qcdark"
    is_qedark = False

    def form_factor_path(self):
        return f"../form_factors/QCDark/{self.material}_final.hdf5"
