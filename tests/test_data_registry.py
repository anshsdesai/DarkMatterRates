import os
from pathlib import Path

from DMeRates.data import registry as reg


def test_repo_root_is_absolute():
    assert reg._REPO_ROOT.is_absolute()
    assert reg._REPO_ROOT.name == "DarkMatterRates"


def test_qcdark2_root_precedence_env_var():
    original = os.environ.get("DMERATES_QCDARK2_ROOT")
    try:
        os.environ["DMERATES_QCDARK2_ROOT"] = "/tmp/custom-qcdark2-root"
        resolved = reg._default_qcdark2_root()
        assert resolved == Path("/tmp/custom-qcdark2-root")
    finally:
        if original is None:
            os.environ.pop("DMERATES_QCDARK2_ROOT", None)
        else:
            os.environ["DMERATES_QCDARK2_ROOT"] = original
