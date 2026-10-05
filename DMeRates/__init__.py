import sys as _sys

# Recorded BEFORE any DMeRates submodule pulls in numericalunits, so
# Constants.py can tell whether user code imported numericalunits first.
# Constants.py re-randomizes the numericalunits base unit scales, which
# invalidates any nu-derived quantities created before this import.
_NUMERICALUNITS_PREIMPORTED = 'numericalunits' in _sys.modules

# Import Constants eagerly so the unit randomization happens at one
# predictable point: the first `import DMeRates` (or any submodule).
from . import Constants  # noqa: E402,F401

__version__ = "0.1.0"
