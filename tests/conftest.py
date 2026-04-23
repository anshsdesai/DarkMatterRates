import os
import sys
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_QCDARK_ROOT = os.path.abspath(os.path.join(REPO_ROOT, '..', 'QCDark'))
UPSTREAM_QEDARK_ROOT = os.path.abspath(os.path.join(REPO_ROOT, '..', 'QEdark', 'QEdark-python'))
UPSTREAM_QCDARK2_ROOT = os.path.abspath(os.path.join(REPO_ROOT, '..', 'QCDark2'))
UPSTREAM_WIMPRATES_ROOT = os.path.abspath(os.path.join(REPO_ROOT, '..', 'wimprates'))

BENCHMARK_MASSES_MEV = [10.0, 100.0, 1000.0]
BENCHMARK_FDM = [0, 2]
DEFAULT_NE_BINS = [1, 2, 3, 4, 5]

# The outer torchinterp1d/ directory (no __init__.py) shadows the installed
# package when pytest runs from the repo root. Insert the inner package path
# first so that "from torchinterp1d import interp1d" resolves correctly.
_torchinterp1d_inner = os.path.join(REPO_ROOT, 'torchinterp1d')
if _torchinterp1d_inner not in sys.path:
    sys.path.insert(0, _torchinterp1d_inner)

def hdf5_path(relative):
    return os.path.join(REPO_ROOT, relative)

# Skip markers for files that require data the user must download
def qcdark2_file(material, variant='composite'):
    suffix = {'composite': 'comp', 'lfe': 'lfe', 'nolfe': 'nolfe'}[variant]
    name = 'diamond' if material == 'Diamond' else material
    return hdf5_path(f'form_factors/QCDark2/{variant}/{name}_{suffix}.h5')

def upstream_qcdark2_file(material, variant='composite'):
    suffix = {'composite': 'comp', 'lfe': 'lfe', 'nolfe': 'nolfe'}[variant]
    name = 'diamond' if material == 'Diamond' else material
    return os.path.join(UPSTREAM_QCDARK2_ROOT, 'dielectric_functions', variant, f'{name}_{suffix}.h5')

def requires_qcdark2(material='Si', variant='composite'):
    path = qcdark2_file(material, variant)
    return pytest.mark.skipif(
        not os.path.exists(path),
        reason=f"QCDark2 file not found: {path}. Download from https://github.com/meganhott/QCDark2"
    )

def requires_upstream_qcdark2(material='Si', variant='composite'):
    path = upstream_qcdark2_file(material, variant)
    return pytest.mark.skipif(
        not os.path.exists(path),
        reason=f"Upstream QCDark2 file not found: {path}"
    )

def requires_upstream_qcdark():
    path = os.path.join(UPSTREAM_QCDARK_ROOT, 'dark_matter_rates.py')
    return pytest.mark.skipif(
        not os.path.exists(path),
        reason=f"QCDark not found at {path}"
    )

def requires_upstream_qedark():
    path = os.path.join(UPSTREAM_QEDARK_ROOT, 'Si_f2.txt')
    return pytest.mark.skipif(
        not os.path.exists(path),
        reason=f"QEDark data not found at {path}"
    )

def requires_upstream_wimprates():
    path = os.path.join(UPSTREAM_WIMPRATES_ROOT, 'wimprates', 'electron.py')
    return pytest.mark.skipif(
        not os.path.exists(path),
        reason=f"wimprates not found at {path}"
    )

def requires_qcdark(material='Si'):
    path = hdf5_path(f'form_factors/QCDark/{material}_final.hdf5')
    return pytest.mark.skipif(
        not os.path.exists(path),
        reason=f"QCDark file not found: {path}"
    )

def requires_qedark(material='Si'):
    path = hdf5_path(f'form_factors/QEDark/{material}_f2.txt')
    return pytest.mark.skipif(
        not os.path.exists(path),
        reason=f"QEDark file not found: {path}"
    )

def requires_wimprates(material='Xe'):
    path = hdf5_path(f'form_factors/wimprates/{material}_dme_ionization_ff.pkl')
    return pytest.mark.skipif(
        not os.path.exists(path),
        reason=f"wimprates form factor file not found: {path}"
    )

@pytest.fixture(autouse=False, scope='session')
def fix_units():
    """Deterministic unit scales for cross-run comparable parity tests."""
    import numericalunits as nu

    nu.reset_units('SI')
    yield
    nu.reset_units()

@pytest.fixture(params=[42, 137, 2718])
def random_nu_seed(request):
    import numericalunits as nu

    nu.reset_units(request.param)
    yield request.param
    nu.reset_units('SI')
