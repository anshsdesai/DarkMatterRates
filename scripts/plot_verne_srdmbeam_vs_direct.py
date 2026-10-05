"""Import and plot Verne SRDMBeam modulated rates against direct SRDM rates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from DMeRates.DMeRate import DMeRate
from DMeRates.data.registry import DataRegistry
from DMeRates.srdm.flux_loader import (
    available_srdmbeam_ring_indices,
    load_srdmbeam_metadata,
)
from scripts.import_verne_srdmbeam_scan import import_scan_manifest
import numericalunits as nu


DEFAULT_DIRECT_SOURCE_ROOT = Path(
    "/home/ansh/Projects/SENSEI/DaMaSCUS-SUN/results/srdm_fdmq2_source_grid_v1"
)


@dataclass(frozen=True)
class RateComparison:
    point_tag: str
    mX_MeV: float
    sigma_e_cm2: float
    FDMn: int
    ring_indices: list[int]
    angles_deg: np.ndarray
    direct_rates: np.ndarray
    modulated_rates: np.ndarray

    @property
    def normalized_rates(self) -> np.ndarray:
        return self.modulated_rates / self.direct_rates.reshape(1, -1)


def _load_scan_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"No Verne scan manifest found: {manifest_path}")
    rows = json.loads(manifest_path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Verne scan manifest must be a JSON list: {manifest_path}")
    return rows


def _parse_ne(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--ne must contain at least one electron bin")
    return values


def _as_rate_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value / (1.0 / (nu.kg * nu.year)), dtype=float).reshape(-1)


def _point_label(row: dict, index: int) -> str:
    return str(row.get("point_tag") or row.get("index") or f"row_{index}")


def _selected_manifest_rows(
    manifest_path: Path,
    *,
    include_incomplete: bool,
    max_points: int | None,
) -> list[tuple[int, dict, str]]:
    selected = []
    for index, row in enumerate(_load_scan_manifest(manifest_path)):
        point_tag = _point_label(row, index)
        if row.get("status") != "complete" and not include_incomplete:
            print(f"skipped: {point_tag} (status={row.get('status')!r})")
            continue
        selected.append((index, row, point_tag))
        if max_points is not None and len(selected) >= max_points:
            break
    return selected


def _slug(value: str) -> str:
    value = value.strip().replace("+", "p").replace("-", "m")
    value = re.sub(r"[^A-Za-z0-9_.=]+", "_", value)
    return value.strip("_")


def _angles_for_rings(metadata: dict, ring_indices: list[int]) -> np.ndarray:
    representative = metadata.get("angle_representative_deg")
    if representative is None:
        representative = metadata.get("file_isoangle_deg")
    if representative is None:
        raise ValueError(
            f"SRDMBeam metadata has no representative angle array: "
            f"{metadata.get('parameter_dir')}"
        )
    angles = np.asarray(representative, dtype=float)
    return np.asarray([angles[int(ring)] for ring in ring_indices], dtype=float)


def _calculate_comparison(
    dmrates: DMeRate,
    row: dict,
    *,
    point_tag: str,
    ne_values: list[int],
    halo_data_root: Path,
    mediator_spin: str,
    screening: str | None,
    variant: str | None,
) -> RateComparison:
    mX_MeV = float(row["mDM_MeV"])
    sigma_e_cm2 = float(row["sigma_e_cm2"])
    FDMn = int(row["FDMn"])

    metadata = load_srdmbeam_metadata(
        mX_MeV,
        sigma_e_cm2,
        FDMn,
        modulated_source="Verne",
        base_data_dir=halo_data_root,
    )
    ring_indices = available_srdmbeam_ring_indices(
        mX_MeV,
        sigma_e_cm2,
        FDMn,
        modulated_source="Verne",
        base_data_dir=halo_data_root,
    )
    if not ring_indices:
        raise FileNotFoundError(
            f"No Verne SRDMBeam rings registered for {point_tag}: "
            f"mX={mX_MeV} MeV, sigma_e={sigma_e_cm2}, FDMn={FDMn}"
        )
    angles_deg = _angles_for_rings(metadata, ring_indices)
    order = np.argsort(angles_deg)
    ring_indices = [ring_indices[int(i)] for i in order]
    angles_deg = angles_deg[order]

    dmrates.update_crosssection(sigma_e_cm2)
    direct = dmrates.calculate_rates(
        mX_MeV,
        "srdm",
        FDMn,
        ne_values,
        sigma_e=sigma_e_cm2,
        mediator_spin=mediator_spin,
        screening=screening,
        variant=variant,
    )
    direct_rates = _as_rate_array(direct)

    modulated_rows = []
    for ring_index in ring_indices:
        value = dmrates.calculate_rates(
            mX_MeV,
            "srdm_modulated",
            FDMn,
            ne_values,
            isoangle=int(ring_index),
            modulated_source="Verne",
            sigma_e=sigma_e_cm2,
            mediator_spin=mediator_spin,
            screening=screening,
            variant=variant,
            srdm_base_data_dir=halo_data_root,
        )
        modulated_rows.append(_as_rate_array(value))
    modulated_rates = np.vstack(modulated_rows)

    if modulated_rates.shape[1] != len(ne_values):
        raise ValueError(
            f"Expected {len(ne_values)} ne columns for {point_tag}, "
            f"got shape {modulated_rates.shape}"
        )
    if direct_rates.shape[0] != len(ne_values):
        raise ValueError(
            f"Expected {len(ne_values)} direct rates for {point_tag}, "
            f"got shape {direct_rates.shape}"
        )

    return RateComparison(
        point_tag=point_tag,
        mX_MeV=mX_MeV,
        sigma_e_cm2=sigma_e_cm2,
        FDMn=FDMn,
        ring_indices=ring_indices,
        angles_deg=angles_deg,
        direct_rates=direct_rates,
        modulated_rates=modulated_rates,
    )


def _write_point_csv(path: Path, comparison: RateComparison, ne_values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = comparison.normalized_rates
    with path.open("w", newline="") as handle:
        fieldnames = ["ring_index", "angle_deg"]
        for ne in ne_values:
            fieldnames.extend(
                [
                    f"rate_ne{ne}_per_kg_year",
                    f"direct_rate_ne{ne}_per_kg_year",
                    f"normalized_to_direct_ne{ne}",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, ring_index in enumerate(comparison.ring_indices):
            row = {
                "ring_index": ring_index,
                "angle_deg": comparison.angles_deg[row_idx],
            }
            for ne_idx, ne in enumerate(ne_values):
                row[f"rate_ne{ne}_per_kg_year"] = comparison.modulated_rates[row_idx, ne_idx]
                row[f"direct_rate_ne{ne}_per_kg_year"] = comparison.direct_rates[ne_idx]
                row[f"normalized_to_direct_ne{ne}"] = normalized[row_idx, ne_idx]
            writer.writerow(row)


def _plot_point(path: Path, comparison: RateComparison, ne_values: list[int]) -> None:
    import matplotlib.pyplot as plt

    if len(ne_values) != 2:
        raise ValueError(
            "Four-panel QA plots expect exactly two ne bins; pass --ne 1,2."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = comparison.normalized_rates
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    fig.suptitle(
        (
            f"{comparison.point_tag}\n"
            f"mX={comparison.mX_MeV:g} MeV, "
            f"sigma_e={comparison.sigma_e_cm2:.1e} cm^2, FDMn={comparison.FDMn}"
        ),
        fontsize=12,
    )

    for ne_idx, ne in enumerate(ne_values):
        raw_axis = axes[ne_idx, 0]
        frac_axis = axes[ne_idx, 1]

        raw_axis.plot(
            comparison.angles_deg,
            comparison.modulated_rates[:, ne_idx],
            marker="o",
            linewidth=1.6,
            markersize=3.2,
            color="tab:blue",
            label="Verne modulated rate",
        )
        raw_axis.axhline(
            comparison.direct_rates[ne_idx],
            linestyle="--",
            linewidth=1.2,
            color="black",
            alpha=0.75,
            label="direct SRDM rate",
        )
        raw_axis.set_title(f"n_e = {ne}: raw rate")
        raw_axis.set_ylabel("Rate [events / kg / year]")
        raw_axis.grid(True, alpha=0.3)
        raw_axis.legend(fontsize=8, loc="best")

        frac_axis.plot(
            comparison.angles_deg,
            normalized[:, ne_idx],
            marker="s",
            linewidth=1.5,
            markersize=3.0,
            color="tab:orange",
            label="Verne / direct",
        )
        frac_axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        frac_axis.set_title(f"n_e = {ne}: fractional modulation")
        frac_axis.set_ylabel("Verne / direct SRDM")
        frac_axis.grid(True, alpha=0.3)
        frac_axis.legend(fontsize=8, loc="best")

    for axis in axes[-1, :]:
        axis.set_xlabel("SRDMBeam representative angle [deg]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Verne daily modulated SRDMBeam rates against matching direct "
            "DaMaSCUS-SUN SRDM rates for each completed point in a Verne scan."
        )
    )
    parser.add_argument("manifest_path", type=Path, help="Verne scan manifest JSON.")
    parser.add_argument(
        "--halo-data-root",
        type=Path,
        default=Path("halo_data"),
        help="DMeRates halo-data root containing modulated/ and srdm/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/verne_srdmbeam_vs_direct"),
        help="Directory for PNG and CSV QA outputs.",
    )
    parser.add_argument("--material", default="Si", help="Target material.")
    parser.add_argument(
        "--form-factor-type",
        default="qcdark2",
        help=(
            "DMeRate form_factor_type. Defaults to qcdark2 for Si production "
            "checks. Noble gases use wimprates automatically."
        ),
    )
    parser.add_argument(
        "--ne",
        default="1,2",
        help="Comma-separated electron-count bins, e.g. '1' or '1,2'.",
    )
    parser.add_argument("--mediator-spin", default="vector")
    parser.add_argument(
        "--screening",
        default="rpa",
        help="Screening mode passed to DMeRate; QCDark2 requires rpa or none.",
    )
    parser.add_argument(
        "--variant",
        default="composite",
        help="QCDark2 dielectric variant.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Attempt manifest rows whose status is not complete.",
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help=(
            "Skip the pre-plot import step. By default this script imports "
            "Verne SRDMBeam files and matching direct SRDM files first."
        ),
    )
    parser.add_argument(
        "--direct-source-root",
        type=Path,
        default=DEFAULT_DIRECT_SOURCE_ROOT,
        help=(
            "DaMaSCUS-SUN source-grid root used for direct SRDM imports. "
            "It should contain scan_manifest.json and points/."
        ),
    )
    parser.add_argument(
        "--no-import-verify",
        action="store_true",
        help="Skip loader-level verification during the import step.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of manifest points plotted.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")

    halo_root = args.halo_data_root.resolve()
    DataRegistry.halo_root = halo_root
    ne_values = _parse_ne(args.ne)
    selected_rows = _selected_manifest_rows(
        args.manifest_path,
        include_incomplete=args.include_incomplete,
        max_points=args.max_points,
    )

    if not args.skip_import:
        print(f"importing Verne/direct SRDM points into {halo_root}")
        import_results = import_scan_manifest(
            args.manifest_path,
            halo_root,
            verify=not args.no_import_verify,
            include_incomplete=args.include_incomplete,
            with_direct_srdm=True,
            direct_source_root=args.direct_source_root,
        )
        imported = [result for result in import_results if result.status == "imported"]
        skipped = [result for result in import_results if result.status == "skipped"]
        print(f"import complete: {len(imported)} imported, {len(skipped)} skipped")

    dmrates = DMeRate(args.material, form_factor_type=args.form_factor_type)

    figures_dir = args.output_dir / "figures"
    csv_dir = args.output_dir / "csv"
    summary_rows: list[dict] = []

    for _index, row, point_tag in selected_rows:
        comparison = _calculate_comparison(
            dmrates,
            row,
            point_tag=point_tag,
            ne_values=ne_values,
            halo_data_root=halo_root,
            mediator_spin=args.mediator_spin,
            screening=args.screening,
            variant=args.variant,
        )
        stem = _slug(point_tag)
        _write_point_csv(csv_dir / f"{stem}.csv", comparison, ne_values)
        _plot_point(figures_dir / f"{stem}.png", comparison, ne_values)

        normalized = comparison.normalized_rates
        for ne_idx, ne in enumerate(ne_values):
            summary_rows.append(
                {
                    "point_tag": point_tag,
                    "mX_MeV": comparison.mX_MeV,
                    "sigma_e_cm2": comparison.sigma_e_cm2,
                    "FDMn": comparison.FDMn,
                    "material": args.material,
                    "ne": ne,
                    "direct_rate_per_kg_year": comparison.direct_rates[ne_idx],
                    "modulated_min_per_kg_year": np.min(comparison.modulated_rates[:, ne_idx]),
                    "modulated_max_per_kg_year": np.max(comparison.modulated_rates[:, ne_idx]),
                    "normalized_min": np.min(normalized[:, ne_idx]),
                    "normalized_max": np.max(normalized[:, ne_idx]),
                    "peak_to_peak_fraction": (
                        np.max(comparison.modulated_rates[:, ne_idx])
                        - np.min(comparison.modulated_rates[:, ne_idx])
                    )
                    / comparison.direct_rates[ne_idx],
                    "figure": str(figures_dir / f"{stem}.png"),
                    "csv": str(csv_dir / f"{stem}.csv"),
                }
            )
        print(f"plotted: {point_tag} -> {figures_dir / f'{stem}.png'}")

    _write_summary(args.output_dir / "summary.csv", summary_rows)
    print(f"summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
