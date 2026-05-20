import argparse
import os
from pathlib import Path

import numpy as np
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


CRYSTAL_SYSTEMS = [
    "cubic",
    "tetragonal",
    "orthorhombic",
    "hexagonal",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", default="data/processed/xrd_patterns.npz")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--theta-min", type=float, default=10.0)
    parser.add_argument("--theta-max", type=float, default=80.0)
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--max-results", type=int, default=2000)

    return parser.parse_args()


def get_api_key():
    key = os.getenv("MP_API_KEY") or os.getenv("PMG_MAPI_KEY")

    if key is None:
        raise RuntimeError(
            "No Materials Project API key found. "
            "Set MP_API_KEY or PMG_MAPI_KEY in your environment."
        )

    return key


def crystal_system_from_structure(structure):
    analyser = SpacegroupAnalyzer(structure, symprec=0.1)
    return analyser.get_crystal_system()


def pattern_to_grid(pattern, theta_grid):
    intensities = np.zeros_like(theta_grid, dtype="float32")

    for theta, intensity in zip(pattern.x, pattern.y):
        idx = np.argmin(np.abs(theta_grid - theta))
        intensities[idx] += intensity

    max_intensity = intensities.max()

    if max_intensity > 0:
        intensities = intensities / max_intensity

    return intensities


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = get_api_key()

    theta_grid = np.linspace(
        args.theta_min,
        args.theta_max,
        args.n_points,
        dtype="float32",
    )

    xrd_calculator = XRDCalculator(wavelength="CuKa")

    X = []
    y = []
    material_ids = []
    formulas = []

    counts = {system: 0 for system in CRYSTAL_SYSTEMS}

    print("\nBuilding lightweight XRD dataset")
    print("------------------------------------------------------------")
    print(f"Target classes: {CRYSTAL_SYSTEMS}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"2θ range: {args.theta_min} to {args.theta_max}")
    print(f"Grid points: {args.n_points}")

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            is_stable=True,
            fields=[
                "material_id",
                "formula_pretty",
                "structure",
            ],
            num_chunks=None,
            chunk_size=1000,
        )

        print("\nQuery returned structures. Simulating XRD patterns...")

        for doc in docs:
            if all(count >= args.samples_per_class for count in counts.values()):
                break

            try:
                structure = doc.structure
                crystal_system = crystal_system_from_structure(structure)

                if crystal_system not in counts:
                    continue

                if counts[crystal_system] >= args.samples_per_class:
                    continue

                pattern = xrd_calculator.get_pattern(
                    structure,
                    two_theta_range=(args.theta_min, args.theta_max),
                )

                intensity_grid = pattern_to_grid(pattern, theta_grid)

                if intensity_grid.max() == 0:
                    continue

                X.append(intensity_grid)
                y.append(crystal_system)
                material_ids.append(str(doc.material_id))
                formulas.append(doc.formula_pretty)

                counts[crystal_system] += 1

                total = sum(counts.values())

                if total % 25 == 0:
                    print(f"Collected {total} patterns | {counts}")

            except Exception as exc:
                print(f"Skipping {getattr(doc, 'material_id', 'unknown')}: {exc}")

            if sum(counts.values()) >= args.max_results:
                break

    X = np.asarray(X, dtype="float32")
    y = np.asarray(y)

    if len(X) == 0:
        raise RuntimeError("No XRD patterns were collected.")

    print("\nFinal class counts")
    print("------------------------------------------------------------")
    for label in CRYSTAL_SYSTEMS:
        print(f"{label:14s}: {(y == label).sum()}")

    np.savez(
        output_path,
        X=X,
        y=y,
        theta=theta_grid,
        labels=np.asarray(CRYSTAL_SYSTEMS),
        material_ids=np.asarray(material_ids),
        formulas=np.asarray(formulas),
    )

    print(f"\nSaved dataset to: {output_path}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    main()