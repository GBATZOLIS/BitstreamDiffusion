"""Decode M0 dashboard LFQ tokens, score geometry, and build overlay assets.

Run this with the dedicated DPLM environment after ``evaluate_m0_dashboard.py``.
The reference coordinates are decoded from the released reference LFQ tokens;
they are a tokenizer-space reference, not experimental native coordinates.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / ".trentinium/eval_shims", ROOT / "external/dplm/src", ROOT / "external/dplm/vendor/openfold"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))


def _prepare_lean_byprot() -> None:
    root = ROOT / "external/dplm/src/byprot"
    def package(name: str, path: Path):
        module = types.ModuleType(name); module.__path__ = [str(path)]; sys.modules[name] = module; return module
    byprot = package("byprot", root); models = package("byprot.models", root / "models"); data = package("byprot.datamodules", root / "datamodules")
    models.MODEL_REGISTRY = {}; data.DATAMODULE_REGISTRY = {}
    def registrar(registry):
        def register(name):
            def decorate(cls): registry[name] = cls; return cls
            return decorate
        return register
    models.register_model = registrar(models.MODEL_REGISTRY); data.register_datamodule = registrar(data.DATAMODULE_REGISTRY)
    byprot.models = models; byprot.datamodules = data


def _dataclass_compat() -> None:
    original = dataclasses.dataclass
    def compatible(cls=None, **kwargs):
        def wrap(candidate):
            try: return original(candidate, **kwargs)
            except ValueError as exc:
                if "mutable default" not in str(exc): raise
                for name in getattr(candidate, "__annotations__", {}):
                    value = getattr(candidate, name, dataclasses.MISSING); kind = getattr(value, "__class__", None)
                    if value is not dataclasses.MISSING and kind is not None and getattr(kind, "__hash__", object.__hash__) is None: kind.__hash__ = object.__hash__
                return original(candidate, **kwargs)
        return wrap if cls is None else wrap(cls)
    dataclasses.dataclass = compatible


def _superpose(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    left = mobile[:, 1].mean(0); right = reference[:, 1].mean(0)
    covariance = (mobile[:, 1] - left).T @ (reference[:, 1] - right)
    u, _, vt = np.linalg.svd(covariance); rotation = u @ vt
    if np.linalg.det(rotation) < 0: u[:, -1] *= -1; rotation = u @ vt
    return (mobile - left) @ rotation + right


def _pdb(path: Path, coords: np.ndarray, label: str) -> None:
    atoms = ("N", "CA", "C", "O"); lines = [f"REMARK 900 {label}"]
    serial = 1
    for residue, row in enumerate(coords, start=1):
        for atom, xyz in zip(atoms, row):
            element = atom[0]
            lines.append(f"ATOM  {serial:5d} {atom:>4s} ALA A{residue:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}")
            serial += 1
    lines.extend(["TER", "END", ""]); path.write_text("\n".join(lines), encoding="utf-8")


def _summarize(records: list[dict]) -> dict:
    out = {"n": len(records)}
    for key in records[0]:
        if isinstance(records[0][key], (int, float)):
            values = np.asarray([r[key] for r in records], dtype=np.float64)
            out[key] = {"mean": float(values.mean()), "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/proteins/m0_v1/protein_eval/dashboard/structure_tokens.npz"))
    parser.add_argument("--tokenizer", default="datasets/dplm_struct_tokenizer")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); out_dir = args.input.parent; pdb_dir = out_dir / "structures"; pdb_dir.mkdir(parents=True, exist_ok=True)
    _dataclass_compat(); _prepare_lean_byprot()
    from evaluation.proteins.dplm_struct_tokenizer import load_struct_tokenizer
    from evaluation.proteins.io import atomic_json
    from evaluation.proteins.structure_metrics import backbone_geometry, lddt, radius_of_gyration, rmsd, secondary_structure_fractions, tm_score
    tokenizer = load_struct_tokenizer(args.tokenizer, device=args.device)
    archive = np.load(args.input, allow_pickle=False); lengths = archive["lengths"].astype(int); names = ["reference", "forward", "shuffled", "unconditional", "marginal"]
    decoded = {name: [] for name in names}
    for i, length in enumerate(lengths):
        print(f"decode [{i + 1:02d}/{len(lengths)}] L={length}", flush=True)
        for name in names: decoded[name].append(tokenizer.decode(archive[name][i, :length].astype(np.uint16)))

    conditions = {}; per_condition = {}
    for name in names[1:]:
        rows = []
        for i, length in enumerate(lengths):
            ref = decoded["reference"][i]; pred = decoded[name][i]; aligned = _superpose(pred, ref); decoded[name][i] = aligned
            geometry = backbone_geometry(aligned); ca = aligned[:, 1]
            rows.append({
                "index": i, "length": int(length), "tm_score_to_decoded_reference": float(tm_score(ca, ref[:, 1])),
                "rmsd_to_decoded_reference": float(rmsd(ca, ref[:, 1], superpose=True)), "lddt_to_decoded_reference": float(lddt(aligned, ref)),
                "radius_of_gyration": float(radius_of_gyration(ca)), "chain_breaks": int(geometry["chain_break_count"]), "ca_clashes": int(geometry["clash_count"]),
                "n_ca_bond_mae": float(geometry["n_ca"]["deviation"]), "ca_c_bond_mae": float(geometry["ca_c"]["deviation"]), "c_n_bond_mae": float(geometry["c_n"]["deviation"]),
                **{f"ss_{key}": float(value) for key, value in secondary_structure_fractions(aligned).items()},
            })
        conditions[name] = _summarize(rows); per_condition[name] = rows

    forward_tm = np.asarray([r["tm_score_to_decoded_reference"] for r in per_condition["forward"]])
    order = np.argsort(forward_tm); exemplar_indices = sorted(set([int(order[0]), int(order[len(order)//2]), int(order[-1])]))
    overlay_payload = []
    colors = {"reference": "#17211d", "forward": "#087f76", "shuffled": "#d06a2d", "unconditional": "#7b4fc4", "marginal": "#1769aa"}
    for index in exemplar_indices:
        item = {"index": index, "length": int(lengths[index]), "forward_tm": float(forward_tm[index]), "tracks": {}}
        for name in names:
            ca = decoded[name][index][:, 1].astype(float)
            item["tracks"][name] = {"color": colors[name], "ca": np.round(ca, 3).tolist()}
            _pdb(pdb_dir / f"target_{index:02d}_{name}.pdb", decoded[name][index], f"M0 {name}; decoded LFQ-token coordinates")
        overlay_payload.append(item)
    (out_dir / "overlays.js").write_text("window.M0_OVERLAYS = " + json.dumps(overlay_payload, separators=(",", ":")) + ";\n", encoding="utf-8")

    maximum = int(lengths.max()); coordinate_archive = {"lengths": lengths}
    for name in names:
        array = np.zeros((len(lengths), maximum, 4, 3), dtype=np.float32)
        for i, coords in enumerate(decoded[name]): array[i, :len(coords)] = coords
        coordinate_archive[name] = array
    np.savez_compressed(out_dir / "decoded_coordinates.npz", **coordinate_archive)
    payload = {
        "schema_version": 1, "reference_semantics": "reference LFQ tokens decoded by the same frozen tokenizer; not experimental/native coordinates",
        "tokenizer": str(args.tokenizer), "conditions": conditions, "per_condition": per_condition, "exemplar_indices": exemplar_indices,
        "tm_score_note": "tmtools exact when installed; otherwise repository Kabsch fallback", "overlay_asset": "overlays.js",
    }
    atomic_json(out_dir / "geometry_metrics.json", payload)
    print(json.dumps(conditions, indent=2))


if __name__ == "__main__":
    main()
